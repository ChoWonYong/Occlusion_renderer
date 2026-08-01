from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from common.io_video import group_annotations_by_image, group_frames_by_video
from mining.tracker import create_boxmot_bytetrack
from train.run import AUG_LEVELS


COCO_TO_PROJECT_CLASS = {
    0: 2,  # person -> person
    1: 3,  # bicycle -> bicycle
    2: 0,  # car -> car
    3: 3,  # motorcycle -> bicycle
    5: 1,  # bus -> truck
    7: 1,  # truck -> truck
}


# Checkpoint file for each saved epoch. ep60 = final EMA weights; ep50 = the
# last-mosaic-epoch snapshot the trainer saves at the no_aug boundary (epoch 50).
EPOCH_CKPT = {
    "ep60": "latest_ckpt.pth.tar",
    "ep50": "last_mosaic_epoch_ckpt.pth.tar",
}


# KITTI label types that must not be scored as background. Its devkit removes
# detections landing on these instead of counting them as false positives —
# "do not count Vans as false positives for cars or Sitting Persons as wrong
# positives for Pedestrians due to their similarity in appearance. (All ignored
# objects are considered as DontCare areas.)" — because they are either
# unlabelled regions or a near-duplicate of an evaluated class.
#
# ``classes.kitti_map`` drops them at conversion, which left them as plain
# background: a detection there had no gt to match and became a false positive.
# Measured on the eval split, that was 53.8% of unmatched person detections and
# 51.6% of car ones.
#
# Value is the classes the region applies to; ``None`` means every class. Van is
# deliberately absent — this project maps it to ``car`` as a positive, which is a
# knowing departure from KITTI's protocol, not an oversight.
IGNORE_REGIONS: dict[str, tuple[str, ...] | None] = {
    "DontCare": None,
    # KITTI Tracking writes "Person" where the object benchmark writes
    # "Person_sitting"; classes.kitti_map only lists the latter, so neither the
    # map nor this table can rely on just one spelling.
    "Person": ("person",),
    "Person_sitting": ("person",),
}

# TrackEval's MOTChallenge preprocessing matches tracker dets against every gt
# row and drops the ones that land on a distractor class, so ignore rows are
# written with this class id. Real gt keeps id 1 (pedestrian), and the distractor
# rows are dropped from the gt set before the metrics are computed.
TRACKEVAL_DISTRACTOR_CLASS = 8  # 'distractor' in MotChallenge2DBox
TRACKEVAL_PEDESTRIAN_CLASS = 1
# gt track ids only have to be unique within a timestep; KITTI's are small
# non-negative ints, so this offset cannot collide with them.
IGNORE_ID_BASE = 900000


def _experiment_dir_name(
    condition: str, aug: str, seed: int, paste_mode: str | None = None, tag: str | None = None
) -> str:
    """Must match ``train.run._experiment_name`` or the checkpoint will not be found."""
    label = "kitti_finetuned" if condition == "kitti" else condition
    suffix = f"_{paste_mode}" if condition == "treatment" and paste_mode else ""
    tag_part = f"_{tag}" if tag else ""
    return f"phase1_{label}{suffix}{tag_part}_{aug}_seed{seed}"


def _checkpoint_path(
    config: dict[str, Any],
    path: Path,
    condition: str,
    seed: int,
    aug: str,
    epoch: str,
    paste_mode: str | None = None,
    tag: str | None = None,
) -> Path:
    output = resolve_path(path.parent, config["train"]["output_dir"])
    return output / _experiment_dir_name(condition, aug, seed, paste_mode, tag) / EPOCH_CKPT[epoch]


def _ignore_regions_by_frame(
    kitti_root: Path, sequence: str, class_names: Sequence[str]
) -> dict[int, list[tuple[str, list[float]]]]:
    """KITTI ignore boxes for one sequence, keyed by 1-based frame number.

    Read straight from the raw labels rather than the converted dataset: these
    rows are an evaluation-protocol concern, and routing them through the COCO
    conversion would put them in front of the *training* set builders too.
    """
    from data.kitti_tracking import load_sequence_labels

    regions: dict[int, list[tuple[str, list[float]]]] = {}
    for frame_index, objects in load_sequence_labels(kitti_root / "training" / "label_02" / f"{sequence}.txt").items():
        for obj in objects:
            if obj.category not in IGNORE_REGIONS:
                continue
            applies_to = IGNORE_REGIONS[obj.category]
            x1, y1, x2, y2 = obj.bbox_xyxy
            if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
                continue
            for name in class_names if applies_to is None else applies_to:
                if name not in class_names:
                    continue
                regions.setdefault(frame_index + 1, []).append(
                    (name, [x1, y1, x2 - x1, y2 - y1])
                )
    return regions


def _load_model(
    config: dict[str, Any],
    config_path_value: Path,
    checkpoint: Path,
    device: str,
    model_source: str,
):
    bytetrack_repo = config_path(config, config_path_value, "paths", "bytetrack_repo")
    if str(bytetrack_repo) not in sys.path:
        sys.path.insert(0, str(bytetrack_repo))
    from yolox.exp import get_exp
    import torch

    if model_source == "coco-pretrained":
        exp_file = bytetrack_repo / "exps" / "default" / "yolox_x.py"
        class_id_map = COCO_TO_PROJECT_CLASS
    elif model_source == "finetuned":
        os.environ["KDS_NUM_CLASSES"] = str(len(config["classes"]["names"]))
        os.environ["KDS_INPUT_SIZE"] = ",".join(
            str(value) for value in config["train"]["input_size"]
        )
        exp_file = resolve_path(config_path_value.parent, config["train"]["exp_file"])
        class_id_map = {index: index for index in range(len(config["classes"]["names"]))}
    else:
        raise ValueError("model_source must be coco-pretrained or finetuned")
    exp = get_exp(str(exp_file), None)
    exp.test_size = tuple(int(value) for value in config["train"]["input_size"])
    exp.test_conf = float(config.get("detector", {}).get("test_conf", 0.01))
    exp.nmsthre = float(config.get("detector", {}).get("nms_thresh", 0.70))
    model = exp.get_model()
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    use_half = bool(config.get("detector", {}).get("fp16", True)) and device == "cuda"
    if use_half:
        model.half()
    return model, exp, class_id_map, use_half


def _remap_detection_classes(
    detections: np.ndarray, class_id_map: dict[int, int]
) -> np.ndarray:
    if detections.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    keep = np.array([int(value) in class_id_map for value in detections[:, 5]], dtype=bool)
    remapped = detections[keep].copy()
    if remapped.size:
        remapped[:, 5] = [class_id_map[int(value)] for value in remapped[:, 5]]
    return remapped.astype(np.float32, copy=False)


def _detect(
    model,
    frame: np.ndarray,
    exp,
    device: str,
    class_id_map: dict[int, int],
    use_half: bool,
) -> np.ndarray:
    import torch
    from yolox.data import ValTransform
    from yolox.utils import postprocess

    transform = ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    processed, _ = transform(frame, None, exp.test_size)
    tensor = torch.from_numpy(processed).unsqueeze(0).float().to(device)
    if use_half:
        tensor = tensor.half()
    with torch.no_grad():
        output = postprocess(model(tensor), exp.num_classes, exp.test_conf, exp.nmsthre)[0]
    if output is None:
        return np.empty((0, 6), dtype=np.float32)
    output = output.detach().cpu().numpy()
    ratio = min(exp.test_size[0] / frame.shape[0], exp.test_size[1] / frame.shape[1])
    boxes = output[:, :4] / ratio
    confidence = output[:, 4] * output[:, 5]
    classes = output[:, 6]
    detections = np.column_stack([boxes, confidence, classes]).astype(np.float32)
    return _remap_detection_classes(detections, class_id_map)


def _write_trackeval_layout(
    root: Path,
    run_name: str,
    class_names: list[str],
    videos: list[dict[str, Any]],
    gt_rows: dict[str, dict[str, list[str]]],
    prediction_rows: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    layouts = []
    for class_name in class_names:
        benchmark = f"KDS_{class_name.upper()}"
        split_name = f"{benchmark}-eval"
        gt_root = root / "gt"
        tracker_root = root / "trackers"
        sequence_names = [str(video["name"]) for video in videos]
        seqmap = root / "seqmaps" / f"{split_name}.txt"
        seqmap.parent.mkdir(parents=True, exist_ok=True)
        seqmap.write_text("name\n" + "\n".join(sequence_names) + "\n", encoding="utf-8")
        for video in videos:
            sequence = str(video["name"])
            sequence_gt = gt_root / split_name / sequence
            (sequence_gt / "gt").mkdir(parents=True, exist_ok=True)
            (sequence_gt / "gt" / "gt.txt").write_text(
                "".join(gt_rows[class_name].get(sequence, [])), encoding="utf-8"
            )
            (sequence_gt / "seqinfo.ini").write_text(
                "[Sequence]\n"
                f"name={sequence}\n"
                f"seqLength={int(video['num_frames'])}\n"
                "frameRate=10\n"
                "imWidth=1242\n"
                "imHeight=375\n"
                "imExt=.png\n",
                encoding="utf-8",
            )
            prediction_file = tracker_root / split_name / run_name / "data" / f"{sequence}.txt"
            prediction_file.parent.mkdir(parents=True, exist_ok=True)
            prediction_file.write_text(
                "".join(prediction_rows[class_name].get(sequence, [])), encoding="utf-8"
            )
        layouts.append(
            {
                "class": class_name,
                "benchmark": benchmark,
                "gt_folder": str(gt_root),
                "tracker_folder": str(tracker_root),
                "seqmap": str(seqmap),
                "sequences": sequence_names,
            }
        )
    return layouts


def run(
    config_file: str | Path,
    condition: str,
    seed: int,
    checkpoint_override: str | Path | None = None,
    skip_metrics: bool = False,
    model_source: str = "finetuned",
    aug: str = "full",
    epoch: str = "ep60",
    paste_mode: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    if condition not in {"kitti", "baseline", "treatment"}:
        raise ValueError("condition must be kitti, baseline, or treatment")
    if model_source not in {"coco-pretrained", "finetuned"}:
        raise ValueError("model_source must be coco-pretrained or finetuned")
    if epoch not in EPOCH_CKPT:
        raise ValueError(f"epoch must be one of {sorted(EPOCH_CKPT)}")
    config, path = load_config(config_file)
    if condition == "treatment" and paste_mode is None:
        # Resolve the same way train.run does, so omitting the flag cannot point
        # the evaluation at a different run than the one that was trained.
        paste_mode = str(config["dataset"].get("paste_mode", "replace"))
    if checkpoint_override:
        checkpoint = Path(checkpoint_override).resolve()
    elif model_source == "coco-pretrained":
        checkpoint = config_path(config, path, "paths", "coco_pretrained_yolox_x")
    else:
        checkpoint = _checkpoint_path(config, path, condition, seed, aug, epoch, paste_mode, tag)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"fine-tuned checkpoint not found: {checkpoint}")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, exp, class_id_map, use_half = _load_model(
        config, path, checkpoint, device, model_source
    )
    dataset_root = resolve_path(path.parent, config["dataset"]["output_dir"])
    dataset = load_json(dataset_root / config["dataset"]["eval_json"])
    frames_by_video = group_frames_by_video(dataset)
    annotations_by_image = group_annotations_by_image(dataset)
    video_by_id = {int(video["id"]): video for video in dataset["videos"]}
    class_names = list(config["classes"]["names"])
    category_name = {int(category["id"]): category["name"] for category in dataset["categories"]}
    class_index = {name: index for index, name in enumerate(class_names)}
    gt_rows: dict[str, dict[str, list[str]]] = {name: defaultdict(list) for name in class_names}
    pred_rows: dict[str, dict[str, list[str]]] = {name: defaultdict(list) for name in class_names}
    run_name = (
        "phase1_coco_pretrained"
        if model_source == "coco-pretrained"
        else f"{_experiment_dir_name(condition, aug, seed, paste_mode, tag)}_{epoch}"
    )

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required") from exc
    kitti_root = config_path(config, path, "paths", "kitti_tracking")
    for video_id, frames in sorted(frames_by_video.items()):
        tracker = create_boxmot_bytetrack(config["tracker"], class_names)
        sequence = str(video_by_id[video_id]["name"])
        ignore_by_frame = _ignore_regions_by_frame(kitti_root, sequence, class_names)
        for frame in frames:
            frame_number = int(frame["frame_index"]) + 1
            image = cv2.imread(str(dataset_root / frame["file_name"]))
            if image is None:
                raise FileNotFoundError(dataset_root / frame["file_name"])
            detections = _detect(
                model, image, exp, device, class_id_map, use_half
            )
            tracks = tracker.update(detections, image)
            for track in tracks:
                x1, y1, x2, y2, track_id, confidence, cls = track[:7]
                cls_index = int(round(cls))
                if not 0 <= cls_index < len(class_names):
                    continue
                name = class_names[cls_index]
                pred_rows[name][sequence].append(
                    f"{frame_number},{int(track_id)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{confidence:.5f},1,-1,-1\n"
                )
            for annotation in annotations_by_image.get(int(frame["id"]), []):
                name = category_name[int(annotation["category_id"])]
                if name not in class_index:
                    continue
                x, y, width, height = (float(value) for value in annotation["bbox"])
                gt_rows[name][sequence].append(
                    f"{frame_number},{int(annotation['track_id'])},{x:.2f},{y:.2f},"
                    f"{width:.2f},{height:.2f},1,{TRACKEVAL_PEDESTRIAN_CLASS},1\n"
                )
            for offset, (name, box) in enumerate(ignore_by_frame.get(frame_number, [])):
                x, y, width, height = box
                gt_rows[name][sequence].append(
                    f"{frame_number},{IGNORE_ID_BASE + offset},{x:.2f},{y:.2f},"
                    f"{width:.2f},{height:.2f},1,{TRACKEVAL_DISTRACTOR_CLASS},1\n"
                )

    output_root = resolve_path(path.parent, config["tracker"]["output_dir"]) / run_name
    layouts = _write_trackeval_layout(
        output_root,
        run_name,
        class_names,
        list(video_by_id.values()),
        gt_rows,
        pred_rows,
    )
    metric_commands: list[list[str]] = []
    trackeval_repo = config_path(config, path, "paths", "trackeval_repo")
    for layout in layouts:
        command = [
            sys.executable,
            str(trackeval_repo / "scripts" / "run_mot_challenge.py"),
            "--GT_FOLDER",
            layout["gt_folder"],
            "--TRACKERS_FOLDER",
            layout["tracker_folder"],
            "--BENCHMARK",
            layout["benchmark"],
            "--SPLIT_TO_EVAL",
            "eval",
            "--SEQ_INFO",
            *layout["sequences"],
            "--TRACKERS_TO_EVAL",
            run_name,
            # On: TrackEval then matches tracker dets against every gt row and
            # drops the ones landing on a distractor (KITTI DontCare / sitting
            # person), which is what makes those regions ignored rather than
            # background. Real gt matches win the assignment first, so a
            # detection on an actual object is never removed by this.
            "--DO_PREPROC",
            "True",
            "--METRICS",
            "HOTA",
            "CLEAR",
            "Identity",
        ]
        metric_commands.append(command)
        if not skip_metrics:
            subprocess.run(command, check=True)
    summary = {
        "condition": condition,
        "model_source": model_source,
        "seed": seed,
        "aug": aug,
        "epoch": epoch,
        "paste_mode": paste_mode,
        "tag": tag,
        "checkpoint": str(checkpoint),
        "tracker": "BoxMOT ByteTrack (ReID disabled)",
        # Which protocol produced these numbers: results from before ignore
        # regions existed are not comparable with results from after.
        "ignore_regions": sorted(IGNORE_REGIONS),
        "trackeval_do_preproc": True,
        "output_dir": str(output_root),
        "trackeval_commands": metric_commands,
    }
    save_json(output_root / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOX-X with BoxMOT ByteTrack")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument(
        "--condition", choices=["kitti", "baseline", "treatment"], default="kitti"
    )
    parser.add_argument(
        "--model-source",
        choices=["coco-pretrained", "finetuned"],
        default="finetuned",
    )
    # Taken from train.run so a new aug level cannot become unevaluatable.
    parser.add_argument("--aug", choices=sorted(AUG_LEVELS), default="full")
    parser.add_argument("--epoch", choices=["ep50", "ep60"], default="ep60")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--paste-mode",
        choices=["append", "replace"],
        default=None,
        help="which treatment run to evaluate; must match how it was trained",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="must match the --tag the run was trained with (train.run --tag)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()
    print(
        run(
            args.config,
            args.condition,
            args.seed,
            args.checkpoint,
            args.skip_metrics,
            args.model_source,
            args.aug,
            args.epoch,
            args.paste_mode,
            args.tag,
        )
    )


if __name__ == "__main__":
    main()
