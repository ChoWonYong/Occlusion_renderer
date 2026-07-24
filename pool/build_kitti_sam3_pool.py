from __future__ import annotations

import argparse
import importlib.util
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from data.kitti_tracking import load_sequence_labels
from pool.build_tracklet_pool import crop_with_context, select_matching_instance
from segment.base import SegBackend


def validate_crop_split(split: Mapping[str, Any]) -> list[str]:
    train = set(str(value) for value in split.get("train_sequences", []))
    evaluation = set(str(value) for value in split.get("eval_sequences", []))
    allowed = set(str(value) for value in split.get("crop_allowed_sequences", []))
    if not allowed:
        raise ValueError("split has no crop_allowed_sequences")
    if not allowed <= train:
        raise ValueError(f"crop sequences outside train split: {sorted(allowed - train)}")
    leakage = allowed & evaluation
    if leakage:
        raise ValueError(f"KITTI crop/eval leakage detected: {sorted(leakage)}")
    return sorted(allowed)


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def inventory(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["kitti_sam3_pool"]
    split_path = resolve_path(path.parent, config["split"]["output"])
    split = load_json(split_path)
    allowed_sequences = validate_crop_split(split)
    eval_sequences = set(str(value) for value in split["eval_sequences"])
    kitti_root = config_path(config, path, "paths", "kitti_tracking")
    image_root = kitti_root / "training" / "image_02"
    label_root = kitti_root / "training" / "label_02"
    class_map = config["classes"]["kitti_map"]
    prompts = settings["prompts"]
    maximum_occlusion = int(settings.get("max_occlusion", 0))
    maximum_truncation = float(settings.get("max_truncation", 0.2))
    minimum_area = float(settings.get("min_bbox_area", 400.0))

    candidates: list[dict[str, Any]] = []
    for sequence in allowed_sequences:
        sequence_dir = image_root / sequence
        frame_paths = sorted(sequence_dir.glob("*.png"))
        if not frame_paths:
            raise FileNotFoundError(f"no KITTI frames found: {sequence_dir}")
        image_width, image_height = _image_size(frame_paths[0])
        labels = load_sequence_labels(label_root / f"{sequence}.txt")
        for frame_index in sorted(labels):
            source_path = sequence_dir / f"{frame_index:06d}.png"
            if not source_path.is_file():
                continue
            for source_index, obj in enumerate(labels[frame_index]):
                if obj.track_id < 0 or obj.category not in prompts or obj.category not in class_map:
                    continue
                if obj.occluded > maximum_occlusion or obj.truncated > maximum_truncation:
                    continue
                x1, y1, x2, y2 = obj.bbox_xyxy
                x1 = max(0.0, min(float(image_width), x1))
                x2 = max(0.0, min(float(image_width), x2))
                y1 = max(0.0, min(float(image_height), y1))
                y2 = max(0.0, min(float(image_height), y2))
                width, height = x2 - x1, y2 - y1
                if width <= 1 or height <= 1 or width * height < minimum_area:
                    continue
                candidates.append(
                    {
                        "candidate_id": len(candidates) + 1,
                        "source_dataset": "KITTI Tracking training",
                        "split_role": "train_crop_allowed",
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "track_id": obj.track_id,
                        "source_index": source_index,
                        "native_category": obj.category,
                        "category": str(class_map[obj.category]),
                        "sam3_prompt": str(prompts[obj.category]),
                        "source_image": str(source_path.resolve()),
                        "source_image_size": [image_width, image_height],
                        "source_bbox_xywh": [x1, y1, width, height],
                        "bbox_area": width * height,
                        "occluded": obj.occluded,
                        "truncated": obj.truncated,
                    }
                )

    leaked = sorted({item["sequence"] for item in candidates} & eval_sequences)
    if leaked:
        raise RuntimeError(f"validation leakage in KITTI crop candidates: {leaked}")
    output_dir = resolve_path(path.parent, settings["output_dir"])
    candidate_path = output_dir / "candidates.json"
    counts = Counter(str(item["category"]) for item in candidates)
    metadata = {
        "description": "KITTI train-only GT bbox candidates for SAM3 static RGBA extraction",
        "split_policy": {
            "source": str(split_path),
            "train_sequences": sorted(str(value) for value in split["train_sequences"]),
            "crop_allowed_sequences": allowed_sequences,
            "eval_sequences": sorted(eval_sequences),
            "disjoint": True,
            "validation_policy": "original images and original GT only; never used as crop source",
        },
        "selection": {
            "max_occlusion": maximum_occlusion,
            "max_truncation": maximum_truncation,
            "min_bbox_area": minimum_area,
            "prompts": dict(prompts),
        },
        "category_counts": dict(sorted(counts.items())),
        "candidates": candidates,
    }
    save_json(candidate_path, metadata)
    summary = {
        "candidates": len(candidates),
        "category_counts": dict(sorted(counts.items())),
        "crop_allowed_sequences": allowed_sequences,
        "eval_sequences": sorted(eval_sequences),
        "leaked_eval_sequences": leaked,
        "output": str(candidate_path),
    }
    save_json(output_dir / "inventory.json", summary)
    return summary


def segment_candidate(
    candidate: Mapping[str, Any],
    segmenter: SegBackend,
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    image = _load_rgb(Path(candidate["source_image"]))
    crop, context, target_box, clipped_box, context_box = crop_with_context(
        image,
        tuple(float(value) for value in candidate["source_bbox_xywh"]),
        float(settings.get("context_padding", 0.2)),
    )
    prompt = str(candidate["sam3_prompt"])
    instances = segmenter.detect_and_mask(context, [prompt])
    matched = select_matching_instance(
        instances,
        target_box,
        float(settings.get("min_gt_iou", 0.3)),
    )
    if matched is None:
        return None
    instance, match_iou = matched
    context_mask = np.asarray(instance.mask, dtype=np.uint8)
    crop_x, crop_y = int(round(target_box[0])), int(round(target_box[1]))
    alpha = (
        context_mask[
            crop_y : crop_y + crop.shape[0],
            crop_x : crop_x + crop.shape[1],
        ]
        != 0
    ).astype(np.uint8) * 255
    mask_area = int(np.count_nonzero(alpha))
    if mask_area < int(settings.get("min_mask_area", 128)):
        return None
    return {
        "image": image,
        "crop": crop,
        "context": context,
        "target_box": target_box,
        "clipped_box": clipped_box,
        "context_box": context_box,
        "instances": instances,
        "selected": instance,
        "match_iou": float(match_iou),
        "alpha": alpha,
        "rgba": np.dstack((crop, alpha)),
        "mask_area": mask_area,
    }


def build(
    config_file: str | Path,
    *,
    candidate_id_override: int | None = None,
    max_instances_override: int | None = None,
    segmenter: SegBackend | None = None,
) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["kitti_sam3_pool"]
    output_dir = resolve_path(path.parent, settings["output_dir"])
    candidate_path = output_dir / "candidates.json"
    if not candidate_path.is_file():
        inventory(path)
    metadata = load_json(candidate_path)
    validate_crop_split(metadata["split_policy"])
    candidates = list(metadata["candidates"])
    if candidate_id_override is not None:
        candidates = [
            item for item in candidates if int(item["candidate_id"]) == candidate_id_override
        ]
        if not candidates:
            raise ValueError(f"KITTI SAM3 candidate_id not found: {candidate_id_override}")
    else:
        random.Random(int(config.get("seed", 0))).shuffle(candidates)

    if segmenter is None:
        from segment.sam3 import Sam3Backend

        segmenter = Sam3Backend(
            config_path(config, path, "paths", "sam3_checkpoint"),
            bpe_path=config_path(config, path, "paths", "sam3_bpe"),
            device=str(settings.get("device", "cuda")),
            confidence_threshold=float(settings.get("confidence_threshold", 0.5)),
            precision=str(settings.get("precision", "auto")),
        )

    maximum = int(
        max_instances_override
        or settings.get("max_instances", len(candidates))
    )
    records: list[dict[str, Any]] = []
    rejected = 0
    for candidate in candidates:
        if len(records) >= maximum:
            break
        result = segment_candidate(candidate, segmenter, settings)
        if result is None:
            rejected += 1
            continue
        relative = (
            Path(str(candidate["category"]))
            / f"kitti_{candidate['sequence']}_{int(candidate['frame_index']):06d}_"
            f"track{int(candidate['track_id'])}_candidate{int(candidate['candidate_id'])}.png"
        )
        from PIL import Image

        destination = output_dir / "rgba" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result["rgba"]).save(destination)
        records.append(
            {
                "id": len(records) + 1,
                **dict(candidate),
                "file_name": str(Path("rgba") / relative),
                "crop_bbox_xywh": result["clipped_box"],
                "sam3_context_bbox_xywh": result["context_box"],
                "mask_area": result["mask_area"],
                "sam3_score": float(result["selected"].score),
                "sam3_gt_match_iou": result["match_iou"],
                "sam3_precision": getattr(segmenter, "precision", "unknown"),
            }
        )
    pool = {
        "schema_version": 1,
        "description": "KITTI train-only static RGBA objects segmented with SAM3",
        "split_policy": metadata["split_policy"],
        "instances": records,
    }
    save_json(output_dir / "pool.json", pool)
    summary = {
        "candidate_id": candidate_id_override,
        "instances": len(records),
        "rejected": rejected,
        "eval_leakage": sorted(
            {record["sequence"] for record in records}
            & set(metadata["split_policy"]["eval_sequences"])
        ),
        "output": str(output_dir / "pool.json"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_track_runs(
    candidates: list[Mapping[str, Any]],
    *,
    min_frames: int,
    max_frames: int | None,
) -> list[list[dict[str, Any]]]:
    """Group clean per-frame candidates into maximal consecutive same-track runs.

    KITTI Tracking is 10 fps, so target == source and no subsampling is needed.
    A non-clean frame is simply absent from ``candidates`` and therefore breaks the
    consecutive run. Runs shorter than ``min_frames`` are dropped; longer runs are
    truncated to the first ``max_frames`` frames.
    """
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")
    by_track: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = (str(candidate["sequence"]), int(candidate["track_id"]))
        by_track.setdefault(key, []).append(dict(candidate))

    runs: list[list[dict[str, Any]]] = []
    for _, items in sorted(by_track.items()):
        ordered = sorted(items, key=lambda item: int(item["frame_index"]))
        current: list[dict[str, Any]] = []
        for candidate in ordered:
            if current and int(candidate["frame_index"]) != int(current[-1]["frame_index"]) + 1:
                if len(current) >= min_frames:
                    runs.append(current if max_frames is None else current[:max_frames])
                current = []
            current.append(candidate)
        if len(current) >= min_frames:
            runs.append(current if max_frames is None else current[:max_frames])
    return runs


def _tracklet_params(settings: Mapping[str, Any]) -> tuple[int, int | None]:
    min_frames = int(settings.get("min_frames", 30))
    raw_max = settings.get("max_frames", 100)
    max_frames = None if raw_max in (None, "null") else int(raw_max)
    return min_frames, max_frames


def tracklet_inventory(config_file: str | Path) -> dict[str, Any]:
    """Report KITTI multi-class tracklet runs without loading SAM3."""
    config, path = load_config(config_file)
    settings = config["kitti_sam3_pool"]
    candidate_path = resolve_path(path.parent, settings["output_dir"]) / "candidates.json"
    if not candidate_path.is_file():
        inventory(path)
    metadata = load_json(candidate_path)
    validate_crop_split(metadata["split_policy"])
    min_frames, max_frames = _tracklet_params(settings)
    runs = build_track_runs(metadata["candidates"], min_frames=min_frames, max_frames=max_frames)
    lengths = [len(run) for run in runs]
    category_counts = Counter(str(run[0]["category"]) for run in runs)
    result = {
        "mode": "kitti_tracklet_runs",
        "min_frames": min_frames,
        "max_frames": max_frames,
        "tracklets": len(runs),
        "distinct_identities": len({(run[0]["sequence"], run[0]["track_id"]) for run in runs}),
        "category_counts": dict(sorted(category_counts.items())),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
    }
    output_dir = resolve_path(path.parent, settings.get("tracklet_output_dir", settings["output_dir"]))
    save_json(output_dir / "tracklet_inventory.json", result)
    result["output"] = str(output_dir / "tracklet_inventory.json")
    return result


def build_tracklets(
    config_file: str | Path,
    *,
    max_tracklets_override: int | None = None,
    segmenter: SegBackend | None = None,
) -> dict[str, Any]:
    """Segment KITTI train-only multi-class tracklets (variable length) with SAM3."""
    config, path = load_config(config_file)
    settings = config["kitti_sam3_pool"]
    candidate_path = resolve_path(path.parent, settings["output_dir"]) / "candidates.json"
    if not candidate_path.is_file():
        inventory(path)
    metadata = load_json(candidate_path)
    validate_crop_split(metadata["split_policy"])
    min_frames, max_frames = _tracklet_params(settings)
    runs = build_track_runs(metadata["candidates"], min_frames=min_frames, max_frames=max_frames)
    random.Random(int(config.get("seed", 0))).shuffle(runs)

    output_dir = resolve_path(path.parent, settings.get("tracklet_output_dir", settings["output_dir"]))
    max_tracklets = int(max_tracklets_override or settings.get("max_tracklets", len(runs)))
    max_per_identity = int(settings.get("max_per_identity", 2))
    if segmenter is None:
        from segment.sam3 import Sam3Backend

        segmenter = Sam3Backend(
            config_path(config, path, "paths", "sam3_checkpoint"),
            bpe_path=config_path(config, path, "paths", "sam3_bpe"),
            device=str(settings.get("device", "cuda")),
            confidence_threshold=float(settings.get("confidence_threshold", 0.5)),
            precision=str(settings.get("precision", "auto")),
        )

    from PIL import Image

    records: list[dict[str, Any]] = []
    rejected = 0
    identity_counts: dict[tuple[str, int], int] = {}
    for run in runs:
        if len(records) >= max_tracklets:
            break
        identity_key = (str(run[0]["sequence"]), int(run[0]["track_id"]))
        if identity_counts.get(identity_key, 0) >= max_per_identity:
            continue
        # Segment every frame, then keep the longest consecutive passing sub-run.
        results = [(candidate, segment_candidate(candidate, segmenter, settings)) for candidate in run]
        best_start = best_length = current_start = current_length = 0
        for index, (_, result) in enumerate(results):
            if result is None:
                current_length = 0
                current_start = index + 1
                continue
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        segmented = [
            (candidate, result)
            for candidate, result in results[best_start : best_start + best_length]
        ]
        if len(segmented) < min_frames:
            rejected += 1
            continue
        kept = [candidate for candidate, _ in segmented]
        identity_counts[identity_key] = identity_counts.get(identity_key, 0) + 1
        tracklet_id = len(records) + 1
        relative_dir = Path(f"tracklet_{tracklet_id:06d}")
        frames: list[dict[str, Any]] = []
        for offset, (candidate, result) in enumerate(segmented):
            relative_file = relative_dir / f"{offset:02d}.png"
            destination = output_dir / relative_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result["rgba"]).save(destination)
            frames.append(
                {
                    "offset": offset,
                    "source_frame": int(candidate["frame_index"]),
                    "source_image": candidate["source_image"],
                    "source_bbox_xywh": candidate["source_bbox_xywh"],
                    "source_image_size": candidate["source_image_size"],
                    "crop_bbox_xywh": result["clipped_box"],
                    "sam3_context_bbox_xywh": result["context_box"],
                    "mask_area": int(result["mask_area"]),
                    "sam3_prompt": str(candidate["sam3_prompt"]),
                    "sam3_score": float(result["selected"].score),
                    "sam3_gt_match_iou": float(result["match_iou"]),
                    "file_name": str(relative_file),
                }
            )
        records.append(
            {
                "id": tracklet_id,
                "category": str(run[0]["category"]),
                "native_category": str(run[0]["native_category"]),
                "source": "KITTI",
                "sequence": str(run[0]["sequence"]),
                "source_track_id": int(run[0]["track_id"]),
                "sam3_prompt": str(run[0]["sam3_prompt"]),
                "source_fps": float(settings.get("target_fps", 10.0)),
                "target_fps": float(settings.get("target_fps", 10.0)),
                "start_frame": int(kept[0]["frame_index"]),
                "end_frame": int(kept[-1]["frame_index"]),
                "length": len(kept),
                "truncated_on_failure": len(kept) < len(run),
                "run_length": len(run),
                "frames": frames,
                "segmentation": {
                    "backend": "sam3",
                    "prompt": str(run[0]["sam3_prompt"]),
                    "identity_assignment": "highest bbox IoU with KITTI GT",
                    "confidence_threshold": float(settings.get("confidence_threshold", 0.5)),
                    "precision": getattr(segmenter, "precision", "unknown"),
                    "min_gt_iou": float(settings.get("min_gt_iou", 0.3)),
                },
            }
        )

    pool = {
        "schema_version": 2,
        "mode": "kitti_multiclass_tracklets",
        "description": "KITTI train-only multi-class variable-length tracklets segmented with SAM3",
        "split_policy": metadata["split_policy"],
        "selection": {
            "min_frames": min_frames,
            "max_frames": max_frames,
            "max_per_identity": max_per_identity,
            "prompts": metadata["selection"]["prompts"],
            "failure_policy": "keep longest consecutive passing sub-run if >= min_frames",
        },
        "tracklets": records,
    }
    save_json(output_dir / "tracklets.json", pool)
    lengths = [record["length"] for record in records]
    summary = {
        "mode": "kitti_multiclass_tracklets",
        "runs": len(runs),
        "tracklets": len(records),
        "rejected": rejected,
        "frames": sum(lengths),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "distinct_identities": len({(record["sequence"], record["source_track_id"]) for record in records}),
        "eval_leakage": sorted(
            {record["sequence"] for record in records}
            & set(metadata["split_policy"]["eval_sequences"])
        ),
        "output": str(output_dir / "tracklets.json"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _relaunch(
    config_file: Path,
    candidate_id: int | None,
    max_instances: int | None,
    *,
    tracklet: bool = False,
) -> int:
    config, path = load_config(config_file)
    python_path = config_path(config, path, "paths", "sam3_python")
    command = [str(python_path), "-m", "pool.build_kitti_sam3_pool", "--config", str(path)]
    if tracklet:
        command.append("--tracklet")
    if candidate_id is not None:
        command.extend(["--candidate-id", str(candidate_id)])
    if max_instances is not None:
        command.extend(["--max-instances", str(max_instances)])
    return subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only KITTI SAM3 RGBA crop / tracklet pool")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--tracklet",
        action="store_true",
        help="build multi-class variable-length tracklets instead of static crops",
    )
    parser.add_argument("--candidate-id", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    if args.inventory_only:
        print(tracklet_inventory(args.config) if args.tracklet else inventory(args.config))
        return
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(
            _relaunch(args.config, args.candidate_id, args.max_instances, tracklet=args.tracklet)
        )
    if args.tracklet:
        print(build_tracklets(args.config, max_tracklets_override=args.max_instances))
    else:
        print(
            build(
                args.config,
                candidate_id_override=args.candidate_id,
                max_instances_override=args.max_instances,
            )
        )


if __name__ == "__main__":
    main()
