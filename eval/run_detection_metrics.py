from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.gpu_budget import enforce_account_gpu_budget
from common.io import load_json, save_json
from common.io_video import group_frames_by_video
from eval.run_boxmot import (
    IGNORE_REGIONS,
    _balanced_video_shards,
    _detect,
    _load_model,
)


GroupSelector = Callable[[Mapping[str, Any]], bool]


def _selected_run_specs(settings: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = settings["runs"]
    enabled = settings.get("enabled_runs")
    if enabled is None:
        return {str(name): spec for name, spec in runs.items()}
    names = [str(name) for name in enabled]
    missing = [name for name in names if name not in runs]
    if missing:
        raise ValueError(f"enabled phase4 runs are undefined: {missing}")
    return {name: runs[name] for name in names}


def _format_metric(value: float) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.2f}"


def _write_comparison_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    comparison = summary["comparison"]
    reference_name = str(comparison["reference_run"])
    target_name = str(comparison["target_run"])
    reference = summary["runs"][reference_name]
    target = summary["runs"][target_name]
    deltas = comparison["delta_target_minus_reference"]
    lines = [
        "# Detector comparison on the held-out KITTI split",
        "",
        f"Reference: **{reference['label']}**",
        f"Target: **{target['label']}**",
        "",
        "All values are percentages; delta is target minus reference.",
        "",
    ]
    evaluation_classes = summary.get("protocol", {}).get("evaluation_classes")
    if evaluation_classes:
        lines.extend(
            [
                "Evaluation classes: **" + ", ".join(evaluation_classes) + "**",
                "",
            ]
        )
    lines.extend(
        [
            "## Combined classes",
            "",
            "| Scope | Metric | Reference | Target | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for group in ("all", "occluded", "non_occluded"):
        for metric in ("AP", "AP50", "AP75", "AR100", "Recall50"):
            lines.append(
                f"| {group} | {metric} | "
                f"{_format_metric(reference['metrics'][group]['combined'][metric])} | "
                f"{_format_metric(target['metrics'][group]['combined'][metric])} | "
                f"{float(deltas[group]['combined'][metric]):+.2f} |"
            )
    lines.extend(
        [
            "",
            "## Per-class overall AP",
            "",
            "| Class | Reference AP | Target AP | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in reference["metrics"]["all"]["per_class"]:
        lines.append(
            f"| {category} | "
            f"{_format_metric(reference['metrics']['all']['per_class'][category]['AP'])} | "
            f"{_format_metric(target['metrics']['all']['per_class'][category]['AP'])} | "
            f"{float(deltas['all']['per_class'][category]['AP']):+.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _annotation_occlusion(annotation: Mapping[str, Any]) -> int:
    return int(annotation.get("kitti", {}).get("occluded", 3))


def _group_selector(name: str) -> GroupSelector:
    if name == "all":
        return lambda annotation: True
    if name == "non_occluded":
        return lambda annotation: _annotation_occlusion(annotation) == 0
    if name == "occluded":
        return lambda annotation: _annotation_occlusion(annotation) in {1, 2}
    raise ValueError(f"unknown occlusion group: {name}")


def _add_protocol_ignore_regions(
    dataset: dict[str, Any], config: Mapping[str, Any], config_file: Path
) -> int:
    """Add KITTI DontCare/sitting-person boxes as category-specific crowd GT.

    This mirrors the ignore policy used by the tracking evaluator. Crowd GT is
    retained in every occlusion subset, so detections landing on excluded
    regions are ignored rather than becoming false positives.
    """
    from data.kitti_tracking import load_sequence_labels

    category_id = {str(item["name"]): int(item["id"]) for item in dataset["categories"]}
    videos = {int(item["id"]): str(item["name"]) for item in dataset["videos"]}
    image_by_video_frame = {
        (int(image["video_id"]), int(image["frame_index"])): image
        for image in dataset["images"]
    }
    next_id = max((int(item["id"]) for item in dataset["annotations"]), default=0) + 1
    kitti_root = config_path(config, config_file, "paths", "kitti_tracking")
    added = 0
    for video_id, sequence in videos.items():
        labels = load_sequence_labels(
            kitti_root / "training" / "label_02" / f"{sequence}.txt"
        )
        for frame_index, objects in labels.items():
            image = image_by_video_frame.get((video_id, frame_index))
            if image is None:
                continue
            for obj in objects:
                if obj.category not in IGNORE_REGIONS:
                    continue
                applies_to = IGNORE_REGIONS[obj.category]
                names = category_id if applies_to is None else applies_to
                x1, y1, x2, y2 = (float(value) for value in obj.bbox_xyxy)
                width, height = x2 - x1, y2 - y1
                if width <= 1.0 or height <= 1.0:
                    continue
                for name in names:
                    if name not in category_id:
                        continue
                    dataset["annotations"].append(
                        {
                            "id": next_id,
                            "image_id": int(image["id"]),
                            "category_id": category_id[name],
                            "bbox": [x1, y1, width, height],
                            "area": width * height,
                            "iscrowd": 1,
                            "protocol_ignore": True,
                        }
                    )
                    next_id += 1
                    added += 1
    return added


def _infer_run_shard(
    config: dict[str, Any],
    path: Path,
    checkpoint: Path,
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    import cv2
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Phase-4 inference requires a CUDA GPU")
    model, exp, class_id_map, use_half = _load_model(
        config, path, checkpoint, device, "finetuned"
    )
    dataset_root = resolve_path(path.parent, config["dataset"]["output_dir"])
    dataset = load_json(dataset_root / config["dataset"]["eval_json"])
    frames_by_video = group_frames_by_video(dataset)
    assigned = _balanced_video_shards(frames_by_video, num_shards)[shard_index]
    category_by_index = {
        index: int(category["id"])
        for index, category in enumerate(dataset["categories"])
    }
    predictions: list[dict[str, Any]] = []
    frame_count = 0
    for video_id in assigned:
        for frame in frames_by_video[video_id]:
            image = cv2.imread(str(dataset_root / frame["file_name"]))
            if image is None:
                raise FileNotFoundError(dataset_root / frame["file_name"])
            detections = _detect(model, image, exp, device, class_id_map, use_half)
            image_height, image_width = image.shape[:2]
            for x1, y1, x2, y2, score, cls in detections:
                class_index = int(round(float(cls)))
                if class_index not in category_by_index:
                    continue
                x1 = min(max(0.0, float(x1)), float(image_width))
                y1 = min(max(0.0, float(y1)), float(image_height))
                x2 = min(max(0.0, float(x2)), float(image_width))
                y2 = min(max(0.0, float(y2)), float(image_height))
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append(
                    {
                        "image_id": int(frame["id"]),
                        "category_id": category_by_index[class_index],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
            frame_count += 1
    del model
    torch.cuda.empty_cache()
    return {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "video_ids": assigned,
        "frames": frame_count,
        "predictions": predictions,
    }


def _merge_shards(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(payloads, key=lambda item: int(item["shard_index"]))
    if [int(item["shard_index"]) for item in ordered] != list(range(len(ordered))):
        raise ValueError("prediction shard indices must be contiguous from zero")
    if any(int(item["num_shards"]) != len(ordered) for item in ordered):
        raise ValueError("prediction shard count mismatch")
    video_ids = [int(video) for item in ordered for video in item["video_ids"]]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("a video was inferred by more than one shard")
    return {
        "frames": sum(int(item["frames"]) for item in ordered),
        "video_ids": sorted(video_ids),
        "predictions": [prediction for item in ordered for prediction in item["predictions"]],
    }


def _metric_mean(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(np.mean(valid) * 100.0) if valid.size else float("nan")


def _extract_metrics(evaluator: Any, categories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    precision = evaluator.eval["precision"]  # T x R x K x A x M
    recall = evaluator.eval["recall"]  # T x K x A x M
    index_50 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.50)))
    index_75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))

    def one(category_index: int | slice) -> dict[str, float]:
        return {
            "AP": _metric_mean(precision[:, :, category_index, 0, 2]),
            "AP50": _metric_mean(precision[index_50, :, category_index, 0, 2]),
            "AP75": _metric_mean(precision[index_75, :, category_index, 0, 2]),
            "AR100": _metric_mean(recall[:, category_index, 0, 2]),
            "Recall50": _metric_mean(recall[index_50, category_index, 0, 2]),
        }

    return {
        "combined": one(slice(None)),
        "per_class": {
            str(category["name"]): one(index)
            for index, category in enumerate(categories)
        },
    }


def evaluate_predictions(
    source_dataset: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    group_name: str,
    *,
    iou_thresholds: Sequence[float] | None = None,
    max_detections: int = 100,
) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    selector = _group_selector(group_name)

    class GroupedCOCOeval(COCOeval):
        def _prepare(self) -> None:
            super()._prepare()
            for annotations in self._gts.values():
                for annotation in annotations:
                    if annotation.get("protocol_ignore"):
                        annotation["ignore"] = 1
                    elif not selector(annotation):
                        annotation["ignore"] = 1

    dataset = copy.deepcopy(dict(source_dataset))
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(list(predictions))
    evaluator = GroupedCOCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(int(image["id"]) for image in dataset["images"])
    evaluator.params.catIds = [int(category["id"]) for category in dataset["categories"]]
    if iou_thresholds is not None:
        evaluator.params.iouThrs = np.asarray(iou_thresholds, dtype=np.float64)
    evaluator.params.maxDets = [1, 10, int(max_detections)]
    evaluator.evaluate()
    evaluator.accumulate()
    metrics = _extract_metrics(evaluator, dataset["categories"])
    target_annotations = [
        annotation
        for annotation in dataset["annotations"]
        if not annotation.get("protocol_ignore") and selector(annotation)
    ]
    metrics["ground_truth"] = {
        "total": len(target_annotations),
        "per_class": {
            str(category["name"]): sum(
                int(annotation["category_id"]) == int(category["id"])
                for annotation in target_annotations
            )
            for category in dataset["categories"]
        },
    }
    return metrics


def _worker(config_file: Path, shard_index: int, num_shards: int) -> dict[str, Any]:
    config, path = load_config(config_file)
    output_dir = resolve_path(path.parent, config["phase4_evaluation"]["output_dir"])
    outputs: dict[str, Any] = {}
    for run_name, spec in _selected_run_specs(config["phase4_evaluation"]).items():
        checkpoint = resolve_path(path.parent, spec["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = _infer_run_shard(
            config,
            path,
            checkpoint,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        destination = output_dir / "_prediction_shards" / f"{run_name}_{shard_index:02d}.json"
        save_json(destination, payload)
        outputs[str(run_name)] = str(destination)
    return outputs


def _launch_workers(config_file: Path, workers: int) -> None:
    config, _ = load_config(config_file)
    enforce_account_gpu_budget(
        config.get("resources", {}), project_limit_key="phase1_eval_max_gpus"
    )
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip() and item.strip() != "-1"
    ]
    if len(visible) != workers or len(set(visible)) != workers:
        raise RuntimeError(
            f"--workers {workers} requires exactly {workers} distinct GPUs in "
            "CUDA_VISIBLE_DEVICES"
        )
    project_root = Path(__file__).resolve().parents[1]
    processes: list[subprocess.Popen[Any]] = []
    for shard_index, device in enumerate(visible):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = device
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "eval.run_detection_metrics",
                    "--config",
                    str(config_file),
                    "--worker-index",
                    str(shard_index),
                    "--worker-count",
                    str(workers),
                ],
                cwd=project_root,
                env=environment,
            )
        )
    return_codes = [process.wait() for process in processes]
    failed = [index for index, code in enumerate(return_codes) if code != 0]
    if failed:
        raise RuntimeError(f"Phase-4 inference workers failed: {failed}")


def run(config_file: str | Path, workers: int = 1) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["phase4_evaluation"]
    output_dir = resolve_path(path.parent, settings["output_dir"])
    _launch_workers(path, workers)

    dataset_root = resolve_path(path.parent, config["dataset"]["output_dir"])
    dataset = load_json(dataset_root / config["dataset"]["eval_json"])
    ignore_regions = _add_protocol_ignore_regions(dataset, config, path)
    run_results: dict[str, Any] = {}
    for run_name, spec in _selected_run_specs(settings).items():
        payloads = [
            load_json(
                output_dir
                / "_prediction_shards"
                / f"{run_name}_{shard_index:02d}.json"
            )
            for shard_index in range(workers)
        ]
        merged = _merge_shards(payloads)
        prediction_file = output_dir / f"predictions_{run_name}.json"
        save_json(prediction_file, merged["predictions"])
        groups = {
            name: evaluate_predictions(
                dataset,
                merged["predictions"],
                name,
                iou_thresholds=settings.get("iou_thresholds"),
                max_detections=int(settings.get("max_detections", 100)),
            )
            for name in ("all", "occluded", "non_occluded")
        }
        run_results[str(run_name)] = {
            "label": str(spec["label"]),
            "checkpoint": str(resolve_path(path.parent, spec["checkpoint"])),
            "training_data": str(spec["training_data"]),
            "frames": int(merged["frames"]),
            "detections": len(merged["predictions"]),
            "predictions": str(prediction_file),
            "metrics": groups,
        }

    comparison = settings.get("comparison", {})
    reference_name = str(comparison.get("reference_run", "baseline"))
    target_name = str(comparison.get("target_run", "confidence_filtered"))
    if reference_name not in run_results or target_name not in run_results:
        raise ValueError(
            f"comparison runs must be enabled: reference={reference_name}, target={target_name}"
        )
    reference = run_results[reference_name]["metrics"]
    target = run_results[target_name]["metrics"]
    deltas = {}
    for group in target:
        deltas[group] = {
            "combined": {
                metric: float(
                    target[group]["combined"][metric]
                    - reference[group]["combined"][metric]
                )
                for metric in target[group]["combined"]
            },
            "per_class": {
                name: {
                    metric: float(
                        target[group]["per_class"][name][metric]
                        - reference[group]["per_class"][name][metric]
                    )
                    for metric in target[group]["per_class"][name]
                }
                for name in target[group]["per_class"]
            },
        }
    summary = {
        "protocol": {
            "metric": "COCO bbox",
            "units": "percentage points",
            "AP": "IoU 0.50:0.95",
            "AR100": "IoU 0.50:0.95, max 100 detections/image",
            "Recall50": "IoU 0.50, max 100 detections/image",
            "occluded": "KITTI occluded in {1,2}",
            "non_occluded": "KITTI occluded == 0",
            "unknown": "KITTI occluded == 3; included in overall, excluded from subsets",
            "ignore_regions": "KITTI DontCare and sitting-person policy shared with tracking eval",
            "paste_jitter": str(settings.get("paste_jitter", "random/mid")),
        },
        "eval_dataset": str(dataset_root / config["dataset"]["eval_json"]),
        "images": len(dataset["images"]),
        "annotations": sum(not item.get("protocol_ignore") for item in dataset["annotations"]),
        "protocol_ignore_regions": ignore_regions,
        "workers": workers,
        "runs": run_results,
        "comparison": {
            "reference_run": reference_name,
            "target_run": target_name,
            "delta_target_minus_reference": deltas,
        },
        "output_dir": str(output_dir),
    }
    if reference_name == "baseline" and target_name == "confidence_filtered":
        summary["delta_confidence_filtered_minus_baseline"] = deltas
    save_json(output_dir / "summary.json", summary)
    _write_comparison_report(output_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate overall and occlusion-stratified COCO bbox metrics"
    )
    parser.add_argument("--config", default="configs/phase4_detection_eval.yaml", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_index is not None:
        if args.worker_count is None:
            parser.error("--worker-index requires --worker-count")
        print(_worker(args.config.resolve(), args.worker_index, args.worker_count))
        return
    print(run(args.config, args.workers))


if __name__ == "__main__":
    main()
