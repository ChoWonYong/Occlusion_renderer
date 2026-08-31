from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from common.config import load_config, resolve_path
from common.io import load_json, save_json
from mining.tracker import create_boxmot_bytetrack


TRACKED_CATEGORIES = ("car", "person")


class Tracker(Protocol):
    def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray: ...


def bbox_iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _load_bgr(path: str | Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for ByteTrack mining") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _detections_array(frame: Mapping[str, Any]) -> np.ndarray:
    rows: list[list[float]] = []
    class_index = {name: index for index, name in enumerate(TRACKED_CATEGORIES)}
    for detection in frame.get("detections", []):
        category = str(detection.get("category", ""))
        if category not in class_index:
            continue
        x1, y1, x2, y2 = (float(value) for value in detection["bbox_xyxy"])
        rows.append([x1, y1, x2, y2, float(detection["score"]), class_index[category]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def _source_detection(
    track: Sequence[float], detections: Sequence[Mapping[str, Any]], category: str
) -> Mapping[str, Any] | None:
    same_class = [item for item in detections if str(item.get("category")) == category]
    if not same_class:
        return None
    box = [float(value) for value in track[:4]]
    return max(same_class, key=lambda item: bbox_iou_xyxy(box, item["bbox_xyxy"]))


def track_manifest(
    payload: Mapping[str, Any],
    tracker_settings: Mapping[str, Any],
    *,
    tracker_factory: Callable[[Mapping[str, Any], list[str]], Tracker] = create_boxmot_bytetrack,
    image_loader: Callable[[str | Path], np.ndarray] = _load_bgr,
    min_bbox_area: float = 400.0,
) -> list[dict[str, Any]]:
    """Associate detector boxes without consulting source annotations."""
    candidates: list[dict[str, Any]] = []
    for sequence in payload.get("sequences", []):
        tracker = tracker_factory(tracker_settings, list(TRACKED_CATEGORIES))
        for frame in sequence.get("frames", []):
            image = image_loader(frame["source_image"])
            detections = _detections_array(frame)
            tracks = np.asarray(tracker.update(detections, image))
            if tracks.size == 0:
                continue
            if tracks.ndim == 1:
                tracks = tracks.reshape(1, -1)
            if tracks.shape[1] < 7:
                raise ValueError(f"ByteTrack output needs at least 7 columns, got {tracks.shape}")
            image_width, image_height = (int(value) for value in frame["source_image_size"])
            for track in tracks:
                x1, y1, x2, y2, track_id, confidence, cls = (float(value) for value in track[:7])
                class_id = int(round(cls))
                if not 0 <= class_id < len(TRACKED_CATEGORIES):
                    continue
                x1, x2 = max(0.0, min(image_width, x1)), max(0.0, min(image_width, x2))
                y1, y2 = max(0.0, min(image_height, y1)), max(0.0, min(image_height, y2))
                width, height = x2 - x1, y2 - y1
                if width <= 1.0 or height <= 1.0 or width * height < min_bbox_area:
                    continue
                category = TRACKED_CATEGORIES[class_id]
                source = _source_detection(track, frame.get("detections", []), category)
                candidates.append(
                    {
                        "source_dataset": str(sequence["source_dataset"]),
                        "sequence": str(sequence["sequence"]),
                        "source_fps": float(sequence["source_fps"]),
                        "target_fps": float(sequence["target_fps"]),
                        "sample_index": int(frame["sample_index"]),
                        "frame_index": int(frame["frame_index"]),
                        "source_image": str(frame["source_image"]),
                        "source_image_size": [image_width, image_height],
                        "source_track_id": int(round(track_id)),
                        "category": category,
                        "sam3_prompt": "human" if category == "person" else "car",
                        "source_bbox_xywh": [x1, y1, width, height],
                        "detector_confidence": float(confidence),
                        "detector_class_name": None if source is None else source.get("coco_class_name"),
                        "detector_class_id": None if source is None else source.get("coco_class_id"),
                        "selection_used_gt": False,
                    }
                )
    return candidates


def build_track_runs(
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_frames: int,
    max_frames: int | None,
    max_per_identity: int,
) -> list[list[dict[str, Any]]]:
    """Keep continuous target-FPS runs produced by ByteTrack."""
    if min_frames <= 0 or max_per_identity <= 0:
        raise ValueError("min_frames and max_per_identity must be positive")
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = (
            str(candidate["source_dataset"]),
            str(candidate["sequence"]),
            int(candidate["source_track_id"]),
            str(candidate["category"]),
        )
        grouped.setdefault(key, []).append(dict(candidate))

    runs: list[list[dict[str, Any]]] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: int(item["sample_index"]))
        contiguous: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for candidate in ordered:
            if current and int(candidate["sample_index"]) != int(current[-1]["sample_index"]) + 1:
                if len(current) >= min_frames:
                    contiguous.append(current)
                current = []
            current.append(candidate)
        if len(current) >= min_frames:
            contiguous.append(current)
        for run in contiguous[:max_per_identity]:
            runs.append(run if max_frames is None else run[:max_frames])
    return runs


def _confidence_stats(run: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores = np.asarray([float(item["detector_confidence"]) for item in run], dtype=np.float64)
    return {
        "min": float(np.min(scores)),
        "p10": float(np.quantile(scores, 0.1)),
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
    }


def mine_detection_tracklets(
    config_file: str | Path,
    *,
    payloads: Sequence[Mapping[str, Any]] | None = None,
    tracker_factory: Callable[[Mapping[str, Any], list[str]], Tracker] = create_boxmot_bytetrack,
    image_loader: Callable[[str | Path], np.ndarray] = _load_bgr,
) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["detector_tracklet_pool"]
    if payloads is None:
        detection_dir = resolve_path(path.parent, config["codetr_detector"]["output_dir"])
        payloads = [
            load_json(detection_dir / "kitti_detections.json"),
            load_json(detection_dir / "mot17_detections.json"),
        ]
    min_frames = int(settings.get("min_frames", 30))
    raw_max = settings.get("max_frames", 100)
    max_frames = None if raw_max in (None, "null") else int(raw_max)
    max_per_identity = int(settings.get("max_per_identity", 2))
    candidates: list[dict[str, Any]] = []
    for payload in payloads:
        candidates.extend(
            track_manifest(
                payload,
                config["tracker"],
                tracker_factory=tracker_factory,
                image_loader=image_loader,
                min_bbox_area=float(settings.get("min_bbox_area", 400.0)),
            )
        )
    runs = build_track_runs(
        candidates,
        min_frames=min_frames,
        max_frames=max_frames,
        max_per_identity=max_per_identity,
    )
    records: list[dict[str, Any]] = []
    for candidate_id, run in enumerate(runs, start=1):
        first, last = run[0], run[-1]
        records.append(
            {
                "candidate_id": candidate_id,
                "category": str(first["category"]),
                "source_dataset": str(first["source_dataset"]),
                "sequence": str(first["sequence"]),
                "source_track_id": int(first["source_track_id"]),
                "source_fps": float(first["source_fps"]),
                "target_fps": float(first["target_fps"]),
                "start_frame": int(first["frame_index"]),
                "end_frame": int(last["frame_index"]),
                "length": len(run),
                "detector_confidence": _confidence_stats(run),
                "frames": [{**dict(item), "offset": offset} for offset, item in enumerate(run)],
            }
        )

    output_dir = resolve_path(path.parent, settings["candidate_output_dir"])
    payload = {
        "schema_version": 1,
        "mode": "codetr_bytetrack_candidates",
        "description": "GT-free Co-DETR boxes associated with ByteTrack before SAM3",
        "selection": {
            "gt_used": False,
            "min_frames": min_frames,
            "max_frames": max_frames,
            "max_per_identity": max_per_identity,
            "quality_filter": "deferred_to_phase2",
            "tracker": dict(config["tracker"]),
        },
        "tracklets": records,
    }
    output_path = output_dir / "tracklets.json"
    save_json(output_path, payload)
    lengths = [record["length"] for record in records]
    summary = {
        "mode": payload["mode"],
        "tracked_frame_boxes": len(candidates),
        "tracklets": len(records),
        "frames": sum(lengths),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "quality_filter": "deferred_to_phase2",
        "gt_used": False,
        "output": str(output_path),
    }
    save_json(output_dir / "summary.json", summary)
    return summary
