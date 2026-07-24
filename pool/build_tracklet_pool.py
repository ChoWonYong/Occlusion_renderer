from __future__ import annotations

import argparse
import importlib.util
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import save_json
from data.mot17 import (
    Mot17Tracklet,
    collect_strict_tracklets,
    collect_variable_fps_tracklets,
    identities,
    image_path,
)
from segment.base import Instance, SegBackend


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(path)


def _clip_bbox(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    x, y, width, height = bbox
    x1 = max(0, min(image_width, int(np.floor(x))))
    y1 = max(0, min(image_height, int(np.floor(y))))
    x2 = max(0, min(image_width, int(np.ceil(x + width))))
    y2 = max(0, min(image_height, int(np.ceil(y + height))))
    return (x1, y1, x2, y2) if x2 - x1 > 1 and y2 - y1 > 1 else None


def crop_with_context(
    image: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    padding_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], list[float]]:
    """Return source bbox crop, padded SAM3 input, and target box in context coordinates."""
    clipped = _clip_bbox(bbox_xywh, image.shape[1], image.shape[0])
    if clipped is None:
        raise ValueError("bbox is empty after clipping")
    x1, y1, x2, y2 = clipped
    pad_x = int(round((x2 - x1) * max(0.0, padding_ratio)))
    pad_y = int(round((y2 - y1) * max(0.0, padding_ratio)))
    context_x1 = max(0, x1 - pad_x)
    context_y1 = max(0, y1 - pad_y)
    context_x2 = min(image.shape[1], x2 + pad_x)
    context_y2 = min(image.shape[0], y2 + pad_y)
    source_crop = image[y1:y2, x1:x2]
    context_crop = image[context_y1:context_y2, context_x1:context_x2]
    target_in_context = [
        float(x1 - context_x1),
        float(y1 - context_y1),
        float(x2 - x1),
        float(y2 - y1),
    ]
    return (
        source_crop,
        context_crop,
        target_in_context,
        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        [
            float(context_x1),
            float(context_y1),
            float(context_x2 - context_x1),
            float(context_y2 - context_y1),
        ],
    )


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def select_matching_instance(
    instances: list[Instance], target_bbox: list[float], min_iou: float
) -> tuple[Instance, float] | None:
    if not instances:
        return None
    ranked = sorted(
        ((instance, bbox_iou(instance.bbox, target_bbox)) for instance in instances),
        key=lambda item: (item[1], item[0].score),
        reverse=True,
    )
    return ranked[0] if ranked[0][1] >= min_iou else None


def inventory(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["tracklet_pool"]
    mot_root = config_path(config, path, "paths", "mot17")
    candidates = collect_strict_tracklets(
        mot_root,
        detector=str(settings.get("detector", "FRCNN")),
        length=int(settings.get("length", 30)),
        visibility_min=float(settings.get("visibility_min", 0.8)),
        stride=int(settings.get("stride", settings.get("length", 30))),
    )
    candidate_records = [
        {
            "candidate_id": index,
            "category": "person",
            "sequence": candidate.sequence,
            "source_track_id": candidate.track_id,
            "start_frame": candidate.start_frame,
            "end_frame": candidate.end_frame,
            "length": len(candidate.objects),
            "visibility_min": min(item.visibility for item in candidate.objects),
            "frames": [
                {
                    "offset": offset,
                    "source_frame": item.frame_index,
                    "source_image": str(image_path(mot_root, candidate, item.frame_index)),
                    "source_bbox_xywh": [float(value) for value in item.bbox],
                    "source_image_size": [candidate.image_width, candidate.image_height],
                    "visibility": item.visibility,
                }
                for offset, item in enumerate(candidate.objects)
            ],
        }
        for index, candidate in enumerate(candidates, start=1)
    ]
    result = {
        "source": "MOT17 train GT",
        "detector_view": str(settings.get("detector", "FRCNN")),
        "visibility_min": float(settings.get("visibility_min", 0.8)),
        "length": int(settings.get("length", 30)),
        "stride": int(settings.get("stride", settings.get("length", 30))),
        "candidate_tracklets": len(candidates),
        "source_identities": len(identities(candidates)),
        "sequences": sorted({candidate.sequence for candidate in candidates}),
        "strict_rule": "all 30 frames are consecutive and every frame has visibility >= 0.8",
    }
    output_dir = resolve_path(path.parent, settings["output_dir"])
    candidates_path = output_dir / "candidates.json"
    save_json(
        candidates_path,
        {
            "description": "MOT17 GT candidates before SAM3; every record is a strict 30-frame tracklet",
            "selection": {
                "visibility_min": float(settings.get("visibility_min", 0.8)),
                "length": int(settings.get("length", 30)),
                "stride": int(settings.get("stride", settings.get("length", 30))),
            },
            "tracklets": candidate_records,
        },
    )
    save_json(output_dir / "inventory.json", result)
    result["output"] = str(output_dir / "inventory.json")
    result["candidates_output"] = str(candidates_path)
    return result


def _longest_passing_run(
    entries: list[tuple[np.ndarray, dict[str, Any]] | None],
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Longest run of consecutive successfully-segmented frames, re-indexed from 0."""
    best_start = best_length = 0
    current_start = current_length = 0
    for index, entry in enumerate(entries):
        if entry is None:
            current_length = 0
            current_start = index + 1
            continue
        if current_length == 0:
            current_start = index
        current_length += 1
        if current_length > best_length:
            best_length = current_length
            best_start = current_start
    run = entries[best_start : best_start + best_length]
    result: list[tuple[np.ndarray, dict[str, Any]]] = []
    for new_offset, item in enumerate(run):
        assert item is not None
        rgba, record = item
        result.append((rgba, {**record, "offset": new_offset}))
    return result


def _segment_frame(
    mot_root: Path,
    tracklet: Mot17Tracklet,
    obj: Any,
    segmenter: SegBackend,
    min_mask_area: int,
    context_padding: float,
    prompt: str,
    min_match_iou: float,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Segment one frame; return (rgba, record) on success or None on any failure."""
    source_path = image_path(mot_root, tracklet, obj.frame_index)
    if not source_path.is_file():
        return None
    image = _load_rgb(source_path)
    try:
        crop, context, target_box, clipped_box, context_box = crop_with_context(
            image, obj.bbox, context_padding
        )
    except ValueError:
        return None
    instances = segmenter.detect_and_mask(context, [prompt])
    matched = select_matching_instance(instances, target_box, min_match_iou)
    if matched is None:
        return None
    instance, match_iou = matched
    context_mask = np.asarray(instance.mask, dtype=np.uint8)
    if context_mask.shape != context.shape[:2]:
        raise ValueError(
            f"SAM3 mask shape {context_mask.shape} does not match context crop {context.shape[:2]}"
        )
    crop_x = int(round(target_box[0]))
    crop_y = int(round(target_box[1]))
    alpha = (
        context_mask[crop_y : crop_y + crop.shape[0], crop_x : crop_x + crop.shape[1]] != 0
    ).astype(np.uint8) * 255
    if int(np.count_nonzero(alpha)) < min_mask_area:
        return None
    rgba = np.dstack((crop, alpha))
    frame_record = {
        "source_frame": obj.frame_index,
        "source_image": str(source_path),
        "source_bbox_xywh": [float(value) for value in obj.bbox],
        "crop_bbox_xywh": clipped_box,
        "sam3_context_bbox_xywh": context_box,
        "source_image_size": [tracklet.image_width, tracklet.image_height],
        "visibility": obj.visibility,
        "mask_area": int(np.count_nonzero(alpha)),
        "sam3_prompt": prompt,
        "sam3_score": float(instance.score),
        "sam3_bbox_xywh_in_context": [float(value) for value in instance.bbox],
        "sam3_gt_match_iou": float(match_iou),
    }
    return rgba, frame_record


def _segment_tracklet(
    mot_root: Path,
    tracklet: Mot17Tracklet,
    segmenter: SegBackend,
    min_mask_area: int,
    context_padding: float,
    prompt: str,
    min_match_iou: float,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Segment every frame, then return the longest consecutive passing sub-run.

    Failures no longer discard the whole tracklet nor only the suffix; the longest
    clean contiguous window (same identity, no gap) is salvaged. Callers keep it if
    it is at least ``min_frames`` long. Every returned frame passed the SAM3
    GT-IoU / mask-area checks.
    """
    entries = [
        _segment_frame(
            mot_root, tracklet, obj, segmenter, min_mask_area, context_padding, prompt, min_match_iou
        )
        for obj in tracklet.objects
    ]
    return _longest_passing_run(entries)


def build(
    config_file: str | Path,
    *,
    max_tracklets_override: int | None = None,
    candidate_id_override: int | None = None,
    segmenter: SegBackend | None = None,
) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["tracklet_pool"]
    mot_root = config_path(config, path, "paths", "mot17")
    output_dir = resolve_path(path.parent, settings["output_dir"])
    length = int(settings.get("length", 30))
    visibility_min = float(settings.get("visibility_min", 0.8))
    stride = int(settings.get("stride", length))
    if length != 30:
        raise ValueError("this experiment requires tracklet_pool.length=30")
    if visibility_min < 0.8:
        raise ValueError("this experiment requires tracklet_pool.visibility_min>=0.8")

    candidates = collect_strict_tracklets(
        mot_root,
        detector=str(settings.get("detector", "FRCNN")),
        length=length,
        visibility_min=visibility_min,
        stride=stride,
    )
    candidate_ids: dict[int, int] = {id(candidate): index for index, candidate in enumerate(candidates, start=1)}
    if candidate_id_override is not None:
        if not 1 <= candidate_id_override <= len(candidates):
            raise ValueError(
                f"candidate_id {candidate_id_override} is out of range 1..{len(candidates)}"
            )
        candidates = [candidates[candidate_id_override - 1]]
    else:
        random.Random(int(config.get("seed", 0))).shuffle(candidates)
    max_tracklets = int(max_tracklets_override or settings.get("max_tracklets", len(candidates)))
    if segmenter is None:
        from segment.sam3 import Sam3Backend

        segmenter = Sam3Backend(
            config_path(config, path, "paths", "sam3_checkpoint"),
            bpe_path=config_path(config, path, "paths", "sam3_bpe"),
            device=str(settings.get("device", "cuda")),
            confidence_threshold=float(settings.get("sam3_confidence_threshold", 0.5)),
            precision=str(settings.get("sam3_precision", "auto")),
        )

    records: list[dict[str, Any]] = []
    rejected = 0
    for candidate in candidates:
        if len(records) >= max_tracklets:
            break
        segmented = _segment_tracklet(
            mot_root,
            candidate,
            segmenter,
            min_mask_area=int(settings.get("min_mask_area", 128)),
            context_padding=float(settings.get("sam3_context_padding", 0.2)),
            prompt=str(settings.get("sam3_text_prompt", "human")),
            min_match_iou=float(settings.get("sam3_min_gt_iou", 0.3)),
        )
        if segmented is None or len(segmented) != 30:
            rejected += 1
            continue
        tracklet_id = len(records) + 1
        relative_dir = Path(f"tracklet_{tracklet_id:06d}")
        frames: list[dict[str, Any]] = []
        for rgba, frame_record in segmented:
            relative_file = relative_dir / f"{int(frame_record['offset']):02d}.png"
            _save_rgba(output_dir / relative_file, rgba)
            frames.append({**frame_record, "file_name": str(relative_file)})
        records.append(
            {
                "id": tracklet_id,
                "category": "person",
                "source": "MOT17",
                "source_candidate_id": candidate_ids[id(candidate)],
                "sequence": candidate.sequence,
                "source_track_id": candidate.track_id,
                "start_frame": candidate.start_frame,
                "end_frame": candidate.end_frame,
                "length": 30,
                "visibility_min": min(item.visibility for item in candidate.objects),
                "frames": frames,
                "segmentation": {
                    "backend": "sam3",
                    "prompt": str(settings.get("sam3_text_prompt", "human")),
                    "identity_assignment": "highest bbox IoU with MOT17 GT",
                    "confidence_threshold": float(settings.get("sam3_confidence_threshold", 0.5)),
                    "precision": getattr(segmenter, "precision", "unknown"),
                    "min_gt_iou": float(settings.get("sam3_min_gt_iou", 0.3)),
                },
            }
        )

    metadata = {
        "version": 1,
        "description": "Strict 30-frame MOT17 person tracklets segmented with SAM3 text prompt",
        "selection": {
            "detector_view": str(settings.get("detector", "FRCNN")),
            "pedestrian_class": 1,
            "visibility_min": visibility_min,
            "length": 30,
            "stride": stride,
            "all_frames_required": True,
            "sam3_text_prompt": str(settings.get("sam3_text_prompt", "human")),
            "sam3_failure_policy": "reject entire tracklet if any frame has no matched human mask",
        },
        "tracklets": records,
    }
    save_json(output_dir / "tracklets.json", metadata)
    summary = {
        "candidates": len(candidates),
        "candidate_id": candidate_id_override,
        "tracklets": len(records),
        "rejected_during_segmentation": rejected,
        "frames": len(records) * 30,
        "output": str(output_dir / "tracklets.json"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _variable_pool_params(settings: dict[str, Any]) -> dict[str, Any]:
    raw_max = settings.get("max_frames", None)
    return {
        "detector": str(settings.get("detector", "FRCNN")),
        "target_fps": float(settings.get("target_fps", 10.0)),
        "min_frames": int(settings.get("min_frames", 30)),
        "visibility_min": float(settings.get("visibility_min", 0.8)),
        "substitution_window": int(settings.get("visibility_substitution_window", 1)),
        "max_frames": None if raw_max in (None, "null") else int(raw_max),
    }


def inventory_variable(config_file: str | Path) -> dict[str, Any]:
    """Validate FPS-aware variable-length GT windows without loading SAM3."""
    config, path = load_config(config_file)
    settings = config["tracklet_pool"]
    mot_root = config_path(config, path, "paths", "mot17")
    params = _variable_pool_params(settings)
    candidates = collect_variable_fps_tracklets(mot_root, **params)
    lengths = [candidate.length for candidate in candidates]
    result = {
        "mode": "fps_aware_variable_length",
        "source": "MOT17 train GT",
        **params,
        "candidate_tracklets": len(candidates),
        "source_identities": len(identities(candidates)),
        "sequences": sorted({candidate.sequence for candidate in candidates}),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "source_fps_seen": sorted({candidate.source_fps for candidate in candidates if candidate.source_fps}),
        "rule": (
            "presence forms continuous runs; visibility>=min is checked only on "
            "resampled target-fps frames with adjacent substitution; length>=min_frames"
        ),
    }
    output_dir = resolve_path(path.parent, settings["output_dir"])
    save_json(output_dir / "inventory_variable.json", result)
    result["output"] = str(output_dir / "inventory_variable.json")
    return result


def build_variable(
    config_file: str | Path,
    *,
    max_tracklets_override: int | None = None,
    segmenter: SegBackend | None = None,
) -> dict[str, Any]:
    """Segment FPS-aware, variable-length MOT17 tracklets with SAM3."""
    config, path = load_config(config_file)
    settings = config["tracklet_pool"]
    mot_root = config_path(config, path, "paths", "mot17")
    output_dir = resolve_path(path.parent, settings["output_dir"])
    params = _variable_pool_params(settings)
    if params["visibility_min"] < 0.8:
        raise ValueError("this experiment requires tracklet_pool.visibility_min>=0.8")
    if params["min_frames"] < 30:
        raise ValueError("this experiment requires tracklet_pool.min_frames>=30")

    candidates = collect_variable_fps_tracklets(mot_root, **params)
    candidate_ids: dict[int, int] = {id(candidate): index for index, candidate in enumerate(candidates, start=1)}
    random.Random(int(config.get("seed", 0))).shuffle(candidates)
    max_tracklets = int(max_tracklets_override or settings.get("max_tracklets", len(candidates)))
    max_per_identity = int(settings.get("max_per_identity", 2))
    if segmenter is None:
        from segment.sam3 import Sam3Backend

        segmenter = Sam3Backend(
            config_path(config, path, "paths", "sam3_checkpoint"),
            bpe_path=config_path(config, path, "paths", "sam3_bpe"),
            device=str(settings.get("device", "cuda")),
            confidence_threshold=float(settings.get("sam3_confidence_threshold", 0.5)),
            precision=str(settings.get("sam3_precision", "auto")),
        )

    prompt = str(settings.get("sam3_text_prompt", "human"))
    records: list[dict[str, Any]] = []
    rejected = 0
    identity_counts: dict[tuple[str, int], int] = {}
    for candidate in candidates:
        if len(records) >= max_tracklets:
            break
        identity_key = (candidate.sequence, candidate.track_id)
        if identity_counts.get(identity_key, 0) >= max_per_identity:
            continue
        segmented = _segment_tracklet(
            mot_root,
            candidate,
            segmenter,
            min_mask_area=int(settings.get("min_mask_area", 128)),
            context_padding=float(settings.get("sam3_context_padding", 0.2)),
            prompt=prompt,
            min_match_iou=float(settings.get("sam3_min_gt_iou", 0.3)),
        )
        # keep the longest passing sub-run if it is long enough.
        if len(segmented) < params["min_frames"]:
            rejected += 1
            continue
        kept_records = [record for _, record in segmented]
        truncated = len(segmented) < candidate.length
        identity_counts[identity_key] = identity_counts.get(identity_key, 0) + 1
        tracklet_id = len(records) + 1
        relative_dir = Path(f"tracklet_{tracklet_id:06d}")
        frames: list[dict[str, Any]] = []
        for rgba, frame_record in segmented:
            relative_file = relative_dir / f"{int(frame_record['offset']):02d}.png"
            _save_rgba(output_dir / relative_file, rgba)
            frames.append({**frame_record, "file_name": str(relative_file)})
        records.append(
            {
                "id": tracklet_id,
                "category": "person",
                "source": "MOT17",
                "source_candidate_id": candidate_ids[id(candidate)],
                "sequence": candidate.sequence,
                "source_track_id": candidate.track_id,
                "source_fps": candidate.source_fps,
                "target_fps": candidate.target_fps,
                "start_frame": int(kept_records[0]["source_frame"]),
                "end_frame": int(kept_records[-1]["source_frame"]),
                "length": len(segmented),
                "truncated_on_failure": truncated,
                "candidate_length": candidate.length,
                "visibility_min": min(float(record["visibility"]) for record in kept_records),
                "frames": frames,
                "segmentation": {
                    "backend": "sam3",
                    "prompt": prompt,
                    "identity_assignment": "highest bbox IoU with MOT17 GT",
                    "confidence_threshold": float(settings.get("sam3_confidence_threshold", 0.5)),
                    "precision": getattr(segmenter, "precision", "unknown"),
                    "min_gt_iou": float(settings.get("sam3_min_gt_iou", 0.3)),
                },
            }
        )

    metadata = {
        "version": 2,
        "mode": "fps_aware_variable_length",
        "description": "FPS-aware variable-length MOT17 person tracklets segmented with SAM3",
        "selection": {
            "pedestrian_class": 1,
            **params,
            "max_per_identity": max_per_identity,
            "sam3_text_prompt": prompt,
            "sam3_failure_policy": "keep longest consecutive passing sub-run if >= min_frames",
        },
        "tracklets": records,
    }
    save_json(output_dir / "tracklets.json", metadata)
    lengths = [record["length"] for record in records]
    summary = {
        "mode": "fps_aware_variable_length",
        "candidates": len(candidates),
        "tracklets": len(records),
        "rejected_during_segmentation": rejected,
        "frames": sum(lengths),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "distinct_identities": len({(record["sequence"], record["source_track_id"]) for record in records}),
        "output": str(output_dir / "tracklets.json"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _relaunch_in_sam3(
    config_file: Path, max_tracklets: int | None, candidate_id: int | None
) -> int:
    config, path = load_config(config_file)
    python_path = config_path(config, path, "paths", "sam3_python")
    if not python_path.is_file():
        raise FileNotFoundError(
            f"kds-sam3 Python not found: {python_path}. See docs/SAM3_WORKFLOW.md"
        )
    command = [str(python_path), "-m", "pool.build_tracklet_pool", "--config", str(path)]
    if max_tracklets is not None:
        command.extend(["--max-tracklets", str(max_tracklets)])
    if candidate_id is not None:
        command.extend(["--candidate-id", str(candidate_id)])
    return subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False).returncode


def _use_variable_mode(config_file: Path, force_strict: bool) -> bool:
    if force_strict:
        return False
    config, _ = load_config(config_file)
    return "target_fps" in config.get("tracklet_pool", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MOT17 human tracklets with SAM3")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--inventory-only", action="store_true", help="validate GT windows without loading SAM3")
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="force the legacy fixed 30-frame path even when target_fps is set",
    )
    parser.add_argument(
        "--candidate-id",
        type=int,
        default=None,
        help="segment one exact candidate ID (legacy strict path only)",
    )
    args = parser.parse_args()
    variable = _use_variable_mode(args.config, args.strict)
    if args.inventory_only:
        result = inventory_variable(args.config) if variable else inventory(args.config)
        print(result)
        return
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(_relaunch_in_sam3(args.config, args.max_tracklets, args.candidate_id))
    if variable:
        result = build_variable(args.config, max_tracklets_override=args.max_tracklets, segmenter=None)
    else:
        result = build(
            args.config,
            max_tracklets_override=args.max_tracklets,
            candidate_id_override=args.candidate_id,
        )
    print(result)


if __name__ == "__main__":
    main()
