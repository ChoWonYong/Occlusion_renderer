from __future__ import annotations

import argparse
import importlib.util
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.gpu_budget import enforce_account_gpu_budget
from common.io import load_json, save_json
from mining.detected_tracklets import mine_detection_tracklets
from pool.build_tracklet_pool import crop_with_context, select_matching_instance
from segment.base import SegBackend


def inventory(config_file: str | Path) -> dict[str, Any]:
    """Create GT-free Co-DETR+ByteTrack candidate runs without loading SAM3."""
    return mine_detection_tracklets(config_file)


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def segment_detector_frame(
    frame: Mapping[str, Any],
    segmenter: SegBackend,
    settings: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Use a detector box—not a GT box—as the crop and SAM3 identity cue."""
    image = _load_rgb(Path(frame["source_image"]))
    try:
        crop, context, target_box, clipped_box, context_box = crop_with_context(
            image,
            tuple(float(value) for value in frame["source_bbox_xywh"]),
            float(settings.get("sam3_context_padding", 0.2)),
        )
    except ValueError:
        return None
    prompt = str(frame["sam3_prompt"])
    instances = segmenter.detect_and_mask(context, [prompt])
    matched = select_matching_instance(
        instances,
        target_box,
        float(settings.get("sam3_min_detector_iou", 0.3)),
    )
    if matched is None:
        return None
    instance, match_iou = matched
    context_mask = np.asarray(instance.mask, dtype=np.uint8)
    if context_mask.shape != context.shape[:2]:
        raise ValueError(
            f"SAM3 mask shape {context_mask.shape} does not match context {context.shape[:2]}"
        )
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
    rgba = np.dstack((crop, alpha))
    record = {
        **dict(frame),
        "crop_bbox_xywh": clipped_box,
        "sam3_context_bbox_xywh": context_box,
        "mask_area": mask_area,
        "sam3_score": float(instance.score),
        "sam3_bbox_xywh_in_context": [float(value) for value in instance.bbox],
        "sam3_detector_match_iou": float(match_iou),
    }
    return rgba, record


def _longest_passing_run(
    entries: list[tuple[np.ndarray, dict[str, Any]] | None],
) -> list[tuple[np.ndarray, dict[str, Any]]]:
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
            best_start, best_length = current_start, current_length
    run = entries[best_start : best_start + best_length]
    return [entry for entry in run if entry is not None]


def _confidence_stats(frames: list[Mapping[str, Any]]) -> dict[str, float]:
    scores = np.asarray([float(frame["detector_confidence"]) for frame in frames])
    return {
        "min": float(np.min(scores)),
        "p10": float(np.quantile(scores, 0.1)),
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
    }


def _create_segmenter(
    config: Mapping[str, Any], path: Path, settings: Mapping[str, Any]
) -> SegBackend:
    enforce_account_gpu_budget(
        config.get("resources", {}), project_limit_key="phase1_sam_max_gpus"
    )
    from segment.sam3 import Sam3Backend

    return Sam3Backend(
        config_path(config, path, "paths", "sam3_checkpoint"),
        bpe_path=config_path(config, path, "paths", "sam3_bpe"),
        device=str(settings.get("device", "cuda")),
        confidence_threshold=float(settings.get("sam3_confidence_threshold", 0.5)),
        precision=str(settings.get("sam3_precision", "auto")),
    )


def _segment_candidate(
    candidate: Mapping[str, Any],
    segmenter: SegBackend,
    settings: Mapping[str, Any],
    output_dir: Path,
    relative_dir: Path,
    *,
    tracklet_id: int,
) -> dict[str, Any] | None:
    minimum_frames = int(settings.get("min_frames", 30))
    segmented = _longest_passing_run(
        [
            segment_detector_frame(frame, segmenter, settings)
            for frame in candidate["frames"]
        ]
    )
    if len(segmented) < minimum_frames:
        return None

    from PIL import Image

    frames: list[dict[str, Any]] = []
    for offset, (rgba, frame_record) in enumerate(segmented):
        relative_file = relative_dir / f"{offset:02d}.png"
        destination = output_dir / relative_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(destination)
        frames.append({**frame_record, "offset": offset, "file_name": str(relative_file)})
    first, last = frames[0], frames[-1]
    return {
        "id": tracklet_id,
        "source_candidate_id": int(candidate["candidate_id"]),
        "category": str(candidate["category"]),
        "source": f"Co-DETR:{candidate['source_dataset']}",
        "source_dataset": str(candidate["source_dataset"]),
        "sequence": str(candidate["sequence"]),
        "source_track_id": int(candidate["source_track_id"]),
        "source_fps": float(candidate["source_fps"]),
        "target_fps": float(candidate["target_fps"]),
        "start_frame": int(first["frame_index"]),
        "end_frame": int(last["frame_index"]),
        "length": len(frames),
        "candidate_length": int(candidate["length"]),
        "truncated_on_sam3_failure": len(frames) < int(candidate["length"]),
        "detector_confidence": _confidence_stats(frames),
        "frames": frames,
        "segmentation": {
            "backend": "sam3",
            "prompt": "per-frame category prompt (car/human)",
            "crop_source": "Co-DETR bbox associated by ByteTrack",
            "identity_assignment": "highest bbox IoU with detector box; no GT",
            "confidence_threshold": float(settings.get("sam3_confidence_threshold", 0.5)),
            "precision": getattr(segmenter, "precision", "unknown"),
            "min_detector_iou": float(settings.get("sam3_min_detector_iou", 0.3)),
            "gt_used": False,
        },
    }


def build(
    config_file: str | Path,
    *,
    max_tracklets_override: int | None = None,
    segmenter: SegBackend | None = None,
) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["detector_tracklet_pool"]
    candidate_dir = resolve_path(path.parent, settings["candidate_output_dir"])
    candidate_path = candidate_dir / "tracklets.json"
    if not candidate_path.is_file():
        inventory(path)
    candidates = list(load_json(candidate_path).get("tracklets", []))
    random.Random(int(config.get("seed", 0))).shuffle(candidates)
    maximum = int(max_tracklets_override or settings.get("max_tracklets", len(candidates)))
    minimum_frames = int(settings.get("min_frames", 30))
    output_dir = resolve_path(path.parent, settings["output_dir"])

    if segmenter is None:
        segmenter = _create_segmenter(config, path, settings)

    records: list[dict[str, Any]] = []
    rejected = 0
    for candidate in candidates:
        if len(records) >= maximum:
            break
        tracklet_id = len(records) + 1
        record = _segment_candidate(
            candidate,
            segmenter,
            settings,
            output_dir,
            Path(f"tracklet_{tracklet_id:06d}"),
            tracklet_id=tracklet_id,
        )
        if record is None:
            rejected += 1
            continue
        records.append(record)

    metadata = {
        "schema_version": 1,
        "mode": "codetr_bytetrack_sam3",
        "description": "GT-free detector-derived variable-length RGBA tracklets",
        "selection": {
            "gt_used": False,
            "min_frames": minimum_frames,
            "max_frames": settings.get("max_frames", 100),
            "quality_filter": "deferred_to_phase2",
            "sam3_failure_policy": "keep longest consecutive passing sub-run if >= min_frames",
        },
        "tracklets": records,
    }
    output_path = output_dir / "tracklets.json"
    save_json(output_path, metadata)
    lengths = [record["length"] for record in records]
    summary = {
        "mode": metadata["mode"],
        "candidates": len(candidates),
        "tracklets": len(records),
        "rejected_during_segmentation": rejected,
        "frames": sum(lengths),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "gt_used": False,
        "output": str(output_path),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_shard(
    config_file: str | Path,
    *,
    shard_index: int,
    num_shards: int,
    segmenter: SegBackend | None = None,
) -> dict[str, Any]:
    """Segment one disjoint shard; final IDs are assigned only during merge."""
    if num_shards < 2 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards), with num_shards >= 2")
    config, path = load_config(config_file)
    settings = config["detector_tracklet_pool"]
    candidate_dir = resolve_path(path.parent, settings["candidate_output_dir"])
    candidates = list(load_json(candidate_dir / "tracklets.json").get("tracklets", []))
    random.Random(int(config.get("seed", 0))).shuffle(candidates)
    assigned = [
        (candidate_order, candidate)
        for candidate_order, candidate in enumerate(candidates)
        if candidate_order % num_shards == shard_index
    ]
    output_dir = resolve_path(path.parent, settings["output_dir"])
    shard_dir = Path("_shards") / f"shard_{shard_index:02d}"
    if segmenter is None:
        segmenter = _create_segmenter(config, path, settings)

    records: list[dict[str, Any]] = []
    rejected = 0
    for candidate_order, candidate in assigned:
        relative_dir = shard_dir / f"candidate_{int(candidate['candidate_id']):06d}"
        record = _segment_candidate(
            candidate,
            segmenter,
            settings,
            output_dir,
            relative_dir,
            tracklet_id=int(candidate["candidate_id"]),
        )
        if record is None:
            rejected += 1
            continue
        records.append(
            {
                **record,
                "candidate_order": candidate_order,
                "shard_index": shard_index,
            }
        )

    payload = {
        "schema_version": 1,
        "mode": "codetr_bytetrack_sam3_shard",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "seed": int(config.get("seed", 0)),
        "assigned_candidates": len(assigned),
        "rejected_during_segmentation": rejected,
        "tracklets": records,
    }
    shard_path = output_dir / "_shards" / f"shard_{shard_index:02d}.json"
    save_json(shard_path, payload)
    return {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "assigned_candidates": len(assigned),
        "tracklets": len(records),
        "rejected_during_segmentation": rejected,
        "output": str(shard_path),
    }


def merge_shards(
    config_file: str | Path,
    *,
    num_shards: int,
    max_tracklets_override: int | None = None,
) -> dict[str, Any]:
    """Merge successful shard outputs in the original deterministic order."""
    config, path = load_config(config_file)
    settings = config["detector_tracklet_pool"]
    output_dir = resolve_path(path.parent, settings["output_dir"])
    records: list[dict[str, Any]] = []
    assigned_candidates = 0
    rejected = 0
    for shard_index in range(num_shards):
        shard_path = output_dir / "_shards" / f"shard_{shard_index:02d}.json"
        payload = load_json(shard_path)
        if int(payload.get("shard_index", -1)) != shard_index:
            raise ValueError(f"wrong shard index in {shard_path}")
        if int(payload.get("num_shards", -1)) != num_shards:
            raise ValueError(f"wrong shard count in {shard_path}")
        if int(payload.get("seed", -1)) != int(config.get("seed", 0)):
            raise ValueError(f"wrong shuffle seed in {shard_path}")
        assigned_candidates += int(payload.get("assigned_candidates", 0))
        rejected += int(payload.get("rejected_during_segmentation", 0))
        records.extend(dict(record) for record in payload.get("tracklets", []))

    records.sort(key=lambda record: int(record["candidate_order"]))
    maximum = int(max_tracklets_override or settings.get("max_tracklets", len(records)))
    selected: list[dict[str, Any]] = []
    for tracklet_id, record in enumerate(records[:maximum], start=1):
        merged = dict(record)
        merged["id"] = tracklet_id
        selected.append(merged)

    minimum_frames = int(settings.get("min_frames", 30))
    metadata = {
        "schema_version": 1,
        "mode": "codetr_bytetrack_sam3",
        "description": "GT-free detector-derived variable-length RGBA tracklets",
        "selection": {
            "gt_used": False,
            "min_frames": minimum_frames,
            "max_frames": settings.get("max_frames", 100),
            "quality_filter": "deferred_to_phase2",
            "sam3_failure_policy": "keep longest consecutive passing sub-run if >= min_frames",
            "parallel_shards": num_shards,
            "merge_order": "seeded candidate order before round-robin sharding",
        },
        "tracklets": selected,
    }
    output_path = output_dir / "tracklets.json"
    save_json(output_path, metadata)
    lengths = [record["length"] for record in selected]
    summary = {
        "mode": metadata["mode"],
        "candidates": assigned_candidates,
        "accepted_candidates_before_cap": len(records),
        "tracklets": len(selected),
        "rejected_during_segmentation": rejected,
        "frames": sum(lengths),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "category_counts": dict(sorted(Counter(record["category"] for record in selected).items())),
        "parallel_shards": num_shards,
        "gt_used": False,
        "output": str(output_path),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def run_parallel(
    config_file: str | Path,
    *,
    workers: int,
    max_tracklets_override: int | None = None,
) -> dict[str, Any]:
    """Launch one isolated SAM3 process per explicitly visible GPU."""
    if workers < 2:
        raise ValueError("parallel SAM3 requires at least two workers")
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
    config, path = load_config(config_file)
    enforce_account_gpu_budget(
        config.get("resources", {}), project_limit_key="phase1_sam_max_gpus"
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
                    "pool.build_detector_tracklet_pool",
                    "--config",
                    str(path),
                    "--shard-index",
                    str(shard_index),
                    "--num-shards",
                    str(workers),
                ],
                cwd=project_root,
                env=environment,
            )
        )
    return_codes = [process.wait() for process in processes]
    failed = [index for index, code in enumerate(return_codes) if code != 0]
    if failed:
        raise RuntimeError(f"SAM3 shard workers failed: {failed}")
    return merge_shards(
        path,
        num_shards=workers,
        max_tracklets_override=max_tracklets_override,
    )


def _relaunch(config_file: Path, max_tracklets: int | None, workers: int) -> int:
    config, path = load_config(config_file)
    python_path = config_path(config, path, "paths", "sam3_python")
    if not python_path.is_file():
        raise FileNotFoundError(f"SAM3 Python not found: {python_path}")
    command = [
        str(python_path), "-m", "pool.build_detector_tracklet_pool",
        "--config", str(path),
    ]
    if max_tracklets is not None:
        command.extend(["--max-tracklets", str(max_tracklets)])
    if workers != 1:
        command.extend(["--workers", str(workers)])
    return subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], check=False
    ).returncode


def _finish_with_quality_filter(config_file: Path, raw_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Run the configured GT-free quality gate after the raw SAM3 pool is complete."""
    config, path = load_config(config_file)
    quality = config.get("detector_tracklet_pool", {}).get("quality_filter", {})
    result: dict[str, Any] = {"raw_pool": dict(raw_summary)}
    if bool(quality.get("enabled", False)):
        from pool.filter_detector_tracklet_pool import run as run_quality_filter

        result["quality_filter"] = run_quality_filter(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine Co-DETR+ByteTrack candidates and build a GT-free SAM3 pool"
    )
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inventory_only:
        print(inventory(args.config))
        return
    config, path = load_config(args.config)
    candidate_path = (
        resolve_path(path.parent, config["detector_tracklet_pool"]["candidate_output_dir"])
        / "tracklets.json"
    )
    if importlib.util.find_spec("sam3") is None:
        if not candidate_path.is_file():
            print(inventory(path))
        raise SystemExit(_relaunch(path, args.max_tracklets, args.workers))
    if args.shard_index is not None:
        if args.num_shards is None:
            parser.error("--shard-index requires --num-shards")
        print(
            build_shard(
                path,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        )
        return
    if args.workers > 1:
        summary = run_parallel(
            path,
            workers=args.workers,
            max_tracklets_override=args.max_tracklets,
        )
        print(_finish_with_quality_filter(path, summary))
        return
    summary = build(path, max_tracklets_override=args.max_tracklets)
    print(_finish_with_quality_filter(path, summary))


if __name__ == "__main__":
    main()
