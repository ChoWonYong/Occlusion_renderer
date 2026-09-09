from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.gpu_budget import enforce_account_gpu_budget
from common.io import load_json, save_json, save_jsonl
from pool.crops import bbox_iou, crop_with_context, select_matching_instance
from segment.base import SegBackend


POLICIES = ("a_bbox", "b_context")
POLICY_DIRS = {
    "a_bbox": "variant_a_bbox_clipped",
    "b_context": "variant_b_context_preserved",
}
MOTS_CLASS_IDS = {"car": 1, "person": 2}
MOTS_IGNORE_ID = 10000


@lru_cache(maxsize=16)
def _load_rgb(path_text: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path_text) as image:
        return np.asarray(image.convert("RGB")).copy()


@lru_cache(maxsize=128)
def _load_mots(path_text: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path_text) as image:
        return np.asarray(image, dtype=np.uint16).copy()


def make_mask_variants(
    crop: np.ndarray,
    context: np.ndarray,
    context_mask: np.ndarray,
    target_box: Sequence[float],
    clipped_box: Sequence[float],
    context_box: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Build A and B from one selected SAM3 context mask."""
    if context_mask.shape != context.shape[:2]:
        raise ValueError(
            f"SAM3 mask shape {context_mask.shape} does not match context {context.shape[:2]}"
        )
    crop_x, crop_y = int(round(float(target_box[0]))), int(round(float(target_box[1])))
    crop_height, crop_width = crop.shape[:2]
    alpha_a = (
        context_mask[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width] != 0
    ).astype(np.uint8) * 255
    if alpha_a.shape != crop.shape[:2]:
        raise ValueError("detector crop lies outside the SAM3 context mask")
    alpha_b = (context_mask != 0).astype(np.uint8) * 255
    return {
        "a_bbox": {
            "rgba": np.dstack((crop, alpha_a)),
            "alpha": alpha_a,
            "crop_bbox_xywh": [float(value) for value in clipped_box],
            "mask_area": int(np.count_nonzero(alpha_a)),
        },
        "b_context": {
            "rgba": np.dstack((context, alpha_b)),
            "alpha": alpha_b,
            "crop_bbox_xywh": [float(value) for value in context_box],
            "mask_area": int(np.count_nonzero(alpha_b)),
        },
    }


def _bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def match_mots_instance(
    labels: np.ndarray,
    category: str,
    detector_bbox_xywh: Sequence[float],
) -> tuple[np.ndarray, int, float, list[float]] | None:
    """Post-hoc GT association by detector-box IoU, independent of A/B masks."""
    class_id = MOTS_CLASS_IDS.get(str(category))
    if class_id is None:
        return None
    ranked: list[tuple[float, int, np.ndarray, list[float]]] = []
    for instance_id in np.unique(labels):
        value = int(instance_id)
        if value == MOTS_IGNORE_ID or value // 1000 != class_id:
            continue
        mask = labels == value
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        ranked.append(
            (
                bbox_iou(
                    [float(item) for item in detector_bbox_xywh],
                    bbox,
                ),
                value,
                mask,
                bbox,
            )
        )
    if not ranked:
        return None
    overlap, instance_id, mask, bbox = max(ranked, key=lambda item: item[0])
    return mask, instance_id, float(overlap), bbox


def evaluate_mask(
    alpha: np.ndarray,
    crop_bbox_xywh: Sequence[float],
    target_mask: np.ndarray,
    ignore_mask: np.ndarray | None = None,
    *,
    complete_recall_threshold: float = 0.9,
) -> dict[str, Any]:
    """Evaluate a cropped alpha mask while counting GT outside the crop as FN."""
    x, y, width, height = (int(round(float(value))) for value in crop_bbox_xywh)
    if alpha.shape != (height, width):
        raise ValueError(
            f"alpha shape {alpha.shape} does not match crop bbox {(height, width)}"
        )
    if target_mask.ndim != 2:
        raise ValueError("target mask must be two-dimensional")
    ignored = np.zeros_like(target_mask, dtype=bool) if ignore_mask is None else ignore_mask.astype(bool)
    if ignored.shape != target_mask.shape:
        raise ValueError("ignore mask shape does not match target mask")
    x2, y2 = x + width, y + height
    if x < 0 or y < 0 or x2 > target_mask.shape[1] or y2 > target_mask.shape[0]:
        raise ValueError("crop bbox lies outside the GT image")
    predicted = alpha != 0
    target_region = target_mask[y:y2, x:x2].astype(bool)
    valid_region = ~ignored[y:y2, x:x2]
    predicted_valid = predicted & valid_region
    true_positive = int(np.count_nonzero(predicted_valid & target_region))
    false_positive = int(np.count_nonzero(predicted_valid & ~target_region))
    target_area = int(np.count_nonzero(target_mask.astype(bool) & ~ignored))
    false_negative = max(0, target_area - true_positive)
    union = true_positive + false_positive + false_negative
    predicted_area = true_positive + false_positive
    iou = true_positive / union if union else 1.0
    precision = true_positive / predicted_area if predicted_area else 0.0
    recall = true_positive / target_area if target_area else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "gt_area": target_area,
        "predicted_valid_area": predicted_area,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "complete": bool(recall >= complete_recall_threshold),
    }


def longest_passing_positions(passing: Sequence[bool]) -> list[int]:
    best_start = best_length = 0
    current_start = current_length = 0
    for index, passed in enumerate(passing):
        if not passed:
            current_length = 0
            current_start = index + 1
            continue
        if current_length == 0:
            current_start = index
        current_length += 1
        if current_length > best_length:
            best_start, best_length = current_start, current_length
    return list(range(best_start, best_start + best_length))


def _create_segmenter(
    config: Mapping[str, Any], path: Path, pool_settings: Mapping[str, Any]
) -> SegBackend:
    from segment.sam3 import Sam3Backend

    return Sam3Backend(
        config_path(config, path, "paths", "sam3_checkpoint"),
        bpe_path=config_path(config, path, "paths", "sam3_bpe"),
        device=str(pool_settings.get("device", "cuda")),
        confidence_threshold=float(pool_settings.get("sam3_confidence_threshold", 0.5)),
        precision=str(pool_settings.get("sam3_precision", "auto")),
    )


def _segment_frame(
    frame: Mapping[str, Any],
    segmenter: SegBackend,
    *,
    padding: float,
    min_detector_iou: float,
    min_mask_area: int,
    mots_root: Path,
    gt_bbox_iou_threshold: float,
    complete_recall_threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base = {
        "source_dataset": str(frame["source_dataset"]),
        "sequence": str(frame["sequence"]),
        "frame_index": int(frame["frame_index"]),
        "source_image": str(frame["source_image"]),
        "source_bbox_xywh": [float(value) for value in frame["source_bbox_xywh"]],
        "category": str(frame["category"]),
        "sam3_prompt": str(frame["sam3_prompt"]),
        "detector_confidence": float(frame["detector_confidence"]),
    }
    image = _load_rgb(str(frame["source_image"]))
    try:
        crop, context, target_box, clipped_box, context_box = crop_with_context(
            image,
            tuple(float(value) for value in frame["source_bbox_xywh"]),
            padding,
        )
    except ValueError:
        return {**base, "status": "invalid_crop"}, {}
    instances = segmenter.detect_and_mask(context, [str(frame["sam3_prompt"])])
    if not instances:
        return {**base, "status": "no_sam3_instance", "sam3_instance_count": 0}, {}
    matched = select_matching_instance(instances, target_box, min_detector_iou)
    if matched is None:
        return {
            **base,
            "status": "no_detector_iou_match",
            "sam3_instance_count": len(instances),
        }, {}
    instance, match_iou = matched
    variants = make_mask_variants(
        crop,
        context,
        np.asarray(instance.mask, dtype=np.uint8),
        target_box,
        clipped_box,
        context_box,
    )
    audit: dict[str, Any] = {
        **base,
        "status": "ok",
        "sam3_instance_count": len(instances),
        "sam3_score": float(instance.score),
        "sam3_bbox_xywh_in_context": [float(value) for value in instance.bbox],
        "sam3_detector_match_iou": float(match_iou),
        "sam3_context_bbox_xywh": [float(value) for value in context_box],
        "policies": {
            policy: {
                "crop_bbox_xywh": variant["crop_bbox_xywh"],
                "mask_area": int(variant["mask_area"]),
                "passes_min_mask_area": int(variant["mask_area"]) >= min_mask_area,
            }
            for policy, variant in variants.items()
        },
    }
    gt: dict[str, Any] = {"status": "not_applicable", "matched": False}
    if str(frame["source_dataset"]) == "KITTI" and str(frame["category"]) in MOTS_CLASS_IDS:
        gt_path = (
            mots_root
            / "instances"
            / str(frame["sequence"])
            / f"{int(frame['frame_index']):06d}.png"
        )
        if not gt_path.is_file():
            gt = {"status": "missing_frame", "matched": False}
        else:
            labels = _load_mots(str(gt_path))
            if labels.shape != image.shape[:2]:
                gt = {
                    "status": "shape_mismatch",
                    "matched": False,
                    "gt_shape": list(labels.shape),
                    "image_shape": list(image.shape[:2]),
                }
            else:
                target = match_mots_instance(labels, str(frame["category"]), clipped_box)
                if target is None:
                    gt = {"status": "no_same_class_instance", "matched": False}
                else:
                    target_mask, instance_id, gt_bbox_iou, gt_bbox = target
                    is_match = gt_bbox_iou >= gt_bbox_iou_threshold
                    gt = {
                        "status": "matched" if is_match else "below_bbox_iou",
                        "matched": is_match,
                        "instance_id": int(instance_id),
                        "bbox_xywh": gt_bbox,
                        "detector_bbox_iou": float(gt_bbox_iou),
                    }
                    if is_match:
                        ignore_mask = labels == MOTS_IGNORE_ID
                        gt["metrics"] = {
                            policy: evaluate_mask(
                                variant["alpha"],
                                variant["crop_bbox_xywh"],
                                target_mask,
                                ignore_mask,
                                complete_recall_threshold=complete_recall_threshold,
                            )
                            for policy, variant in variants.items()
                        }
    audit["gt"] = gt
    return audit, {policy: variant["rgba"] for policy, variant in variants.items()}


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(path)


def _process_candidate(
    candidate: Mapping[str, Any],
    candidate_order: int,
    segmenter: SegBackend,
    *,
    pool_settings: Mapping[str, Any],
    experiment: Mapping[str, Any],
    mots_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    minimum_frames = int(experiment.get("min_frames", pool_settings.get("min_frames", 30)))
    minimum_area = int(experiment.get("min_mask_area", pool_settings.get("min_mask_area", 128)))
    frame_results: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for frame in candidate["frames"]:
        frame_results.append(
            _segment_frame(
                frame,
                segmenter,
                padding=float(experiment.get("context_padding", 0.2)),
                min_detector_iou=float(pool_settings.get("sam3_min_detector_iou", 0.3)),
                min_mask_area=minimum_area,
                mots_root=mots_root,
                gt_bbox_iou_threshold=float(experiment.get("gt_bbox_iou_threshold", 0.5)),
                complete_recall_threshold=float(experiment.get("complete_recall_threshold", 0.9)),
            )
        )
    audits = [result[0] for result in frame_results]
    policy_payloads: dict[str, Any] = {}
    candidate_id = int(candidate["candidate_id"])
    for policy in POLICIES:
        passing = [
            audit.get("status") == "ok"
            and bool(audit.get("policies", {}).get(policy, {}).get("passes_min_mask_area", False))
            for audit in audits
        ]
        positions = longest_passing_positions(passing)
        accepted = len(positions) >= minimum_frames
        retained_frames: list[dict[str, Any]] = []
        if accepted:
            for offset, position in enumerate(positions):
                audit, images = frame_results[position]
                relative_file = (
                    Path(POLICY_DIRS[policy])
                    / f"candidate_{candidate_id:06d}"
                    / f"{int(audit['frame_index']):06d}.png"
                )
                _save_rgba(output_dir / relative_file, images[policy])
                policy_record = audit["policies"][policy]
                retained_frames.append(
                    {
                        "offset": offset,
                        "file_name": str(relative_file),
                        "source_image": audit["source_image"],
                        "sequence": audit["sequence"],
                        "frame_index": int(audit["frame_index"]),
                        "source_bbox_xywh": audit["source_bbox_xywh"],
                        "crop_bbox_xywh": policy_record["crop_bbox_xywh"],
                        "sam3_context_bbox_xywh": audit["sam3_context_bbox_xywh"],
                        "mask_area": int(policy_record["mask_area"]),
                        "sam3_score": float(audit["sam3_score"]),
                        "sam3_detector_match_iou": float(audit["sam3_detector_match_iou"]),
                        "detector_confidence": float(audit["detector_confidence"]),
                        "gt": audit.get("gt", {}),
                    }
                )
        policy_payloads[policy] = {
            "accepted": accepted,
            "longest_passing_length": len(positions),
            "retained_positions": positions if accepted else [],
            "frames": retained_frames,
        }
    return {
        "candidate_order": int(candidate_order),
        "candidate_id": candidate_id,
        "category": str(candidate["category"]),
        "source_dataset": str(candidate["source_dataset"]),
        "sequence": str(candidate["sequence"]),
        "source_track_id": int(candidate["source_track_id"]),
        "candidate_length": int(candidate["length"]),
        "frame_audits": audits,
        "policies": policy_payloads,
    }


def _load_jsonl_recover(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    damaged = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                damaged = True
                break
    if damaged:
        save_jsonl(path, rows)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ordered_candidates(
    config: Mapping[str, Any], path: Path
) -> tuple[list[dict[str, Any]], Path]:
    settings = config["detector_tracklet_pool"]
    candidate_path = (
        resolve_path(path.parent, settings["candidate_output_dir"]) / "tracklets.json"
    )
    candidates = [dict(item) for item in load_json(candidate_path).get("tracklets", [])]
    random.Random(int(config.get("seed", 0))).shuffle(candidates)
    return candidates, candidate_path


def _signature(
    config: Mapping[str, Any], path: Path, *, workers: int, candidate_path: Path
) -> dict[str, Any]:
    experiment = config["sam3_context_ablation"]
    pool_settings = config["detector_tracklet_pool"]
    stat = candidate_path.stat()
    return {
        "schema_version": 1,
        "workers": workers,
        "seed": int(config.get("seed", 0)),
        "candidate_path": str(candidate_path),
        "candidate_size": stat.st_size,
        "candidate_mtime_ns": stat.st_mtime_ns,
        "context_padding": float(experiment.get("context_padding", 0.2)),
        "min_mask_area": int(experiment.get("min_mask_area", 128)),
        "min_frames": int(experiment.get("min_frames", 30)),
        "max_tracklets": int(experiment.get("max_tracklets", 200)),
        "sam3_confidence_threshold": float(pool_settings.get("sam3_confidence_threshold", 0.5)),
        "sam3_min_detector_iou": float(pool_settings.get("sam3_min_detector_iou", 0.3)),
        "gt_bbox_iou_threshold": float(experiment.get("gt_bbox_iou_threshold", 0.5)),
        "complete_recall_threshold": float(experiment.get("complete_recall_threshold", 0.9)),
        "pairing": "one SAM3 inference and one bbox-IoU-selected instance per frame",
    }


def _prepare_output(config_file: Path, *, workers: int) -> tuple[dict[str, Any], Path]:
    config, path = load_config(config_file)
    candidates, candidate_path = _ordered_candidates(config, path)
    output_dir = resolve_path(path.parent, config["sam3_context_ablation"]["output_dir"])
    signature = _signature(config, path, workers=workers, candidate_path=candidate_path)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = load_json(metadata_path)
        if existing != signature:
            raise RuntimeError(
                f"existing ablation metadata differs from this run: {metadata_path}"
            )
    else:
        save_json(metadata_path, signature)
    return {**signature, "candidates": len(candidates)}, output_dir


def run_shard(config_file: str | Path, *, shard_index: int, num_shards: int) -> dict[str, Any]:
    config, path = load_config(config_file)
    candidates, candidate_path = _ordered_candidates(config, path)
    experiment = config["sam3_context_ablation"]
    pool_settings = config["detector_tracklet_pool"]
    output_dir = resolve_path(path.parent, experiment["output_dir"])
    signature = _signature(config, path, workers=num_shards, candidate_path=candidate_path)
    if load_json(output_dir / "metadata.json") != signature:
        raise RuntimeError("ablation metadata does not match shard configuration")
    assigned = [
        (order, candidate)
        for order, candidate in enumerate(candidates)
        if order % num_shards == shard_index
    ]
    shard_path = output_dir / "_shards" / f"shard_{shard_index:02d}.jsonl"
    existing = _load_jsonl_recover(shard_path)
    by_candidate = {int(row["candidate_id"]): row for row in existing}
    pending = [item for item in assigned if int(item[1]["candidate_id"]) not in by_candidate]
    if pending:
        segmenter = _create_segmenter(config, path, pool_settings)
        mots_root = resolve_path(path.parent, experiment["mots_root"])
        for completed, (candidate_order, candidate) in enumerate(pending, start=1):
            row = _process_candidate(
                candidate,
                candidate_order,
                segmenter,
                pool_settings=pool_settings,
                experiment=experiment,
                mots_root=mots_root,
                output_dir=output_dir,
            )
            _append_jsonl(shard_path, row)
            by_candidate[int(row["candidate_id"])] = row
            print(
                f"shard {shard_index}: {len(existing) + completed}/{len(assigned)} "
                f"candidate={row['candidate_id']} "
                f"A={row['policies']['a_bbox']['longest_passing_length']} "
                f"B={row['policies']['b_context']['longest_passing_length']}",
                flush=True,
            )
    missing = [
        int(candidate["candidate_id"])
        for _, candidate in assigned
        if int(candidate["candidate_id"]) not in by_candidate
    ]
    if missing:
        raise RuntimeError(f"shard {shard_index} is incomplete: {missing[:10]}")
    done = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "assigned_candidates": len(assigned),
        "completed_candidates": len(assigned),
        "output": str(shard_path),
    }
    save_json(output_dir / "_shards" / f"shard_{shard_index:02d}.done.json", done)
    return done


def _policy_summary(
    rows: Sequence[Mapping[str, Any]], policy: str, maximum: int
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    accepted = [row for row in rows if bool(row["policies"][policy]["accepted"])]
    selected = accepted[:maximum]
    summary = {
        "candidates": len(rows),
        "accepted_before_cap": len(accepted),
        "tracklets_after_cap": len(selected),
        "frames_before_cap": sum(
            int(row["policies"][policy]["longest_passing_length"]) for row in accepted
        ),
        "frames_after_cap": sum(
            int(row["policies"][policy]["longest_passing_length"]) for row in selected
        ),
        "category_counts_after_cap": dict(
            sorted(Counter(str(row["category"]) for row in selected).items())
        ),
        "source_counts_after_cap": dict(
            sorted(Counter(str(row["source_dataset"]) for row in selected).items())
        ),
    }
    return summary, selected


def _write_policy_manifest(
    output_dir: Path,
    policy: str,
    selected: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> None:
    tracklets: list[dict[str, Any]] = []
    for tracklet_id, row in enumerate(selected, start=1):
        frames = list(row["policies"][policy]["frames"])
        tracklets.append(
            {
                "id": tracklet_id,
                "source_candidate_id": int(row["candidate_id"]),
                "category": str(row["category"]),
                "source_dataset": str(row["source_dataset"]),
                "sequence": str(row["sequence"]),
                "source_track_id": int(row["source_track_id"]),
                "start_frame": int(frames[0]["frame_index"]),
                "end_frame": int(frames[-1]["frame_index"]),
                "length": len(frames),
                "candidate_length": int(row["candidate_length"]),
                "frames": frames,
                "segmentation": {
                    "backend": "sam3",
                    "paired_ablation_policy": policy,
                    "context_padding": float(experiment.get("context_padding", 0.2)),
                    "gt_used_for_selection": False,
                },
            }
        )
    destination = output_dir / POLICY_DIRS[policy]
    save_json(
        destination / "tracklets.json",
        {
            "schema_version": 1,
            "mode": f"sam3_crop_ablation_{policy}",
            "selection": {
                "min_frames": int(experiment.get("min_frames", 30)),
                "min_mask_area": int(experiment.get("min_mask_area", 128)),
                "gt_used": False,
            },
            "tracklets": tracklets,
        },
    )
    save_json(destination / "summary.json", dict(summary))


def _metric_summary(audits: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    metrics = [
        audit["gt"]["metrics"][policy]
        for audit in audits
        if audit.get("gt", {}).get("matched")
    ]
    if not metrics:
        return {
            "frames": 0,
            "macro": {"iou": 0.0, "precision": 0.0, "recall": 0.0},
            "micro": {"iou": 0.0, "precision": 0.0, "recall": 0.0},
            "complete_frames": 0,
            "complete_rate": 0.0,
        }
    macro = {
        name: float(np.mean([float(metric[name]) for metric in metrics]))
        for name in ("iou", "precision", "recall")
    }
    tp = sum(int(metric["tp"]) for metric in metrics)
    fp = sum(int(metric["fp"]) for metric in metrics)
    fn = sum(int(metric["fn"]) for metric in metrics)
    complete = sum(bool(metric["complete"]) for metric in metrics)
    return {
        "frames": len(metrics),
        "macro": macro,
        "micro": {
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        },
        "pixels": {"tp": tp, "fp": fp, "fn": fn},
        "complete_frames": complete,
        "complete_rate": complete / len(metrics),
    }


def _paired_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audits = [audit for row in rows for audit in row["frame_audits"]]
    matched = [audit for audit in audits if audit.get("gt", {}).get("matched")]
    by_class: dict[str, Any] = {}
    for category in ("car", "person"):
        class_audits = [audit for audit in matched if audit["category"] == category]
        by_class[category] = {
            policy: _metric_summary(class_audits, policy) for policy in POLICIES
        }
    result = {
        "frame_accounting": {
            "candidate_frames": len(audits),
            "sam3_common_success": sum(audit["status"] == "ok" for audit in audits),
            "sam3_status_counts": dict(sorted(Counter(audit["status"] for audit in audits).items())),
            "mots_status_counts": dict(
                sorted(Counter(audit.get("gt", {}).get("status", "not_run") for audit in audits).items())
            ),
            "mots_matched_frames": len(matched),
        },
        "overall": {policy: _metric_summary(matched, policy) for policy in POLICIES},
        "per_class": by_class,
    }
    for group_name, group_audits in [("overall", matched)] + [
        (category, [audit for audit in matched if audit["category"] == category])
        for category in ("car", "person")
    ]:
        if not group_audits:
            continue
        deltas = {
            name: [
                float(audit["gt"]["metrics"]["b_context"][name])
                - float(audit["gt"]["metrics"]["a_bbox"][name])
                for audit in group_audits
            ]
            for name in ("iou", "precision", "recall")
        }
        paired = {
            "frames": len(group_audits),
            "mean_delta_b_minus_a": {
                name: float(np.mean(values)) for name, values in deltas.items()
            },
            "additional_true_positive_pixels": sum(
                int(audit["gt"]["metrics"]["b_context"]["tp"])
                - int(audit["gt"]["metrics"]["a_bbox"]["tp"])
                for audit in group_audits
            ),
            "additional_false_positive_pixels": sum(
                int(audit["gt"]["metrics"]["b_context"]["fp"])
                - int(audit["gt"]["metrics"]["a_bbox"]["fp"])
                for audit in group_audits
            ),
        }
        if group_name == "overall":
            result["paired"] = paired
        else:
            result["per_class"][group_name]["paired"] = paired
    return result


def _write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    metrics = summary["mask_metrics"]
    a = metrics["overall"]["a_bbox"]
    b = metrics["overall"]["b_context"]
    delta = metrics["paired"]["mean_delta_b_minus_a"]
    lines = [
        "# SAM3 crop-policy A/B result",
        "",
        "A clips the selected SAM3 mask back to the Co-DETR bbox. B preserves the full 20% context crop.",
        "Each pair comes from the same SAM3 inference and the same bbox-IoU-selected instance.",
        "MOTS GT is used only for this post-hoc audit.",
        "",
        "## Tracklet retention",
        "",
        "| Policy | Accepted before cap | Tracklets after cap | Frames after cap |",
        "|---|---:|---:|---:|",
        f"| A bbox clip | {summary['tracklets']['a_bbox']['accepted_before_cap']} | {summary['tracklets']['a_bbox']['tracklets_after_cap']} | {summary['tracklets']['a_bbox']['frames_after_cap']} |",
        f"| B context | {summary['tracklets']['b_context']['accepted_before_cap']} | {summary['tracklets']['b_context']['tracklets_after_cap']} | {summary['tracklets']['b_context']['frames_after_cap']} |",
        "",
        "## MOTS-matched frame metrics",
        "",
        "| Policy | Frames | Mask IoU | Precision | Recall | Complete recall>=0.90 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| A bbox clip | {a['frames']} | {a['macro']['iou']:.4f} | {a['macro']['precision']:.4f} | {a['macro']['recall']:.4f} | {a['complete_rate']:.4f} |",
        f"| B context | {b['frames']} | {b['macro']['iou']:.4f} | {b['macro']['precision']:.4f} | {b['macro']['recall']:.4f} | {b['complete_rate']:.4f} |",
        f"| B - A |  | {delta['iou']:+.4f} | {delta['precision']:+.4f} | {delta['recall']:+.4f} |  |",
        "",
        "## Per class",
        "",
        "| Class | Policy | Frames | IoU | Precision | Recall | Complete rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for category in ("car", "person"):
        for policy, label in (("a_bbox", "A"), ("b_context", "B")):
            item = metrics["per_class"][category][policy]
            lines.append(
                f"| {category} | {label} | {item['frames']} | {item['macro']['iou']:.4f} | "
                f"{item['macro']['precision']:.4f} | {item['macro']['recall']:.4f} | "
                f"{item['complete_rate']:.4f} |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_shards(config_file: str | Path, *, num_shards: int) -> dict[str, Any]:
    config, path = load_config(config_file)
    experiment = config["sam3_context_ablation"]
    output_dir = resolve_path(path.parent, experiment["output_dir"])
    rows: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        done_path = output_dir / "_shards" / f"shard_{shard_index:02d}.done.json"
        if not done_path.is_file():
            raise RuntimeError(f"missing completed shard marker: {done_path}")
        shard_path = output_dir / "_shards" / f"shard_{shard_index:02d}.jsonl"
        rows.extend(_load_jsonl_recover(shard_path))
    rows.sort(key=lambda row: int(row["candidate_order"]))
    candidates, _ = _ordered_candidates(config, path)
    if len(rows) != len(candidates):
        raise RuntimeError(f"merged {len(rows)} candidates, expected {len(candidates)}")
    maximum = int(experiment.get("max_tracklets", 200))
    tracklet_summaries: dict[str, Any] = {}
    for policy in POLICIES:
        policy_summary, selected = _policy_summary(rows, policy, maximum)
        tracklet_summaries[policy] = policy_summary
        _write_policy_manifest(output_dir, policy, selected, policy_summary, experiment)
    mask_metrics = _paired_metrics(rows)
    summary = {
        "mode": "sam3_context_crop_ablation",
        "pairing": "one SAM3 inference and one selected instance per frame",
        "context_padding": float(experiment.get("context_padding", 0.2)),
        "candidates": len(rows),
        "tracklets": tracklet_summaries,
        "mask_metrics": mask_metrics,
        "gt_usage": "post-hoc audit only; never used for SAM3 selection or tracklet acceptance",
    }
    save_json(output_dir / "summary.json", summary)
    _write_report(output_dir, summary)
    return summary


def run_parallel(config_file: str | Path, *, workers: int) -> dict[str, Any]:
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip() and item.strip() != "-1"
    ]
    if len(visible) != workers or len(set(visible)) != workers:
        raise RuntimeError(
            f"--workers {workers} requires exactly {workers} distinct CUDA_VISIBLE_DEVICES"
        )
    config, path = load_config(config_file)
    enforce_account_gpu_budget(
        config.get("resources", {}), project_limit_key="phase1_sam_max_gpus"
    )
    _prepare_output(path, workers=workers)
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
                    "eval.compare_sam3_crop_policies",
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
        raise RuntimeError(f"SAM3 A/B shard workers failed: {failed}")
    return merge_shards(path, num_shards=workers)


def _relaunch(config_file: Path, workers: int) -> int:
    config, path = load_config(config_file)
    python_path = config_path(config, path, "paths", "sam3_python")
    if not python_path.is_file():
        raise FileNotFoundError(f"SAM3 Python not found: {python_path}")
    return subprocess.run(
        [
            str(python_path),
            "-m",
            "eval.compare_sam3_crop_policies",
            "--config",
            str(path),
            "--workers",
            str(workers),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired SAM3 bbox-clip versus context-mask audit")
    parser.add_argument("--config", type=Path, default=Path("configs/sam3_context_ablation.yaml"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path_value = args.config.expanduser().resolve()
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(_relaunch(config_path_value, args.workers))
    if args.shard_index is not None:
        if args.num_shards is None:
            parser.error("--shard-index requires --num-shards")
        print(
            run_shard(
                config_path_value,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        )
        return
    if args.workers > 1:
        result = run_parallel(config_path_value, workers=args.workers)
    else:
        _prepare_output(config_path_value, workers=1)
        run_shard(config_path_value, shard_index=0, num_shards=1)
        result = merge_shards(config_path_value, num_shards=1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
