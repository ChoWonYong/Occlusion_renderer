from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from common.config import load_config, resolve_path
from common.io import load_json, save_json
from eval.compare_tracklet_pools import GroundTruthIndex, audit_detector_pool
from pool.filter_detector_tracklet_pool import filter_tracklet


def _simulate(
    records: list[Mapping[str, Any]],
    *,
    threshold: float,
    min_frames: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        _, filtered = filter_tracklet(
            record,
            threshold=threshold,
            min_frames=min_frames,
        )
        if filtered is not None:
            kept.append(filtered)
    return kept


def _delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "matched_rate_iou50",
        "clean_frame_precision",
        "clean_rate_among_matched_frames",
        "mean_best_iou",
        "identity_purity_diagnostic",
    )
    return {key: float(after[key]) - float(before[key]) for key in keys}


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    filter_settings = config["detector_tracklet_pool"]["quality_filter"]
    audit_settings = config["confidence_filter_audit"]
    input_dir = resolve_path(path.parent, filter_settings["input_pool_dir"])
    output_dir = resolve_path(path.parent, filter_settings["output_dir"])
    raw = load_json(input_dir / "tracklets.json")
    filtered = load_json(output_dir / "tracklets.json")
    records = [dict(record) for record in raw.get("tracklets", [])]
    min_frames = int(filter_settings["min_frames"])
    selected_threshold = float(filter_settings["detector_confidence_min"])
    gt = GroundTruthIndex(config, path)

    before = audit_detector_pool(raw, gt)
    after = audit_detector_pool(filtered, gt)
    sweep = []
    for threshold in (float(value) for value in audit_settings["thresholds"]):
        kept = _simulate(records, threshold=threshold, min_frames=min_frames)
        audit = audit_detector_pool({"tracklets": kept}, gt)
        sweep.append(
            {
                "threshold": threshold,
                "tracklets": len(kept),
                "frames": audit["frames"],
                "tracklet_retention": len(kept) / len(records) if records else 0.0,
                "frame_retention": audit["frames"] / before["frames"] if before["frames"] else 0.0,
                "category_counts": dict(sorted(Counter(item["category"] for item in kept).items())),
                "matched_rate_iou50": audit["matched_rate_iou50"],
                "clean_frame_precision": audit["clean_frame_precision"],
                "clean_rate_among_matched_frames": audit["clean_rate_among_matched_frames"],
            }
        )

    report = {
        "protocol": {
            "selection_gt_used": False,
            "audit_gt_used": True,
            "identity_purity_used_for_selection": False,
            "filter_signal": "per-frame Co-DETR confidence only",
            "selected_threshold": selected_threshold,
            "min_consecutive_frames": min_frames,
            "note": "GT is read only after filtering to audit the fixed confidence policy",
        },
        "before_filter": before,
        "after_filter": after,
        "after_minus_before": _delta(after, before),
        "threshold_sweep": sweep,
    }
    output_path = resolve_path(path.parent, audit_settings["output_dir"]) / "report.json"
    save_json(output_path, report)
    report["output"] = str(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc GT audit of the detector-confidence quality filter"
    )
    parser.add_argument("--config", default="configs/phase2_detector_conf.yaml", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
