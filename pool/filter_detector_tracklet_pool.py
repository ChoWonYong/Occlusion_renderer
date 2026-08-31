from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.config import load_config, resolve_path
from common.io import load_json, save_json


def longest_passing_window(flags: Sequence[bool]) -> tuple[int, int]:
    """Return the first longest half-open True run."""
    best_start = best_end = 0
    current_start: int | None = None
    for index, passed in enumerate([*flags, False]):
        if passed and current_start is None:
            current_start = index
        elif not passed and current_start is not None:
            if index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
            current_start = None
    return best_start, best_end


def classify_decision(
    *,
    accepted: bool,
    input_length: int,
    pass_frames: int,
    longest_run: int,
    min_frames: int = 30,
) -> str:
    if accepted:
        return "accepted_high" if longest_run == input_length else "accepted_trimmed"
    return "rejected_fragmented" if pass_frames >= min_frames else "rejected_low"


def _confidence_stats(frames: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores = np.asarray(
        [float(frame["detector_confidence"]) for frame in frames],
        dtype=np.float64,
    )
    return {
        "min": float(np.min(scores)),
        "p10": float(np.quantile(scores, 0.1)),
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
    }


def filter_tracklet(
    record: Mapping[str, Any],
    *,
    threshold: float,
    min_frames: int,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply a per-frame confidence gate and keep one consecutive passing run."""
    frames = [dict(frame) for frame in record.get("frames", [])]
    flags = [
        math.isfinite(float(frame.get("detector_confidence", float("nan"))))
        and float(frame["detector_confidence"]) >= threshold
        for frame in frames
    ]
    start, end = longest_passing_window(flags)
    run_length = end - start
    accepted = run_length >= min_frames
    pass_frames = sum(flags)
    scenario = classify_decision(
        accepted=accepted,
        input_length=len(frames),
        pass_frames=pass_frames,
        longest_run=run_length,
        min_frames=min_frames,
    )
    frame_decisions = []
    for index, (frame, passed) in enumerate(zip(frames, flags)):
        frame_decisions.append(
            {
                "raw_offset": index,
                "sample_index": int(frame.get("sample_index", index)),
                "frame_index": int(frame.get("frame_index", index)),
                "detector_confidence": float(frame["detector_confidence"]),
                "passed": passed,
                "retained": accepted and start <= index < end,
            }
        )
    decision = {
        "source_tracklet_id": int(record["id"]),
        "source_candidate_id": record.get("source_candidate_id"),
        "category": str(record["category"]),
        "source_dataset": str(record.get("source_dataset", "unknown")),
        "sequence": str(record.get("sequence", "unknown")),
        "source_track_id": record.get("source_track_id"),
        "input_length": len(frames),
        "passing_frames": pass_frames,
        "longest_passing_start": start,
        "longest_passing_end": end,
        "longest_passing_length": run_length,
        "accepted": accepted,
        "scenario": scenario,
        "reason": (
            f"longest confidence-passing run {run_length} >= {min_frames}"
            if accepted
            else f"longest confidence-passing run {run_length} < {min_frames}"
        ),
        "frames": frame_decisions,
    }
    if not accepted:
        return decision, None

    retained = frames[start:end]
    rewritten: list[dict[str, Any]] = []
    for offset, frame in enumerate(retained):
        updated = dict(frame)
        updated["raw_offset"] = int(frame.get("offset", start + offset))
        updated["offset"] = offset
        if input_dir is not None and output_dir is not None:
            source = (input_dir / str(frame["file_name"])).resolve()
            updated["file_name"] = os.path.relpath(source, output_dir)
        rewritten.append(updated)
    filtered = dict(record)
    filtered["source_tracklet_id"] = int(record["id"])
    filtered["frames"] = rewritten
    filtered["length"] = len(rewritten)
    filtered["start_frame"] = int(rewritten[0]["frame_index"])
    filtered["end_frame"] = int(rewritten[-1]["frame_index"])
    filtered["detector_confidence"] = _confidence_stats(rewritten)
    filtered["quality_filter"] = {
        "signal": "per-frame Co-DETR confidence",
        "threshold": threshold,
        "run_policy": "first longest consecutive passing run",
        "raw_length": len(frames),
        "raw_start_offset": start,
        "raw_end_offset_exclusive": end,
        "identity_purity_used": False,
        "gt_used": False,
    }
    return decision, filtered


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["detector_tracklet_pool"]["quality_filter"]
    if not bool(settings.get("enabled", False)):
        raise ValueError("detector_tracklet_pool.quality_filter.enabled must be true")
    threshold = float(settings["detector_confidence_min"])
    min_frames = int(settings.get("min_frames", config["detector_tracklet_pool"]["min_frames"]))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("detector_confidence_min must be in [0, 1]")
    if min_frames <= 0:
        raise ValueError("quality-filter min_frames must be positive")

    input_dir = resolve_path(path.parent, settings["input_pool_dir"])
    output_dir = resolve_path(path.parent, settings["output_dir"])
    source_file = input_dir / "tracklets.json"
    if not source_file.is_file():
        raise FileNotFoundError(f"raw detector pool not found: {source_file}")
    raw = load_json(source_file)

    decisions: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for record in raw.get("tracklets", []):
        decision, filtered = filter_tracklet(
            record,
            threshold=threshold,
            min_frames=min_frames,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        decisions.append(decision)
        if filtered is not None:
            updated = dict(filtered)
            updated["id"] = len(accepted) + 1
            accepted.append(updated)

    metadata = {
        "schema_version": 1,
        "mode": "codetr_bytetrack_sam3_confidence_filtered",
        "description": "GT-free Phase-2 pool filtered only by per-frame detector confidence",
        "source_pool": str(source_file),
        "selection": {
            "gt_used": False,
            "identity_purity_used": False,
            "detector_confidence_min": threshold,
            "min_frames": min_frames,
            "run_policy": "first longest consecutive passing run",
            "rejected_tracklets_remain_in_decision_manifest": True,
        },
        "tracklets": accepted,
    }
    output_file = output_dir / "tracklets.json"
    decisions_file = output_dir / "decisions.json"
    save_json(output_file, metadata)
    save_json(
        decisions_file,
        {
            "schema_version": 1,
            "source_pool": str(source_file),
            "filter": metadata["selection"],
            "tracklets": decisions,
        },
    )
    before_frames = sum(int(item.get("length", len(item.get("frames", [])))) for item in raw.get("tracklets", []))
    after_frames = sum(int(item["length"]) for item in accepted)
    summary = {
        "mode": metadata["mode"],
        "detector_confidence_min": threshold,
        "min_frames": min_frames,
        "tracklets_before": len(raw.get("tracklets", [])),
        "tracklets_after": len(accepted),
        "tracklets_rejected": len(decisions) - len(accepted),
        "frames_before": before_frames,
        "frames_after": after_frames,
        "tracklet_retention": len(accepted) / len(decisions) if decisions else 0.0,
        "frame_retention": after_frames / before_frames if before_frames else 0.0,
        "category_counts_after": dict(sorted(Counter(item["category"] for item in accepted).items())),
        "scenario_counts": dict(sorted(Counter(item["scenario"] for item in decisions).items())),
        "gt_used": False,
        "identity_purity_used": False,
        "output": str(output_file),
        "decisions": str(decisions_file),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a raw Co-DETR+SAM3 tracklet pool by detector confidence"
    )
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
