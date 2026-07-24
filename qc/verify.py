from __future__ import annotations

from typing import Any, Mapping

from common.schema import validate_video_dataset


def verify_synthetic_video_dataset(dataset: Mapping[str, Any], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    validate_video_dataset(dataset)
    invalid_ratios = [
        annotation["id"]
        for annotation in dataset.get("annotations", [])
        if "occlusion_ratio" in annotation and not 0.0 <= float(annotation["occlusion_ratio"]) <= 1.0
    ]
    invalid_events = [
        event["event_id"]
        for event in events
        if not int(event["frame_start"]) <= int(event["frame_peak"]) <= int(event["frame_end"])
        or int(event["duration"]) != int(event["frame_end"]) - int(event["frame_start"]) + 1
    ]
    return {
        "ok": not invalid_ratios and not invalid_events,
        "invalid_ratio_annotations": invalid_ratios,
        "invalid_events": invalid_events,
        "videos": len(dataset.get("videos", [])),
        "frames": len(dataset.get("images", [])),
        "events": len(events),
    }

