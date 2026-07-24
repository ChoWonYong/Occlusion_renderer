from __future__ import annotations

from typing import Any, Mapping


def create_boxmot_bytetrack(config: Mapping[str, Any], class_names: list[str]):
    """Create the fixed Phase-1 tracker: BoxMOT ByteTrack, no ReID features."""
    try:
        from boxmot.trackers.bbox import ByteTrack
    except ImportError as exc:
        raise RuntimeError("Install the local BoxMOT repository in the active Conda environment") from exc
    return ByteTrack(
        min_conf=float(config.get("min_conf", 0.10)),
        track_thresh=float(config.get("track_thresh", 0.45)),
        match_thresh=float(config.get("match_thresh", 0.80)),
        track_buffer=int(config.get("track_buffer", 30)),
        frame_rate=int(config.get("frame_rate", 10)),
        per_class=bool(config.get("per_class", True)),
        class_ids=list(range(len(class_names))),
        class_names=class_names,
    )

