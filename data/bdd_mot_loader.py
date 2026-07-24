from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io import load_json
from common.schema import validate_video_dataset


def load_converted_bdd_mot(annotation_file: str | Path, frames_root: str | Path) -> dict[str, Any]:
    """Load output from `python -m bdd100k.label.to_coco -m box_track`.

    Phase 2 deliberately builds on the official converter instead of maintaining
    a second Scalabel parser. This adapter only normalizes paths and validates the
    temporal fields needed by the KDS pipeline.
    """
    dataset = load_json(annotation_file)
    root = Path(frames_root).expanduser().resolve()
    images_by_id: dict[int, dict[str, Any]] = {}
    for image in dataset.get("images", []):
        if "frame_index" not in image:
            if "frame_id" not in image:
                raise ValueError("BDD MOT image requires frame_id or frame_index")
            image["frame_index"] = int(image["frame_id"])
        image["source_path"] = str((root / image["file_name"]).resolve())
        images_by_id[int(image["id"])] = image
    for annotation in dataset.get("annotations", []):
        if "track_id" not in annotation:
            if "instance_id" not in annotation:
                raise ValueError("BDD MOT annotation requires track_id or instance_id")
            annotation["track_id"] = int(annotation["instance_id"])
        image = images_by_id[int(annotation["image_id"])]
        annotation.setdefault("video_id", int(image["video_id"]))
        annotation.setdefault("frame_index", int(image["frame_index"]))
    validate_video_dataset(dataset)
    return dataset
