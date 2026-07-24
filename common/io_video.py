from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from common.io import load_json, save_json, save_jsonl


def group_annotations_by_image(dataset: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        grouped[int(annotation["image_id"])].append(annotation)
    return dict(grouped)


def group_frames_by_video(dataset: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image in dataset.get("images", []):
        grouped[int(image["video_id"])].append(image)
    for frames in grouped.values():
        frames.sort(key=lambda frame: int(frame["frame_index"]))
    return dict(grouped)


def write_manifest(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    save_jsonl(path, rows)


__all__ = [
    "group_annotations_by_image",
    "group_frames_by_video",
    "load_json",
    "save_json",
    "write_manifest",
]

