from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class KittiTrackObject:
    frame_index: int
    track_id: int
    category: str
    truncated: float
    occluded: int
    alpha: float
    bbox_xyxy: tuple[float, float, float, float]
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float
    score: float | None = None


def parse_tracking_line(line: str) -> KittiTrackObject:
    fields = line.split()
    if len(fields) not in (17, 18):
        raise ValueError(f"expected 17 or 18 KITTI Tracking fields, got {len(fields)}")
    return KittiTrackObject(
        frame_index=int(fields[0]),
        track_id=int(fields[1]),
        category=fields[2],
        truncated=float(fields[3]),
        occluded=int(fields[4]),
        alpha=float(fields[5]),
        bbox_xyxy=tuple(float(value) for value in fields[6:10]),
        dimensions_hwl=tuple(float(value) for value in fields[10:13]),
        location_xyz=tuple(float(value) for value in fields[13:16]),
        rotation_y=float(fields[16]),
        score=float(fields[17]) if len(fields) == 18 else None,
    )


def load_sequence_labels(path: str | Path) -> dict[int, list[KittiTrackObject]]:
    by_frame: dict[int, list[KittiTrackObject]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = parse_tracking_line(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            by_frame.setdefault(obj.frame_index, []).append(obj)
    return by_frame


def discover_sequences(root: str | Path) -> list[str]:
    image_root = Path(root) / "training" / "image_02"
    if not image_root.is_dir():
        raise FileNotFoundError(f"KITTI Tracking image root not found: {image_root}")
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc
    with Image.open(path) as image:
        return image.size


def convert_tracking_to_video_coco(
    root: str | Path,
    sequence_ids: Iterable[str],
    class_map: Mapping[str, str],
) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    category_names = list(dict.fromkeys(class_map.values()))
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(category_names)]
    category_ids = {category["name"]: int(category["id"]) for category in categories}
    videos: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    annotation_id = 1

    for video_id, sequence_id in enumerate(sequence_ids, start=1):
        sequence_dir = dataset_root / "training" / "image_02" / sequence_id
        label_path = dataset_root / "training" / "label_02" / f"{sequence_id}.txt"
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"sequence directory not found: {sequence_dir}")
        if not label_path.is_file():
            raise FileNotFoundError(f"sequence labels not found: {label_path}")
        labels_by_frame = load_sequence_labels(label_path)
        frame_paths = sorted(
            path for path in sequence_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        videos.append({"id": video_id, "name": sequence_id, "fps": 10, "num_frames": len(frame_paths)})
        for frame_path in frame_paths:
            frame_index = int(frame_path.stem)
            width, height = _image_size(frame_path)
            current_image_id = image_id
            images.append(
                {
                    "id": current_image_id,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "frame_id": frame_index + 1,
                    "file_name": f"{sequence_id}/{frame_path.name}",
                    "source_path": str(frame_path.resolve()),
                    "width": width,
                    "height": height,
                }
            )
            image_id += 1
            for source_index, obj in enumerate(labels_by_frame.get(frame_index, [])):
                mapped_category = class_map.get(obj.category)
                if mapped_category is None or obj.track_id < 0:
                    continue
                x1, y1, x2, y2 = obj.bbox_xyxy
                x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
                y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
                box_width, box_height = x2 - x1, y2 - y1
                if box_width <= 1.0 or box_height <= 1.0:
                    continue
                bbox = [x1, y1, box_width, box_height]
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "track_id": obj.track_id,
                        "category_id": category_ids[mapped_category],
                        "bbox": bbox,
                        "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                        "kitti": {
                            "source_index": source_index,
                            "native_category": obj.category,
                            "truncated": obj.truncated,
                            "occluded": obj.occluded,
                            "alpha": obj.alpha,
                            "dimensions_hwl": list(obj.dimensions_hwl),
                            "location_xyz": list(obj.location_xyz),
                            "rotation_y": obj.rotation_y,
                        },
                    }
                )
                annotation_id += 1

    return {
        "info": {"description": "KITTI Tracking converted to temporal COCO"},
        "videos": videos,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

