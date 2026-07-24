from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from common.io import save_json


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_CLASS_MAP = {
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Tram": "truck",
    "Pedestrian": "person",
    "Person_sitting": "person",
    "Cyclist": "bicycle",
}


@dataclass(frozen=True)
class KittiObject:
    category: str
    truncated: float
    occluded: int
    alpha: float
    bbox_xyxy: tuple[float, float, float, float]
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float
    score: float | None = None


def parse_kitti_line(line: str) -> KittiObject:
    fields = line.split()
    if len(fields) not in (15, 16):
        raise ValueError(f"expected 15 or 16 KITTI fields, got {len(fields)}")
    values = [float(value) for value in fields[1:]]
    return KittiObject(
        category=fields[0],
        truncated=values[0],
        occluded=int(values[1]),
        alpha=values[2],
        bbox_xyxy=(values[3], values[4], values[5], values[6]),
        dimensions_hwl=(values[7], values[8], values[9]),
        location_xyz=(values[10], values[11], values[12]),
        rotation_y=values[13],
        score=values[14] if len(values) == 15 else None,
    )


def load_kitti_labels(path: str | Path) -> list[KittiObject]:
    objects: list[KittiObject] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                objects.append(parse_kitti_line(stripped))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return objects


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc
    with Image.open(path) as image:
        return image.size


def _iter_images(images_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def convert_kitti_to_coco(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    if not images_path.is_dir():
        raise FileNotFoundError(f"KITTI image directory not found: {images_path}")
    if not labels_path.is_dir():
        raise FileNotFoundError(f"KITTI label directory not found: {labels_path}")

    mapping = dict(class_map or DEFAULT_CLASS_MAP)
    category_names = list(dict.fromkeys(mapping.values()))
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(category_names)]
    category_ids = {category["name"]: category["id"] for category in categories}

    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, image_path in enumerate(_iter_images(images_path), start=1):
        width, height = _image_size(image_path)
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "source_path": str(image_path.resolve()),
            }
        )
        label_path = labels_path / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for source_index, obj in enumerate(load_kitti_labels(label_path)):
            mapped_name = mapping.get(obj.category)
            if mapped_name is None:
                continue
            x1, y1, x2, y2 = obj.bbox_xyxy
            x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
            y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 1.0 or box_height <= 1.0:
                continue
            bbox = [x1, y1, box_width, box_height]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[mapped_name],
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
        "info": {"description": "KITTI detection labels converted for KDS occlusion synthesis"},
        "images": coco_images,
        "annotations": annotations,
        "categories": categories,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert KITTI detection labels to COCO JSON")
    parser.add_argument("--images", required=True, type=Path, help="KITTI training/image_2 directory")
    parser.add_argument("--labels", required=True, type=Path, help="KITTI training/label_2 directory")
    parser.add_argument("--output", required=True, type=Path, help="Output COCO JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    converted = convert_kitti_to_coco(args.images, args.labels)
    save_json(args.output, converted)
    print(
        f"saved {len(converted['images'])} images and "
        f"{len(converted['annotations'])} annotations to {args.output}"
    )


if __name__ == "__main__":
    main()

