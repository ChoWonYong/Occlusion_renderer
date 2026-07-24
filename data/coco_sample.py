from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def polygons_to_crop_mask(
    polygons: Iterable[list[float]], crop_xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc

    x1, y1, x2, y2 = crop_xyxy
    mask_image = Image.new("L", (x2 - x1, y2 - y1), 0)
    drawer = ImageDraw.Draw(mask_image)
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            continue
        points = [(float(polygon[index]) - x1, float(polygon[index + 1]) - y1) for index in range(0, len(polygon), 2)]
        drawer.polygon(points, fill=1)
    return np.asarray(mask_image, dtype=np.uint8)


class CocoDonorPool:
    """On-demand COCO cutout loader; COCO val non-crowd polygons need no pycocotools."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        categories: Iterable[str] | None = None,
        min_area: float = 256.0,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {self.images_dir}")
        if not self.annotations_path.is_file():
            raise FileNotFoundError(f"COCO annotations not found: {self.annotations_path}")

        with self.annotations_path.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        self.images = {int(image["id"]): image for image in dataset["images"]}
        self.category_names = {int(cat["id"]): str(cat["name"]) for cat in dataset["categories"]}
        allowed = set(categories) if categories else set(self.category_names.values())
        self.by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in allowed}
        for annotation in dataset["annotations"]:
            category = self.category_names.get(int(annotation["category_id"]))
            segmentation = annotation.get("segmentation")
            if (
                category not in allowed
                or int(annotation.get("iscrowd", 0)) != 0
                or float(annotation.get("area", 0.0)) < min_area
                or not isinstance(segmentation, list)
                or not segmentation
            ):
                continue
            self.by_category.setdefault(category, []).append(annotation)
        self.by_category = {name: anns for name, anns in self.by_category.items() if anns}
        if not self.by_category:
            raise ValueError("no usable non-crowd polygon annotations matched donor categories")

    @property
    def available_categories(self) -> list[str]:
        return sorted(self.by_category)

    def sample(self, rng: random.Random, category: str | None = None) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc

        selected_category = category if category in self.by_category else rng.choice(self.available_categories)
        for _ in range(30):
            annotation = rng.choice(self.by_category[selected_category])
            image_info = self.images[int(annotation["image_id"])]
            image_path = self.images_dir / image_info["file_name"]
            if not image_path.is_file():
                continue
            x, y, width, height = (float(value) for value in annotation["bbox"])
            image_width, image_height = int(image_info["width"]), int(image_info["height"])
            x1 = max(0, int(np.floor(x)))
            y1 = max(0, int(np.floor(y)))
            x2 = min(image_width, int(np.ceil(x + width)))
            y2 = min(image_height, int(np.ceil(y + height)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            mask = polygons_to_crop_mask(annotation["segmentation"], (x1, y1, x2, y2))
            if int(mask.sum()) < 32:
                continue
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB").crop((x1, y1, x2, y2))).copy()
            return {
                "image": rgb,
                "mask": mask,
                "category": selected_category,
                "source_image_id": int(annotation["image_id"]),
                "source_annotation_id": int(annotation["id"]),
                "source_file_name": image_info["file_name"],
            }
        raise RuntimeError(f"could not load a valid COCO donor for category {selected_category!r}")

