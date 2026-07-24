from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import save_json
from data.coco_sample import CocoDonorPool


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    pool_config = config["pool"]
    if pool_config.get("source") != "coco_gt":
        raise ValueError(
            "Only source=coco_gt is runnable before a SAM3 backend is selected. "
            "KITTI crop extraction must use training split sequences through segment.SegBackend."
        )
    output_dir = resolve_path(path.parent, pool_config["output_dir"])
    requested_categories = list(config["occluder_class_dist"])
    donor_pool = CocoDonorPool(
        config_path(config, path, "paths", "coco_images"),
        config_path(config, path, "paths", "coco_annotations"),
        categories=requested_categories,
        min_area=float(pool_config.get("min_area", 256)),
    )
    samples_per_class = int(pool_config.get("samples_per_class", 200))
    rng = random.Random(int(config.get("seed", 0)))
    records: list[dict[str, Any]] = []
    seen: set[int] = set()

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc

    for category in donor_pool.available_categories:
        saved = 0
        attempts = 0
        while saved < samples_per_class and attempts < samples_per_class * 50:
            attempts += 1
            donor = donor_pool.sample(rng, category)
            annotation_id = int(donor["source_annotation_id"])
            if annotation_id in seen:
                continue
            seen.add(annotation_id)
            rgb = np.asarray(donor["image"], dtype=np.uint8)
            alpha = (np.asarray(donor["mask"]) != 0).astype(np.uint8) * 255
            rgba = np.dstack([rgb, alpha])
            relative = Path(category) / f"coco_{donor['source_image_id']}_{annotation_id}.png"
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba).save(destination)
            records.append(
                {
                    "id": len(records) + 1,
                    "category": category,
                    "file_name": str(relative),
                    "width": int(rgba.shape[1]),
                    "height": int(rgba.shape[0]),
                    "source": {
                        "dataset": "COCO val2017",
                        "image_id": int(donor["source_image_id"]),
                        "annotation_id": annotation_id,
                        "file_name": donor["source_file_name"],
                        "backend": "coco_gt_polygon",
                    },
                }
            )
            saved += 1
    if not records:
        raise RuntimeError("occluder pool is empty")
    metadata = {
        "schema_version": 2,
        "identity_rule": "One RGBA patch and one augmentation sample are reused for an entire sequence.",
        "instances": records,
    }
    save_json(output_dir / "pool.json", metadata)
    return {"output_dir": str(output_dir), "instances": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the Phase-1 COCO occluder pool")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    args = parser.parse_args()
    result = run(args.config)
    print(f"saved {result['instances']} occluders to {result['output_dir']}")


if __name__ == "__main__":
    main()
