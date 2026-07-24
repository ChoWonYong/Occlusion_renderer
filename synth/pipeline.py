from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.io import save_json, save_jsonl
from common.schema import (
    bbox_to_mask,
    compute_ratio,
    encode_binary_mask,
    mask_to_bbox,
    occlusion_level,
    validate_extended_annotation,
)
from data.coco_sample import CocoDonorPool
from data.kitti2coco import convert_kitti_to_coco
from synth.compositor import composite_cutout
from synth.placer import find_placement, place_mask


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install -r requirements.txt") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def _path(config_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def _range_pair(value: Sequence[Any], name: str, cast: type = float) -> tuple[Any, Any]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = cast(value[0]), cast(value[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    return lower, upper


def _weighted_category(
    rng: random.Random,
    distribution: Mapping[str, float],
    available: Sequence[str],
) -> str:
    names = [name for name in available if float(distribution.get(name, 0.0)) > 0.0]
    if not names:
        return rng.choice(list(available))
    weights = [float(distribution[name]) for name in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _save_rgb(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        path, format="JPEG", quality=jpeg_quality, subsampling=0
    )


def _save_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask) != 0).astype(np.uint8) * 255).save(path)


def _validate_config(config: Mapping[str, Any]) -> None:
    for section in ("paths", "coco", "kitti", "synthesis"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"config section {section!r} is required")
    rho_min, rho_max = _range_pair(config["synthesis"].get("rho_dist", [0.2, 0.7]), "rho_dist")
    if not 0.0 < rho_min <= rho_max < 1.0:
        raise ValueError("rho_dist must be inside (0, 1)")
    inserts_min, inserts_max = _range_pair(
        config["synthesis"].get("inserts_per_image", [1, 1]), "inserts_per_image", int
    )
    if inserts_min < 1:
        raise ValueError("inserts_per_image must be positive")
    allowed_blends = {"none", "gaussian", "alpha"}
    blends = set(config["synthesis"].get("blend_methods", allowed_blends))
    if not blends or not blends.issubset(allowed_blends):
        raise ValueError(f"blend_methods must be drawn from {sorted(allowed_blends)}")


def run(config_path: str | Path, num_images_override: int | None = None) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _load_yaml(config_file)
    _validate_config(config)
    config_dir = config_file.parent
    paths = config["paths"]
    coco_config = config["coco"]
    kitti_config = config["kitti"]
    synth_config = config["synthesis"]

    seed = int(config.get("seed", 0))
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    num_images = int(num_images_override or synth_config.get("num_images", 100))
    if num_images < 1:
        raise ValueError("num_images must be positive")

    output_dir = _path(config_dir, paths["output_dir"])
    images_output_dir = output_dir / "images"
    masks_output_dir = output_dir / "occluder_masks"
    annotations_output = output_dir / "annotations" / "instances_synthetic.json"
    manifest_output = output_dir / "placement_manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    kitti_images_dir = _path(config_dir, paths["kitti_images"])
    kitti_labels_dir = _path(config_dir, paths["kitti_labels"])
    base_dataset = convert_kitti_to_coco(kitti_images_dir, kitti_labels_dir)
    source_images = {int(image["id"]): image for image in base_dataset["images"]}
    source_annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in base_dataset["annotations"]:
        source_annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    target_names = set(kitti_config.get("target_categories", ["car", "truck", "person", "bicycle"]))
    category_name_by_id = {int(cat["id"]): cat["name"] for cat in base_dataset["categories"]}
    max_base_occlusion = int(kitti_config.get("max_base_occlusion", 0))
    min_target_area = float(kitti_config.get("min_target_area", 400.0))
    eligible_by_image: dict[int, list[dict[str, Any]]] = {}
    for image_id, annotations in source_annotations_by_image.items():
        eligible = [
            annotation
            for annotation in annotations
            if category_name_by_id[int(annotation["category_id"])] in target_names
            and int(annotation.get("kitti", {}).get("occluded", 3)) <= max_base_occlusion
            and float(annotation.get("area", 0.0)) >= min_target_area
        ]
        if eligible:
            eligible_by_image[image_id] = eligible
    if not eligible_by_image:
        raise ValueError("no KITTI target matches category, area, and base-occlusion filters")

    donor_distribution = {
        str(name): float(weight)
        for name, weight in coco_config.get(
            "occluder_class_dist",
            {"car": 0.40, "person": 0.25, "truck": 0.15, "bus": 0.10, "bicycle": 0.10},
        ).items()
    }
    donor_pool = CocoDonorPool(
        _path(config_dir, paths["coco_images"]),
        _path(config_dir, paths["coco_annotations"]),
        categories=donor_distribution,
        min_area=float(coco_config.get("min_donor_area", 256.0)),
    )

    output_categories = copy.deepcopy(base_dataset["categories"])
    category_id_by_name = {cat["name"]: int(cat["id"]) for cat in output_categories}
    for donor_name in donor_pool.available_categories:
        if donor_name not in category_id_by_name:
            category_id_by_name[donor_name] = max(category_id_by_name.values(), default=0) + 1
            output_categories.append({"id": category_id_by_name[donor_name], "name": donor_name})

    rho_range = _range_pair(synth_config.get("rho_dist", [0.2, 0.7]), "rho_dist")
    scale_range = _range_pair(synth_config.get("scale_range", [0.6, 1.4]), "scale_range")
    inserts_range = _range_pair(
        synth_config.get("inserts_per_image", [1, 1]), "inserts_per_image", int
    )
    blend_methods = list(synth_config.get("blend_methods", ["none", "gaussian", "alpha"]))
    placement_trials = int(synth_config.get("placement_trials", 160))
    rho_tolerance = float(synth_config.get("rho_tolerance", 0.06))
    save_masks = bool(synth_config.get("save_masks", True))
    jpeg_quality = int(synth_config.get("jpeg_quality", 95))
    max_attempts = int(synth_config.get("max_attempts_per_image", 80)) * num_images

    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    next_annotation_id = 1
    available_image_ids = sorted(eligible_by_image)
    attempts = 0

    while len(output_images) < num_images and attempts < max_attempts:
        attempts += 1
        source_image_id = rng.choice(available_image_ids)
        source_image = source_images[source_image_id]
        source_path = Path(source_image["source_path"])
        background = _load_rgb(source_path)
        image_height, image_width = background.shape[:2]
        insert_count = rng.randint(inserts_range[0], inserts_range[1])
        target_candidates = eligible_by_image[source_image_id]
        if len(target_candidates) < insert_count:
            continue
        selected_targets = rng.sample(target_candidates, k=insert_count)

        staged_image_id = len(output_images) + 1
        staged_annotations: list[dict[str, Any]] = []
        source_to_staged: dict[int, dict[str, Any]] = {}
        staged_next_id = next_annotation_id
        for source_annotation in source_annotations_by_image.get(source_image_id, []):
            copied = copy.deepcopy(source_annotation)
            copied["id"] = staged_next_id
            copied["image_id"] = staged_image_id
            copied["source_annotation_id"] = int(source_annotation["id"])
            source_to_staged[int(source_annotation["id"])] = copied
            staged_annotations.append(copied)
            staged_next_id += 1

        composed = background
        inserted: list[dict[str, Any]] = []
        placement_failed = False
        for selected_target in selected_targets:
            donor_category = _weighted_category(
                rng, donor_distribution, donor_pool.available_categories
            )
            donor = donor_pool.sample(rng, donor_category)
            rho_target = rng.uniform(rho_range[0], rho_range[1])
            placement = find_placement(
                donor["mask"],
                selected_target["bbox"],
                (image_height, image_width),
                rho_target,
                np_rng,
                scale_range=scale_range,
                trials=placement_trials,
                tolerance=rho_tolerance,
            )
            if placement is None:
                placement_failed = True
                break
            blend_method = rng.choice(blend_methods)
            composed = composite_cutout(
                composed,
                donor["image"],
                donor["mask"],
                placement,
                blend_method,
                gaussian_radius=float(synth_config.get("gaussian_radius", 1.5)),
            )
            full_mask = place_mask(donor["mask"], placement, (image_height, image_width))
            occluder_id = staged_next_id
            staged_next_id += 1
            occluder_bbox = mask_to_bbox(full_mask)
            target_staged = source_to_staged[int(selected_target["id"])]
            occluder_annotation = {
                "id": occluder_id,
                "image_id": staged_image_id,
                "category_id": category_id_by_name[donor["category"]],
                "bbox": occluder_bbox,
                "segmentation": encode_binary_mask(full_mask),
                "area": int(full_mask.sum()),
                "iscrowd": 0,
                "synthetic": True,
                "occludes_ids": [int(target_staged["id"])],
                "provenance": {
                    "dataset": "COCO val2017",
                    "src_image_id": donor["source_image_id"],
                    "src_annotation_id": donor["source_annotation_id"],
                    "src_file_name": donor["source_file_name"],
                },
            }
            staged_annotations.append(occluder_annotation)
            inserted.append(
                {
                    "target_source_annotation_id": int(selected_target["id"]),
                    "target_annotation_id": int(target_staged["id"]),
                    "occluder_annotation_id": occluder_id,
                    "donor": {
                        "category": donor["category"],
                        "source_image_id": donor["source_image_id"],
                        "source_annotation_id": donor["source_annotation_id"],
                        "source_file_name": donor["source_file_name"],
                    },
                    "placement": {
                        "x": placement.x,
                        "y": placement.y,
                        "width": placement.width,
                        "height": placement.height,
                        "flip_horizontal": placement.flip_horizontal,
                    },
                    "rho_target": placement.rho_target,
                    "rho_placement": placement.rho_actual,
                    "blend_method": blend_method,
                    "mask": full_mask,
                }
            )
        if placement_failed:
            continue

        for target in selected_targets:
            staged_target = source_to_staged[int(target["id"])]
            amodal_mask = bbox_to_mask(target["bbox"], image_height, image_width)
            covering = [
                item
                for item in inserted
                if np.logical_and(item["mask"] != 0, amodal_mask != 0).any()
            ]
            all_occluders = np.zeros_like(amodal_mask)
            for item in covering:
                all_occluders |= item["mask"]
            visible_mask = np.logical_and(amodal_mask != 0, all_occluders == 0).astype(np.uint8)
            ratio = compute_ratio(visible_mask, amodal_mask)
            visible_bbox = mask_to_bbox(visible_mask)
            for item in covering:
                occluder_annotation = next(
                    annotation
                    for annotation in staged_annotations
                    if int(annotation["id"]) == int(item["occluder_annotation_id"])
                )
                target_annotation_id = int(staged_target["id"])
                if target_annotation_id not in occluder_annotation["occludes_ids"]:
                    occluder_annotation["occludes_ids"].append(target_annotation_id)
                if int(item["target_source_annotation_id"]) == int(target["id"]):
                    item["rho_actual"] = ratio
            staged_target.update(
                {
                    "bbox": visible_bbox,
                    "visible_bbox": visible_bbox,
                    "segmentation": encode_binary_mask(visible_mask),
                    "area": int(visible_mask.sum()),
                    "amodal_bbox": [float(value) for value in target["bbox"]],
                    "amodal_segmentation": encode_binary_mask(amodal_mask),
                    "occlusion_ratio": ratio,
                    "occlusion_level": occlusion_level(ratio),
                    "occluder_ids": [int(item["occluder_annotation_id"]) for item in covering],
                    "synthetic": True,
                    "provenance": {
                        "src_dataset": "KITTI",
                        "src_image_id": source_image_id,
                        "src_annotation_id": int(target["id"]),
                        "target_mask_source": "kitti_bbox_proxy",
                        "paste_params": {
                            "rho_targets": [float(item["rho_target"]) for item in covering],
                            "blend_methods": [item["blend_method"] for item in covering],
                            "backend": "coco_ground_truth_polygon",
                        },
                    },
                }
            )
            validate_extended_annotation(staged_target)

        output_index = len(output_images)
        output_name = f"{source_path.stem}__a2a_{output_index:06d}.jpg"
        _save_rgb(images_output_dir / output_name, composed, jpeg_quality)
        if save_masks:
            for insert_index, item in enumerate(inserted):
                mask_name = f"{Path(output_name).stem}__occ_{insert_index:02d}.png"
                _save_mask(masks_output_dir / mask_name, item["mask"])
                item["mask_file_name"] = str(Path("occluder_masks") / mask_name)

        for item in inserted:
            item.pop("mask", None)
        output_images.append(
            {
                "id": staged_image_id,
                "file_name": output_name,
                "width": image_width,
                "height": image_height,
                "source": {
                    "dataset": "KITTI",
                    "image_id": source_image_id,
                    "file_name": source_image["file_name"],
                },
            }
        )
        output_annotations.extend(staged_annotations)
        next_annotation_id = staged_next_id
        manifest_rows.append(
            {
                "output_image_id": staged_image_id,
                "output_file_name": output_name,
                "source_image_id": source_image_id,
                "source_file_name": source_image["file_name"],
                "seed": seed,
                "insertions": inserted,
            }
        )

    if len(output_images) < num_images:
        raise RuntimeError(
            f"generated only {len(output_images)}/{num_images} images after {attempts} attempts; "
            "increase placement tolerance/trials or relax target filters"
        )

    output_dataset = {
        "info": {
            "description": "KDS A2a synthetic occlusion: COCO val2017 cutouts pasted on KITTI",
            "seed": seed,
            "target_mask_caveat": "KITTI has boxes but no instance masks; bbox masks are amodal proxies.",
        },
        "images": output_images,
        "annotations": output_annotations,
        "categories": output_categories,
    }
    save_json(annotations_output, output_dataset)
    save_jsonl(manifest_output, manifest_rows)
    save_json(output_dir / "run_config.json", config)
    summary = {
        "generated_images": len(output_images),
        "annotations": len(output_annotations),
        "attempts": attempts,
        "output_dir": str(output_dir),
        "annotations_file": str(annotations_output),
        "manifest_file": str(manifest_output),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run COCO val2017 -> KITTI A2a copy-paste")
    parser.add_argument("--config", required=True, type=Path, help="A2a YAML config")
    parser.add_argument("--num-images", type=int, default=None, help="Override config for a quick smoke run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args.config, args.num_images)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
