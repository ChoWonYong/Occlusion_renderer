from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any, Iterable

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from data.kitti_tracking import convert_tracking_to_video_coco


def _safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.symlink_to(source.resolve())


def _materialize_images(
    dataset: dict[str, Any], output_root: Path, prefix: str, source_key: str = "source_path"
) -> dict[str, Any]:
    result = copy.deepcopy(dataset)
    for image in result["images"]:
        if source_key in image:
            source = Path(image[source_key])
        else:
            source = Path(dataset["root"]) / image["file_name"]
        relative = Path("images") / prefix / image["file_name"]
        _safe_symlink(source, output_root / relative)
        image["file_name"] = str(relative)
        image.pop("source_path", None)
    return result


def _append_dataset(destination: dict[str, Any], source: dict[str, Any]) -> None:
    video_map: dict[int, int] = {}
    image_map: dict[int, int] = {}
    next_video_id = max((int(video["id"]) for video in destination["videos"]), default=0) + 1
    next_image_id = max((int(image["id"]) for image in destination["images"]), default=0) + 1
    next_annotation_id = max((int(ann["id"]) for ann in destination["annotations"]), default=0) + 1
    for video in source.get("videos", []):
        copied = copy.deepcopy(video)
        video_map[int(video["id"])] = next_video_id
        copied["id"] = next_video_id
        next_video_id += 1
        destination["videos"].append(copied)
    for image in source["images"]:
        copied = copy.deepcopy(image)
        image_map[int(image["id"])] = next_image_id
        copied["id"] = next_image_id
        if "video_id" in copied:
            copied["video_id"] = video_map[int(image["video_id"])]
        next_image_id += 1
        destination["images"].append(copied)
    for annotation in source["annotations"]:
        copied = copy.deepcopy(annotation)
        copied["id"] = next_annotation_id
        copied["image_id"] = image_map[int(annotation["image_id"])]
        if "video_id" in copied:
            copied["video_id"] = video_map[int(annotation["video_id"])]
        next_annotation_id += 1
        destination["annotations"].append(copied)


def _empty_like(dataset: dict[str, Any], description: str) -> dict[str, Any]:
    return {
        "info": {"description": description},
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(dataset["categories"]),
    }


def _filter_paste_frames(dataset: dict[str, Any]) -> dict[str, Any]:
    """Keep only synthetic frames that actually carry a pasted occluder.

    Frames of the synthesized sequences where no occluder was rendered are
    near-duplicates of the original and would double clean frames in the
    treatment set, so they are dropped; only frames with a synthetic occluder
    annotation are kept.
    """
    paste_image_ids = {
        int(annotation["image_id"])
        for annotation in dataset.get("annotations", [])
        if annotation.get("synthetic_occluder") is True
    }
    result = copy.deepcopy(dataset)
    result["images"] = [image for image in dataset["images"] if int(image["id"]) in paste_image_ids]
    kept_ids = {int(image["id"]) for image in result["images"]}
    result["annotations"] = [
        copy.deepcopy(annotation)
        for annotation in dataset["annotations"]
        if int(annotation["image_id"]) in kept_ids
    ]
    kept_video_ids = {int(image["video_id"]) for image in result["images"] if "video_id" in image}
    result["videos"] = [video for video in dataset.get("videos", []) if int(video["id"]) in kept_video_ids]
    return result


def _augment_equal_budget(
    source: dict[str, Any], count: int, output_root: Path, seed: int
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageEnhance
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc
    rng = random.Random(seed)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in source["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    result = _empty_like(source, "Equal-budget simple augmentation for Phase-1 baseline")
    for index in range(count):
        image_info = rng.choice(source["images"])
        source_path = output_root / image_info["file_name"]
        method = "horizontal_flip" if index % 2 == 0 else "color_jitter"
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            if method == "horizontal_flip":
                augmented = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            else:
                augmented = ImageEnhance.Color(ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))).enhance(
                    rng.uniform(0.85, 1.15)
                )
        relative = Path("images") / "baseline_aug" / f"aug_{index:08d}.jpg"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        augmented.save(destination, quality=95, subsampling=0)
        video_id = index + 1
        image_id = index + 1
        result["videos"].append({"id": video_id, "name": f"baseline_aug_{index:08d}", "num_frames": 1})
        result["images"].append(
            {
                "id": image_id,
                "video_id": video_id,
                "frame_index": 0,
                "frame_id": 1,
                "file_name": str(relative),
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
                "augmentation": method,
            }
        )
        for source_annotation in annotations_by_image.get(int(image_info["id"]), []):
            annotation = copy.deepcopy(source_annotation)
            annotation["id"] = len(result["annotations"]) + 1
            annotation["image_id"] = image_id
            annotation["video_id"] = video_id
            annotation["frame_index"] = 0
            annotation["track_id"] = int(source_annotation["track_id"])
            if method == "horizontal_flip":
                x, y, width, height = (float(value) for value in annotation["bbox"])
                annotation["bbox"] = [float(image_info["width"]) - x - width, y, width, height]
                annotation["segmentation"] = []
            result["annotations"].append(annotation)
    return result


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    split = load_json(resolve_path(path.parent, config["split"]["output"]))
    kitti_root = config_path(config, path, "paths", "kitti_tracking")
    class_map = config["classes"]["kitti_map"]
    train_source = convert_tracking_to_video_coco(kitti_root, split["train_sequences"], class_map)
    eval_source = convert_tracking_to_video_coco(kitti_root, split["eval_sequences"], class_map)
    synthetic_root = resolve_path(
        path.parent,
        config["dataset"].get("synthetic_output_dir", config["synthesis"]["output_dir"]),
    )
    synthetic = load_json(synthetic_root / "annotations.json")
    # 2026-07-23: treatment uses only frames that actually carry a pasted occluder
    # (avoids duplicating clean frames). Set dataset.synthetic_frames: all to keep
    # the whole synthesized sequences instead.
    if str(config["dataset"].get("synthetic_frames", "paste_only")) == "paste_only":
        synthetic = _filter_paste_frames(synthetic)
    synthetic["root"] = str(synthetic_root)
    output_root = resolve_path(path.parent, config["dataset"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    train_local = _materialize_images(train_source, output_root, "original_train")
    eval_local = _materialize_images(eval_source, output_root, "original_eval")
    synthetic_local = copy.deepcopy(synthetic)
    for image in synthetic_local["images"]:
        source = synthetic_root / image["file_name"]
        relative = Path("images") / "synthetic" / Path(image["file_name"]).relative_to("frames")
        _safe_symlink(source, output_root / relative)
        image["file_name"] = str(relative)
    synthetic_local.pop("root", None)

    treatment = _empty_like(train_local, "Treatment: original KITTI train split plus paste frames")
    _append_dataset(treatment, train_local)
    _append_dataset(treatment, synthetic_local)

    # Equal-budget baseline is optional (dataset.baseline_equal_budget). When on,
    # the baseline pads the original train with the same number of simple
    # flip/color-jitter frames as the added paste frames; when off, the baseline
    # is the original train split alone.
    baseline = _empty_like(train_local, "Baseline: original KITTI train split")
    _append_dataset(baseline, train_local)
    equal_budget = bool(config["dataset"].get("baseline_equal_budget", True))
    if equal_budget and synthetic_local["images"]:
        baseline_aug = _augment_equal_budget(
            train_local, len(synthetic_local["images"]), output_root, int(config["seed"])
        )
        _append_dataset(baseline, baseline_aug)

    annotation_dir = output_root / "annotations"
    baseline_path = output_root / config["dataset"]["baseline_train_json"]
    treatment_path = output_root / config["dataset"]["treatment_train_json"]
    eval_path = output_root / config["dataset"]["eval_json"]
    save_json(baseline_path, baseline)
    save_json(treatment_path, treatment)
    save_json(eval_path, eval_local)
    summary = {
        "baseline_images": len(baseline["images"]),
        "treatment_images": len(treatment["images"]),
        "equal_budget": len(baseline["images"]) == len(treatment["images"]),
        "eval_images": len(eval_local["images"]),
        "output_dir": str(output_root),
    }
    save_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build equal-budget Phase-1 YOLOX baseline/treatment datasets")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
