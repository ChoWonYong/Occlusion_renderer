from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

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


def paste_components(
    synthetic: dict[str, Any]
) -> list[dict[str, Any]]:
    """Group pasted tracklets into units that can be toggled independently.

    A rendered frame bakes in every occluder active at that moment, so a single
    tracklet cannot be switched off without re-rendering. Tracklets that share
    any frame are therefore merged into one component; each frame then belongs to
    exactly one component, and toggling a component is well defined.

    Returns one entry per component with its ``tracks`` and the synthetic
    ``image_ids`` it covers.
    """
    frames_by_track: dict[tuple[int, int], set[int]] = {}
    for annotation in synthetic.get("annotations", []):
        if annotation.get("synthetic_occluder") is not True:
            continue
        key = (int(annotation.get("video_id", 0)), int(annotation["track_id"]))
        frames_by_track.setdefault(key, set()).add(int(annotation["image_id"]))

    parent: dict[tuple[int, int], tuple[int, int]] = {key: key for key in frames_by_track}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[int, tuple[int, int]] = {}
    for key, image_ids in frames_by_track.items():
        for image_id in image_ids:
            if image_id in owner:
                union(owner[image_id], key)
            else:
                owner[image_id] = key

    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for key, image_ids in frames_by_track.items():
        root = find(key)
        entry = grouped.setdefault(root, {"tracks": [], "image_ids": set()})
        entry["tracks"].append(key)
        entry["image_ids"] |= image_ids
    components = [
        {"tracks": sorted(entry["tracks"]), "image_ids": sorted(entry["image_ids"])}
        for entry in grouped.values()
    ]
    components.sort(key=lambda component: component["image_ids"][0] if component["image_ids"] else -1)
    return components


def _replace_with_paste_frames(
    train_local: dict[str, Any],
    synthetic_local: dict[str, Any],
    probability: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Swap selected original frames for their pasted renders, keeping the count.

    Appending the synthetic frames instead would give the treatment arm ~1.9x the
    images and therefore ~1.9x the gradient steps at the same epoch count, which
    the removed equal-budget padding used to hide. Replacing keeps both arms at
    the original frame count so the only difference is what is in the pixels.

    Selection happens per component from :func:`paste_components`, not per frame:
    a frame-wise coin flip would leave an occluder present on one frame and gone
    on the next, destroying the temporal continuity the method exists to create.
    """
    rng = random.Random(seed)
    components = paste_components(synthetic_local)
    selected_images: set[int] = set()
    selected_components = 0
    for component in components:
        if rng.random() < float(probability):
            selected_components += 1
            selected_images.update(component["image_ids"])

    synthetic_by_source: dict[int, dict[str, Any]] = {}
    for image in synthetic_local["images"]:
        if int(image["id"]) not in selected_images:
            continue
        source = image.get("source", {})
        if "image_id" not in source:
            raise KeyError("synthetic images must record source.image_id to replace originals")
        synthetic_by_source[int(source["image_id"])] = image
    synthetic_annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in synthetic_local["annotations"]:
        synthetic_annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    result = _empty_like(train_local, "Treatment: original KITTI train split with paste frames swapped in")
    result["videos"] = copy.deepcopy(train_local.get("videos", []))
    train_annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in train_local["annotations"]:
        train_annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    replaced = 0
    for image in train_local["images"]:
        original_id = int(image["id"])
        pasted = synthetic_by_source.get(original_id)
        if pasted is None:
            copied = copy.deepcopy(image)
            annotations = copy.deepcopy(train_annotations.get(original_id, []))
        else:
            replaced += 1
            copied = copy.deepcopy(image)
            copied["file_name"] = pasted["file_name"]
            copied["pasted"] = True
            annotations = copy.deepcopy(synthetic_annotations.get(int(pasted["id"]), []))
        result["images"].append(copied)
        for annotation in annotations:
            annotation["id"] = len(result["annotations"]) + 1
            annotation["image_id"] = original_id
            annotation["video_id"] = int(copied.get("video_id", annotation.get("video_id", 0)))
            result["annotations"].append(annotation)

    stats = {
        "components": len(components),
        "components_selected": selected_components,
        "frames_replaced": replaced,
        "paste_probability": float(probability),
    }
    return result, stats


def run(config_file: str | Path, paste_mode: str | None = None) -> dict[str, Any]:
    """Build the eval set plus a baseline and a treatment training set.

    The baseline is the plain original KITTI train split. Copy-paste occlusion is
    the method under test, so the arms differ only by it — every other knob
    (mosaic, mixup, flip, HSV, schedule) is identical and applied online by the
    trainer.

    ``paste_mode`` is ``"append"`` (default) or ``"replace"``; see
    :func:`_replace_with_paste_frames` for why the two exist.
    """
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

    mode = str(paste_mode or config["dataset"].get("paste_mode", "replace"))
    if mode not in {"append", "replace"}:
        raise ValueError("dataset.paste_mode must be 'append' or 'replace'")
    paste_stats: dict[str, Any] = {}
    if mode == "append":
        treatment = _empty_like(train_local, "Treatment: original KITTI train split plus paste frames")
        _append_dataset(treatment, train_local)
        _append_dataset(treatment, synthetic_local)
    else:
        treatment, paste_stats = _replace_with_paste_frames(
            train_local,
            synthetic_local,
            float(config["dataset"].get("paste_probability", 1.0)),
            int(config["seed"]),
        )

    baseline = _empty_like(train_local, "Baseline: original KITTI train split")
    _append_dataset(baseline, train_local)

    treatment_key = config["dataset"].get(
        f"treatment_{mode}_train_json", config["dataset"]["treatment_train_json"]
    )
    baseline_path = output_root / config["dataset"]["baseline_train_json"]
    treatment_path = output_root / treatment_key
    eval_path = output_root / config["dataset"]["eval_json"]
    save_json(baseline_path, baseline)
    save_json(treatment_path, treatment)
    save_json(eval_path, eval_local)
    summary = {
        "paste_mode": mode,
        "baseline_images": len(baseline["images"]),
        "treatment_images": len(treatment["images"]),
        "baseline_annotations": len(baseline["annotations"]),
        "treatment_annotations": len(treatment["annotations"]),
        # Equal iteration budget is what "replace" buys: same image count, so the
        # same number of gradient steps per epoch in both arms.
        "matched_image_budget": len(baseline["images"]) == len(treatment["images"]),
        "eval_images": len(eval_local["images"]),
        "treatment_train_json": str(treatment_path.relative_to(output_root)),
        "output_dir": str(output_root),
    }
    summary.update(paste_stats)
    save_json(output_root / f"summary_{mode}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase-1 YOLOX baseline/treatment datasets")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument(
        "--paste-mode",
        choices=["append", "replace"],
        default=None,
        help=(
            "append (default): paste frames are added, so the treatment arm sees more "
            "images and more gradient steps. replace: paste frames swap in for their "
            "source frames, matching the baseline's image and iteration budget."
        ),
    )
    args = parser.parse_args()
    print(run(args.config, args.paste_mode))


if __name__ == "__main__":
    main()
