from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json, save_jsonl
from common.io_video import group_annotations_by_image, group_frames_by_video
from common.schema import encode_binary_mask, mask_to_bbox, validate_video_dataset
from data.kitti_tracking import convert_tracking_to_video_coco
from depth.order import front_only_order
from label.compute import label_frame
from label.events import derive_events
from pool.augment import augment_identity
from qc.verify import verify_synthetic_video_dataset
from synth.trajectory import sample_crossing_trajectory
from synth.video_compositor import composite_frame


def _load_rgba(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA")).copy()


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, quality=95, subsampling=0)


def _weighted_pool_instance(
    records: list[dict[str, Any]], distribution: Mapping[str, float], rng: random.Random
) -> dict[str, Any]:
    available = sorted({record["category"] for record in records})
    categories = [category for category in available if float(distribution.get(category, 0.0)) > 0]
    if not categories:
        categories = available
    category = rng.choices(categories, [float(distribution.get(name, 1.0)) for name in categories], k=1)[0]
    return rng.choice([record for record in records if record["category"] == category])


def _select_victim(
    frames: list[dict[str, Any]],
    annotations_by_image: Mapping[int, list[dict[str, Any]]],
    duration: int,
    rng: random.Random,
) -> tuple[dict[str, Any], int] | None:
    half = duration // 2
    candidates: list[tuple[dict[str, Any], int]] = []
    for position, frame in enumerate(frames):
        if position < half or position + (duration - half) >= len(frames):
            continue
        for annotation in annotations_by_image.get(int(frame["id"]), []):
            if (
                int(annotation.get("kitti", {}).get("occluded", 3)) == 0
                and float(annotation.get("area", 0.0)) >= 400.0
            ):
                candidates.append((annotation, position))
    return rng.choice(candidates) if candidates else None


def run(config_file: str | Path, max_sequences_override: int | None = None) -> dict[str, Any]:
    config, path = load_config(config_file)
    split = load_json(resolve_path(path.parent, config["split"]["output"]))
    pool_dir = resolve_path(path.parent, config["pool"]["output_dir"])
    pool_metadata = load_json(pool_dir / "pool.json")
    pool_records = list(pool_metadata["instances"])
    if not pool_records:
        raise ValueError("occluder pool is empty")

    source_dataset = convert_tracking_to_video_coco(
        config_path(config, path, "paths", "kitti_tracking"),
        split["train_sequences"],
        config["classes"]["kitti_map"],
    )
    frames_by_video = group_frames_by_video(source_dataset)
    annotations_by_image = group_annotations_by_image(source_dataset)
    source_video_by_id = {int(video["id"]): video for video in source_dataset["videos"]}
    output_dir = resolve_path(path.parent, config["synthesis"]["output_dir"])
    output_frames_dir = output_dir / "frames"
    synthesis = config["synthesis"]
    max_sequences = int(max_sequences_override or synthesis.get("max_sequences", len(frames_by_video)))
    copies_per_source = int(synthesis.get("sequences_per_source", 1))
    rng = random.Random(int(config.get("seed", 0)))

    output_dataset: dict[str, Any] = {
        "info": {
            "description": "KDS v2 Phase-1 video copy-paste on KITTI Tracking",
            "depth_mode": synthesis.get("depth_mode", "occluder_front"),
            "amodal_mask_caveat": "KITTI bbox masks are used until SAM3 target masks are available.",
        },
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(source_dataset["categories"]),
    }
    category_id_by_name = {category["name"]: int(category["id"]) for category in output_dataset["categories"]}
    manifests: list[dict[str, Any]] = []
    occluder_tracks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    next_event_id = 1
    generated = 0

    for source_video_id in sorted(frames_by_video):
        if generated >= max_sequences:
            break
        source_frames = frames_by_video[source_video_id]
        if not source_frames:
            continue
        for copy_index in range(copies_per_source):
            if generated >= max_sequences:
                break
            duration_min, duration_max = (int(value) for value in synthesis.get("event_duration", [16, 40]))
            duration_max = min(duration_max, max(2, len(source_frames) - 2))
            if duration_min > duration_max:
                continue
            duration = rng.randint(duration_min, duration_max)
            selected = _select_victim(source_frames, annotations_by_image, duration, rng)
            if selected is None:
                continue
            victim, peak_position = selected
            start_position = peak_position - duration // 2
            start_frame = int(source_frames[start_position]["frame_index"])

            pool_record = _weighted_pool_instance(pool_records, config["occluder_class_dist"], rng)
            rgba = _load_rgba(pool_dir / pool_record["file_name"])
            rgba, augmentation = augment_identity(rgba, rng)
            # Pool records already carry project class names (the builders resolve
            # them through classes.kitti_map), so this only has to reject a class
            # the target dataset does not carry.
            target_category = pool_record["category"]
            if target_category not in category_id_by_name:
                continue
            alpha_area = int(np.count_nonzero(rgba[..., 3]))
            peak_rho = rng.uniform(*[float(value) for value in synthesis.get("peak_rho", [0.2, 0.7])])
            motion_model = rng.choice(list(synthesis.get("motion_models", ["const_vel"])))
            first_frame = source_frames[0]
            trajectory = sample_crossing_trajectory(
                image_size=(int(first_frame["width"]), int(first_frame["height"])),
                victim_bbox=victim["bbox"],
                patch_size=(rgba.shape[1], rgba.shape[0]),
                start_frame=start_frame,
                duration=duration,
                motion_model=motion_model,
                scale_range=tuple(float(value) for value in synthesis.get("scale_range", [0.6, 1.4])),
                scale_rate_range=tuple(float(value) for value in synthesis.get("scale_rate", [-0.002, 0.004])),
                rng=rng,
                patch_mask_area=alpha_area,
                peak_rho=peak_rho,
            )
            placement_by_frame = {placement.frame_index: placement for placement in trajectory.placements}
            output_video_id = generated + 1
            source_name = source_video_by_id[source_video_id]["name"]
            output_name = f"{source_name}_synth_{copy_index:02d}"
            occluder_track_id = 90000 + output_video_id
            blend_method = rng.choice(list(synthesis.get("blend_methods", ["alpha"])))
            output_dataset["videos"].append(
                {"id": output_video_id, "name": output_name, "fps": 10, "num_frames": len(source_frames)}
            )
            histories: list[dict[str, Any]] = []
            track_frame_records: list[dict[str, Any]] = []
            for frame in source_frames:
                frame_index = int(frame["frame_index"])
                background_annotations = annotations_by_image.get(int(frame["id"]), [])
                background = _load_rgb(Path(frame["source_path"]))
                placement = placement_by_frame.get(frame_index)
                effective_mask = np.zeros(background.shape[:2], dtype=np.uint8)
                if placement is not None:
                    if synthesis.get("depth_mode", "occluder_front") != "occluder_front":
                        raise RuntimeError("Phase-1 currently supports depth_mode=occluder_front; DA-V2 adapter is next.")
                    order = front_only_order([int(annotation["track_id"]) for annotation in background_annotations])
                    background, effective_mask = composite_frame(
                        background,
                        background_annotations,
                        rgba,
                        placement,
                        order,
                        blend_method=blend_method,
                    )
                relative_frame = Path(output_name) / f"{frame_index:06d}.jpg"
                _save_rgb(output_frames_dir / relative_frame, background)
                current_image_id = next_image_id
                next_image_id += 1
                output_dataset["images"].append(
                    {
                        "id": current_image_id,
                        "video_id": output_video_id,
                        "frame_index": frame_index,
                        "frame_id": frame_index + 1,
                        "file_name": str(Path("frames") / relative_frame),
                        "width": int(frame["width"]),
                        "height": int(frame["height"]),
                        "source": {"video": source_name, "image_id": int(frame["id"])},
                    }
                )
                for source_annotation in background_annotations:
                    annotation = copy.deepcopy(source_annotation)
                    annotation["id"] = next_annotation_id
                    annotation["image_id"] = current_image_id
                    annotation["video_id"] = output_video_id
                    next_annotation_id += 1
                    update = label_frame(
                        source_annotation,
                        effective_mask,
                        background.shape[:2],
                        [occluder_track_id] if placement is not None else [],
                    )
                    update["provenance"]["base_kitti_occluded"] = int(
                        source_annotation.get("kitti", {}).get("occluded", -1)
                    )
                    annotation.update(update)
                    if float(update["occlusion_ratio"]) > 0:
                        histories.append(
                            {
                                "frame_index": frame_index,
                                "victim_track": int(source_annotation["track_id"]),
                                "occluder_track": occluder_track_id,
                                "occlusion_ratio": float(update["occlusion_ratio"]),
                            }
                        )
                    output_dataset["annotations"].append(annotation)
                if placement is not None and effective_mask.any():
                    bbox = mask_to_bbox(effective_mask)
                    output_dataset["annotations"].append(
                        {
                            "id": next_annotation_id,
                            "image_id": current_image_id,
                            "video_id": output_video_id,
                            "frame_index": frame_index,
                            "track_id": occluder_track_id,
                            "category_id": category_id_by_name[target_category],
                            "bbox": bbox,
                            "visible_bbox": bbox,
                            "amodal_bbox": bbox,
                            "segmentation": encode_binary_mask(effective_mask),
                            "area": int(effective_mask.sum()),
                            "iscrowd": 0,
                            "occlusion_ratio": 0.0,
                            "occlusion_level": 0,
                            "occluder_ids": [],
                            "synthetic_occluder": True,
                            "provenance": {"pool_instance_id": int(pool_record["id"])},
                        }
                    )
                    next_annotation_id += 1
                    track_frame_records.append(
                        {"frame": frame_index, "bbox": bbox, "mask_ref": None}
                    )

            video_events = derive_events(
                histories,
                output_video_id,
                entry_speed=abs(float(trajectory.motion.v0[0])),
                first_event_id=next_event_id,
            )
            events.extend(event.to_dict() for event in video_events)
            next_event_id += len(video_events)
            occluder_tracks.append(
                {
                    "track_id": occluder_track_id,
                    "video_id": output_video_id,
                    "category": target_category,
                    "synthetic": True,
                    "source": pool_record["source"],
                    "motion": trajectory.to_dict()["motion"],
                    "frames": track_frame_records,
                    "depth_plane": 0.0,
                }
            )
            manifests.append(
                {
                    "output_video_id": output_video_id,
                    "output_video": output_name,
                    "source_video": source_name,
                    "victim_track": int(victim["track_id"]),
                    "occluder_track": occluder_track_id,
                    "pool_instance": pool_record,
                    "augmentation": augmentation,
                    "rho_target_peak": peak_rho,
                    "blend_method": blend_method,
                    "trajectory": trajectory.to_dict(),
                    "depth_mode": synthesis.get("depth_mode", "occluder_front"),
                }
            )
            generated += 1

    if generated == 0:
        raise RuntimeError("no synthetic sequence was generated; inspect split, pool, and victim filters")
    validate_video_dataset(output_dataset)
    qc_result = verify_synthetic_video_dataset(output_dataset, events)
    if not qc_result["ok"]:
        raise RuntimeError(f"synthetic dataset QC failed: {qc_result}")
    save_json(output_dir / "annotations.json", output_dataset)
    save_json(output_dir / "occluder_tracks.json", occluder_tracks)
    save_json(output_dir / "events.json", events)
    save_jsonl(output_dir / "manifest.jsonl", manifests)
    summary = {
        "sequences": generated,
        "frames": len(output_dataset["images"]),
        "annotations": len(output_dataset["annotations"]),
        "events": len(events),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "qc.json", qc_result)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate moving copy-paste occlusion on KITTI Tracking")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()
    summary = run(args.config, args.max_sequences)
    print(summary)


if __name__ == "__main__":
    main()
