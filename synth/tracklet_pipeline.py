from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json, save_jsonl
from common.io_video import group_annotations_by_image, group_frames_by_video
from common.schema import bbox_to_mask, compute_ratio, validate_video_dataset
from data.kitti_tracking import convert_tracking_to_video_coco
from label.compute import label_frame_multi, label_synthetic_occluder
from label.events import derive_events
from qc.verify import verify_synthetic_video_dataset
from synth.tracklet_compositor import RenderedLayer, TrackletLayer, composite_tracklet_layers


@dataclass(frozen=True)
class ScheduledTracklet:
    record: dict[str, Any]
    track_id: int
    start_position: int
    boundary_side: str | None
    target_height_fraction: float | None
    source_reference_height: float | None
    translation_xy: tuple[float, float]
    victim_anchor: dict[str, Any] | None

    @property
    def end_position(self) -> int:
        return self.start_position + 29


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _load_rgba(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA")).copy()


def _save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, quality=95, subsampling=0)


def map_source_bbox(
    frame_record: Mapping[str, Any],
    target_size: tuple[int, int],
    *,
    position_mode: str = "normalized_source",
    boundary_side: str | None = None,
    boundary_visible_fraction: float = 0.6,
    target_height_fraction: float | None = None,
    source_reference_height: float | None = None,
    height_fraction_range: tuple[float, float] | None = None,
    translation_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float]:
    """Map the source crop's original coordinates to the target image."""
    x, y, width, height = (float(value) for value in frame_record["crop_bbox_xywh"])
    source_width, source_height = (float(value) for value in frame_record["source_image_size"])
    target_width, target_height = target_size
    source_crop_height = height
    source_center_x_fraction = (x + width / 2.0) / source_width
    source_bottom_fraction = (y + height) / source_height
    if position_mode == "normalized_source":
        x = x * target_width / source_width
        y = y * target_height / source_height
    elif position_mode != "source_pixels":
        raise ValueError(f"unsupported position_mode: {position_mode}")

    if target_height_fraction is not None:
        if source_reference_height is None or source_reference_height <= 0:
            raise ValueError("source_reference_height must be positive for KITTI scale matching")
        relative_motion_scale = source_crop_height / source_reference_height
        realized_fraction = target_height_fraction * relative_motion_scale
        if height_fraction_range is not None:
            realized_fraction = float(np.clip(realized_fraction, *height_fraction_range))
        scale = target_height * realized_fraction / source_crop_height
        width, height = width * scale, height * scale
        # Scaling around the source top-left would make a smaller pedestrian
        # float above the road. Preserve the normalized bottom-center anchor.
        x = source_center_x_fraction * target_width - width / 2.0
        y = source_bottom_fraction * target_height - height
    elif position_mode == "normalized_source":
        # Legacy fallback used when no KITTI scale statistics are requested.
        scale = target_height / source_height
        width, height = width * scale, height * scale

    x += float(translation_xy[0])
    y += float(translation_xy[1])

    if boundary_side is not None:
        visible = float(np.clip(boundary_visible_fraction, 0.05, 1.0))
        if boundary_side == "left":
            x = -(1.0 - visible) * width
        elif boundary_side == "right":
            x = target_width - visible * width
        else:
            raise ValueError(f"unsupported boundary side: {boundary_side}")
    return x, y, width, height


def _bbox_intersection_ratio(occluder: tuple[float, ...], victim: list[float]) -> float:
    ox, oy, ow, oh = occluder
    vx, vy, vw, vh = (float(value) for value in victim)
    intersection_width = max(0.0, min(ox + ow, vx + vw) - max(ox, vx))
    intersection_height = max(0.0, min(oy + oh, vy + vh) - max(oy, vy))
    victim_area = max(0.0, vw * vh)
    return intersection_width * intersection_height / victim_area if victim_area > 0 else 0.0


def _select_victim_overlap_alignment(
    record: Mapping[str, Any],
    source_frames: list[dict[str, Any]],
    annotations_by_image: Mapping[int, list[dict[str, Any]]],
    synthesis: Mapping[str, Any],
    target_height_fraction: float,
    source_reference_height: float,
    height_fraction_range: tuple[float, float],
) -> tuple[int, tuple[float, float], dict[str, Any]]:
    """Choose one constant translation that maximizes overlap with one KITTI victim track."""
    anchor_offset = int(np.clip(int(synthesis.get("victim_anchor_offset", 15)), 0, 29))
    minimum_area = float(synthesis.get("victim_min_area", 400.0))
    max_base_occlusion = int(synthesis.get("victim_max_base_occlusion", 0))
    best: tuple[tuple[int, float], int, tuple[float, float], dict[str, Any]] | None = None
    position_mode = str(synthesis.get("position_mode", "normalized_source"))
    for start_position in range(len(source_frames) - 29):
        anchor_frame = source_frames[start_position + anchor_offset]
        anchor_box = map_source_bbox(
            record["frames"][anchor_offset],
            (int(anchor_frame["width"]), int(anchor_frame["height"])),
            position_mode=position_mode,
            target_height_fraction=target_height_fraction,
            source_reference_height=source_reference_height,
            height_fraction_range=height_fraction_range,
        )
        for victim in annotations_by_image.get(int(anchor_frame["id"]), []):
            if float(victim.get("area", 0.0)) < minimum_area:
                continue
            if int(victim.get("kitti", {}).get("occluded", 3)) > max_base_occlusion:
                continue
            vx, vy, vw, vh = (float(value) for value in victim["bbox"])
            ox, oy, ow, oh = anchor_box
            translation = (
                vx + vw / 2.0 - (ox + ow / 2.0),
                vy + vh - (oy + oh),
            )
            hit_frames = 0
            ratio_sum = 0.0
            for offset in range(30):
                frame = source_frames[start_position + offset]
                mapped = map_source_bbox(
                    record["frames"][offset],
                    (int(frame["width"]), int(frame["height"])),
                    position_mode=position_mode,
                    target_height_fraction=target_height_fraction,
                    source_reference_height=source_reference_height,
                    height_fraction_range=height_fraction_range,
                    translation_xy=translation,
                )
                same_track = next(
                    (
                        annotation
                        for annotation in annotations_by_image.get(int(frame["id"]), [])
                        if int(annotation["track_id"]) == int(victim["track_id"])
                    ),
                    None,
                )
                if same_track is None:
                    continue
                ratio = _bbox_intersection_ratio(mapped, same_track["bbox"])
                if ratio > 0:
                    hit_frames += 1
                    ratio_sum += ratio
            score = (hit_frames, ratio_sum)
            anchor = {
                "track_id": int(victim["track_id"]),
                "category_id": int(victim["category_id"]),
                "frame_index": int(anchor_frame["frame_index"]),
                "frame_position": start_position + anchor_offset,
                "bbox_xywh": [float(value) for value in victim["bbox"]],
                "predicted_overlap_frames": hit_frames,
                "predicted_overlap_ratio_sum": ratio_sum,
            }
            candidate = (score, start_position, translation, anchor)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None or best[0][0] == 0:
        raise RuntimeError("no clean KITTI victim can overlap the source-preserving tracklet trajectory")
    return best[1], best[2], best[3]


def _schedule_tracklets(
    records: list[dict[str, Any]],
    frame_count: int,
    output_video_id: int,
    synthesis: Mapping[str, Any],
    rng: random.Random,
    scale_stats: Mapping[str, float] | None = None,
    source_frames: list[dict[str, Any]] | None = None,
    annotations_by_image: Mapping[int, list[dict[str, Any]]] | None = None,
) -> list[ScheduledTracklet]:
    if frame_count < 30:
        return []
    count_min, count_max = (int(value) for value in synthesis.get("tracklets_per_sequence", [2, 4]))
    count = min(len(records), rng.randint(count_min, count_max))
    selected = rng.sample(records, count)
    boundary_count = max(0, min(int(synthesis.get("boundary_shift_count", 2)), 2, count))
    boundary_sides = ["left", "right"][:boundary_count]
    schedules: list[ScheduledTracklet] = []
    for index, record in enumerate(selected):
        if int(record.get("length", 0)) != 30 or len(record.get("frames", [])) != 30:
            raise ValueError(f"tracklet {record.get('id')} is not exactly 30 frames")
        source_frame_indices = [int(frame["source_frame"]) for frame in record["frames"]]
        if source_frame_indices != list(
            range(source_frame_indices[0], source_frame_indices[0] + 30)
        ):
            raise ValueError(f"tracklet {record.get('id')} contains a frame gap")
        if min(float(frame["visibility"]) for frame in record["frames"]) < 0.8:
            raise ValueError(f"tracklet {record.get('id')} contains visibility < 0.8")
        reference_height = float(
            np.median([float(frame["crop_bbox_xywh"][3]) for frame in record["frames"]])
        )
        target_height_fraction = None
        if scale_stats is not None:
            target_height_fraction = rng.triangular(
                float(scale_stats["lower"]),
                float(scale_stats["upper"]),
                float(scale_stats["mean"]),
            )
        start_position = rng.randint(0, frame_count - 30)
        translation = (0.0, 0.0)
        victim_anchor = None
        placement_mode = str(synthesis.get("placement_mode", "source_coordinates"))
        if placement_mode == "victim_overlap_translation" and index >= boundary_count:
            if scale_stats is None or target_height_fraction is None:
                raise ValueError("victim overlap placement requires KITTI scale statistics")
            if source_frames is None or annotations_by_image is None:
                raise ValueError("victim overlap placement requires KITTI frames and annotations")
            start_position, translation, victim_anchor = _select_victim_overlap_alignment(
                record,
                source_frames,
                annotations_by_image,
                synthesis,
                target_height_fraction,
                reference_height,
                (float(scale_stats["lower"]), float(scale_stats["upper"])),
            )
        elif placement_mode != "source_coordinates":
            raise ValueError(f"unsupported tracklet placement_mode: {placement_mode}")
        schedules.append(
            ScheduledTracklet(
                record=record,
                track_id=900000 + output_video_id * 100 + index,
                start_position=start_position,
                boundary_side=boundary_sides[index] if index < boundary_count else None,
                target_height_fraction=target_height_fraction,
                source_reference_height=reference_height,
                translation_xy=translation,
                victim_anchor=victim_anchor,
            )
        )
    return schedules


def kitti_person_height_stats(
    dataset: Mapping[str, Any],
    *,
    max_occluded: int = 0,
    max_truncated: float = 0.2,
    stddevs: float = 1.0,
) -> dict[str, float | int | str]:
    """Measure clean KITTI person bbox heights relative to their image height."""
    category_ids = {
        int(category["id"])
        for category in dataset.get("categories", [])
        if category.get("name") == "person"
    }
    image_heights = {
        int(image["id"]): float(image["height"])
        for image in dataset.get("images", [])
    }
    values: list[float] = []
    for annotation in dataset.get("annotations", []):
        if int(annotation["category_id"]) not in category_ids:
            continue
        kitti = annotation.get("kitti", {})
        if int(kitti.get("occluded", 3)) > max_occluded:
            continue
        if float(kitti.get("truncated", 1.0)) > max_truncated:
            continue
        image_height = image_heights.get(int(annotation["image_id"]), 0.0)
        if image_height > 0:
            values.append(float(annotation["bbox"][3]) / image_height)
    if not values:
        raise ValueError("no KITTI person boxes satisfy the configured scale reference filters")
    samples = np.asarray(values, dtype=np.float64)
    mean = float(samples.mean())
    std = float(samples.std())
    lower = max(0.01, mean - stddevs * std)
    upper = min(0.95, mean + stddevs * std)
    if not lower < upper:
        raise ValueError("KITTI person scale interval is empty")
    return {
        "mode": "kitti_person_bbox_height_mean_std",
        "count": len(values),
        "mean": mean,
        "std": std,
        "stddevs": float(stddevs),
        "lower": lower,
        "upper": upper,
        "max_occluded": int(max_occluded),
        "max_truncated": float(max_truncated),
        "sampling": "triangular_with_mode_at_mean",
    }


def _pairwise_ratio(victim_bbox: list[float], mask: np.ndarray, shape: tuple[int, int]) -> float:
    amodal = bbox_to_mask(victim_bbox, shape[0], shape[1])
    visible = np.logical_and(amodal != 0, np.asarray(mask) == 0).astype(np.uint8)
    return compute_ratio(visible, amodal)


def run(config_file: str | Path, max_sequences_override: int | None = None) -> dict[str, Any]:
    config, path = load_config(config_file)
    split = load_json(resolve_path(path.parent, config["split"]["output"]))
    pool_dir = resolve_path(path.parent, config["tracklet_pool"]["output_dir"])
    pool_metadata = load_json(pool_dir / "tracklets.json")
    pool_records = list(pool_metadata.get("tracklets", []))
    if not pool_records:
        raise ValueError("MOT17 SAM tracklet pool is empty")

    source_dataset = convert_tracking_to_video_coco(
        config_path(config, path, "paths", "kitti_tracking"),
        split["train_sequences"],
        config["classes"]["kitti_map"],
    )
    frames_by_video = group_frames_by_video(source_dataset)
    annotations_by_image = group_annotations_by_image(source_dataset)
    source_video_by_id = {int(video["id"]): video for video in source_dataset["videos"]}
    synthesis = config["tracklet_synthesis"]
    scale_mode = str(synthesis.get("scale_mode", "normalized_source"))
    scale_stats: dict[str, float | int | str] | None = None
    if scale_mode == "kitti_person_bbox_height_mean_std":
        scale_stats = kitti_person_height_stats(
            source_dataset,
            max_occluded=int(synthesis.get("scale_reference_max_occlusion", 0)),
            max_truncated=float(synthesis.get("scale_reference_max_truncation", 0.2)),
            stddevs=float(synthesis.get("scale_stddevs", 1.0)),
        )
    elif scale_mode != "normalized_source":
        raise ValueError(f"unsupported tracklet scale_mode: {scale_mode}")
    output_dir = resolve_path(path.parent, synthesis["output_dir"])
    output_frames_dir = output_dir / "frames"
    rng = random.Random(int(config.get("seed", 0)))
    max_sequences = int(max_sequences_override or synthesis.get("max_sequences", len(frames_by_video)))

    output_dataset: dict[str, Any] = {
        "info": {
            "description": "Strict MOT17-SAM tracklet copy-paste on KITTI Tracking",
            "tracklet_rule": "same identity, exactly 30 consecutive frames, visibility >= 0.8 in every frame",
            "paste_order": "large_first",
            "position_mode": synthesis.get("position_mode", "normalized_source"),
            "scale_policy": scale_stats or {"mode": "normalized_source"},
            "amodal_mask_caveat": "KITTI victim amodal masks use bounding-box proxies.",
        },
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(source_dataset["categories"]),
    }
    category_id_by_name = {category["name"]: int(category["id"]) for category in output_dataset["categories"]}
    if "person" not in category_id_by_name:
        raise ValueError("KITTI class map must include person")

    manifests: list[dict[str, Any]] = []
    occluder_tracks: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    generated = 0

    for source_video_id in sorted(frames_by_video):
        if generated >= max_sequences:
            break
        source_frames = frames_by_video[source_video_id]
        output_video_id = generated + 1
        schedules = _schedule_tracklets(
            pool_records,
            len(source_frames),
            output_video_id,
            synthesis,
            rng,
            scale_stats=scale_stats,
            source_frames=source_frames,
            annotations_by_image=annotations_by_image,
        )
        if not schedules:
            continue
        source_name = str(source_video_by_id[source_video_id]["name"])
        output_name = f"{source_name}_tracklet_synth"
        output_dataset["videos"].append(
            {"id": output_video_id, "name": output_name, "fps": 10, "num_frames": len(source_frames)}
        )
        track_frames: dict[int, list[dict[str, Any]]] = {schedule.track_id: [] for schedule in schedules}

        for frame_position, frame in enumerate(source_frames):
            frame_index = int(frame["frame_index"])
            background_annotations = annotations_by_image.get(int(frame["id"]), [])
            background = _load_rgb(Path(frame["source_path"]))
            active: list[TrackletLayer] = []
            for schedule in schedules:
                if not schedule.start_position <= frame_position <= schedule.end_position:
                    continue
                offset = frame_position - schedule.start_position
                frame_record = schedule.record["frames"][offset]
                bbox = map_source_bbox(
                    frame_record,
                    (background.shape[1], background.shape[0]),
                    position_mode=str(synthesis.get("position_mode", "normalized_source")),
                    boundary_side=schedule.boundary_side,
                    boundary_visible_fraction=float(synthesis.get("boundary_visible_fraction", 0.6)),
                    target_height_fraction=schedule.target_height_fraction,
                    source_reference_height=schedule.source_reference_height,
                    height_fraction_range=(
                        (float(scale_stats["lower"]), float(scale_stats["upper"]))
                        if scale_stats is not None
                        else None
                    ),
                    translation_xy=schedule.translation_xy,
                )
                active.append(
                    TrackletLayer(
                        track_id=schedule.track_id,
                        rgba=_load_rgba(pool_dir / frame_record["file_name"]),
                        bbox_xywh=bbox,
                        provenance={
                            "tracklet_id": int(schedule.record["id"]),
                            "mot17_sequence": schedule.record["sequence"],
                            "mot17_track_id": int(schedule.record["source_track_id"]),
                            "mot17_frame": int(frame_record["source_frame"]),
                            "mot17_bbox_xywh": frame_record["source_bbox_xywh"],
                            "position_mode": synthesis.get("position_mode", "normalized_source"),
                            "boundary_shift": schedule.boundary_side,
                            "scale_mode": scale_mode,
                            "target_height_fraction": schedule.target_height_fraction,
                            "translation_xy": list(schedule.translation_xy),
                            "victim_anchor": schedule.victim_anchor,
                        },
                    )
                )
            background, rendered = composite_tracklet_layers(
                background,
                active,
                blend_method=str(synthesis.get("blend_method", "none")),
            )
            masks_by_id = {layer.track_id: layer.amodal_mask for layer in rendered}

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
                annotation.update(
                    label_frame_multi(source_annotation, masks_by_id, background.shape[:2])
                )
                annotation["id"] = next_annotation_id
                annotation["image_id"] = current_image_id
                annotation["video_id"] = output_video_id
                annotation["provenance"]["base_kitti_occluded"] = int(
                    source_annotation.get("kitti", {}).get("occluded", -1)
                )
                next_annotation_id += 1
                output_dataset["annotations"].append(annotation)
                for layer in rendered:
                    ratio = _pairwise_ratio(source_annotation["bbox"], layer.amodal_mask, background.shape[:2])
                    if ratio > 0:
                        histories.append(
                            {
                                "frame_index": frame_index,
                                "video_id": output_video_id,
                                "victim_track": int(source_annotation["track_id"]),
                                "occluder_track": layer.track_id,
                                "occlusion_ratio": ratio,
                            }
                        )

            for layer in rendered:
                annotation = {
                    "id": next_annotation_id,
                    "image_id": current_image_id,
                    "video_id": output_video_id,
                    "frame_index": frame_index,
                    "track_id": layer.track_id,
                    "category_id": category_id_by_name["person"],
                    "iscrowd": 0,
                    "paste_order": layer.paste_order,
                }
                annotation.update(
                    label_synthetic_occluder(
                        visible_mask=layer.visible_mask,
                        amodal_mask=layer.amodal_mask,
                        occluder_ids=layer.occluder_ids,
                        provenance=layer.provenance,
                    )
                )
                output_dataset["annotations"].append(annotation)
                next_annotation_id += 1
                track_frames[layer.track_id].append(
                    {
                        "frame": frame_index,
                        "bbox": annotation["bbox"],
                        "amodal_bbox": annotation["amodal_bbox"],
                        "occlusion_ratio": annotation["occlusion_ratio"],
                        "occlusion_level": annotation["occlusion_level"],
                        "occluder_ids": annotation["occluder_ids"],
                        "bbox_height_fraction": float(annotation["amodal_bbox"][3])
                        / float(background.shape[0]),
                    }
                )

        for schedule in schedules:
            if len(track_frames[schedule.track_id]) != 30:
                raise RuntimeError(f"synthetic track {schedule.track_id} did not appear in exactly 30 frames")
            occluder_tracks.append(
                {
                    "track_id": schedule.track_id,
                    "video_id": output_video_id,
                    "category": "person",
                    "synthetic": True,
                    "source": {
                        "dataset": "MOT17",
                        "sequence": schedule.record["sequence"],
                        "track_id": schedule.record["source_track_id"],
                        "tracklet_id": schedule.record["id"],
                    },
                    "start_position": schedule.start_position,
                    "duration": 30,
                    "boundary_shift": schedule.boundary_side,
                    "scale_mode": scale_mode,
                    "target_height_fraction": schedule.target_height_fraction,
                    "source_reference_height": schedule.source_reference_height,
                    "translation_xy": list(schedule.translation_xy),
                    "victim_anchor": schedule.victim_anchor,
                    "frames": track_frames[schedule.track_id],
                }
            )
        manifests.append(
            {
                "output_video_id": output_video_id,
                "output_video": output_name,
                "source_video": source_name,
                "paste_order": "large_first_per_frame",
                "blend_method": synthesis.get("blend_method", "none"),
                "tracklets": [
                    {
                        "synthetic_track_id": schedule.track_id,
                        "pool_tracklet_id": schedule.record["id"],
                        "mot17_identity": [schedule.record["sequence"], schedule.record["source_track_id"]],
                        "start_position": schedule.start_position,
                        "end_position": schedule.end_position,
                        "boundary_shift": schedule.boundary_side,
                        "scale_mode": scale_mode,
                        "target_height_fraction": schedule.target_height_fraction,
                        "translation_xy": list(schedule.translation_xy),
                        "victim_anchor": schedule.victim_anchor,
                    }
                    for schedule in schedules
                ],
            }
        )
        generated += 1

    if generated == 0:
        raise RuntimeError("no sequence generated; KITTI videos must contain at least 30 frames")
    events: list[dict[str, Any]] = []
    next_event_id = 1
    for video_id in range(1, generated + 1):
        video_histories = [record for record in histories if int(record["video_id"]) == video_id]
        video_events = derive_events(video_histories, video_id, entry_speed=0.0, first_event_id=next_event_id)
        events.extend(event.to_dict() for event in video_events)
        next_event_id += len(video_events)

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
        "synthetic_tracklets": len(occluder_tracks),
        "events": len(events),
        "scale_policy": scale_stats or {"mode": "normalized_source"},
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "qc.json", qc_result)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paste strict MOT17 SAM tracklets into KITTI Tracking")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()
    print(run(args.config, args.max_sequences))


if __name__ == "__main__":
    main()
