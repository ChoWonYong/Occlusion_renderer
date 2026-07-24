"""Event-based, variable-length occlusion synthesis pipeline (Step 7).

Wires the confirmed 2026-07-23 policy end to end:
- merge the MOT17 + KITTI multi-class tracklet pools;
- per KITTI train sequence, plan victim events by density and class ratio
  (``synth.scheduler``);
- for each event, size the occluder relative to its victim and search a
  target-peak, complete-event placement (``synth.placement`` + ``synth.geometry``),
  retrying with alternative tracklets on failure;
- pack events so at most 2 occluders are ever concurrent;
- render large-first and emit extended GT with the ``amodal_original`` detector
  bbox policy, synthetic occluder tracks, occlusion events, and QC.

The legacy fixed-30-frame ``synth.tracklet_pipeline`` is left untouched.
"""

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
from common.schema import validate_video_dataset
from data.kitti_tracking import convert_tracking_to_video_coco
from label.compute import label_frame_multi, label_synthetic_occluder
from label.events import derive_events
from qc.verify import verify_synthetic_video_dataset
from synth.geometry import search_event_placement
from synth.placement import sample_target_peak
from synth.scheduler import EventPlan, PoolTracklet, merge_pools, plan_schedule, pool_by_category
from synth.tracklet_compositor import TrackletLayer, composite_tracklet_layers


@dataclass
class ScheduledRender:
    synthetic_track_id: int
    occluder: PoolTracklet
    start_position: int
    exposure_length: int
    reference_height: float
    source_reference_height: float
    translation: tuple[float, float]
    kind: str
    victim_track_id: int | None
    target_peak: float
    achieved_peak: float
    peak_frame: int | None


def _load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _load_rgba(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA")).copy()


def _save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, quality=95, subsampling=0)


def _load_pool(config: Mapping[str, Any], config_dir: Path) -> list[PoolTracklet]:
    """Load and merge the MOT17 + KITTI tracklet pools, resolving RGBA paths."""
    metadatas: list[dict[str, Any]] = []
    candidates = [
        ("tracklet_pool", "output_dir"),
        ("kitti_sam3_pool", "tracklet_output_dir"),
    ]
    for section, key in candidates:
        directory = config.get(section, {}).get(key)
        if not directory:
            continue
        base_dir = resolve_path(config_dir, directory)
        pool_file = base_dir / "tracklets.json"
        if not pool_file.is_file():
            continue
        metadata = load_json(pool_file)
        for record in metadata.get("tracklets", []):
            for frame in record.get("frames", []):
                frame["rgba_path"] = str(base_dir / frame["file_name"])
        metadatas.append(metadata)
    if not metadatas:
        raise FileNotFoundError("no tracklet pool found; run the MOT17 / KITTI SAM3 pool builders")
    return merge_pools(metadatas)


def _frames_annotations(
    source_frames: list[dict[str, Any]],
    annotations_by_image: Mapping[int, list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    return [list(annotations_by_image.get(int(frame["id"]), [])) for frame in source_frames]


def _victim_positions(
    frames_annotations: list[list[dict[str, Any]]], victim_track_id: int
) -> dict[int, list[float]]:
    positions: dict[int, list[float]] = {}
    for position, annotations in enumerate(frames_annotations):
        for annotation in annotations:
            if int(annotation["track_id"]) == victim_track_id:
                positions[position] = [float(value) for value in annotation["bbox"]]
                break
    return positions


def _resolve_placement(
    event: EventPlan,
    victim_positions: dict[int, list[float]],
    victim_class: str,
    grouped_pool: Mapping[str, list[PoolTracklet]],
    synthesis: Mapping[str, Any],
    target_size: tuple[int, int],
    sequence_length: int,
    rng: random.Random,
) -> tuple[PoolTracklet, dict[str, Any]] | None:
    """Search a target-peak placement, retrying with alternative same-class tracklets."""
    multipliers = list(synthesis.get("scale_search_multipliers", [1.0]))
    factors = dict(synthesis["class_height_factor"])
    effective = tuple(synthesis.get("effective_event_frames", [8, 20]))
    accept_kwargs = {
        "effective_range": (int(effective[0]), int(effective[1])),
        "peak_max": float(synthesis.get("peak_rho_max", 0.80)),
        "end_max": float(synthesis.get("event_end_rho_max", 0.05)),
    }
    target_peak, _, band = sample_target_peak(dict(synthesis["peak_rho_distribution"]), rng)

    tried: set[int] = set()
    attempts = [event.occluder] + [t for t in grouped_pool.get(event.occluder.category, [])]
    for occluder in attempts:
        if occluder.tracklet_id in tried:
            continue
        tried.add(occluder.tracklet_id)
        exposure = min(event.exposure_length, occluder.length)
        occluder_frames = occluder.frames[:exposure]
        best = search_event_placement(
            occluder_frames,
            victim_positions,
            occluder_class=occluder.category,
            victim_class=victim_class,
            class_height_factor=factors,
            target_peak=target_peak,
            band=band,
            multipliers=multipliers,
            target_size=target_size,
            sequence_length=sequence_length,
            accept_kwargs=accept_kwargs,
        )
        if best is not None:
            best["target_peak"] = target_peak
            best["exposure_length"] = exposure
            return occluder, best
    return None


def _fits_concurrency(
    occupancy: list[int], start: int, length: int, max_concurrent: int, allow_concurrent: bool
) -> bool:
    ceiling = max_concurrent if allow_concurrent else 1
    return all(occupancy[position] < ceiling for position in range(start, start + length))


def run(config_file: str | Path, max_sequences_override: int | None = None) -> dict[str, Any]:
    config, path = load_config(config_file)
    synthesis = config["tracklet_synthesis"]
    split = load_json(resolve_path(path.parent, config["split"]["output"]))
    pool = _load_pool(config, path.parent)
    grouped_pool = pool_by_category(pool)

    source_dataset = convert_tracking_to_video_coco(
        config_path(config, path, "paths", "kitti_tracking"),
        split["train_sequences"],
        config["classes"]["kitti_map"],
    )
    frames_by_video = group_frames_by_video(source_dataset)
    annotations_by_image = group_annotations_by_image(source_dataset)
    video_by_id = {int(video["id"]): video for video in source_dataset["videos"]}
    name_by_category_id = {int(c["id"]): str(c["name"]) for c in source_dataset["categories"]}
    category_id_by_name = {str(c["name"]): int(c["id"]) for c in source_dataset["categories"]}

    detector_bbox_policy = str(synthesis.get("victim_detector_bbox_policy", "amodal_original"))
    blend_method = str(synthesis.get("blend_method", "none"))
    max_concurrent = int(synthesis.get("max_concurrent_occluders", 2))
    rng = random.Random(int(config.get("seed", 0)))
    output_dir = resolve_path(path.parent, synthesis["output_dir"])
    output_frames_dir = output_dir / "frames"
    max_sequences = int(max_sequences_override or synthesis.get("max_sequences", len(frames_by_video)))

    output_dataset: dict[str, Any] = {
        "info": {
            "description": "Event-based multi-class tracklet copy-paste on KITTI Tracking",
            "detector_bbox_policy": detector_bbox_policy,
            "scale_policy": "victim_relative",
            "amodal_mask_caveat": "KITTI victim amodal masks use bounding-box proxies.",
        },
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(source_dataset["categories"]),
    }

    occluder_tracks: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    scheduling_summaries: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    generated = 0

    for source_video_id in sorted(frames_by_video):
        if generated >= max_sequences:
            break
        source_frames = frames_by_video[source_video_id]
        sequence_length = len(source_frames)
        if sequence_length < int(synthesis.get("tracklet_min_frames", 30)):
            continue
        target_size = (int(source_frames[0]["width"]), int(source_frames[0]["height"]))
        frames_annotations = _frames_annotations(source_frames, annotations_by_image)
        plan = plan_schedule(frames_annotations, pool, synthesis, rng)

        output_video_id = generated + 1
        occupancy = [0] * sequence_length
        scheduled: list[ScheduledRender] = []
        next_synth_track = 900000 + output_video_id * 1000
        rejected = 0
        for event in plan["events"]:
            if event.kind != "victim" or event.victim_track_id is None:
                continue  # boundary events handled separately below
            victim_positions = _victim_positions(frames_annotations, event.victim_track_id)
            if not victim_positions:
                rejected += 1
                continue
            victim_category_id = next(
                int(annotation["category_id"])
                for annotations in frames_annotations
                for annotation in annotations
                if int(annotation["track_id"]) == event.victim_track_id
            )
            victim_class = name_by_category_id[victim_category_id]
            resolved = _resolve_placement(
                event, victim_positions, victim_class, grouped_pool, synthesis,
                target_size, sequence_length, rng,
            )
            if resolved is None:
                rejected += 1
                continue
            occluder, best = resolved
            if not _fits_concurrency(
                occupancy, best["start_position"], best["exposure_length"],
                max_concurrent, event.allow_concurrent,
            ):
                rejected += 1
                continue
            for position in range(best["start_position"], best["start_position"] + best["exposure_length"]):
                occupancy[position] += 1
            scheduled.append(
                ScheduledRender(
                    synthetic_track_id=next_synth_track,
                    occluder=occluder,
                    start_position=int(best["start_position"]),
                    exposure_length=int(best["exposure_length"]),
                    reference_height=float(best["reference_height"]),
                    source_reference_height=float(best["source_reference_height"]),
                    translation=tuple(best["translation"]),
                    kind="victim",
                    victim_track_id=int(event.victim_track_id),
                    target_peak=float(best["target_peak"]),
                    achieved_peak=float(best["achieved_peak"]),
                    peak_frame=int(best["peak_frame"]),
                )
            )
            next_synth_track += 1

        if not scheduled:
            continue

        source_name = str(video_by_id[source_video_id]["name"])
        output_name = f"{source_name}_event_synth"
        output_dataset["videos"].append(
            {"id": output_video_id, "name": output_name, "fps": 10, "num_frames": sequence_length}
        )
        track_frame_counts: dict[int, int] = {s.synthetic_track_id: 0 for s in scheduled}

        for position, frame in enumerate(source_frames):
            from synth.geometry import map_occluder_box

            frame_index = int(frame["frame_index"])
            background = _load_rgb(Path(frame["source_path"]))
            active: list[TrackletLayer] = []
            for sched in scheduled:
                if not (sched.start_position <= position < sched.start_position + sched.exposure_length):
                    continue
                offset = position - sched.start_position
                frame_record = sched.occluder.frames[offset]
                box = map_occluder_box(
                    frame_record,
                    (background.shape[1], background.shape[0]),
                    reference_height=sched.reference_height,
                    source_reference_height=sched.source_reference_height,
                    translation=sched.translation,
                )
                active.append(
                    TrackletLayer(
                        track_id=sched.synthetic_track_id,
                        rgba=_load_rgba(frame_record["rgba_path"]),
                        bbox_xywh=box,
                        provenance={
                            "tracklet_id": int(sched.occluder.tracklet_id),
                            "category": sched.occluder.category,
                            "source": sched.occluder.source,
                            "identity": list(sched.occluder.identity_key),
                            "victim_track_id": sched.victim_track_id,
                            "target_peak": sched.target_peak,
                            "translation_xy": list(sched.translation),
                        },
                    )
                )
            background, rendered = composite_tracklet_layers(background, active, blend_method=blend_method)
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

            for source_annotation in frames_annotations[position]:
                annotation = copy.deepcopy(source_annotation)
                annotation.update(
                    label_frame_multi(
                        source_annotation, masks_by_id, background.shape[:2],
                        detector_bbox_policy=detector_bbox_policy,
                    )
                )
                annotation["id"] = next_annotation_id
                annotation["image_id"] = current_image_id
                annotation["video_id"] = output_video_id
                annotation["provenance"]["base_kitti_occluded"] = int(
                    source_annotation.get("kitti", {}).get("occluded", -1)
                )
                next_annotation_id += 1
                output_dataset["annotations"].append(annotation)
                for occluder_id in annotation["occluder_ids"]:
                    histories.append(
                        {
                            "frame_index": frame_index,
                            "video_id": output_video_id,
                            "victim_track": int(source_annotation["track_id"]),
                            "occluder_track": int(occluder_id),
                            "occlusion_ratio": float(annotation["occlusion_ratio"]),
                        }
                    )

            for layer in rendered:
                occluder_annotation = {
                    "id": next_annotation_id,
                    "image_id": current_image_id,
                    "video_id": output_video_id,
                    "frame_index": frame_index,
                    "track_id": layer.track_id,
                    "category_id": category_id_by_name[layer.provenance["category"]],
                    "iscrowd": 0,
                    "paste_order": layer.paste_order,
                }
                occluder_annotation.update(
                    label_synthetic_occluder(
                        visible_mask=layer.visible_mask,
                        amodal_mask=layer.amodal_mask,
                        occluder_ids=layer.occluder_ids,
                        provenance=layer.provenance,
                    )
                )
                output_dataset["annotations"].append(occluder_annotation)
                next_annotation_id += 1
                track_frame_counts[layer.track_id] += 1

        for sched in scheduled:
            # The occluder is pasted for its exposure window but may leave the image
            # frame away from the peak (it enters/exits along its own trajectory), so
            # rendered_frames <= exposure_length is expected. It must be visible at
            # least during the rho>=0.1 event.
            rendered_frames = track_frame_counts[sched.synthetic_track_id]
            if rendered_frames == 0:
                continue
            occluder_tracks.append(
                {
                    "track_id": sched.synthetic_track_id,
                    "video_id": output_video_id,
                    "category": sched.occluder.category,
                    "synthetic": True,
                    "source": {
                        "dataset": sched.occluder.source,
                        "identity": list(sched.occluder.identity_key),
                        "tracklet_id": sched.occluder.tracklet_id,
                    },
                    "start_position": sched.start_position,
                    "exposure_length": sched.exposure_length,
                    "rendered_frames": rendered_frames,
                    "victim_track_id": sched.victim_track_id,
                    "target_peak": sched.target_peak,
                    "achieved_peak": sched.achieved_peak,
                    "peak_frame": sched.peak_frame,
                    "translation_xy": list(sched.translation),
                }
            )
        summary = dict(plan["summary"])
        summary.update({"video": output_name, "scheduled": len(scheduled), "rejected": rejected,
                         "peak_occupancy": max(occupancy) if occupancy else 0})
        scheduling_summaries.append(summary)
        generated += 1

    if generated == 0:
        raise RuntimeError("no synthetic sequence generated")

    events: list[dict[str, Any]] = []
    next_event_id = 1
    for video_id in range(1, generated + 1):
        video_histories = [record for record in histories if int(record["video_id"]) == video_id]
        video_events = derive_events(video_histories, video_id, entry_speed=0.0, first_event_id=next_event_id)
        events.extend(event.to_dict() for event in video_events)
        next_event_id += len(video_events)

    validate_video_dataset(output_dataset)
    qc_result = verify_synthetic_video_dataset(output_dataset, events)
    save_json(output_dir / "annotations.json", output_dataset)
    save_json(output_dir / "occluder_tracks.json", occluder_tracks)
    save_json(output_dir / "events.json", events)
    save_jsonl(output_dir / "scheduling.jsonl", scheduling_summaries)
    summary = {
        "sequences": generated,
        "frames": len(output_dataset["images"]),
        "annotations": len(output_dataset["annotations"]),
        "synthetic_tracklets": len(occluder_tracks),
        "events": len(events),
        "detector_bbox_policy": detector_bbox_policy,
        "qc_ok": bool(qc_result.get("ok", False)),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "qc.json", qc_result)
    if not qc_result.get("ok", False):
        raise RuntimeError(f"synthetic dataset QC failed: {qc_result}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-based multi-class tracklet synthesis on KITTI")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()
    print(run(args.config, args.max_sequences))


if __name__ == "__main__":
    main()
