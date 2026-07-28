"""Event-based, variable-length occlusion synthesis pipeline (Step 7).

Wires the 2026-07-27 policy end to end:
- merge the MOT17 + KITTI multi-class tracklet pools;
- per KITTI train sequence, plan victim events by density and class ratio
  (``synth.scheduler``);
- for each event, draw physical parameters (occluder height from its class's real
  size range, lateral offset from the separation prior) and keep the first draw
  whose *mask-based* rho forms a complete single-peaked event
  (``synth.placement`` + ``synth.geometry``), retrying with alternative tracklets
  only when the drawn tracklet never passes;
- allow at most ``max_occluders_per_victim`` occluders on any one victim; events
  on different victims no longer compete for a global frame budget;
- render large-first and emit extended GT with the ``amodal_original`` detector
  bbox policy, ignore flags on effectively-invisible victims, synthetic occluder
  tracks, occlusion events, and QC.

The legacy fixed-30-frame ``synth.tracklet_pipeline`` is left untouched.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json, save_jsonl
from common.io_video import group_annotations_by_image, group_frames_by_video
from common.schema import validate_video_dataset
from data.kitti_tracking import convert_tracking_to_video_coco
from label.compute import label_frame_multi, label_synthetic_occluder
from label.events import derive_events
from qc.verify import verify_synthetic_video_dataset
from synth.geometry import AlphaIntegralCache, sample_event_placement
from synth.placement import mask_cover_ratio, mid_height_factor, sample_height_factor
from synth.scheduler import (
    EventPlan,
    PoolTracklet,
    claim_identity,
    identity_available,
    merge_pools,
    plan_schedule,
    pool_by_category,
)
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
    achieved_peak: float
    band: str | None
    peak_frame: int | None
    height_factor: float
    lateral_offset_fraction: float


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


def _boxes_by_position(frames_annotations: list[list[dict[str, Any]]]) -> list[list[list[float]]]:
    """Every real annotation's box per output frame position."""
    return [
        [[float(value) for value in annotation["bbox"]] for annotation in annotations]
        for annotations in frames_annotations
    ]


def _boxes_overlap(first: Sequence[float], second: Sequence[float]) -> bool:
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    return min(ax + aw, bx + bw) > max(ax, bx) and min(ay + ah, by + bh) > max(ay, by)


class SceneBudget:
    """Keeps pasted occluders from burying anything, across the whole sequence.

    The event gate only scores an occluder against its own victim, so on its own
    it lets a placement pass while covering an unrelated object it happens to
    drive past — 118 of 174 over-cap frames in a full run, none of them the
    occluder's intended victim — and it cannot see other scheduled events at all.

    Two rules, checked together before a placement is accepted:

    - a candidate must not overlap an already-placed occluder on any frame. Beyond
      keeping pasted objects from covering each other, disjoint occluders have
      disjoint masks, which is what makes the second rule exact;
    - accumulated coverage of every real annotation stays at or below the cap.
      With disjoint occluders the union over an object's box equals the sum of the
      individual contributions, so summing is exact rather than conservative.

    Box overlap (not mask overlap) is the disjointness test: it is O(1) and errs
    toward rejecting, so the masks behind it are certainly disjoint.
    """

    def __init__(self, boxes_by_position: list[list[list[float]]], rho_max: float) -> None:
        self._boxes = boxes_by_position
        self._rho_max = float(rho_max)
        self._committed: list[dict[int, float]] = [{} for _ in boxes_by_position]
        self._placed: list[list[tuple[float, ...]]] = [[] for _ in boxes_by_position]

    def accepts(
        self, start: int, occluder_boxes: Sequence[Any], integrals: Sequence[Any]
    ) -> bool:
        for offset, (box, integral) in enumerate(zip(occluder_boxes, integrals)):
            position = start + offset
            if not 0 <= position < len(self._boxes):
                continue
            for placed in self._placed[position]:
                if _boxes_overlap(box, placed):
                    return False
            committed = self._committed[position]
            for index, other in enumerate(self._boxes[position]):
                total = committed.get(index, 0.0) + mask_cover_ratio(integral, box, other)
                if total > self._rho_max:
                    return False
        return True

    def commit(self, start: int, occluder_boxes: Sequence[Any], integrals: Sequence[Any]) -> None:
        for offset, (box, integral) in enumerate(zip(occluder_boxes, integrals)):
            position = start + offset
            if not 0 <= position < len(self._boxes):
                continue
            self._placed[position].append(tuple(float(value) for value in box))
            committed = self._committed[position]
            for index, other in enumerate(self._boxes[position]):
                value = mask_cover_ratio(integral, box, other)
                if value > 0:
                    committed[index] = committed.get(index, 0.0) + value


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
    integral_cache: AlphaIntegralCache,
    usage: dict[tuple[Any, ...], int],
    used_in_sequence: set[tuple[Any, ...]],
    scene_check: Any = None,
) -> tuple[PoolTracklet, dict[str, Any]] | None:
    """Sample a placement, retrying with alternative same-class tracklets.

    Retries respect both identity-reuse rules; before 2026-07-27 they ignored the
    usage map entirely, which let one identity be pasted three times against a
    cap of two. Retries are also capped: exhausting the whole 47-tracklet car
    pool used to account for 74% of all search work while producing nothing,
    because a placement that fails does so on the class pair's geometry, not on
    which particular crop was drawn.
    """
    ranges = dict(synthesis["class_height_range"])
    effective = tuple(synthesis.get("effective_event_frames", [8, 20]))
    accept_kwargs = {
        "effective_range": (int(effective[0]), int(effective[1])),
        "peak_max": float(synthesis.get("peak_rho_max", 1.00)),
        "end_max": float(synthesis.get("event_end_rho_max", 0.05)),
        "gate_floor": float(synthesis.get("event_gate_floor", 0.20)),
    }
    max_lateral = float(synthesis.get("max_lateral_offset_fraction", 1.15))
    draws = int(synthesis.get("placement_draws", 24))
    max_tracklets = int(synthesis.get("placement_max_tracklets", 8))
    factor_victim = mid_height_factor(ranges, victim_class)

    candidates = [event.occluder] + [
        tracklet
        for tracklet in grouped_pool.get(event.occluder.category, [])
        if tracklet.tracklet_id != event.occluder.tracklet_id
    ]
    tried = 0
    for occluder in candidates:
        if tried >= max_tracklets:
            break
        if occluder is not event.occluder and not identity_available(
            occluder, usage, used_in_sequence, int(synthesis.get("max_per_identity", 2))
        ):
            continue
        tried += 1
        exposure = min(event.exposure_length, occluder.length)
        occluder_frames = occluder.frames[:exposure]
        integrals = integral_cache.for_frames(occluder_frames)
        placement = sample_event_placement(
            occluder_frames,
            victim_positions,
            factor_occluder=sample_height_factor(ranges, occluder.category, rng),
            factor_victim=factor_victim,
            integrals=integrals,
            target_size=target_size,
            sequence_length=sequence_length,
            max_lateral_fraction=max_lateral,
            rng=rng,
            attempts=draws,
            accept_kwargs=accept_kwargs,
            scene_check=scene_check,
        )
        if placement is not None:
            placement["exposure_length"] = exposure
            placement["integrals"] = integrals
            if occluder is not event.occluder:
                claim_identity(occluder, usage, used_in_sequence)
            return occluder, placement
    return None


def _fits_victim_budget(occupancy: list[int], start: int, length: int, ceiling: int) -> bool:
    """At most ``ceiling`` occluders on *this victim* at once.

    Until 2026-07-27 the counter was global to the frame, so an event occluding a
    pedestrian on the left of the road competed with one occluding a car on the
    right; that discarded 44% of successfully placed events while leaving an
    average of only 0.52 occluders on screen.
    """
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
    max_per_victim = int(synthesis.get("max_occluders_per_victim", 2))
    rng = random.Random(int(config.get("seed", 0)))
    integral_cache = AlphaIntegralCache()
    identity_usage: dict[tuple[Any, ...], int] = {}
    output_dir = resolve_path(path.parent, synthesis["output_dir"])
    output_frames_dir = output_dir / "frames"
    max_sequences = int(max_sequences_override or synthesis.get("max_sequences", len(frames_by_video)))

    output_dataset: dict[str, Any] = {
        "info": {
            "description": "Event-based multi-class tracklet copy-paste on KITTI Tracking",
            "detector_bbox_policy": detector_bbox_policy,
            "scale_policy": "class_height_range_sampled",
            "rho_method": "occluder_mask_integral_image",
            "rho_control": "lateral_offset",
            "ignore_rule": "none: every victim the baseline trains on stays a positive",
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
        plan = plan_schedule(frames_annotations, pool, synthesis, rng, usage=identity_usage)
        used_in_sequence = plan["used_identities"]
        scene = SceneBudget(
            _boxes_by_position(frames_annotations), float(synthesis.get("peak_rho_max", 0.90))
        )

        output_video_id = generated + 1
        occupancy = [0] * sequence_length
        victim_occupancy: dict[int, list[int]] = {}
        scheduled: list[ScheduledRender] = []
        next_synth_track = 900000 + output_video_id * 1000
        rejected = 0
        for event in plan["events"]:
            if event.victim_track_id is None:
                rejected += 1
                continue
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
                target_size, sequence_length, rng, integral_cache,
                identity_usage, used_in_sequence, scene.accepts,
            )
            if resolved is None:
                rejected += 1
                continue
            occluder, best = resolved
            budget = victim_occupancy.setdefault(int(event.victim_track_id), [0] * sequence_length)
            if not _fits_victim_budget(
                budget, best["start_position"], best["exposure_length"], max_per_victim
            ):
                rejected += 1
                continue
            # Only now does this placement become part of the scene the next
            # candidate has to fit around.
            scene.commit(best["start_position"], best["occluder_boxes"], best["integrals"])
            for position in range(best["start_position"], best["start_position"] + best["exposure_length"]):
                budget[position] += 1
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
                    achieved_peak=float(best["achieved_peak"]),
                    band=best["band"],
                    peak_frame=int(best["peak_frame"]),
                    height_factor=float(best["height_factor"]),
                    lateral_offset_fraction=float(best["lateral_offset_fraction"]),
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
                            "achieved_peak": sched.achieved_peak,
                            "band": sched.band,
                            "height_factor": sched.height_factor,
                            "lateral_offset_fraction": sched.lateral_offset_fraction,
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
                    "achieved_peak": sched.achieved_peak,
                    "band": sched.band,
                    "peak_frame": sched.peak_frame,
                    "height_factor": sched.height_factor,
                    "lateral_offset_fraction": sched.lateral_offset_fraction,
                    "translation_xy": list(sched.translation),
                }
            )
        summary = dict(plan["summary"])
        summary.update({"video": output_name, "scheduled": len(scheduled), "rejected": rejected,
                         "peak_occupancy": max(occupancy) if occupancy else 0,
                         "mean_occupancy": (sum(occupancy) / len(occupancy)) if occupancy else 0.0})
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
