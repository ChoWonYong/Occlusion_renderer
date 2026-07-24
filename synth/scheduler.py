"""Category / FPS-aware occlusion-event scheduler (Step 4).

This module decides *what* to paste and *how much*, decoupled from the heavy
rendering loop. It merges the multi-class tracklet pools, sizes the number of
victim events by sequence length, assigns occluder tracklets by the confirmed
class ratio with an identity-reuse cap, samples the paste-exposure length, and
flags the ~15% of events allowed to run concurrently.

The exact temporal placement and the translation/scale that realise a target
peak rho are resolved later, in the Step-5 placement search. Here we only
produce the assignments and the concurrency budget.

Confirmed policy (2026-07-23):
- victim events per 100 frames = 3, capped by the number of distinct clean
  victim tracks, one event per victim (no victim reuse);
- occluder class ratio car/person/bicycle = 60/30/10 (truck excluded);
- occluder identity reuse capped by ``max_per_identity``;
- paste exposure L ~ uniform[tracklet_min, tracklet_length];
- max 2 concurrent occluders, ~15% of events allowed concurrent;
- boundary-shift events are auxiliary and do not consume the victim quota.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PoolTracklet:
    tracklet_id: int          # unique id across the merged pool
    category: str             # car / person / bicycle
    source: str               # MOT17 / KITTI
    length: int
    frames: list[dict[str, Any]]
    identity_key: tuple[Any, ...]  # (source, sequence, source_track_id) for reuse capping
    record: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class VictimTrack:
    track_id: int
    category_id: int
    presence: tuple[int, ...]  # frame positions where the victim is clean and large enough

    @property
    def presence_frames(self) -> int:
        return len(self.presence)


@dataclass
class EventPlan:
    kind: str                       # "victim" | "boundary"
    occluder: PoolTracklet
    exposure_length: int
    victim_track_id: int | None = None
    victim_category_id: int | None = None
    boundary_side: str | None = None
    allow_concurrent: bool = False


def merge_pools(metadatas: Sequence[Mapping[str, Any]]) -> list[PoolTracklet]:
    """Merge one or more ``tracklets.json`` payloads into a single pool.

    Each metadata is the dict loaded from a pool builder's ``tracklets.json``
    (MOT17 ``build_variable`` or KITTI ``build_tracklets``). Category comes from
    each record; the identity key combines source/sequence/track so the reuse
    cap treats the same real object as one identity.
    """
    pool: list[PoolTracklet] = []
    next_id = 1
    for metadata in metadatas:
        for record in metadata.get("tracklets", []):
            frames = list(record.get("frames", []))
            length = int(record.get("length", len(frames)))
            if length != len(frames):
                raise ValueError(f"tracklet {record.get('id')} length {length} != {len(frames)} frames")
            if length < 1:
                raise ValueError(f"tracklet {record.get('id')} has no frames")
            source = str(record.get("source", "unknown"))
            identity_key = (
                source,
                record.get("sequence"),
                record.get("source_track_id"),
            )
            pool.append(
                PoolTracklet(
                    tracklet_id=next_id,
                    category=str(record["category"]),
                    source=source,
                    length=length,
                    frames=frames,
                    identity_key=identity_key,
                    record=dict(record),
                )
            )
            next_id += 1
    return pool


def pool_by_category(pool: Sequence[PoolTracklet]) -> dict[str, list[PoolTracklet]]:
    grouped: dict[str, list[PoolTracklet]] = {}
    for tracklet in pool:
        grouped.setdefault(tracklet.category, []).append(tracklet)
    return grouped


def victim_event_count(frame_count: int, per_100_frames: float) -> int:
    """Number of victim events proportional to sequence length (min 1)."""
    if frame_count <= 0:
        return 0
    return max(1, int(round(frame_count * float(per_100_frames) / 100.0)))


def select_eligible_victims(
    frames_annotations: Sequence[Sequence[Mapping[str, Any]]],
    *,
    min_area: float,
    max_base_occlusion: int,
    min_presence_frames: int,
) -> list[VictimTrack]:
    """Distinct KITTI victim tracks that are clean and large enough often enough.

    ``frames_annotations[position]`` is the list of background annotations on that
    output frame position. A track is eligible if it appears clean
    (``kitti.occluded <= max_base_occlusion``) with ``area >= min_area`` on at
    least ``min_presence_frames`` positions.
    """
    presence: dict[int, list[int]] = {}
    category_of: dict[int, int] = {}
    for position, annotations in enumerate(frames_annotations):
        for annotation in annotations:
            if float(annotation.get("area", 0.0)) < min_area:
                continue
            if int(annotation.get("kitti", {}).get("occluded", 3)) > max_base_occlusion:
                continue
            track_id = int(annotation["track_id"])
            presence.setdefault(track_id, []).append(position)
            category_of.setdefault(track_id, int(annotation["category_id"]))
    victims: list[VictimTrack] = []
    for track_id, positions in presence.items():
        if len(positions) >= min_presence_frames:
            victims.append(
                VictimTrack(
                    track_id=track_id,
                    category_id=category_of[track_id],
                    presence=tuple(sorted(positions)),
                )
            )
    victims.sort(key=lambda victim: victim.track_id)
    return victims


def sample_categories(count: int, class_ratio: Mapping[str, float], rng: random.Random) -> list[str]:
    """Apportion ``count`` occluders to categories by ratio (largest remainder)."""
    if count <= 0:
        return []
    categories = list(class_ratio)
    weights = [max(0.0, float(class_ratio[category])) for category in categories]
    total = sum(weights)
    if total <= 0:
        raise ValueError("class_ratio must have a positive total weight")
    quotas = [count * weight / total for weight in weights]
    base = [int(math.floor(quota)) for quota in quotas]
    remainder = count - sum(base)
    order = sorted(range(len(categories)), key=lambda i: quotas[i] - base[i], reverse=True)
    for step in range(remainder):
        base[order[step % len(categories)]] += 1
    result: list[str] = []
    for category, number in zip(categories, base):
        result.extend([category] * number)
    rng.shuffle(result)
    return result


def sample_occluders(
    categories: Sequence[str],
    grouped_pool: Mapping[str, list[PoolTracklet]],
    rng: random.Random,
    *,
    max_per_identity: int,
    usage: dict[tuple[Any, ...], int] | None = None,
) -> tuple[list[PoolTracklet], int]:
    """Pick one tracklet per requested category, respecting the reuse cap.

    ``usage`` (identity_key -> count) can be shared across calls so that victim
    and boundary events draw from a common reuse budget. Returns the chosen
    tracklets and the number of picks that had to exceed the cap because the
    category pool was saturated.
    """
    usage = {} if usage is None else usage
    shuffled = {category: rng.sample(items, len(items)) for category, items in grouped_pool.items()}
    cursor = {category: 0 for category in shuffled}
    chosen: list[PoolTracklet] = []
    over_cap = 0
    for category in categories:
        pool = shuffled.get(category, [])
        if not pool:
            raise ValueError(f"no occluder tracklets available for category {category!r}")
        size = len(pool)
        pick: PoolTracklet | None = None
        for step in range(size):
            index = (cursor[category] + step) % size
            candidate = pool[index]
            if usage.get(candidate.identity_key, 0) < max_per_identity:
                pick = candidate
                cursor[category] = (index + 1) % size
                break
        if pick is None:
            pick = min(pool, key=lambda tracklet: usage.get(tracklet.identity_key, 0))
            over_cap += 1
        usage[pick.identity_key] = usage.get(pick.identity_key, 0) + 1
        chosen.append(pick)
    return chosen, over_cap


def paste_exposure_length(length: int, rng: random.Random, *, min_frames: int = 30) -> int:
    """Uniform paste-exposure length in ``[min_frames, length]`` (inclusive)."""
    if length < min_frames:
        raise ValueError(f"tracklet length {length} is shorter than min_frames {min_frames}")
    return rng.randint(min_frames, length)


def plan_schedule(
    frames_annotations: Sequence[Sequence[Mapping[str, Any]]],
    pool: Sequence[PoolTracklet],
    synthesis: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Produce the occlusion-event plan for one background sequence.

    Returns a dict with the ``events`` (list[EventPlan]) and a ``summary`` of the
    scheduling decisions for QC.
    """
    frame_count = len(frames_annotations)
    class_ratio = dict(synthesis["class_ratio"])
    min_frames = int(synthesis.get("tracklet_min_frames", 30))
    max_per_identity = int(synthesis.get("max_per_identity", 2))
    effective_low = int(synthesis.get("effective_event_frames", [8, 20])[0])
    min_presence = int(synthesis.get("min_victim_presence", effective_low))

    grouped = pool_by_category(pool)
    victims = select_eligible_victims(
        frames_annotations,
        min_area=float(synthesis.get("victim_min_area", 400.0)),
        max_base_occlusion=int(synthesis.get("victim_max_base_occlusion", 0)),
        min_presence_frames=min_presence,
    )

    target = victim_event_count(frame_count, float(synthesis.get("victim_events_per_100_frames", 3.0)))
    cap_to_victims = bool(synthesis.get("cap_events_to_distinct_victims", True))
    event_count = min(target, len(victims)) if cap_to_victims else target
    chosen_victims = rng.sample(victims, event_count) if event_count <= len(victims) else list(victims)

    usage: dict[tuple[Any, ...], int] = {}
    categories = sample_categories(event_count, class_ratio, rng)
    occluders, over_cap = sample_occluders(
        categories, grouped, rng, max_per_identity=max_per_identity, usage=usage
    )

    events: list[EventPlan] = []
    for victim, occluder in zip(chosen_victims, occluders):
        events.append(
            EventPlan(
                kind="victim",
                occluder=occluder,
                exposure_length=paste_exposure_length(occluder.length, rng, min_frames=min_frames),
                victim_track_id=victim.track_id,
                victim_category_id=victim.category_id,
            )
        )

    # ~double_occluder_probability of the victim events may run concurrently.
    double_probability = float(synthesis.get("double_occluder_probability", 0.15))
    concurrent_count = int(math.floor(len(events) * double_probability))
    for event in rng.sample(events, concurrent_count) if concurrent_count else []:
        event.allow_concurrent = True

    # Boundary-shift events are auxiliary: they do not consume the victim quota.
    boundary_events: list[EventPlan] = []
    if bool(synthesis.get("boundary_events_are_auxiliary", True)):
        boundary_max = int(synthesis.get("boundary_shift_count", 2))
        boundary_sides = ["left", "right"][: max(0, min(2, boundary_max))]
        if boundary_sides:
            boundary_categories = sample_categories(len(boundary_sides), class_ratio, rng)
            boundary_occluders, boundary_over = sample_occluders(
                boundary_categories, grouped, rng, max_per_identity=max_per_identity, usage=usage
            )
            over_cap += boundary_over
            for side, occluder in zip(boundary_sides, boundary_occluders):
                boundary_events.append(
                    EventPlan(
                        kind="boundary",
                        occluder=occluder,
                        exposure_length=paste_exposure_length(occluder.length, rng, min_frames=min_frames),
                        boundary_side=side,
                        allow_concurrent=True,
                    )
                )

    all_events = events + boundary_events
    actual_class_counts: dict[str, int] = {}
    for event in all_events:
        actual_class_counts[event.occluder.category] = actual_class_counts.get(event.occluder.category, 0) + 1

    summary = {
        "frame_count": frame_count,
        "target_victim_events": target,
        "distinct_eligible_victims": len(victims),
        "victim_events": len(events),
        "boundary_events": len(boundary_events),
        "concurrent_flagged": concurrent_count,
        "max_concurrent_occluders": int(synthesis.get("max_concurrent_occluders", 2)),
        "class_distribution": dict(sorted(actual_class_counts.items())),
        "identity_reuse_over_cap": over_cap,
        "distinct_occluder_identities": len({event.occluder.identity_key for event in all_events}),
    }
    return {"events": all_events, "summary": summary}
