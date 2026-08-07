"""Category / FPS-aware occlusion-event scheduler (Step 4).

This module decides *what* to paste and *how much*, decoupled from the heavy
rendering loop. It merges the multi-class tracklet pools, sizes the number of
victim events by sequence length, assigns occluder tracklets by the configured
class ratio under the identity-reuse rules, and samples the paste-exposure
length.

Where the event actually lands and how strongly it occludes are resolved later,
in the generative placement sampling (``synth.placement`` + ``synth.geometry``).
Here we only produce the assignments.

Confirmed policy (2026-07-23, revised 2026-07-27):
- victim events per 100 frames = 3, capped by the number of distinct clean
  victim tracks, one event per victim (no victim reuse);
- occluder class ratio car/person = 65/35 (truck and bicycle excluded);
- occluder identity reuse: at most ``max_per_identity`` across the whole run and
  never twice within one background sequence. The caller passes a persistent
  ``usage`` map so the cap survives across sequences and across the placement
  retries in ``synth.event_pipeline`` — those retries previously bypassed it,
  which let a single identity appear three times;
- paste exposure L ~ uniform[tracklet_min, tracklet_length];
- concurrency is enforced per victim by the pipeline, not as a global
  frame budget (2026-07-27: the global counter discarded 44% of successfully
  placed events and left an average of 0.52 occluders on screen);
- boundary-shift (truncation) events were removed on 2026-07-27: they were
  planned here but never rendered, and they are a different augmentation axis
  that would confound the occlusion A/B.
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
    kind: str                       # "victim" (boundary events were removed 2026-07-27)
    occluder: PoolTracklet
    exposure_length: int
    victim_track_id: int | None = None
    victim_category_id: int | None = None


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


def identity_available(
    tracklet: PoolTracklet,
    usage: Mapping[tuple[Any, ...], int],
    used_in_sequence: Sequence[tuple[Any, ...]] | set[tuple[Any, ...]],
    max_per_identity: int,
) -> bool:
    """Both reuse rules: global cap, and never twice in one background sequence."""
    if tracklet.identity_key in used_in_sequence:
        return False
    return usage.get(tracklet.identity_key, 0) < max_per_identity


def claim_identity(
    tracklet: PoolTracklet,
    usage: dict[tuple[Any, ...], int],
    used_in_sequence: set[tuple[Any, ...]],
) -> None:
    """Record a tracklet as used so later picks and retries see it."""
    usage[tracklet.identity_key] = usage.get(tracklet.identity_key, 0) + 1
    used_in_sequence.add(tracklet.identity_key)


def sample_occluders(
    categories: Sequence[str],
    grouped_pool: Mapping[str, list[PoolTracklet]],
    rng: random.Random,
    *,
    max_per_identity: int,
    usage: dict[tuple[Any, ...], int] | None = None,
    used_in_sequence: set[tuple[Any, ...]] | None = None,
) -> tuple[list[PoolTracklet], int]:
    """Pick one tracklet per requested category, respecting both reuse rules.

    ``usage`` (identity_key -> count) is shared across sequences so the global cap
    holds for the whole run; ``used_in_sequence`` is reset per background sequence
    so the same real object never appears twice in one clip. Returns the chosen
    tracklets and the number of picks that had to break a rule because the
    category pool was saturated.
    """
    usage = {} if usage is None else usage
    used_in_sequence = set() if used_in_sequence is None else used_in_sequence
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
            if identity_available(candidate, usage, used_in_sequence, max_per_identity):
                pick = candidate
                cursor[category] = (index + 1) % size
                break
        if pick is None:
            pick = min(pool, key=lambda tracklet: usage.get(tracklet.identity_key, 0))
            over_cap += 1
        claim_identity(pick, usage, used_in_sequence)
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
    usage: dict[tuple[Any, ...], int] | None = None,
) -> dict[str, Any]:
    """Produce the occlusion-event plan for one background sequence.

    Pass the same ``usage`` map on every call so the global identity cap applies
    across the whole run; the per-sequence exclusion set is created here and
    returned so the pipeline's placement retries obey the same rules.

    Returns a dict with the ``events`` (list[EventPlan]), the per-sequence
    ``used_identities`` set, and a ``summary`` of the scheduling decisions.
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

    usage = {} if usage is None else usage
    used_in_sequence: set[tuple[Any, ...]] = set()
    categories = sample_categories(event_count, class_ratio, rng)
    occluders, over_cap = sample_occluders(
        categories,
        grouped,
        rng,
        max_per_identity=max_per_identity,
        usage=usage,
        used_in_sequence=used_in_sequence,
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

    actual_class_counts: dict[str, int] = {}
    for event in events:
        actual_class_counts[event.occluder.category] = actual_class_counts.get(event.occluder.category, 0) + 1

    summary = {
        "frame_count": frame_count,
        "target_victim_events": target,
        "distinct_eligible_victims": len(victims),
        "victim_events": len(events),
        "max_occluders_per_victim": int(synthesis.get("max_occluders_per_victim", 2)),
        "class_distribution": dict(sorted(actual_class_counts.items())),
        "identity_reuse_over_cap": over_cap,
        "distinct_occluder_identities": len({event.occluder.identity_key for event in events}),
    }
    return {"events": events, "used_identities": used_in_sequence, "summary": summary}
