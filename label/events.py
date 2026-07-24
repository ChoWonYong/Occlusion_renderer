from __future__ import annotations

from typing import Any, Iterable, Mapping

from common.schema import OcclusionEvent


def derive_events(
    histories: Iterable[Mapping[str, Any]],
    video_id: int,
    entry_speed: float,
    first_event_id: int = 1,
) -> list[OcclusionEvent]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in histories:
        if float(row["occlusion_ratio"]) <= 0.0:
            continue
        key = (int(row["victim_track"]), int(row["occluder_track"]))
        grouped.setdefault(key, []).append(row)
    events: list[OcclusionEvent] = []
    event_id = first_event_id
    for (victim_track, occluder_track), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["frame_index"]))
        runs: list[list[Mapping[str, Any]]] = []
        for row in rows:
            if not runs or int(row["frame_index"]) > int(runs[-1][-1]["frame_index"]) + 1:
                runs.append([row])
            else:
                runs[-1].append(row)
        for run in runs:
            peak = max(run, key=lambda row: float(row["occlusion_ratio"]))
            peak_ratio = float(peak["occlusion_ratio"])
            events.append(
                OcclusionEvent(
                    event_id=event_id,
                    video_id=int(video_id),
                    victim_track=victim_track,
                    occluder_track=occluder_track,
                    frame_start=int(run[0]["frame_index"]),
                    frame_peak=int(peak["frame_index"]),
                    frame_end=int(run[-1]["frame_index"]),
                    peak_ratio=peak_ratio,
                    duration=int(run[-1]["frame_index"]) - int(run[0]["frame_index"]) + 1,
                    entry_speed=float(entry_speed),
                    fully_occluded=peak_ratio >= 0.95,
                )
            )
            event_id += 1
    return events

