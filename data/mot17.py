from __future__ import annotations

import configparser
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Mot17Object:
    frame_index: int
    track_id: int
    bbox: tuple[float, float, float, float]
    mark: int
    category_id: int
    visibility: float


@dataclass(frozen=True)
class Mot17Tracklet:
    sequence: str
    track_id: int
    objects: tuple[Mot17Object, ...]
    image_width: int
    image_height: int
    image_extension: str
    # Populated by the FPS-aware builder; None for legacy strict windows.
    source_fps: float | None = None
    target_fps: float | None = None

    @property
    def start_frame(self) -> int:
        return self.objects[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.objects[-1].frame_index

    @property
    def length(self) -> int:
        return len(self.objects)


def parse_gt(path: str | Path) -> list[Mot17Object]:
    """Parse MOTChallenge GT rows (frame,id,x,y,w,h,mark,class,visibility)."""
    records: list[Mot17Object] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle), start=1):
            if not fields:
                continue
            if len(fields) < 9:
                raise ValueError(f"{path}:{line_number}: expected at least 9 MOT fields")
            try:
                records.append(
                    Mot17Object(
                        frame_index=int(float(fields[0])),
                        track_id=int(float(fields[1])),
                        bbox=tuple(float(value) for value in fields[2:6]),
                        mark=int(float(fields[6])),
                        category_id=int(float(fields[7])),
                        visibility=float(fields[8]),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid MOT row") from exc
    return records


def _sequence_info(sequence_dir: Path) -> tuple[int, int, str]:
    parser = configparser.ConfigParser()
    if not parser.read(sequence_dir / "seqinfo.ini"):
        raise FileNotFoundError(f"MOT17 seqinfo.ini not found: {sequence_dir}")
    section = parser["Sequence"]
    return int(section["imWidth"]), int(section["imHeight"]), section.get("imExt", ".jpg")


def sequence_frame_rate(sequence_dir: str | Path, default: float = 30.0) -> float:
    """Read frameRate from MOT17 seqinfo.ini (defaults to 30 fps if absent)."""
    parser = configparser.ConfigParser()
    if not parser.read(Path(sequence_dir) / "seqinfo.ini"):
        raise FileNotFoundError(f"MOT17 seqinfo.ini not found: {sequence_dir}")
    section = parser["Sequence"]
    rate = float(section.get("frameRate", default))
    if rate <= 0.0:
        raise ValueError(f"invalid MOT17 frameRate {rate} in {sequence_dir}")
    return rate


def _consecutive_runs(objects: list[Mot17Object]) -> list[list[Mot17Object]]:
    """Split frame-ordered objects into maximal runs of consecutive frames."""
    runs: list[list[Mot17Object]] = []
    current: list[Mot17Object] = []
    for item in objects:
        if current and item.frame_index != current[-1].frame_index + 1:
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)
    return runs


def resample_run(
    run: list[Mot17Object],
    source_fps: float,
    target_fps: float,
    *,
    min_frames: int,
    visibility_min: float,
    substitution_window: int = 1,
    max_frames: int | None = None,
) -> list[Mot17Object] | None:
    """Resample one continuous identity run to ``target_fps``.

    The run must be a list of objects on consecutive source frames (identity
    never disappears). Target sampling positions are laid over the run at
    ``source_fps / target_fps`` spacing. Visibility is checked only at the
    sampled positions; a sample whose frame has ``visibility < visibility_min``
    is replaced by the nearest frame within ``substitution_window`` that meets
    it. The longest strictly increasing contiguous span is returned, or ``None``
    if it is shorter than ``min_frames``.
    """
    import math

    if not run:
        return None
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps values must be positive")
    if substitution_window < 0:
        raise ValueError("substitution_window must be non-negative")

    step = source_fps / target_fps
    present = {obj.frame_index: obj for obj in run}
    first = run[0].frame_index
    last = run[-1].frame_index
    total_targets = int(math.floor((last - first) / step + 1e-9)) + 1

    resolved: list[int | None] = []
    for k in range(total_targets):
        ideal = first + k * step
        center = int(math.floor(ideal + 0.5))  # round half up (avoid banker's rounding)
        best: int | None = None
        best_distance = 0.0
        for frame in range(center - substitution_window, center + substitution_window + 1):
            obj = present.get(frame)
            if obj is None or obj.visibility < visibility_min:
                continue
            distance = abs(frame - ideal)
            if best is None or distance < best_distance or (distance == best_distance and frame < best):
                best = frame
                best_distance = distance
        resolved.append(best)

    best_span: list[int] = []
    current: list[int] = []
    previous: int | None = None
    for frame in resolved:
        if frame is None:
            if len(current) > len(best_span):
                best_span = current
            current = []
            previous = None
            continue
        if previous is not None and frame <= previous:
            # Duplicate or non-increasing selection: cut and restart here.
            if len(current) > len(best_span):
                best_span = current
            current = []
        current.append(frame)
        previous = frame
    if len(current) > len(best_span):
        best_span = current

    if len(best_span) < min_frames:
        return None
    if max_frames is not None and len(best_span) > max_frames:
        best_span = best_span[:max_frames]
    return [present[frame] for frame in best_span]


def variable_fps_tracklets(
    sequence_dir: str | Path,
    *,
    target_fps: float = 10.0,
    min_frames: int = 30,
    visibility_min: float = 0.8,
    substitution_window: int = 1,
    max_frames: int | None = None,
    pedestrian_class: int = 1,
) -> list[Mot17Tracklet]:
    """Build FPS-aware, variable-length tracklets (>= ``min_frames`` target frames).

    Presence (mark==1, pedestrian, non-degenerate bbox) forms the continuous
    runs; visibility is enforced only on the resampled frames, so a low-visibility
    frame between two samples no longer splits the identity's run.
    """
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")
    if not 0.0 <= visibility_min <= 1.0:
        raise ValueError("visibility_min must be in [0, 1]")

    directory = Path(sequence_dir).expanduser().resolve()
    width, height, extension = _sequence_info(directory)
    source_fps = sequence_frame_rate(directory)

    present = [
        item
        for item in parse_gt(directory / "gt" / "gt.txt")
        if item.mark == 1
        and item.category_id == pedestrian_class
        and item.bbox[2] > 1.0
        and item.bbox[3] > 1.0
    ]
    by_identity: dict[int, list[Mot17Object]] = {}
    for item in present:
        by_identity.setdefault(item.track_id, []).append(item)

    output: list[Mot17Tracklet] = []
    for track_id, objects in sorted(by_identity.items()):
        ordered = sorted(objects, key=lambda item: item.frame_index)
        for run in _consecutive_runs(ordered):
            sampled = resample_run(
                run,
                source_fps,
                target_fps,
                min_frames=min_frames,
                visibility_min=visibility_min,
                substitution_window=substitution_window,
                max_frames=max_frames,
            )
            if sampled is None:
                continue
            output.append(
                Mot17Tracklet(
                    sequence=directory.name,
                    track_id=track_id,
                    objects=tuple(sampled),
                    image_width=width,
                    image_height=height,
                    image_extension=extension,
                    source_fps=source_fps,
                    target_fps=float(target_fps),
                )
            )
    return output


def collect_variable_fps_tracklets(
    root: str | Path,
    *,
    detector: str = "FRCNN",
    target_fps: float = 10.0,
    min_frames: int = 30,
    visibility_min: float = 0.8,
    substitution_window: int = 1,
    max_frames: int | None = None,
) -> list[Mot17Tracklet]:
    result: list[Mot17Tracklet] = []
    for sequence in discover_train_sequences(root, detector):
        result.extend(
            variable_fps_tracklets(
                sequence,
                target_fps=target_fps,
                min_frames=min_frames,
                visibility_min=visibility_min,
                substitution_window=substitution_window,
                max_frames=max_frames,
            )
        )
    return result


def discover_train_sequences(root: str | Path, detector: str = "FRCNN") -> list[Path]:
    train_root = Path(root).expanduser().resolve() / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"MOT17 train directory not found: {train_root}")
    suffix = f"-{detector.upper()}"
    sequences = sorted(path for path in train_root.iterdir() if path.is_dir() and path.name.endswith(suffix))
    if not sequences:
        raise FileNotFoundError(f"no MOT17 {detector} training sequences found under {train_root}")
    return sequences


def strict_tracklets(
    sequence_dir: str | Path,
    *,
    length: int = 30,
    visibility_min: float = 0.8,
    stride: int | None = None,
    pedestrian_class: int = 1,
) -> list[Mot17Tracklet]:
    """Return fixed-length windows with no missing or low-visibility frame.

    Filtering happens before consecutive runs are formed, so one low-visibility
    row splits the identity's run. With the default stride equal to ``length``,
    windows do not overlap.
    """
    if length <= 0:
        raise ValueError("tracklet length must be positive")
    if not 0.0 <= visibility_min <= 1.0:
        raise ValueError("visibility_min must be in [0, 1]")
    window_stride = length if stride is None else stride
    if window_stride <= 0:
        raise ValueError("tracklet stride must be positive")

    directory = Path(sequence_dir).expanduser().resolve()
    width, height, extension = _sequence_info(directory)
    eligible = [
        item
        for item in parse_gt(directory / "gt" / "gt.txt")
        if item.mark == 1
        and item.category_id == pedestrian_class
        and item.visibility >= visibility_min
        and item.bbox[2] > 1.0
        and item.bbox[3] > 1.0
    ]
    by_identity: dict[int, list[Mot17Object]] = {}
    for item in eligible:
        by_identity.setdefault(item.track_id, []).append(item)

    output: list[Mot17Tracklet] = []
    for track_id, objects in sorted(by_identity.items()):
        ordered = sorted(objects, key=lambda item: item.frame_index)
        runs: list[list[Mot17Object]] = []
        current: list[Mot17Object] = []
        for item in ordered:
            if current and item.frame_index != current[-1].frame_index + 1:
                runs.append(current)
                current = []
            current.append(item)
        if current:
            runs.append(current)

        for run in runs:
            for start in range(0, len(run) - length + 1, window_stride):
                window = tuple(run[start : start + length])
                if len(window) != length:
                    continue
                output.append(
                    Mot17Tracklet(
                        sequence=directory.name,
                        track_id=track_id,
                        objects=window,
                        image_width=width,
                        image_height=height,
                        image_extension=extension,
                    )
                )
    return output


def collect_strict_tracklets(
    root: str | Path,
    *,
    detector: str = "FRCNN",
    length: int = 30,
    visibility_min: float = 0.8,
    stride: int | None = None,
) -> list[Mot17Tracklet]:
    result: list[Mot17Tracklet] = []
    for sequence in discover_train_sequences(root, detector):
        result.extend(
            strict_tracklets(
                sequence,
                length=length,
                visibility_min=visibility_min,
                stride=stride,
            )
        )
    return result


def image_path(root: str | Path, tracklet: Mot17Tracklet, frame_index: int) -> Path:
    return (
        Path(root).expanduser().resolve()
        / "train"
        / tracklet.sequence
        / "img1"
        / f"{frame_index:06d}{tracklet.image_extension}"
    )


def identities(tracklets: Iterable[Mot17Tracklet]) -> set[tuple[str, int]]:
    return {(tracklet.sequence, tracklet.track_id) for tracklet in tracklets}
