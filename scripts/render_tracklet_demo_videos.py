"""Render one demo video per synthetic occluder tracklet.

Each video covers only the frames where that pasted tracklet is on screen and
colour-codes the boxes so the composited object is obvious: magenta = pasted
tracklet, amber = the real object it occludes (with its rho), green = untouched
KITTI ground truth.

    kds-render-tracklet-demos --config configs/phase1_kitti.yaml --count 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from common.config import load_config, resolve_path
from common.io import load_json, save_json

REPO_ROOT = Path(__file__).resolve().parents[1]

# BGR. The pasted object gets the one colour nothing in a driving scene has.
COLOR_SYNTH = (255, 0, 255)
COLOR_SYNTH_OTHER = (200, 110, 200)
COLOR_VICTIM = (0, 190, 255)
COLOR_REAL = (100, 220, 100)
COLOR_HEADER = (28, 28, 32)
COLOR_TEXT = (240, 240, 240)
COLOR_MUTED = (165, 165, 172)
COLOR_PLOT = (110, 110, 120)

HEADER_HEIGHT = 108
MAX_VICTIM_TAGS = 3
BAND_ORDER = ("heavy", "moderate", "mild")
# Skewed to heavy: those are the events the method is actually about.
BAND_MIX = {"heavy": 0.4, "moderate": 0.3, "mild": 0.3}

_FONT_CANDIDATES = {
    False: (
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
    True: (
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
}

# Korean labels need a CJK font; _Text falls back to these when none is found.
LABELS_KO = {
    "legend_synth": "합성 occluder (paste)",
    "legend_synth_other": "같은 프레임의 다른 합성체",
    "legend_victim": "가려진 victim (원본 GT)",
    "legend_real": "원본 KITTI GT",
    "background": "배경",
    "exposure": "노출",
    "frames_unit": "프레임",
    "victim": "victim",
    "covered": "이 프레임에서 가린 객체",
    "occluder": "합성 tracklet",
}
LABELS_EN = {
    "legend_synth": "synthetic occluder (paste)",
    "legend_synth_other": "other paste in the same frame",
    "legend_victim": "occluded victim (real GT)",
    "legend_real": "real KITTI GT",
    "background": "background",
    "exposure": "exposure",
    "frames_unit": "frames",
    "victim": "victim",
    "covered": "objects covered in this frame",
    "occluder": "synthetic tracklet",
}


class _Text:
    """PIL text drawing on a BGR canvas, one image round-trip per frame."""

    def __init__(self) -> None:
        from PIL import ImageFont

        self._image_font = ImageFont
        self._cache: dict[tuple[int, bool], Any] = {}
        self.unicode_ok = any(Path(p).exists() for p in _FONT_CANDIDATES[False][:3])

    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._cache:
            font = None
            for candidate in _FONT_CANDIDATES[bold]:
                if Path(candidate).exists():
                    try:
                        font = self._image_font.truetype(candidate, size)
                        break
                    except OSError:
                        continue
            self._cache[key] = font or self._image_font.load_default()
        return self._cache[key]

    def measure(self, text: str, size: int, bold: bool = False) -> tuple[int, int]:
        left, top, right, bottom = self.font(size, bold).getbbox(text)
        return int(right - left), int(bottom - top)

    def draw(self, canvas: np.ndarray, items: Sequence[tuple[int, int, str, int, tuple[int, int, int], bool]]) -> np.ndarray:
        if not items:
            return canvas
        from PIL import Image, ImageDraw

        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for x, y, text, size, color, bold in items:
            draw.text((x, y), text, font=self.font(size, bold), fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


@dataclass
class TrackRow:
    """One synthetic occluder tracklet plus the stats used to pick demos."""

    track_id: int
    video_id: int
    video_name: str
    sequence: str
    category: str
    source: str
    identity: tuple[Any, ...]
    band: str
    peak: float
    victim_track_id: int
    victim_category: str
    frames: list[int]
    median_height: float
    median_width: float
    rho: dict[int, float] = field(default_factory=dict)

    @property
    def legibility(self) -> float:
        """How well the paste reads on screen; saturates once it is clearly visible."""
        return min(self.median_height, 200.0) / 200.0


def _index_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    videos = {int(v["id"]): v for v in dataset["videos"]}
    categories = {int(c["id"]): c["name"] for c in dataset["categories"]}
    images: dict[tuple[int, int], dict[str, Any]] = {}
    for image in dataset["images"]:
        images[(int(image["video_id"]), int(image["frame_index"]))] = image
    per_frame: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    per_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in dataset["annotations"]:
        key = (int(ann["video_id"]), int(ann["frame_index"]))
        per_frame[key].append(ann)
        per_track[int(ann["track_id"])].append(ann)
    return {
        "videos": videos,
        "categories": categories,
        "images": images,
        "per_frame": dict(per_frame),
        "per_track": dict(per_track),
    }


def _sequence_name(video_name: str) -> str:
    return video_name.split("_", 1)[0]


def _collect_rows(tracks: Sequence[Mapping[str, Any]], index: Mapping[str, Any]) -> list[TrackRow]:
    rows: list[TrackRow] = []
    for track in tracks:
        track_id = int(track["track_id"])
        video_id = int(track["video_id"])
        anns = [a for a in index["per_track"].get(track_id, []) if int(a["video_id"]) == video_id]
        if not anns:
            continue
        victim_id = int(track["victim_track_id"])
        victim_anns = [a for a in index["per_track"].get(victim_id, []) if int(a["video_id"]) == video_id]
        victim_category = index["categories"].get(int(victim_anns[0]["category_id"]), "?") if victim_anns else "?"
        frames = sorted(int(a["frame_index"]) for a in anns)
        video_name = index["videos"][video_id]["name"]
        rows.append(
            TrackRow(
                track_id=track_id,
                video_id=video_id,
                video_name=video_name,
                sequence=_sequence_name(video_name),
                category=str(track["category"]),
                source=str(track["source"]["dataset"]),
                identity=tuple(track["source"]["identity"]),
                band=str(track["band"]),
                peak=float(track["achieved_peak"]),
                victim_track_id=victim_id,
                victim_category=victim_category,
                frames=frames,
                median_height=float(median(a["bbox"][3] for a in anns)),
                median_width=float(median(a["bbox"][2] for a in anns)),
                rho={int(a["frame_index"]): float(a.get("occlusion_ratio", 0.0)) for a in victim_anns},
            )
        )
    return rows


def _band_quota(count: int, available: Mapping[str, int]) -> dict[str, int]:
    quota = {band: min(available.get(band, 0), int(round(count * share))) for band, share in BAND_MIX.items()}
    # Rounding and short bands both leave slack; hand it to whoever still has candidates.
    while sum(quota.values()) != count:
        if sum(quota.values()) < count:
            options = [b for b in BAND_ORDER if quota[b] < available.get(b, 0)]
            if not options:
                break
            quota[options[0]] += 1
        else:
            options = [b for b in reversed(BAND_ORDER) if quota[b] > 0]
            quota[options[0]] -= 1
    return quota


def _select_rows(rows: Sequence[TrackRow], count: int) -> list[TrackRow]:
    """Pick legible pastes while spreading over band, background and class pair."""
    available = Counter(row.band for row in rows)
    quota = _band_quota(count, available)
    chosen: list[TrackRow] = []
    used_sequence: Counter[str] = Counter()
    used_pair: Counter[tuple[str, str]] = Counter()
    used_source: Counter[str] = Counter()
    for band in BAND_ORDER:
        pool = [row for row in rows if row.band == band]
        for _ in range(quota.get(band, 0)):
            candidates = [row for row in pool if row not in chosen]
            if not candidates:
                break
            picked = len(chosen)
            best = max(
                candidates,
                key=lambda row: (
                    row.legibility
                    - 0.35 * used_sequence[row.sequence]
                    - 0.25 * used_pair[(row.category, row.victim_category)]
                    # Share, not count: cars are 68% of the pool and a per-use
                    # penalty would keep pushing every car occluder to the back.
                    - 0.50 * used_source[row.source] / max(picked, 1),
                    row.peak,
                    -row.track_id,
                ),
            )
            chosen.append(best)
            used_sequence[best.sequence] += 1
            used_pair[(best.category, best.victim_category)] += 1
            used_source[best.source] += 1
    order = {band: i for i, band in enumerate(BAND_ORDER)}
    chosen.sort(key=lambda row: (order[row.band], -row.peak))
    return chosen


def _put_box(canvas: np.ndarray, bbox: Sequence[float], color: tuple[int, int, int], thickness: int) -> tuple[int, int]:
    x, y, w, h = (float(v) for v in bbox)
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    return p1


def _chip(
    canvas: np.ndarray,
    anchor: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    typo: _Text,
    size: int,
    min_y: int = 0,
    placed: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, str, int, tuple[int, int, int], bool]:
    """Filled label tag above the box (below it when the box hugs the frame top)."""
    text_w, text_h = typo.measure(text, size, bold=True)
    pad_x, pad_y = 5, 3
    box_w, box_h = text_w + 2 * pad_x, text_h + 2 * pad_y + 2
    x = min(max(anchor[0], 0), canvas.shape[1] - box_w)
    step = box_h + 2
    # Victims cluster around the paste, so their tags collide. Stack them
    # upwards into the background rather than down over the objects; only fall
    # back to below the anchor once the stack reaches the header.
    candidates = [anchor[1] - box_h - 2 - k * step for k in range(5)]
    candidates += [anchor[1] + 2 + k * step for k in range(5)]
    candidates = [y for y in candidates if min_y <= y <= canvas.shape[0] - box_h]
    if not candidates:
        candidates = [max(min_y, min(anchor[1], canvas.shape[0] - box_h))]
    y = candidates[0]
    if placed is not None:
        for candidate in candidates:
            overlaps = any(
                x < other[2] and other[0] < x + box_w and candidate < other[3] and other[1] < candidate + box_h
                for other in placed
            )
            if not overlaps:
                y = candidate
                break
        placed.append((x, y, x + box_w, y + box_h))
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), color, -1)
    return (x + pad_x, y + pad_y - 1, text, size, (16, 16, 16), True)


def _draw_sparkline(
    canvas: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    series: Sequence[float | None],
    cursor: int,
) -> None:
    """rho(t) over the tracklet's exposure, with a cursor on the current frame."""
    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (44, 44, 50), -1)
    for level in (0.5, 1.0):
        y = int(y0 + height - level * height)
        cv2.line(canvas, (x0, y), (x0 + width, y), (70, 70, 78), 1)
    if len(series) < 2:
        return
    step = width / (len(series) - 1)

    def point(i: int, value: float) -> tuple[int, int]:
        return int(x0 + i * step), int(y0 + height - min(max(value, 0.0), 1.0) * height)

    # Frames where the victim has no annotation break the curve instead of
    # being bridged by a line that was never measured.
    segment: list[tuple[int, int]] = []
    for i, value in enumerate(series):
        if value is None:
            if len(segment) >= 2:
                cv2.polylines(canvas, [np.asarray(segment, np.int32)], False, COLOR_VICTIM, 2, lineType=cv2.LINE_AA)
            segment = []
            continue
        segment.append(point(i, value))
    if len(segment) >= 2:
        cv2.polylines(canvas, [np.asarray(segment, np.int32)], False, COLOR_VICTIM, 2, lineType=cv2.LINE_AA)

    current = series[cursor] if 0 <= cursor < len(series) else None
    cx = int(x0 + cursor * step)
    cv2.line(canvas, (cx, y0), (cx, y0 + height), COLOR_PLOT, 1)
    if current is not None:
        cv2.circle(canvas, point(cursor, current), 4, COLOR_VICTIM, -1, lineType=cv2.LINE_AA)


def _render_frame(
    frame: np.ndarray,
    row: TrackRow,
    frame_index: int,
    position: int,
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    series: Sequence[float | None],
    typo: _Text,
    labels: Mapping[str, str],
) -> np.ndarray:
    height, width = frame.shape[:2]
    canvas = cv2.copyMakeBorder(frame, HEADER_HEIGHT, 0, 0, 0, cv2.BORDER_CONSTANT, value=COLOR_HEADER)
    offset = HEADER_HEIGHT
    text_items: list[tuple[int, int, str, int, tuple[int, int, int], bool]] = []

    focus = next((a for a in annotations if int(a["track_id"]) == row.track_id), None)
    victim_ids = {row.victim_track_id}
    victim_ids.update(
        int(a["track_id"]) for a in annotations if row.track_id in (a.get("occluder_ids") or [])
    )

    for ann in annotations:
        track_id = int(ann["track_id"])
        if track_id == row.track_id:
            continue
        bbox = list(ann["bbox"])
        bbox[1] += offset
        if ann.get("synthetic_occluder"):
            _put_box(canvas, bbox, COLOR_SYNTH_OTHER, 2)
        elif track_id not in victim_ids:
            _put_box(canvas, bbox, COLOR_REAL, 1)

    victim_tags: list[tuple[tuple[int, int], str, float, bool]] = []
    for ann in annotations:
        track_id = int(ann["track_id"])
        if track_id not in victim_ids or ann.get("synthetic_occluder"):
            continue
        bbox = list(ann["bbox"])
        bbox[1] += offset
        corner = _put_box(canvas, bbox, COLOR_VICTIM, 2)
        rho = float(ann.get("occlusion_ratio", 0.0))
        name = categories.get(int(ann["category_id"]), "?")
        text = f"{labels['victim']} #{track_id} {name}  rho {rho:.2f}"
        victim_tags.append((corner, text, rho, track_id == row.victim_track_id))
    # In dense traffic one paste can clip a handful of objects; every one keeps
    # its amber box, but only the hardest-hit few are worth a legible tag.
    victim_tags.sort(key=lambda tag: (not tag[3], -tag[2]))

    focus_corner = None
    if focus is not None:
        bbox = list(focus["bbox"])
        bbox[1] += offset
        focus_corner = _put_box(canvas, bbox, COLOR_SYNTH, 3)

    # The paste's own tag keeps its spot; victim tags move around it.
    placed: list[tuple[int, int, int, int]] = []
    if focus_corner is not None:
        text_items.append(
            _chip(
                canvas,
                focus_corner,
                f"SYNTH #{row.track_id} {row.category}",
                COLOR_SYNTH,
                typo,
                16,
                min_y=offset,
                placed=placed,
            )
        )
    for corner, text, _, _ in victim_tags[:MAX_VICTIM_TAGS]:
        text_items.append(_chip(canvas, corner, text, COLOR_VICTIM, typo, 15, min_y=offset, placed=placed))

    identity = "/".join(str(part) for part in row.identity)
    covered = f"  ·  {labels['covered']} {len(victim_tags)}" if len(victim_tags) > 1 else ""
    text_items.append((14, 8, f"{labels['occluder']} #{row.track_id}  ({row.category} <- {identity})", 21, COLOR_TEXT, True))
    text_items.append(
        (
            14,
            38,
            f"{labels['background']} KITTI {row.sequence}  ·  band {row.band.upper()}"
            f"  ·  peak rho {row.peak:.2f}  ·  {labels['exposure']} {len(row.frames)}{labels['frames_unit']}"
            f"  ·  {labels['victim']} #{row.victim_track_id} {row.victim_category}{covered}",
            17,
            COLOR_MUTED,
            False,
        )
    )

    legend_x = 14
    for color, key in (
        (COLOR_SYNTH, "legend_synth"),
        (COLOR_SYNTH_OTHER, "legend_synth_other"),
        (COLOR_VICTIM, "legend_victim"),
        (COLOR_REAL, "legend_real"),
    ):
        cv2.rectangle(canvas, (legend_x, 70), (legend_x + 22, 86), color, -1)
        text_items.append((legend_x + 30, 68, labels[key], 16, COLOR_TEXT, False))
        legend_x += 30 + typo.measure(labels[key], 16)[0] + 24

    plot_w, plot_h = 300, 46
    plot_x = width - plot_w - 16
    current = series[position] if 0 <= position < len(series) else None
    current_text = (
        f"{labels['victim']} #{row.victim_track_id}  rho "
        + ("-" if current is None else f"{current:.2f}")
    )
    counter = f"frame {frame_index:06d}   {position + 1}/{len(series)}"
    text_items.append((plot_x - typo.measure(counter, 17, True)[0] - 22, 12, counter, 17, COLOR_TEXT, True))
    text_items.append((plot_x, 10, current_text, 17, COLOR_VICTIM, True))
    _draw_sparkline(canvas, (plot_x, 52), (plot_w, plot_h), series, position)
    text_items.append((plot_x + 4, 50, "rho 1.0", 12, COLOR_MUTED, False))

    return typo.draw(canvas, text_items)


def _find_ffmpeg(explicit: str | Path | None) -> str | None:
    for candidate in (explicit, os.environ.get("KDS_FFMPEG")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class _Encoder:
    """H.264 through ffmpeg when available, OpenCV mp4v otherwise."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, ffmpeg: str | None, crf: int) -> None:
        self.path = path
        self.size = size
        self.backend = "ffmpeg-libx264" if ffmpeg else "opencv-mp4v"
        path.parent.mkdir(parents=True, exist_ok=True)
        if ffmpeg:
            self._process = subprocess.Popen(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-",
                    "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
                ],
                stdin=subprocess.PIPE,
            )
            self._writer = None
        else:
            self._process = None
            self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open a video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.write(frame.tobytes())
        else:
            assert self._writer is not None
            self._writer.write(frame)

    def close(self) -> None:
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.close()
            code = self._process.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {code} for {self.path}")
        else:
            assert self._writer is not None
            self._writer.release()


def _render_demo(
    row: TrackRow,
    index: Mapping[str, Any],
    synthesis_dir: Path,
    destination: Path,
    *,
    pad: int,
    fps: float,
    ffmpeg: str | None,
    crf: int,
    typo: _Text,
    labels: Mapping[str, str],
) -> dict[str, Any]:
    num_frames = int(index["videos"][row.video_id]["num_frames"])
    first = max(0, row.frames[0] - pad)
    last = min(num_frames - 1, row.frames[-1] + pad)
    positions = list(range(first, last + 1))
    series: list[float | None] = [row.rho.get(position) for position in positions]

    encoder: _Encoder | None = None
    written = 0
    try:
        for offset, frame_index in enumerate(positions):
            image = index["images"].get((row.video_id, frame_index))
            if image is None:
                raise FileNotFoundError(f"no image entry for {row.video_name} frame {frame_index}")
            path = synthesis_dir / image["file_name"]
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"failed to read synthesized frame {path}")
            canvas = _render_frame(
                frame,
                row,
                frame_index,
                offset,
                index["per_frame"].get((row.video_id, frame_index), []),
                index["categories"],
                series,
                typo,
                labels,
            )
            # libx264 with yuv420p needs even dimensions.
            pad_x = canvas.shape[1] % 2
            pad_y = canvas.shape[0] % 2
            if pad_x or pad_y:
                canvas = cv2.copyMakeBorder(canvas, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=COLOR_HEADER)
            if encoder is None:
                encoder = _Encoder(destination, (canvas.shape[1], canvas.shape[0]), fps, ffmpeg, crf)
            encoder.write(canvas)
            written += 1
    finally:
        if encoder is not None:
            encoder.close()

    peak_rho = max((value for value in series if value is not None), default=0.0)
    return {
        "file": destination.name,
        "track_id": row.track_id,
        "video": row.video_name,
        "sequence": row.sequence,
        "occluder_category": row.category,
        "occluder_source": row.source,
        "occluder_identity": list(row.identity),
        "band": row.band,
        "peak_rho": round(row.peak, 4),
        "victim_track_id": row.victim_track_id,
        "victim_category": row.victim_category,
        "victim_peak_rho_in_clip": round(peak_rho, 4),
        "median_paste_box_hw": [round(row.median_height, 1), round(row.median_width, 1)],
        "tracklet_frames": [row.frames[0], row.frames[-1]],
        "clip_frames": [first, last],
        "frames": written,
        "fps": fps,
        "codec": encoder.backend if encoder is not None else None,
    }


def run(
    config_file: str | Path,
    output: str | Path,
    *,
    count: int = 10,
    tracks: Sequence[int] | None = None,
    pad: int = 0,
    fps: float = 10.0,
    crf: int = 18,
    ffmpeg: str | Path | None = None,
    language: str = "ko",
) -> dict[str, Any]:
    config, config_file_path = load_config(config_file)
    synthesis_dir = resolve_path(config_file_path.parent, config["tracklet_synthesis"]["output_dir"])
    dataset = load_json(synthesis_dir / "annotations.json")
    occluders = load_json(synthesis_dir / "occluder_tracks.json")
    index = _index_dataset(dataset)
    rows = _collect_rows(occluders, index)
    if not rows:
        raise RuntimeError(f"no synthetic occluder tracklets found under {synthesis_dir}")

    if tracks:
        by_id = {row.track_id: row for row in rows}
        missing = [track_id for track_id in tracks if track_id not in by_id]
        if missing:
            raise KeyError(f"unknown occluder track ids: {missing}")
        selected = [by_id[track_id] for track_id in tracks]
    else:
        selected = _select_rows(rows, min(count, len(rows)))

    typo = _Text()
    labels = LABELS_KO if (language == "ko" and typo.unicode_ok) else LABELS_EN
    ffmpeg_bin = _find_ffmpeg(ffmpeg)
    destination_dir = Path(output).expanduser()
    if not destination_dir.is_absolute():
        destination_dir = (REPO_ROOT / destination_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for order, row in enumerate(selected, start=1):
        name = (
            f"{order:02d}_{row.band}_{row.sequence}_occ{row.track_id}"
            f"_{row.category}_on_{row.victim_category}.mp4"
        )
        entry = _render_demo(
            row,
            index,
            synthesis_dir,
            destination_dir / name,
            pad=pad,
            fps=fps,
            ffmpeg=ffmpeg_bin,
            crf=crf,
            typo=typo,
            labels=labels,
        )
        entries.append(entry)
        print(
            f"[{order:2d}/{len(selected)}] {name}  frames={entry['frames']}"
            f"  peak_rho={entry['peak_rho']:.2f}  codec={entry['codec']}"
        )

    summary = {
        "source": str(synthesis_dir),
        "output_dir": str(destination_dir),
        "count": len(entries),
        "pad": pad,
        "fps": fps,
        "colors_bgr": {
            "synthetic_occluder": list(COLOR_SYNTH),
            "other_synthetic_occluder": list(COLOR_SYNTH_OTHER),
            "occluded_victim": list(COLOR_VICTIM),
            "real_kitti_gt": list(COLOR_REAL),
        },
        "legend": {
            "magenta": "synthetic occluder tracklet pasted by the renderer (focus of the clip)",
            "light magenta": "another synthetic occluder present in the same frame",
            "amber": "real KITTI object occluded by the pasted tracklet, labelled with rho",
            "green": "real KITTI ground truth left untouched",
        },
        "demos": entries,
    }
    save_json(destination_dir / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-tracklet occlusion demo videos")
    parser.add_argument("--config", default=REPO_ROOT / "configs" / "phase1_kitti.yaml", type=Path)
    parser.add_argument("--output", default=Path("visualizations/tracklet_demos"), type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--tracks", default="", help="comma separated occluder track ids (overrides --count)")
    parser.add_argument("--pad", type=int, default=0, help="context frames kept around the tracklet")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg binary (defaults to KDS_FFMPEG, PATH, imageio-ffmpeg)")
    parser.add_argument("--language", choices=["ko", "en"], default="ko")
    args = parser.parse_args()
    track_ids = [int(part) for part in args.tracks.split(",") if part.strip()]
    summary = run(
        args.config,
        args.output,
        count=args.count,
        tracks=track_ids or None,
        pad=args.pad,
        fps=args.fps,
        crf=args.crf,
        ffmpeg=args.ffmpeg,
        language=args.language,
    )
    print(f"wrote {summary['count']} demo videos to {summary['output_dir']}")


if __name__ == "__main__":
    main()
