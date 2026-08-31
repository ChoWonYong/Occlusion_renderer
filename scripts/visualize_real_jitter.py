from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from scripts.render_tracklet_demo_videos import _Encoder, _find_ffmpeg
from synth.paste_jitter import (
    IDENTITY,
    jitter_box,
    jitter_rgba,
    real_policy_from_config,
    resolve_preset,
    sample_sequence,
)
from synth.tracklet_compositor import TrackletLayer, composite_tracklet_layers


BACKGROUND = (26, 26, 30)
TEXT = (245, 245, 245)
MUTED = (175, 180, 185)


def _fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    left = (width - resized.shape[1]) // 2
    top = (height - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA")).copy()


def _select_tracklet(records: Sequence[Mapping[str, Any]], frame_limit: int) -> Mapping[str, Any]:
    eligible = [record for record in records if len(record.get("frames", [])) >= frame_limit]
    if not eligible:
        eligible = [record for record in records if record.get("frames")]
    if not eligible:
        raise RuntimeError("the filtered detector pool has no renderable tracklet")

    def score(record: Mapping[str, Any]) -> tuple[float, float, int]:
        frames = record["frames"]
        areas = [
            float(frame["crop_bbox_xywh"][2]) * float(frame["crop_bbox_xywh"][3])
            for frame in frames
        ]
        confidence = median(float(frame["detector_confidence"]) for frame in frames)
        return median(areas), confidence, len(frames)

    return max(eligible, key=score)


def _base_box(background: np.ndarray, rgba: np.ndarray, category: str) -> tuple[float, ...]:
    target_height = background.shape[0] * (0.25 if category == "car" else 0.36)
    scale = target_height / max(1, rgba.shape[0])
    width = rgba.shape[1] * scale
    height = rgba.shape[0] * scale
    if width > background.shape[1] * 0.42:
        correction = background.shape[1] * 0.42 / width
        width *= correction
        height *= correction
    center_x = background.shape[1] * 0.72
    bottom_y = background.shape[0] * 0.92
    return (center_x - width / 2.0, bottom_y - height, width, height)


def _paste(background: np.ndarray, rgba: np.ndarray, jitter: Any, category: str) -> np.ndarray:
    base = _base_box(background, rgba, category)
    box = jitter_box(base, (rgba.shape[1], rgba.shape[0]), jitter)
    output, _ = composite_tracklet_layers(
        background,
        [
            TrackletLayer(
                track_id=1,
                rgba=jitter_rgba(rgba, jitter, apply_real_effects=False),
                bbox_xywh=box,
                post_resize_jitter=jitter,
            )
        ],
        blend_method="none",
    )
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def _contact_sheet(entries: Sequence[Mapping[str, Any]], destination: Path) -> None:
    previews = [cv2.imread(str(entry["preview"]), cv2.IMREAD_COLOR) for entry in entries]
    if any(image is None for image in previews):
        raise RuntimeError("could not read a real-jitter preview")
    tile_width = 960
    tiles = [
        cv2.resize(image, (tile_width, round(image.shape[0] * tile_width / image.shape[1])))
        for image in previews
    ]
    tile_height = max(image.shape[0] for image in tiles)
    rows = []
    for index in range(0, len(tiles), 2):
        row = tiles[index : index + 2]
        while len(row) < 2:
            row.append(np.full_like(tiles[0], BACKGROUND))
        rows.append(
            np.hstack(
                [
                    cv2.copyMakeBorder(
                        image,
                        0,
                        tile_height - image.shape[0],
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=BACKGROUND,
                    )
                    for image in row
                ]
            )
        )
    cv2.imwrite(str(destination), np.vstack(rows))


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    synthesis = config["tracklet_synthesis"]
    visual = config["real_jitter_visualization"]
    pool_dir = resolve_path(path.parent, synthesis["pool_sources"][0])
    pool = load_json(pool_dir / "tracklets.json")
    requested_frames = int(visual.get("frames", 50))
    tracklet = _select_tracklet(pool["tracklets"], requested_frames)
    frames = tracklet["frames"][: min(requested_frames, len(tracklet["frames"]))]

    background_sequence = str(visual.get("background_sequence", "0000"))
    background_dir = (
        config_path(config, path, "paths", "kitti_tracking")
        / "training"
        / "image_02"
        / background_sequence
    )
    backgrounds = sorted([*background_dir.glob("*.png"), *background_dir.glob("*.jpg")])
    if not backgrounds:
        raise FileNotFoundError(f"no visualization backgrounds found: {background_dir}")

    jitter_section = synthesis["paste_jitter"]
    real_policy = real_policy_from_config(jitter_section["real"])
    random_ranges = resolve_preset("mid")
    assert random_ranges is not None
    random_jitters = sample_sequence(random_ranges, len(frames), random.Random(1301))
    output_dir = resolve_path(path.parent, visual["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_size = tuple(int(value) for value in visual.get("panel_size", [640, 360]))
    panel_width, panel_height = panel_size
    header_height, footer_height = 82, 58
    canvas_size = (panel_width * 3, header_height + panel_height + footer_height)
    ffmpeg = _find_ffmpeg(None)
    entries: list[dict[str, Any]] = []

    for order, scenario in enumerate(visual["scenarios"]):
        real_jitters = sample_sequence(
            real_policy,
            len(frames),
            random.Random(2300 + order),
            scenario=str(scenario),
        )
        video = output_dir / f"{order + 1:02d}_{scenario}.mp4"
        preview = output_dir / f"{order + 1:02d}_{scenario}.jpg"
        encoder = _Encoder(
            video,
            canvas_size,
            float(visual.get("fps", 10)),
            ffmpeg,
            int(visual.get("crf", 18)),
        )
        changed_pixels: list[int] = []
        alpha_preserved = True
        try:
            for index, frame in enumerate(frames):
                rgba = _load_rgba(pool_dir / frame["file_name"])
                background = _load_rgb(backgrounds[index % len(backgrounds)])
                original = _paste(background, rgba, IDENTITY, str(tracklet["category"]))
                random_panel = _paste(
                    background, rgba, random_jitters[index], str(tracklet["category"])
                )
                transformed = jitter_rgba(rgba, real_jitters[index])
                alpha_preserved &= np.array_equal(transformed[..., 3], rgba[..., 3])
                changed_pixels.append(
                    int(
                        np.count_nonzero(
                            np.any(transformed[..., :3] != rgba[..., :3], axis=2)
                            & (rgba[..., 3] > 0)
                        )
                    )
                )
                real_panel = _paste(
                    background, rgba, real_jitters[index], str(tracklet["category"])
                )
                panels = [_fit(item, panel_size) for item in (original, random_panel, real_panel)]
                canvas = np.full((canvas_size[1], canvas_size[0], 3), BACKGROUND, dtype=np.uint8)
                canvas[header_height : header_height + panel_height] = np.hstack(panels)
                cv2.putText(
                    canvas,
                    f"REAL JITTER: {str(scenario).upper()}  |  same tracklet / background / placement",
                    (18, 31),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    TEXT,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"automatic filtered tracklet #{tracklet['id']}  {tracklet['category']}  frame {index + 1}/{len(frames)}",
                    (18, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    MUTED,
                    2,
                    cv2.LINE_AA,
                )
                for panel_index, label in enumerate(
                    ("NO JITTER", "DEFAULT: RANDOM MID", f"REAL: {str(scenario).upper()}")
                ):
                    cv2.putText(
                        canvas,
                        label,
                        (panel_index * panel_width + 16, header_height + 29),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.64,
                        TEXT,
                        2,
                        cv2.LINE_AA,
                    )
                footer = {
                    "day": "brightness only",
                    "night": "lower brightness",
                    "tunnel": "lower brightness + warm vignette",
                    "rain": "sparse 2x2 local elastic RGB displacement",
                    "snow": "weak 2x2 elastic displacement + sparse white RGB pixels",
                }[str(scenario)]
                cv2.putText(
                    canvas,
                    f"{footer}  |  alpha mask unchanged: YES",
                    (18, header_height + panel_height + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    MUTED,
                    2,
                    cv2.LINE_AA,
                )
                encoder.write(canvas)
                if index == len(frames) // 2:
                    cv2.imwrite(str(preview), canvas)
        finally:
            encoder.close()
        entries.append(
            {
                "scenario": str(scenario),
                "video": str(video),
                "preview": str(preview),
                "codec": encoder.backend,
                "frames": len(frames),
                "alpha_preserved": bool(alpha_preserved),
                "mean_changed_crop_pixels": float(np.mean(changed_pixels)),
            }
        )

    contact_sheet = output_dir / "contact_sheet.jpg"
    _contact_sheet(entries, contact_sheet)
    summary = {
        "source_pool": str(pool_dir / "tracklets.json"),
        "source_policy": "automatic Co-DETR/ByteTrack/SAM3 + confidence >= 0.60",
        "tracklet_id": int(tracklet["id"]),
        "category": str(tracklet["category"]),
        "source_dataset": str(tracklet.get("source_dataset", tracklet.get("source"))),
        "comparison_policy": {
            "same_tracklet": True,
            "same_background_frames": True,
            "same_base_position_and_scale": True,
            "default_mode": "random/mid",
            "real_scenario_constant_per_event": True,
            "real_effects_modify_alpha": False,
        },
        "contact_sheet": str(contact_sheet),
        "videos": entries,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize random jitter against real scenarios")
    parser.add_argument("--config", default="configs/phase3_real_jitter.yaml", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
