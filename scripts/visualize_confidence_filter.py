from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from scripts.render_tracklet_demo_videos import _Encoder, _find_ffmpeg


PASS_COLOR = (70, 205, 90)
FAIL_COLOR = (70, 70, 230)
RETAIN_COLOR = (255, 210, 50)
TEXT_COLOR = (245, 245, 245)
MUTED_COLOR = (175, 175, 180)
HEADER_COLOR = (28, 28, 32)


def _fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _load_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _load_bgra(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"expected BGRA crop: {path}")
    return image


def _annotate_source(
    image: np.ndarray,
    rgba: np.ndarray,
    frame: Mapping[str, Any],
    *,
    passed: bool,
    threshold: float,
) -> np.ndarray:
    output = image.copy()
    x, y, width, height = (int(round(value)) for value in frame["source_bbox_xywh"])
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(output.shape[1], x + width), min(output.shape[0], y + height)
    if x2 > x1 and y2 > y1:
        mask = cv2.resize(rgba[:, :, 3], (width, height), interpolation=cv2.INTER_NEAREST)
        mask = mask[y1 - y : y2 - y, x1 - x : x2 - x] > 0
        region = output[y1:y2, x1:x2]
        tint = np.full_like(region, (210, 40, 210))
        region[mask] = cv2.addWeighted(region[mask], 0.55, tint[mask], 0.45, 0)
        color = PASS_COLOR if passed else FAIL_COLOR
        cv2.rectangle(output, (x1, y1), (x2, y2), color, max(2, output.shape[1] // 500))
    confidence = float(frame["detector_confidence"])
    text = f"conf {confidence:.3f}  {'PASS' if passed else 'FAIL'}  (threshold {threshold:.2f})"
    cv2.putText(
        output,
        text,
        (max(8, x1), max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, output.shape[1] / 1800),
        PASS_COLOR if passed else FAIL_COLOR,
        2,
        cv2.LINE_AA,
    )
    return output


def _paste_controlled(
    background: np.ndarray,
    rgba: np.ndarray,
    category: str,
) -> np.ndarray:
    output = background.copy()
    target_height = round(output.shape[0] * (0.24 if category == "car" else 0.34))
    scale = target_height / rgba.shape[0]
    target_width = max(1, round(rgba.shape[1] * scale))
    target_height = max(1, target_height)
    if target_width > round(output.shape[1] * 0.38):
        scale *= (output.shape[1] * 0.38) / target_width
        target_width = max(1, round(rgba.shape[1] * scale))
        target_height = max(1, round(rgba.shape[0] * scale))
    resized = cv2.resize(rgba, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    x = min(output.shape[1] - target_width, round(output.shape[1] * 0.70))
    y = min(output.shape[0] - target_height, round(output.shape[0] * 0.91) - target_height)
    x, y = max(0, x), max(0, y)
    rgb = resized[:, :, :3].astype(np.float32)
    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
    region = output[y : y + target_height, x : x + target_width].astype(np.float32)
    output[y : y + target_height, x : x + target_width] = np.clip(
        rgb * alpha + region * (1.0 - alpha), 0, 255
    ).astype(np.uint8)
    cv2.rectangle(output, (x, y), (x + target_width, y + target_height), (220, 60, 220), 2)
    return output


def _dark_label(panel: np.ndarray, first: str, second: str = "") -> np.ndarray:
    output = cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0)
    size = cv2.getTextSize(first, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    x = max(10, (output.shape[1] - size[0]) // 2)
    y = output.shape[0] // 2
    cv2.putText(output, first, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, FAIL_COLOR, 2, cv2.LINE_AA)
    if second:
        size = cv2.getTextSize(second, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        x = max(10, (output.shape[1] - size[0]) // 2)
        cv2.putText(
            output, second, (x, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA
        )
    return output


def _timeline(
    canvas: np.ndarray,
    decision: Mapping[str, Any],
    current: int,
    *,
    top: int,
) -> None:
    frames = decision["frames"]
    left, right = 40, canvas.shape[1] - 40
    width = right - left
    for index, frame in enumerate(frames):
        x1 = left + round(width * index / len(frames))
        x2 = left + round(width * (index + 1) / len(frames))
        color = PASS_COLOR if frame["passed"] else FAIL_COLOR
        cv2.rectangle(canvas, (x1, top), (max(x1 + 1, x2), top + 18), color, -1)
        if frame["retained"]:
            cv2.rectangle(canvas, (x1, top + 20), (max(x1 + 1, x2), top + 28), RETAIN_COLOR, -1)
    marker = left + round(width * (current + 0.5) / len(frames))
    cv2.line(canvas, (marker, top - 4), (marker, top + 32), TEXT_COLOR, 2)
    cv2.putText(
        canvas,
        "green=confidence pass   red=fail   yellow=retained longest run",
        (left, top + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )


def _record_visual_score(record: Mapping[str, Any], decision: Mapping[str, Any]) -> float:
    areas = [
        float(frame["source_bbox_xywh"][2]) * float(frame["source_bbox_xywh"][3])
        for frame in record["frames"]
    ]
    area = median(areas) if areas else 0.0
    scenario = str(decision["scenario"])
    if scenario == "accepted_high":
        quality = min(float(frame["detector_confidence"]) for frame in record["frames"])
    elif scenario == "accepted_trimmed":
        quality = float(decision["input_length"] - decision["longest_passing_length"])
    elif scenario == "rejected_fragmented":
        quality = float(decision["passing_frames"])
    else:
        quality = float(decision["input_length"] - decision["passing_frames"])
    return area + 1000.0 * quality


def select_scenarios(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    specifications: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    record_by_id = {int(record["id"]): record for record in records}
    selected = []
    used: set[int] = set()
    for spec in specifications:
        eligible = []
        for decision in decisions:
            tracklet_id = int(decision["source_tracklet_id"])
            if tracklet_id in used or str(decision["scenario"]) != str(spec["outcome"]):
                continue
            if spec.get("category") and str(decision["category"]) != str(spec["category"]):
                continue
            if spec.get("source_dataset") and str(decision["source_dataset"]) != str(spec["source_dataset"]):
                continue
            record = record_by_id[tracklet_id]
            eligible.append((record, decision))
        if not eligible:
            raise RuntimeError(f"no tracklet matches visualization scenario {spec}")
        record, decision = max(
            eligible,
            key=lambda item: (_record_visual_score(item[0], item[1]), -int(item[0]["id"])),
        )
        used.add(int(record["id"]))
        selected.append((str(spec["name"]), record, decision))
    return selected


def _preview_index(decision: Mapping[str, Any]) -> int:
    frames = decision["frames"]
    if decision["scenario"] == "accepted_high":
        return len(frames) // 2
    removed = [index for index, frame in enumerate(frames) if not frame["retained"]]
    if removed:
        return min(removed, key=lambda index: float(frames[index]["detector_confidence"]))
    return len(frames) // 2


def _render_scenario(
    name: str,
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    raw_dir: Path,
    backgrounds: Sequence[Path],
    background_offset: int,
    destination: Path,
    preview: Path,
    panel_size: tuple[int, int],
    fps: float,
    crf: int,
    ffmpeg: str | None,
    threshold: float,
) -> dict[str, Any]:
    panel_width, panel_height = panel_size
    header_height, footer_height = 78, 82
    canvas_size = (panel_width * 3, header_height + panel_height + footer_height)
    encoder = _Encoder(destination, canvas_size, fps, ffmpeg, crf)
    preview_at = _preview_index(decision)
    try:
        for index, (frame, frame_decision) in enumerate(zip(record["frames"], decision["frames"])):
            rgba = _load_bgra(raw_dir / frame["file_name"])
            source = _annotate_source(
                _load_bgr(frame["source_image"]),
                rgba,
                frame,
                passed=bool(frame_decision["passed"]),
                threshold=threshold,
            )
            background = _load_bgr(backgrounds[(background_offset + index) % len(backgrounds)])
            before = _paste_controlled(background, rgba, str(record["category"]))
            if frame_decision["retained"]:
                after = _paste_controlled(background, rgba, str(record["category"]))
            elif decision["accepted"]:
                after = _dark_label(background, "REMOVED BY FILTER", "outside retained consecutive run")
            else:
                after = _dark_label(
                    background,
                    "TRACKLET REJECTED",
                    f"longest passing run {decision['longest_passing_length']} < 30",
                )
            panels = [_fit(source, panel_size), _fit(before, panel_size), _fit(after, panel_size)]
            canvas = np.full((canvas_size[1], canvas_size[0], 3), HEADER_COLOR, dtype=np.uint8)
            canvas[header_height : header_height + panel_height] = np.hstack(panels)
            status = "ACCEPT" if decision["accepted"] else "REJECT"
            color = PASS_COLOR if decision["accepted"] else FAIL_COLOR
            cv2.putText(
                canvas,
                f"{name}  |  raw tracklet #{record['id']}  {record['category']} / {record.get('source_dataset')}",
                (18, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{status}: {decision['reason']}  |  frame {index + 1}/{len(record['frames'])}",
                (18, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                color,
                2,
                cv2.LINE_AA,
            )
            for panel_index, label in enumerate(
                ("SOURCE + Co-DETR/SAM3", "BEFORE: RAW PASTE", "AFTER: CONF FILTER")
            ):
                cv2.putText(
                    canvas,
                    label,
                    (panel_index * panel_width + 16, header_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    TEXT_COLOR,
                    2,
                    cv2.LINE_AA,
                )
            _timeline(canvas, decision, index, top=header_height + panel_height + 10)
            encoder.write(canvas)
            if index == preview_at:
                cv2.imwrite(str(preview), canvas)
    finally:
        encoder.close()
    return {
        "name": name,
        "video": str(destination),
        "preview": str(preview),
        "codec": encoder.backend,
        "source_tracklet_id": int(record["id"]),
        "category": str(record["category"]),
        "source_dataset": str(record.get("source_dataset")),
        "sequence": str(record.get("sequence")),
        "scenario": str(decision["scenario"]),
        "accepted": bool(decision["accepted"]),
        "input_length": int(decision["input_length"]),
        "passing_frames": int(decision["passing_frames"]),
        "longest_passing_length": int(decision["longest_passing_length"]),
        "frames": len(record["frames"]),
    }


def _contact_sheet(entries: Sequence[Mapping[str, Any]], destination: Path) -> None:
    previews = [_load_bgr(entry["preview"]) for entry in entries]
    tile_width = 960
    tiles = [
        cv2.resize(image, (tile_width, round(image.shape[0] * tile_width / image.shape[1])))
        for image in previews
    ]
    tile_height = max(image.shape[0] for image in tiles)
    rows = []
    for index in range(0, len(tiles), 2):
        row_tiles = tiles[index : index + 2]
        while len(row_tiles) < 2:
            row_tiles.append(np.full_like(tiles[0], HEADER_COLOR))
        padded = [
            cv2.copyMakeBorder(
                image, 0, tile_height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=HEADER_COLOR
            )
            for image in row_tiles
        ]
        rows.append(np.hstack(padded))
    cv2.imwrite(str(destination), np.vstack(rows))


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    filter_settings = config["detector_tracklet_pool"]["quality_filter"]
    visual = config["confidence_filter_visualization"]
    raw_dir = resolve_path(path.parent, filter_settings["input_pool_dir"])
    filtered_dir = resolve_path(path.parent, filter_settings["output_dir"])
    output_dir = resolve_path(path.parent, visual["output_dir"])
    raw = load_json(raw_dir / "tracklets.json")
    decisions = load_json(filtered_dir / "decisions.json")
    selected = select_scenarios(
        raw["tracklets"],
        decisions["tracklets"],
        visual["scenarios"],
    )
    kitti_root = config_path(config, path, "paths", "kitti_tracking")
    sequence = str(visual.get("background_sequence", "0000"))
    background_dir = kitti_root / "training" / "image_02" / sequence
    backgrounds = sorted([*background_dir.glob("*.png"), *background_dir.glob("*.jpg")])
    if not backgrounds:
        raise FileNotFoundError(f"no visualization backgrounds found: {background_dir}")
    panel_size = tuple(int(value) for value in visual.get("panel_size", [640, 360]))
    ffmpeg = _find_ffmpeg(None)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for order, (name, record, decision) in enumerate(selected):
        entries.append(
            _render_scenario(
                name,
                record,
                decision,
                raw_dir=raw_dir,
                backgrounds=backgrounds,
                background_offset=order * 17,
                destination=output_dir / f"{order + 1:02d}_{name}.mp4",
                preview=output_dir / f"{order + 1:02d}_{name}.jpg",
                panel_size=panel_size,
                fps=float(visual.get("fps", 10)),
                crf=int(visual.get("crf", 18)),
                ffmpeg=ffmpeg,
                threshold=float(filter_settings["detector_confidence_min"]),
            )
        )
    contact_sheet = output_dir / "contact_sheet.jpg"
    _contact_sheet(entries, contact_sheet)
    summary = {
        "source_pool": str(raw_dir / "tracklets.json"),
        "decision_manifest": str(filtered_dir / "decisions.json"),
        "output_dir": str(output_dir),
        "comparison_policy": {
            "same_raw_tracklet": True,
            "same_background_frames": True,
            "same_position_and_scale": True,
            "additional_jitter": False,
            "rejected_tracklets_visible_before_filter": True,
        },
        "contact_sheet": str(contact_sheet),
        "videos": entries,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render controlled before/after videos for detector-confidence filtering"
    )
    parser.add_argument("--config", default="configs/phase2_detector_conf.yaml", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
