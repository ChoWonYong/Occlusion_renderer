from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from pool.build_tracklet_pool import crop_with_context, inventory, select_matching_instance
from segment.base import Instance


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _checkerboard(height: int, width: int, tile: int = 12) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cells = (xx // tile + yy // tile) % 2
    values = np.where(cells[..., None] == 0, 225, 180)
    return np.repeat(values, 3, axis=2).astype(np.uint8)


def _save_visuals(
    output_dir: Path,
    prefix: str,
    context: np.ndarray,
    target_box: list[float],
    instances: list[Instance],
    selected: Instance,
    source_crop: np.ndarray,
    crop_mask: np.ndarray,
) -> dict[str, str]:
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    context_image = Image.fromarray(context)
    draw = ImageDraw.Draw(context_image)
    tx, ty, tw, th = target_box
    draw.rectangle((tx, ty, tx + tw, ty + th), outline=(0, 255, 255), width=3)
    for instance in instances:
        x, y, width, height = instance.bbox
        color = (0, 255, 0) if instance is selected else (255, 190, 0)
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
        draw.text((x + 2, max(0, y - 12)), f"{instance.score:.2f}", fill=color)

    binary = np.asarray(crop_mask, dtype=bool)
    alpha = binary.astype(np.uint8) * 255
    rgba = np.dstack((source_crop, alpha))
    overlay = source_crop.astype(np.float32)
    overlay[binary] = overlay[binary] * 0.45 + np.asarray([255, 40, 40], dtype=np.float32) * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    checker = _checkerboard(source_crop.shape[0], source_crop.shape[1])
    checker[binary] = source_crop[binary]

    context_path = output_dir / f"{prefix}_context_detections.png"
    crop_path = output_dir / f"{prefix}_crop.png"
    overlay_path = output_dir / f"{prefix}_overlay.png"
    rgba_path = output_dir / f"{prefix}_rgba.png"
    comparison_path = output_dir / f"{prefix}_comparison.png"
    context_image.save(context_path)
    Image.fromarray(source_crop).save(crop_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(rgba).save(rgba_path)

    label_height = 28
    panel_width = source_crop.shape[1]
    sheet = Image.new("RGB", (panel_width * 3, source_crop.shape[0] + label_height), "white")
    for index, (label, panel) in enumerate(
        [("MOT17 bbox crop", source_crop), ("SAM3 human mask", overlay), ("RGBA on checker", checker)]
    ):
        sheet.paste(Image.fromarray(panel), (index * panel_width, label_height))
        ImageDraw.Draw(sheet).text((index * panel_width + 6, 7), label, fill=(20, 20, 20))
    sheet.save(comparison_path)
    return {
        "context_detections": str(context_path),
        "crop": str(crop_path),
        "overlay": str(overlay_path),
        "rgba": str(rgba_path),
        "comparison": str(comparison_path),
    }


def run(
    config_file: str | Path,
    *,
    candidate_id: int = 1,
    frame_offset: int = 15,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    from segment.sam3 import Sam3Backend

    config, path = load_config(config_file)
    settings = config["tracklet_pool"]
    pool_dir = resolve_path(path.parent, settings["output_dir"])
    candidates_path = pool_dir / "candidates.json"
    if not candidates_path.is_file():
        inventory(path)
    metadata = load_json(candidates_path)
    by_id = {int(record["candidate_id"]): record for record in metadata["tracklets"]}
    if candidate_id not in by_id:
        raise ValueError(f"candidate_id {candidate_id} not found; valid range is 1..{len(by_id)}")
    if not 0 <= frame_offset < 30:
        raise ValueError("frame_offset must be in [0, 29]")
    candidate = by_id[candidate_id]
    frame = candidate["frames"][frame_offset]
    source_path = Path(frame["source_image"])
    image = _load_rgb(source_path)
    source_crop, context, target_box, clipped_box, context_box = crop_with_context(
        image,
        tuple(float(value) for value in frame["source_bbox_xywh"]),
        float(settings.get("sam3_context_padding", 0.2)),
    )

    segmenter = Sam3Backend(
        config_path(config, path, "paths", "sam3_checkpoint"),
        bpe_path=config_path(config, path, "paths", "sam3_bpe"),
        device=str(settings.get("device", "cuda")),
        confidence_threshold=float(settings.get("sam3_confidence_threshold", 0.5)),
        precision=str(settings.get("sam3_precision", "auto")),
    )
    prompt = str(settings.get("sam3_text_prompt", "human"))
    instances = segmenter.detect_and_mask(context, [prompt])
    minimum_iou = float(settings.get("sam3_min_gt_iou", 0.3))
    matched = select_matching_instance(instances, target_box, minimum_iou)
    if matched is None:
        raise RuntimeError(
            f"SAM3 found {len(instances)} '{prompt}' instances, but none reached GT IoU {minimum_iou}"
        )
    selected, match_iou = matched
    crop_x, crop_y = int(round(target_box[0])), int(round(target_box[1]))
    crop_mask = np.asarray(selected.mask, dtype=np.uint8)[
        crop_y : crop_y + source_crop.shape[0], crop_x : crop_x + source_crop.shape[1]
    ]
    mask_area = int(np.count_nonzero(crop_mask))
    crop_area = int(crop_mask.size)
    coverage = mask_area / crop_area if crop_area else 0.0
    minimum_area = int(settings.get("min_mask_area", 128))
    warnings: list[str] = []
    if mask_area < minimum_area:
        warnings.append(f"mask area {mask_area} is below min_mask_area={minimum_area}")
    if coverage < 0.10:
        warnings.append("mask covers less than 10% of the MOT17 bbox crop")
    if coverage > 0.95:
        warnings.append("mask covers more than 95% of the MOT17 bbox crop")

    output_dir = (
        Path(output_override).expanduser().resolve()
        if output_override is not None
        else resolve_path(path.parent, "../visualizations/sam3_smoke")
    )
    prefix = f"candidate_{candidate_id:06d}_offset_{frame_offset:02d}"
    paths = _save_visuals(
        output_dir, prefix, context, target_box, instances, selected, source_crop, crop_mask
    )
    result = {
        "ok": mask_area >= minimum_area and 0.10 <= coverage <= 0.95 and match_iou >= minimum_iou,
        "backend": "sam3",
        "precision": segmenter.precision,
        "text_prompt": prompt,
        "candidate_id": candidate_id,
        "sequence": candidate["sequence"],
        "source_track_id": candidate["source_track_id"],
        "frame_offset": frame_offset,
        "source_frame": frame["source_frame"],
        "source_image": str(source_path),
        "source_bbox_xywh": frame["source_bbox_xywh"],
        "clipped_crop_bbox_xywh": clipped_box,
        "sam3_context_bbox_xywh": context_box,
        "detections": len(instances),
        "selected_score": float(selected.score),
        "selected_bbox_xywh_in_context": selected.bbox,
        "selected_gt_iou": float(match_iou),
        "mask_area": mask_area,
        "crop_area": crop_area,
        "mask_coverage": coverage,
        "warnings": warnings,
        "outputs": paths,
    }
    report_path = output_dir / f"{prefix}_report.json"
    save_json(report_path, result)
    result["report"] = str(report_path)
    return result


def _relaunch(config_file: Path, candidate_id: int, frame_offset: int, output: Path | None) -> int:
    config, path = load_config(config_file)
    python_path = config_path(config, path, "paths", "sam3_python")
    if not python_path.is_file():
        raise FileNotFoundError(f"kds-sam3 Python not found: {python_path}. See docs/SAM3_WORKFLOW.md")
    command = [
        str(python_path),
        "-m",
        "scripts.sam3_smoke",
        "--config",
        str(path),
        "--candidate-id",
        str(candidate_id),
        "--frame-offset",
        str(frame_offset),
    ]
    if output is not None:
        command.extend(["--output", str(output)])
    return subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3 'human' prompt on one MOT17 candidate frame")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--candidate-id", type=int, default=1)
    parser.add_argument("--frame-offset", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(_relaunch(args.config, args.candidate_id, args.frame_offset, args.output))
    result = run(
        args.config,
        candidate_id=args.candidate_id,
        frame_offset=args.frame_offset,
        output_override=args.output,
    )
    print(result)
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
