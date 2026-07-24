from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common.config import load_config, resolve_path
from common.io import load_json, save_json


def _checker_composite(rgba: np.ndarray, tile: int = 10) -> np.ndarray:
    patch = np.asarray(rgba, dtype=np.uint8)
    yy, xx = np.indices(patch.shape[:2])
    cells = (xx // tile + yy // tile) % 2
    base = np.repeat(np.where(cells[..., None] == 0, 225, 180), 3, axis=2).astype(np.float32)
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    return np.clip(patch[..., :3] * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)


def run(
    config_file: str | Path,
    *,
    count: int = 10,
    columns: int = 10,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    config, path = load_config(config_file)
    pool_dir = resolve_path(path.parent, config["tracklet_pool"]["output_dir"])
    metadata = load_json(pool_dir / "tracklets.json")
    records = list(metadata.get("tracklets", []))[:count]
    if not records:
        raise ValueError(f"no generated SAM tracklets found in {pool_dir / 'tracklets.json'}")
    output_dir = (
        Path(output_override).expanduser().resolve()
        if output_override is not None
        else resolve_path(path.parent, "../visualizations/mot17_sam_tracklets")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    for record in records:
        patches: list[np.ndarray] = []
        for frame in record["frames"]:
            with Image.open(pool_dir / frame["file_name"]) as image:
                patches.append(_checker_composite(np.asarray(image.convert("RGBA"))))
        cell_width = max(patch.shape[1] for patch in patches)
        cell_height = max(patch.shape[0] for patch in patches) + 20
        rows = (len(patches) + columns - 1) // columns
        sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
        draw = ImageDraw.Draw(sheet)
        for index, patch in enumerate(patches):
            row, column = divmod(index, columns)
            x = column * cell_width + (cell_width - patch.shape[1]) // 2
            y = row * cell_height + 18
            sheet.paste(Image.fromarray(patch), (x, y))
            draw.text((column * cell_width + 3, row * cell_height + 2), f"{index:02d}", fill=(20, 20, 20))
        destination = output_dir / f"tracklet_{int(record['id']):06d}.jpg"
        sheet.save(destination, quality=95)
        outputs.append(str(destination))
    result = {"tracklets": len(outputs), "output_dir": str(output_dir), "files": outputs}
    save_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 30-frame contact sheets for generated SAM tracklets")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(run(args.config, count=args.count, columns=args.columns, output_override=args.output))


if __name__ == "__main__":
    main()
