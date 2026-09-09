from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

from common.config import config_path, load_config


YOLOX_X_COCO_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official COCO-pretrained YOLOX-X checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, path = load_config(args.config)
    destination = config_path(config, path, "paths", "coco_pretrained_yolox_x")
    if destination.exists() and not args.force:
        print(f"already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {YOLOX_X_COCO_URL}")
    urllib.request.urlretrieve(YOLOX_X_COCO_URL, partial)
    if partial.stat().st_size < 100_000_000:
        raise RuntimeError(f"downloaded checkpoint is unexpectedly small: {partial.stat().st_size} bytes")
    partial.replace(destination)
    print(f"saved: {destination}")


if __name__ == "__main__":
    main()

