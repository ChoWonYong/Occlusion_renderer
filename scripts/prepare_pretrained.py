"""Fetch the COCO-pretrained YOLOX-X checkpoint used as the fine-tuning start.

The weights are not ours: YOLOX-X is trained and released by YOLOX
(Megvii-BaseDetection, Apache-2.0). We only fine-tune from them, and follow
ByteTrack (Yifu Zhang, MIT) in placing the file under <ByteTrack>/pretrained.

ByteTrack's README routes to the YOLOX v0.1.0 model zoo, but that asset is a
*different* file (793,388,371 bytes) from the one every reported run started
from. The md5 below pins the exact 0.1.1rc0 asset actually used, so a checkpoint
swap can never pass silently.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

from common.config import config_path, load_config


YOLOX_X_COCO_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"
YOLOX_X_COCO_MD5 = "c58e4a3a710e3a464bf472aaf6f4891a"
YOLOX_X_COCO_BYTES = 793_463_373


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official COCO-pretrained YOLOX-X checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, path = load_config(args.config)
    destination = config_path(config, path, "paths", "coco_pretrained_yolox_x")
    if destination.exists() and not args.force:
        actual = file_md5(destination)
        if actual != YOLOX_X_COCO_MD5:
            raise RuntimeError(
                f"{destination} is not the checkpoint the reported runs used "
                f"(md5 {actual}, expected {YOLOX_X_COCO_MD5}); "
                "re-download with --force"
            )
        print(f"already exists and matches md5 {YOLOX_X_COCO_MD5}: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {YOLOX_X_COCO_URL}")
    urllib.request.urlretrieve(YOLOX_X_COCO_URL, partial)
    size, actual = partial.stat().st_size, file_md5(partial)
    if actual != YOLOX_X_COCO_MD5:
        partial.unlink()
        raise RuntimeError(
            f"downloaded checkpoint does not match the pinned YOLOX-X weights: "
            f"md5 {actual} ({size} bytes), expected {YOLOX_X_COCO_MD5} ({YOLOX_X_COCO_BYTES} bytes)"
        )
    partial.replace(destination)
    print(f"saved: {destination} (md5 {YOLOX_X_COCO_MD5})")


if __name__ == "__main__":
    main()

