from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from common.config import config_path, load_config, resolve_path
from common.io import save_json


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    add("python>=3.10", sys.version_info >= (3, 10), platform.python_version())
    for module in ["numpy", "PIL", "yaml", "cv2", "torch", "pycocotools"]:
        add(f"python:{module}", importlib.util.find_spec(module) is not None, "installed" if importlib.util.find_spec(module) else "missing")
    for module in ["boxmot", "yolox"]:
        add(f"python:{module}", importlib.util.find_spec(module) is not None, "installed" if importlib.util.find_spec(module) else "missing")
    sam3_python = config_path(config, path, "paths", "sam3_python")
    add("SAM3 Python", sam3_python.is_file(), str(sam3_python), True)
    if sam3_python.is_file():
        runtime = subprocess.run(
            [
                str(sam3_python),
                "-c",
                (
                    "import sys,torch,sam3; "
                    "print(f'python={sys.version_info.major}.{sys.version_info.minor} '"
                    "f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        detail = (runtime.stdout or runtime.stderr).strip()
        add("SAM3 runtime", runtime.returncode == 0 and "cuda=True" in detail, detail, True)

    path_checks = [
        ("ByteTrack repo", config_path(config, path, "paths", "bytetrack_repo"), True),
        ("BoxMOT repo", config_path(config, path, "paths", "boxmot_repo"), True),
        ("TrackEval repo", config_path(config, path, "paths", "trackeval_repo"), True),
        ("KITTI Tracking images", config_path(config, path, "paths", "kitti_tracking") / "training" / "image_02", True),
        ("KITTI Tracking labels", config_path(config, path, "paths", "kitti_tracking") / "training" / "label_02", True),
        ("MOT17 train", config_path(config, path, "paths", "mot17") / "train", True),
        ("SAM3 source", config_path(config, path, "paths", "sam3_repo") / "sam3", True),
        ("SAM3 checkpoint", config_path(config, path, "paths", "sam3_checkpoint"), True),
        ("SAM3 BPE vocabulary", config_path(config, path, "paths", "sam3_bpe"), True),
        ("COCO YOLOX-X checkpoint", config_path(config, path, "paths", "coco_pretrained_yolox_x"), True),
    ]
    # Only the crop-policy ablation config declares MOTS ground truth.
    ablation = config.get("sam3_context_ablation")
    if ablation:
        path_checks.append(
            ("KITTI MOTS ground truth", resolve_path(path.parent, ablation["mots_root"]), True)
        )
    for name, checked_path, required in path_checks:
        add(name, checked_path.exists(), str(checked_path), required)
    result = {
        "ok": all(check["ok"] for check in checks if check["required"]),
        "config": str(path),
        "checks": checks,
    }
    output = resolve_path(path.parent, config["paths"]["artifacts"]) / "preflight.json"
    save_json(output, result)
    result["output"] = str(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the environment, repos, data, and checkpoints")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    args = parser.parse_args()
    result = run(args.config)
    for check in result["checks"]:
        marker = "OK" if check["ok"] else "MISSING"
        print(f"[{marker:7}] {check['name']}: {check['detail']}")
    print(f"report: {result['output']}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
