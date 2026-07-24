import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common.io import load_json
from pool.build_kitti_sam3_pool import (
    build,
    build_track_runs,
    build_tracklets,
    inventory,
    validate_crop_split,
)
from scripts.kitti_sam3_track_video import select_consecutive_window
from segment.base import Instance


class _FakeSegmenter:
    precision = "float16"

    def detect_and_mask(self, image: np.ndarray, prompts: list[str]) -> list[Instance]:
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        return [
            Instance(
                bbox=[0.0, 0.0, float(image.shape[1]), float(image.shape[0])],
                mask=mask,
                category=prompts[0],
                score=0.95,
            )
        ]


class KittiSam3PoolTest(unittest.TestCase):
    def test_inventory_and_build_never_use_eval_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kitti = root / "KITTI" / "training"
            for sequence in ("0000", "0001"):
                images = kitti / "image_02" / sequence
                images.mkdir(parents=True)
                Image.fromarray(np.full((40, 60, 3), 100, dtype=np.uint8)).save(
                    images / "000000.png"
                )
            labels = kitti / "label_02"
            labels.mkdir()
            row = "0 7 Car 0.0 0 0.0 10 5 40 35 1.5 1.6 4.0 0 0 10 0"
            (labels / "0000.txt").write_text(row + "\n", encoding="utf-8")
            (labels / "0001.txt").write_text(row + "\n", encoding="utf-8")
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "train_sequences": ["0000"],
                        "eval_sequences": ["0001"],
                        "crop_allowed_sequences": ["0000"],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "pool"
            config = {
                "seed": 0,
                "paths": {"kitti_tracking": str(root / "KITTI")},
                "classes": {"kitti_map": {"Car": "car"}},
                "split": {"output": str(split)},
                "kitti_sam3_pool": {
                    "output_dir": str(output),
                    "prompts": {"Car": "car"},
                    "max_occlusion": 0,
                    "max_truncation": 0.2,
                    "min_bbox_area": 100,
                    "min_mask_area": 10,
                    "context_padding": 0.0,
                    "min_gt_iou": 0.1,
                    "max_instances": 1,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            summary = inventory(config_path)
            self.assertEqual(summary["candidates"], 1)
            self.assertEqual(summary["leaked_eval_sequences"], [])
            result = build(config_path, segmenter=_FakeSegmenter())
            self.assertEqual(result["instances"], 1)
            self.assertEqual(result["eval_leakage"], [])
            pool = load_json(output / "pool.json")
            self.assertEqual(pool["instances"][0]["sequence"], "0000")
            self.assertEqual(pool["instances"][0]["sam3_prompt"], "car")
            self.assertTrue((output / pool["instances"][0]["file_name"]).is_file())

    def test_track_runs_split_on_gaps_and_truncate_to_max(self) -> None:
        candidates = (
            [{"sequence": "0000", "track_id": 7, "frame_index": f, "category": "car"} for f in range(0, 40)]
            + [{"sequence": "0000", "track_id": 7, "frame_index": f, "category": "car"} for f in range(45, 70)]
            + [{"sequence": "0000", "track_id": 9, "frame_index": f, "category": "car"} for f in range(0, 10)]
        )
        runs = build_track_runs(candidates, min_frames=30, max_frames=35)
        # track 7: run 0..39 -> truncated to 35; run 45..69 (25 frames) dropped (<30);
        # track 9: 10 frames dropped.
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]), 35)
        self.assertEqual([runs[0][0]["frame_index"], runs[0][-1]["frame_index"]], [0, 34])

    def test_builds_kitti_multiclass_tracklet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kitti = root / "KITTI" / "training"
            images = kitti / "image_02" / "0000"
            images.mkdir(parents=True)
            labels = kitti / "label_02"
            labels.mkdir()
            rows = []
            for frame in range(0, 30):
                Image.fromarray(np.full((40, 60, 3), 100, dtype=np.uint8)).save(
                    images / f"{frame:06d}.png"
                )
                rows.append(f"{frame} 7 Car 0.0 0 0.0 10 5 40 35 1.5 1.6 4.0 0 0 10 0")
            (labels / "0000.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "train_sequences": ["0000"],
                        "eval_sequences": ["0001"],
                        "crop_allowed_sequences": ["0000"],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "static"
            tracklet_output = root / "tracklets"
            config = {
                "seed": 0,
                "paths": {"kitti_tracking": str(root / "KITTI")},
                "classes": {"kitti_map": {"Car": "car"}},
                "split": {"output": str(split)},
                "kitti_sam3_pool": {
                    "output_dir": str(output),
                    "tracklet_output_dir": str(tracklet_output),
                    "prompts": {"Car": "car"},
                    "max_occlusion": 0,
                    "max_truncation": 0.2,
                    "min_bbox_area": 100,
                    "min_mask_area": 10,
                    "context_padding": 0.0,
                    "min_gt_iou": 0.1,
                    "target_fps": 10,
                    "min_frames": 30,
                    "max_frames": 100,
                    "max_tracklets": 5,
                    "max_per_identity": 2,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            summary = build_tracklets(config_path, segmenter=_FakeSegmenter())
            self.assertEqual(summary["mode"], "kitti_multiclass_tracklets")
            self.assertEqual(summary["tracklets"], 1)
            self.assertEqual(summary["eval_leakage"], [])
            self.assertEqual(summary["category_counts"], {"car": 1})
            pool = load_json(tracklet_output / "tracklets.json")
            tracklet = pool["tracklets"][0]
            self.assertEqual(tracklet["length"], 30)
            self.assertEqual(tracklet["category"], "car")
            self.assertEqual(tracklet["sequence"], "0000")
            self.assertEqual(len(tracklet["frames"]), 30)
            self.assertTrue((tracklet_output / tracklet["frames"][29]["file_name"]).is_file())

    def test_rejects_crop_sequence_in_eval_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside train|leakage"):
            validate_crop_split(
                {
                    "train_sequences": ["0000"],
                    "eval_sequences": ["0001"],
                    "crop_allowed_sequences": ["0001"],
                }
            )

    def test_selects_only_exact_consecutive_track_window(self) -> None:
        candidates = [
            {"sequence": "0000", "track_id": 7, "frame_index": frame}
            for frame in (10, 11, 12, 14)
        ]
        selected = select_consecutive_window(
            candidates, sequence="0000", track_id=7, length=3
        )
        self.assertEqual([item["frame_index"] for item in selected], [10, 11, 12])
        with self.assertRaisesRegex(ValueError, "missing requested frames"):
            select_consecutive_window(
                candidates,
                sequence="0000",
                track_id=7,
                length=3,
                start_frame=12,
            )


if __name__ == "__main__":
    unittest.main()
