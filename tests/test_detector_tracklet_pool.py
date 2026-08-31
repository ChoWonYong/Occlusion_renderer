import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common.io import load_json, save_json
from pool.build_detector_tracklet_pool import build, merge_shards
from segment.base import Instance


class _FakeSegmenter:
    precision = "float16"

    def detect_and_mask(self, image: np.ndarray, prompts: list[str]) -> list[Instance]:
        return [
            Instance(
                bbox=[0.0, 0.0, float(image.shape[1]), float(image.shape[0])],
                mask=np.ones(image.shape[:2], dtype=np.uint8),
                category=prompts[0],
                score=0.95,
            )
        ]


class DetectorTrackletPoolTest(unittest.TestCase):
    def test_detector_boxes_drive_crop_and_sam_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images"
            image_dir.mkdir()
            frames = []
            for index in range(3):
                path = image_dir / f"{index:06d}.png"
                Image.new("RGB", (30, 20), color=(100, 100, 100)).save(path)
                frames.append(
                    {
                        "offset": index,
                        "source_dataset": "KITTI",
                        "sequence": "0000",
                        "source_fps": 10,
                        "target_fps": 10,
                        "sample_index": index,
                        "frame_index": index,
                        "source_image": str(path),
                        "source_image_size": [30, 20],
                        "source_track_id": 7,
                        "category": "car",
                        "sam3_prompt": "car",
                        "source_bbox_xywh": [5, 4, 12, 8],
                        "detector_confidence": 0.8,
                        "selection_used_gt": False,
                    }
                )
            candidate_dir = root / "candidates"
            save_json(
                candidate_dir / "tracklets.json",
                {
                    "tracklets": [
                        {
                            "candidate_id": 1,
                            "category": "car",
                            "source_dataset": "KITTI",
                            "sequence": "0000",
                            "source_track_id": 7,
                            "source_fps": 10,
                            "target_fps": 10,
                            "length": 3,
                            "frames": frames,
                        }
                    ]
                },
            )
            config = {
                "seed": 0,
                "paths": {"sam3_checkpoint": "unused", "sam3_bpe": "unused"},
                "detector_tracklet_pool": {
                    "candidate_output_dir": str(candidate_dir),
                    "output_dir": str(root / "pool"),
                    "min_frames": 3,
                    "max_frames": 100,
                    "min_mask_area": 1,
                    "sam3_context_padding": 0,
                    "sam3_min_detector_iou": 0.1,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = build(config_path, segmenter=_FakeSegmenter())
            self.assertEqual(summary["tracklets"], 1)
            pool = load_json(summary["output"])
            tracklet = pool["tracklets"][0]
            self.assertEqual(tracklet["frames"][0]["source_bbox_xywh"], [5, 4, 12, 8])
            self.assertIn("sam3_detector_match_iou", tracklet["frames"][0])
            self.assertNotIn("sam3_gt_match_iou", tracklet["frames"][0])
            self.assertFalse(tracklet["segmentation"]["gt_used"])
            with Image.open(root / "pool" / "tracklet_000001" / "00.png") as crop:
                self.assertEqual(crop.size, (12, 8))

    def test_parallel_shards_merge_in_original_candidate_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pool"
            config = {
                "seed": 7,
                "detector_tracklet_pool": {
                    "output_dir": str(output),
                    "min_frames": 1,
                    "max_frames": 100,
                    "max_tracklets": 2,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            def record(candidate_order: int, candidate_id: int, category: str) -> dict:
                return {
                    "id": candidate_id,
                    "source_candidate_id": candidate_id,
                    "candidate_order": candidate_order,
                    "shard_index": candidate_order % 2,
                    "category": category,
                    "length": 1,
                    "frames": [{"file_name": f"candidate_{candidate_id}/00.png"}],
                }

            shard_root = output / "_shards"
            save_json(
                shard_root / "shard_00.json",
                {
                    "shard_index": 0,
                    "num_shards": 2,
                    "seed": 7,
                    "assigned_candidates": 2,
                    "rejected_during_segmentation": 0,
                    "tracklets": [record(2, 30, "car"), record(0, 10, "person")],
                },
            )
            save_json(
                shard_root / "shard_01.json",
                {
                    "shard_index": 1,
                    "num_shards": 2,
                    "seed": 7,
                    "assigned_candidates": 1,
                    "rejected_during_segmentation": 0,
                    "tracklets": [record(1, 20, "car")],
                },
            )
            summary = merge_shards(config_path, num_shards=2)
            self.assertEqual(summary["candidates"], 3)
            self.assertEqual(summary["accepted_candidates_before_cap"], 3)
            self.assertEqual(summary["parallel_shards"], 2)
            merged = load_json(summary["output"])
            self.assertEqual(
                [record["source_candidate_id"] for record in merged["tracklets"]],
                [10, 20],
            )
            self.assertEqual([record["id"] for record in merged["tracklets"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
