import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from common.io import load_json
from mining.detected_tracklets import build_track_runs, mine_detection_tracklets


class _FakeTracker:
    def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        tracks = []
        for detection in detections:
            x1, y1, x2, y2, score, cls = detection
            tracks.append([x1, y1, x2, y2, 100 + int(cls), score, cls])
        return np.asarray(tracks, dtype=np.float32).reshape(-1, 7)


def _tracker_factory(settings, class_names):
    assert class_names == ["car", "person"]
    return _FakeTracker()


class DetectedTrackletsTest(unittest.TestCase):
    def test_gap_breaks_target_fps_run(self) -> None:
        candidates = [
            {
                "source_dataset": "MOT17", "sequence": "S", "source_track_id": 1,
                "category": "person", "sample_index": index,
            }
            for index in (0, 1, 2, 4, 5)
        ]
        runs = build_track_runs(candidates, min_frames=3, max_frames=100, max_per_identity=2)
        self.assertEqual([[item["sample_index"] for item in run] for run in runs], [[0, 1, 2]])

    def test_mines_two_classes_without_quality_filter_or_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "tracker": {"min_conf": 0.1, "track_thresh": 0.45},
                "codetr_detector": {"output_dir": str(root / "detections")},
                "detector_tracklet_pool": {
                    "candidate_output_dir": str(root / "candidates"),
                    "min_frames": 3,
                    "max_frames": 100,
                    "max_per_identity": 2,
                    "min_bbox_area": 1,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            frames = []
            for index in range(3):
                frames.append(
                    {
                        "sample_index": index,
                        "frame_index": index * 3 + 1,
                        "source_image": "unused.jpg",
                        "source_image_size": [30, 20],
                        "detections": [
                            {"bbox_xyxy": [1, 1, 11, 9], "score": 0.9, "category": "car", "coco_class_name": "car"},
                            {"bbox_xyxy": [20, 1, 25, 15], "score": 0.7, "category": "person", "coco_class_name": "person"},
                        ],
                    }
                )
            payload = {
                "sequences": [
                    {"source_dataset": "MOT17", "sequence": "S", "source_fps": 30, "target_fps": 10, "frames": frames}
                ]
            }
            summary = mine_detection_tracklets(
                config_path,
                payloads=[payload],
                tracker_factory=_tracker_factory,
                image_loader=lambda _: np.zeros((20, 30, 3), dtype=np.uint8),
            )
            self.assertEqual(summary["tracklets"], 2)
            self.assertEqual(summary["category_counts"], {"car": 1, "person": 1})
            self.assertEqual(summary["quality_filter"], "deferred_to_phase2")
            metadata = load_json(summary["output"])
            self.assertFalse(metadata["selection"]["gt_used"])
            self.assertNotIn("visibility", metadata["tracklets"][0]["frames"][0])


if __name__ == "__main__":
    unittest.main()
