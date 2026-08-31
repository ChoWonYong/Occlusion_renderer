import tempfile
import unittest
from pathlib import Path

import yaml

from common.io import load_json, save_json
from pool.filter_detector_tracklet_pool import (
    filter_tracklet,
    longest_passing_window,
    run,
)
from common.config import load_config


def _record(tracklet_id: int, confidences: list[float], category: str = "car") -> dict:
    return {
        "id": tracklet_id,
        "category": category,
        "source_dataset": "KITTI",
        "sequence": "0000",
        "source_track_id": 4,
        "length": len(confidences),
        "frames": [
            {
                "offset": index,
                "sample_index": index,
                "frame_index": index,
                "detector_confidence": confidence,
                "file_name": f"rgba/{tracklet_id}_{index}.png",
            }
            for index, confidence in enumerate(confidences)
        ],
    }


class ConfidenceFilterTest(unittest.TestCase):
    def test_project_default_is_automatic_and_confidence_filtered(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config, _ = load_config(root / "configs" / "default.yaml")
        quality = config["detector_tracklet_pool"]["quality_filter"]
        self.assertTrue(quality["enabled"])
        self.assertEqual(float(quality["detector_confidence_min"]), 0.60)
        self.assertEqual(
            config["tracklet_synthesis"]["pool_sources"],
            ["../artifacts/phase2/codetr_sam3_tracklets_conf60"],
        )
        self.assertEqual(config["tracklet_synthesis"]["paste_jitter"]["mode"], "random")

    def test_longest_window_uses_first_run_on_tie(self) -> None:
        self.assertEqual(longest_passing_window([True, True, False, True, True]), (0, 2))

    def test_accepted_tracklet_keeps_only_longest_run(self) -> None:
        decision, filtered = filter_tracklet(
            _record(7, [0.8, 0.2, 0.7, 0.9, 0.8]),
            threshold=0.6,
            min_frames=3,
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["scenario"], "accepted_trimmed")
        self.assertEqual(decision["longest_passing_start"], 2)
        self.assertIsNotNone(filtered)
        assert filtered is not None
        self.assertEqual(filtered["length"], 3)
        self.assertEqual([frame["offset"] for frame in filtered["frames"]], [0, 1, 2])
        self.assertEqual([frame["raw_offset"] for frame in filtered["frames"]], [2, 3, 4])
        self.assertFalse(filtered["quality_filter"]["gt_used"])

    def test_rejected_fragmented_tracklet_stays_in_decisions(self) -> None:
        decision, filtered = filter_tracklet(
            _record(8, [0.8, 0.8, 0.1, 0.9, 0.9]),
            threshold=0.6,
            min_frames=3,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["scenario"], "rejected_fragmented")
        self.assertIsNone(filtered)

    def test_run_preserves_raw_pool_and_writes_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            output_dir = root / "filtered"
            save_json(
                raw_dir / "tracklets.json",
                {
                    "tracklets": [
                        _record(1, [0.8, 0.9, 0.7]),
                        _record(2, [0.8, 0.1, 0.8]),
                    ]
                },
            )
            config = {
                "detector_tracklet_pool": {
                    "min_frames": 2,
                    "quality_filter": {
                        "enabled": True,
                        "detector_confidence_min": 0.6,
                        "min_frames": 2,
                        "input_pool_dir": str(raw_dir),
                        "output_dir": str(output_dir),
                    },
                }
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = run(config_path)
            self.assertEqual(summary["tracklets_before"], 2)
            self.assertEqual(summary["tracklets_after"], 1)
            self.assertEqual(summary["tracklets_rejected"], 1)
            self.assertTrue((raw_dir / "tracklets.json").is_file())
            decisions = load_json(output_dir / "decisions.json")["tracklets"]
            self.assertEqual(len(decisions), 2)
            self.assertFalse(decisions[1]["accepted"])


if __name__ == "__main__":
    unittest.main()
