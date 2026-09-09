import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common.io import load_json
from synth.event_pipeline import run


class EventPipelineIntegrationTest(unittest.TestCase):
    def test_end_to_end_event_synthesis_with_amodal_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kitti = root / "KITTI"
            images = kitti / "training" / "image_02" / "0000"
            labels = kitti / "training" / "label_02"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            # 40 frames, 200x100, one stationary pedestrian victim (COCO bbox [80,20,40,40]).
            rows = []
            for frame in range(40):
                Image.fromarray(np.full((100, 200, 3), 90, dtype=np.uint8)).save(images / f"{frame:06d}.png")
                rows.append(f"{frame} 5 Pedestrian 0.0 0 0.0 80 20 120 60 1.7 0.5 0.5 0 0 10 0")
            (labels / "0000.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

            # MOT17-style person tracklet whose crop sweeps across the victim.
            pool = root / "pool"
            frames = []
            for offset in range(20):
                cf = 0.3 + offset * 0.02
                relative = Path("tracklet_000001") / f"{offset:02d}.png"
                patch = np.zeros((100, 50, 4), dtype=np.uint8)
                patch[..., :3] = (200, 40, 40)
                patch[..., 3] = 255
                (pool / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(patch).save(pool / relative)
                frames.append(
                    {
                        "offset": offset,
                        "source_frame": offset,
                        "crop_bbox_xywh": [cf * 1000 - 50, 400, 100, 200],
                        "source_image_size": [1000, 1000],
                        "file_name": str(relative),
                    }
                )
            (pool / "tracklets.json").write_text(
                json.dumps(
                    {
                        "tracklets": [
                            {
                                "id": 1,
                                "category": "person",
                                "source": "MOT17",
                                "sequence": "M-02",
                                "source_track_id": 1,
                                "length": 20,
                                "frames": frames,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            split = root / "split.json"
            split.write_text(json.dumps({"train_sequences": ["0000"]}), encoding="utf-8")
            config = {
                "seed": 0,
                "paths": {"kitti_tracking": str(kitti)},
                "classes": {"kitti_map": {"Pedestrian": "person"}},
                "split": {"output": str(split)},
                "tracklet_synthesis": {
                    "output_dir": str(root / "out"),
                    "pool_sources": [str(pool)],
                    "max_sequences": 1,
                    "class_ratio": {"person": 1.0},
                    "class_height_range": {
                        "car": [0.9, 1.1],
                        "truck": [1.7, 1.9],
                        "person": [1.1, 1.3],
                        "bicycle": [0.8, 1.0],
                    },
                    "max_lateral_offset_fraction": 1.15,
                    "placement_draws": 64,
                    "placement_max_tracklets": 4,
                    "victim_events_per_100_frames": 3.0,
                    "tracklet_min_frames": 20,
                    "effective_event_frames": [8, 20],
                    "peak_rho_max": 1.00,
                    "event_gate_floor": 0.20,
                    "event_end_rho_max": 0.05,
                    "victim_detector_bbox_policy": "amodal_original",
                    "victim_min_area": 100,
                    "victim_max_base_occlusion": 0,
                    "max_occluders_per_victim": 2,
                    "blend_method": "none",
                    "paste_jitter": {"mode": "random", "preset": "mid"},
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            summary = run(config_path)
            self.assertEqual(summary["sequences"], 1)
            self.assertTrue(summary["qc_ok"])
            self.assertEqual(summary["synthetic_tracklets"], 1)
            self.assertEqual(summary["detector_bbox_policy"], "amodal_original")
            self.assertEqual(summary["jitter_mode"], "random")

            tracks = load_json(root / "out" / "occluder_tracks.json")
            self.assertEqual(tracks[0]["category"], "person")
            self.assertEqual(tracks[0]["exposure_length"], 20)
            # difficulty is derived, so only the gate is asserted, not a band
            self.assertGreaterEqual(tracks[0]["achieved_peak"], 0.20)
            self.assertLessEqual(tracks[0]["achieved_peak"], 1.0)
            self.assertIn(tracks[0]["band"], {"mild", "moderate", "heavy"})
            # the sampled size stays inside the class's physical range
            self.assertTrue(1.1 <= tracks[0]["height_factor"] <= 1.3)
            self.assertTrue(abs(tracks[0]["lateral_offset_fraction"]) <= 1.15)

            dataset = load_json(root / "out" / "annotations.json")
            # victim annotations keep the amodal (original) box as the detector bbox
            victim = next(
                a for a in dataset["annotations"]
                if a.get("detector_bbox_policy") == "amodal_original"
                and float(a.get("occlusion_ratio", 0)) > 0
            )
            self.assertEqual(victim["bbox"], victim["amodal_bbox"])
            # The visible *bbox* can still equal the amodal one when the occluder
            # cuts through the middle and leaves pixels on both sides, so compare
            # areas rather than boxes. ``area`` tracks the detector target (the
            # amodal box); ``visible_area`` is what the paste actually left.
            self.assertLess(int(victim["visible_area"]), int(victim["area"]))

            # no victim is dropped from training, however heavily it is covered:
            # MOTDataset skips iscrowd != 0 and area == 0, and those become
            # negatives rather than ignore regions
            for annotation in dataset["annotations"]:
                if annotation.get("synthetic_occluder") is True:
                    continue
                self.assertEqual(annotation["iscrowd"], 0)
                self.assertGreater(int(annotation["area"]), 0)

            # at least one event was derived
            events = load_json(root / "out" / "events.json")
            self.assertGreaterEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
