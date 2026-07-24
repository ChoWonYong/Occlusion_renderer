import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common.io import load_json
from synth.tracklet_pipeline import run


class TrackletPipelineIntegrationTest(unittest.TestCase):
    def test_two_tracklets_remain_present_for_exactly_30_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kitti = root / "KITTI"
            images = kitti / "training" / "image_02" / "0000"
            labels = kitti / "training" / "label_02"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (labels / "0000.txt").write_text("", encoding="utf-8")
            for frame in range(30):
                Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(images / f"{frame:06d}.png")

            pool = root / "pool"
            records = []
            for tracklet_id, color in [(1, (255, 0, 0)), (2, (0, 255, 0))]:
                frames = []
                for offset in range(30):
                    relative = Path(f"tracklet_{tracklet_id:06d}") / f"{offset:02d}.png"
                    rgba = np.zeros((10 + tracklet_id, 8 + tracklet_id, 4), dtype=np.uint8)
                    rgba[..., :3] = color
                    rgba[..., 3] = 255
                    (pool / relative).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgba).save(pool / relative)
                    frames.append(
                        {
                            "offset": offset,
                            "source_frame": offset + 1,
                            "source_bbox_xywh": [10, 10, rgba.shape[1], rgba.shape[0]],
                            "crop_bbox_xywh": [10, 10, rgba.shape[1], rgba.shape[0]],
                            "source_image_size": [60, 40],
                            "visibility": 0.9,
                            "file_name": str(relative),
                        }
                    )
                records.append(
                    {
                        "id": tracklet_id,
                        "length": 30,
                        "sequence": "MOT17-02-FRCNN",
                        "source_track_id": tracklet_id,
                        "frames": frames,
                    }
                )
            (pool / "tracklets.json").write_text(
                json.dumps({"tracklets": records}), encoding="utf-8"
            )
            split = root / "split.json"
            split.write_text(json.dumps({"train_sequences": ["0000"]}), encoding="utf-8")
            config = {
                "seed": 3,
                "paths": {"kitti_tracking": str(kitti)},
                "classes": {"kitti_map": {"Pedestrian": "person"}},
                "split": {"output": str(split)},
                "tracklet_pool": {"output_dir": str(pool)},
                "tracklet_synthesis": {
                    "output_dir": str(root / "output"),
                    "max_sequences": 1,
                    "tracklets_per_sequence": [2, 2],
                    "boundary_shift_count": 2,
                    "position_mode": "normalized_source",
                    "blend_method": "none",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            summary = run(config_path)
            self.assertEqual(summary["synthetic_tracklets"], 2)
            tracks = load_json(root / "output" / "occluder_tracks.json")
            self.assertEqual([len(track["frames"]) for track in tracks], [30, 30])
            self.assertEqual({track["boundary_shift"] for track in tracks}, {"left", "right"})


if __name__ == "__main__":
    unittest.main()
