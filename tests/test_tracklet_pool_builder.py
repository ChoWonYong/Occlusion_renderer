import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common.io import load_json
from pool.build_tracklet_pool import build, build_variable, select_matching_instance
from segment.base import Instance


class _FakeSegmenter:
    def detect_and_mask(self, image: np.ndarray, prompts: list[str]) -> list[Instance]:
        self.last_shape = image.shape
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        mask[[0, -1], :] = 0
        mask[:, [0, -1]] = 0
        return [
            Instance(
                bbox=[0.0, 0.0, float(image.shape[1]), float(image.shape[0])],
                mask=mask,
                category="human",
                score=0.9,
            )
        ]


class TrackletPoolBuilderTest(unittest.TestCase):
    def test_builds_30_rgba_crops_from_one_strict_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "MOT17" / "train" / "MOT17-02-FRCNN"
            (sequence / "gt").mkdir(parents=True)
            (sequence / "img1").mkdir()
            (sequence / "seqinfo.ini").write_text(
                "[Sequence]\nimWidth=40\nimHeight=30\nimExt=.jpg\n", encoding="utf-8"
            )
            rows = []
            for frame in range(1, 31):
                rows.append(f"{frame},7,10,5,12,20,1,1,0.8")
                Image.fromarray(np.full((30, 40, 3), 120, dtype=np.uint8)).save(
                    sequence / "img1" / f"{frame:06d}.jpg"
                )
            (sequence / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            output = root / "pool"
            config = {
                "seed": 0,
                "paths": {"mot17": str(root / "MOT17")},
                "tracklet_pool": {
                    "output_dir": str(output),
                    "detector": "FRCNN",
                    "visibility_min": 0.8,
                    "length": 30,
                    "stride": 30,
                    "max_tracklets": 1,
                    "min_mask_area": 10,
                    "sam3_text_prompt": "human",
                    "sam3_context_padding": 0.0,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            fake = _FakeSegmenter()
            summary = build(config_path, segmenter=fake)
            self.assertEqual(summary["tracklets"], 1)
            metadata = load_json(output / "tracklets.json")
            self.assertEqual(len(metadata["tracklets"][0]["frames"]), 30)
            self.assertEqual(fake.last_shape, (20, 12, 3))
            self.assertTrue((output / "tracklet_000001" / "29.png").is_file())
            self.assertEqual(metadata["tracklets"][0]["segmentation"]["backend"], "sam3")
            self.assertEqual(metadata["tracklets"][0]["segmentation"]["prompt"], "human")

    def test_builds_fps_aware_variable_length_tracklet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "MOT17" / "train" / "MOT17-02-FRCNN"
            (sequence / "gt").mkdir(parents=True)
            (sequence / "img1").mkdir()
            # 30 fps source, 90 consecutive frames -> 30 target frames at 10 fps.
            (sequence / "seqinfo.ini").write_text(
                "[Sequence]\nimWidth=40\nimHeight=30\nimExt=.jpg\nframeRate=30\n",
                encoding="utf-8",
            )
            rows = []
            for frame in range(1, 91):
                rows.append(f"{frame},7,10,5,12,20,1,1,0.9")
                Image.fromarray(np.full((30, 40, 3), 120, dtype=np.uint8)).save(
                    sequence / "img1" / f"{frame:06d}.jpg"
                )
            (sequence / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            output = root / "pool"
            config = {
                "seed": 0,
                "paths": {"mot17": str(root / "MOT17")},
                "tracklet_pool": {
                    "output_dir": str(output),
                    "detector": "FRCNN",
                    "visibility_min": 0.8,
                    "target_fps": 10,
                    "min_frames": 30,
                    "visibility_substitution_window": 1,
                    "max_per_identity": 2,
                    "max_tracklets": 5,
                    "min_mask_area": 10,
                    "sam3_text_prompt": "human",
                    "sam3_context_padding": 0.0,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            summary = build_variable(config_path, segmenter=_FakeSegmenter())
            self.assertEqual(summary["mode"], "fps_aware_variable_length")
            self.assertEqual(summary["tracklets"], 1)
            metadata = load_json(output / "tracklets.json")
            tracklet = metadata["tracklets"][0]
            self.assertEqual(tracklet["length"], 30)
            self.assertEqual(tracklet["source_fps"], 30.0)
            self.assertEqual(tracklet["target_fps"], 10.0)
            self.assertEqual(len(tracklet["frames"]), 30)
            # every-3rd source frame, starting at frame 1
            self.assertEqual(
                [frame["source_frame"] for frame in tracklet["frames"][:3]], [1, 4, 7]
            )
            self.assertTrue((output / "tracklet_000001" / "29.png").is_file())

    def test_matches_human_instance_by_mot_bbox_iou(self) -> None:
        shape = (30, 40)
        target = Instance([8, 5, 10, 20], np.ones(shape, dtype=np.uint8), "human", 0.7)
        neighbor = Instance([22, 5, 10, 20], np.ones(shape, dtype=np.uint8), "human", 0.95)
        matched = select_matching_instance([neighbor, target], [7, 4, 12, 22], min_iou=0.3)
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertIs(matched[0], target)


if __name__ == "__main__":
    unittest.main()
