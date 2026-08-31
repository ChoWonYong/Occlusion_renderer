import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data.detection_sources import (
    collect_kitti_sources,
    collect_mot17_sources,
    sample_frame_paths,
)


class DetectionSourcesTest(unittest.TestCase):
    @staticmethod
    def _image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), color=(100, 100, 100)).save(path)

    def test_resampling_is_gt_free_and_round_half_up(self) -> None:
        paths = [Path(f"{frame:06d}.jpg") for frame in range(1, 91)]
        sampled = sample_frame_paths(paths, source_fps=30.0, target_fps=10.0)
        self.assertEqual([int(path.stem) for path in sampled[:3]], [1, 4, 7])
        self.assertEqual(len(sampled), 30)

    def test_kitti_uses_only_crop_allowed_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sequence in ("0000", "0001"):
                self._image(root / "KITTI" / "training" / "image_02" / sequence / "000000.png")
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "train_sequences": ["0000"],
                        "eval_sequences": ["0001"],
                        "crop_allowed_sequences": ["0000"],
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "paths": {"kitti_tracking": str(root / "KITTI")},
                "split": {"output": str(split_path)},
            }
            sequences = collect_kitti_sources(config, root / "config.yaml")
            self.assertEqual([item["sequence"] for item in sequences], ["0000"])
            self.assertNotIn("label", sequences[0]["frames"][0])

    def test_mot17_resamples_images_without_gt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "MOT17" / "train" / "MOT17-02-FRCNN"
            sequence.mkdir(parents=True)
            (sequence / "seqinfo.ini").write_text(
                "[Sequence]\nframeRate=30\nimExt=.jpg\n", encoding="utf-8"
            )
            for frame in range(1, 91):
                self._image(sequence / "img1" / f"{frame:06d}.jpg")
            config = {
                "paths": {"mot17": str(root / "MOT17")},
                "detector_tracklet_pool": {
                    "mot17_detector_view": "FRCNN",
                    "target_fps": 10,
                },
            }
            sequences = collect_mot17_sources(config, root / "config.yaml")
            self.assertEqual(len(sequences[0]["frames"]), 30)
            self.assertFalse((sequence / "gt").exists())


if __name__ == "__main__":
    unittest.main()
