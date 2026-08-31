import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from mining.codetr import (
    aspect_fallback_category,
    export_detections,
    map_detector_category,
)


class _FakeDetector:
    def detect(self, image: np.ndarray) -> list[dict]:
        return [
            {"bbox_xyxy": [1, 1, 11, 9], "score": 0.9, "coco_class_name": "car"},
            {"bbox_xyxy": [5, 1, 9, 9], "score": 0.8, "coco_class_name": "person"},
            {"bbox_xyxy": [1, 1, 8, 8], "score": 0.7, "coco_class_name": "bicycle"},
            {"bbox_xyxy": [1, 1, 8, 8], "score": 0.001, "coco_class_name": "truck"},
        ]


class CoDetrExportTest(unittest.TestCase):
    def test_class_mapping_prefers_detector_and_corrects_aspect_fallback(self) -> None:
        mapping = {"person": "person", "car": "car", "bus": "car", "truck": "car"}
        self.assertEqual(map_detector_category("truck", [0, 0, 8, 4], mapping, use_aspect_fallback=True), "car")
        self.assertIsNone(map_detector_category("bicycle", [0, 0, 4, 8], mapping, use_aspect_fallback=True))
        self.assertEqual(aspect_fallback_category([0, 0, 4, 8]), "person")
        self.assertEqual(aspect_fallback_category([0, 0, 8, 4]), "car")

    def test_low_confidence_never_overrides_detector_class_with_aspect(self) -> None:
        mapping = {"person": "person", "car": "car"}
        self.assertEqual(
            map_detector_category(
                "car",
                [0, 0, 4, 12],
                mapping,
                use_aspect_fallback=True,
                detector_confidence=0.11,
            ),
            "car",
        )

    def test_exports_only_selected_coco_classes_without_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "KITTI" / "training" / "image_02" / "0000"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(image_dir / "000000.png")
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
            config = {
                "paths": {"kitti_tracking": str(root / "KITTI")},
                "split": {"output": str(split)},
                "codetr_detector": {
                    "output_dir": str(root / "detections"),
                    "score_threshold": 0.01,
                    "coco_class_map": {"person": "person", "car": "car", "bus": "car", "truck": "car"},
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = export_detections(config_path, "kitti", backend=_FakeDetector())
            self.assertEqual(summary["detections"], 2)
            payload = json.loads(Path(summary["output"]).read_text(encoding="utf-8"))
            categories = [item["category"] for item in payload["sequences"][0]["frames"][0]["detections"]]
            self.assertEqual(categories, ["car", "person"])
            self.assertFalse(payload["detector"]["gt_used"])


if __name__ == "__main__":
    unittest.main()
