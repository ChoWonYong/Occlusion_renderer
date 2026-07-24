import unittest

import numpy as np

from eval.run_boxmot import COCO_TO_PROJECT_CLASS, _remap_detection_classes


class EvalMappingTest(unittest.TestCase):
    def test_coco_classes_are_mapped_and_unrelated_classes_are_dropped(self) -> None:
        detections = np.array(
            [
                [0, 0, 10, 10, 0.9, 0],
                [0, 0, 10, 10, 0.8, 2],
                [0, 0, 10, 10, 0.7, 5],
                [0, 0, 10, 10, 0.6, 16],
            ],
            dtype=np.float32,
        )
        result = _remap_detection_classes(detections, COCO_TO_PROJECT_CLASS)
        self.assertEqual(result.shape, (3, 6))
        np.testing.assert_array_equal(result[:, 5], np.array([2, 0, 1]))


if __name__ == "__main__":
    unittest.main()
