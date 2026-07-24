import unittest

from data.kitti2coco import parse_kitti_line


class KittiParserTest(unittest.TestCase):
    def test_parses_official_detection_row(self) -> None:
        row = "Car 0.00 0 -1.57 10.0 20.0 110.0 80.0 1.5 1.6 3.7 1.0 1.5 20.0 0.3"
        parsed = parse_kitti_line(row)
        self.assertEqual(parsed.category, "Car")
        self.assertEqual(parsed.occluded, 0)
        self.assertEqual(parsed.bbox_xyxy, (10.0, 20.0, 110.0, 80.0))
        self.assertIsNone(parsed.score)


if __name__ == "__main__":
    unittest.main()

