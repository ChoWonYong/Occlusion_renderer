import unittest

from data.kitti_tracking import parse_tracking_line


class KittiTrackingParserTest(unittest.TestCase):
    def test_parses_tracking_row(self) -> None:
        row = "3 17 Car 0.00 0 -1.57 10 20 110 80 1.5 1.6 3.7 1.0 1.5 20.0 0.3"
        parsed = parse_tracking_line(row)
        self.assertEqual(parsed.frame_index, 3)
        self.assertEqual(parsed.track_id, 17)
        self.assertEqual(parsed.category, "Car")
        self.assertEqual(parsed.bbox_xyxy, (10.0, 20.0, 110.0, 80.0))


if __name__ == "__main__":
    unittest.main()

