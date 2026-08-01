import tempfile
import unittest
from pathlib import Path

from eval.run_boxmot import IGNORE_REGIONS, _ignore_regions_by_frame


CLASS_NAMES = ["car", "truck", "person", "bicycle"]


def _line(frame, track_id, category, x1, y1, x2, y2):
    """One KITTI Tracking label row: 17 fields, bbox at 6:10."""
    return (
        f"{frame} {track_id} {category} 0 0 0.0 "
        f"{x1} {y1} {x2} {y2} 1.5 1.6 4.0 1.0 2.0 8.0 0.1"
    )


class IgnoreRegionTest(unittest.TestCase):
    def _regions(self, lines, class_names=CLASS_NAMES):
        with tempfile.TemporaryDirectory() as directory:
            label_dir = Path(directory) / "training" / "label_02"
            label_dir.mkdir(parents=True)
            (label_dir / "0001.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            return _ignore_regions_by_frame(Path(directory), "0001", class_names)

    def test_dontcare_applies_to_every_class(self) -> None:
        regions = self._regions([_line(0, -1, "DontCare", 10, 20, 60, 80)])
        self.assertEqual(sorted(name for name, _ in regions[1]), sorted(CLASS_NAMES))
        for _, box in regions[1]:
            self.assertEqual(box, [10.0, 20.0, 50.0, 60.0])

    def test_sitting_person_applies_to_person_only(self) -> None:
        """A car detection on a sitting person is a genuine false positive; only
        the pedestrian class is the near-duplicate KITTI wants ignored."""
        for spelling in ("Person", "Person_sitting"):
            regions = self._regions([_line(3, 7, spelling, 10, 20, 60, 80)])
            self.assertEqual([name for name, _ in regions[4]], ["person"], spelling)

    def test_frame_numbers_are_one_based(self) -> None:
        """KITTI frames are 0-based; the MOTChallenge files this feeds are not."""
        regions = self._regions([_line(0, -1, "DontCare", 1, 2, 30, 40)])
        self.assertIn(1, regions)
        self.assertNotIn(0, regions)

    def test_evaluated_classes_are_not_treated_as_ignore(self) -> None:
        lines = [
            _line(0, 1, "Car", 10, 20, 60, 80),
            _line(0, 2, "Pedestrian", 10, 20, 60, 80),
            _line(0, 3, "Van", 10, 20, 60, 80),
            _line(0, 4, "Cyclist", 10, 20, 60, 80),
        ]
        self.assertEqual(self._regions(lines), {})

    def test_van_is_deliberately_not_ignored(self) -> None:
        """KITTI ignores Van; this project maps it to car as a positive. The
        departure is intentional, so guard it against a silent change."""
        self.assertNotIn("Van", IGNORE_REGIONS)

    def test_degenerate_boxes_are_dropped(self) -> None:
        regions = self._regions([_line(0, -1, "DontCare", 10, 20, 10.5, 80)])
        self.assertEqual(regions, {})

    def test_unknown_class_name_is_skipped(self) -> None:
        """A class absent from this run's head must not produce gt rows for it."""
        regions = self._regions([_line(0, 7, "Person", 10, 20, 60, 80)], ["car", "truck"])
        self.assertEqual(regions, {})

    def test_multiple_regions_in_one_frame_are_all_kept(self) -> None:
        regions = self._regions(
            [_line(0, -1, "DontCare", 10, 20, 60, 80), _line(0, -1, "DontCare", 70, 20, 120, 80)]
        )
        self.assertEqual(sum(1 for name, _ in regions[1] if name == "car"), 2)


if __name__ == "__main__":
    unittest.main()
