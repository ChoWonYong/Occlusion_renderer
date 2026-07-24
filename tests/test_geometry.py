import unittest

from synth.geometry import alignment_translation, map_occluder_box, search_event_placement
from synth.placement import PEAK_BANDS

FACTORS = {"car": 1.0, "truck": 1.8, "person": 1.2, "bicycle": 0.9}


class MapOccluderBoxTest(unittest.TestCase):
    def test_bottom_center_anchor_and_scale(self) -> None:
        frame = {"crop_bbox_xywh": [100, 400, 100, 200], "source_image_size": [1000, 1000]}
        box = map_occluder_box(frame, (200, 100), reference_height=40, source_reference_height=200)
        # center_x_fraction 0.15, bottom 0.6, scale 0.2 -> 20x40 box at (20, 20)
        self.assertEqual(box, (20.0, 20.0, 20.0, 40.0))

    def test_relative_motion_scales_with_source_height(self) -> None:
        # a taller source crop than the reference is drawn proportionally larger
        frame = {"crop_bbox_xywh": [0, 0, 100, 400], "source_image_size": [1000, 1000]}
        box = map_occluder_box(frame, (200, 100), reference_height=40, source_reference_height=200)
        self.assertAlmostEqual(box[3], 80.0)  # 40 * (400/200)


class AlignmentTranslationTest(unittest.TestCase):
    def test_aligns_bottom_center(self) -> None:
        translation = alignment_translation((20, 20, 20, 40), (80, 20, 40, 40))
        self.assertEqual(translation, (70.0, 0.0))


class SearchEventPlacementTest(unittest.TestCase):
    def _occluder_frames(self) -> list[dict]:
        frames = []
        for offset in range(20):
            cf = 0.3 + offset * 0.02  # sweep the occluder horizontally across the victim
            frames.append(
                {
                    "crop_bbox_xywh": [cf * 1000 - 50, 400, 100, 200],
                    "source_image_size": [1000, 1000],
                }
            )
        return frames

    def test_finds_complete_moderate_event(self) -> None:
        victim = {position: [80.0, 20.0, 40.0, 40.0] for position in range(40)}
        best = search_event_placement(
            self._occluder_frames(),
            victim,
            occluder_class="person",
            victim_class="person",
            class_height_factor=FACTORS,
            target_peak=0.5,
            band=PEAK_BANDS["moderate"],
            multipliers=[1.0],
            target_size=(200, 100),
            sequence_length=40,
        )
        self.assertIsNotNone(best)
        assert best is not None
        self.assertAlmostEqual(best["achieved_peak"], 0.5, places=3)
        self.assertEqual(best["scale_multiplier"], 1.0)
        # start must keep the whole exposure inside the sequence
        self.assertTrue(0 <= best["start_position"] <= 20)
        self.assertEqual(best["rho_series"][0], 0.0)  # event begins un-occluded

    def test_returns_none_when_victim_never_present_in_window(self) -> None:
        victim = {0: [80.0, 20.0, 40.0, 40.0]}  # only one frame, cannot host a full event
        best = search_event_placement(
            self._occluder_frames(),
            victim,
            occluder_class="person",
            victim_class="person",
            class_height_factor=FACTORS,
            target_peak=0.5,
            band=PEAK_BANDS["moderate"],
            multipliers=[1.0],
            target_size=(200, 100),
            sequence_length=40,
        )
        self.assertIsNone(best)


if __name__ == "__main__":
    unittest.main()
