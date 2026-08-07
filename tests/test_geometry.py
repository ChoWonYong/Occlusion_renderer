import random
import unittest

import numpy as np

from synth.geometry import alignment_translation, map_occluder_box, sample_event_placement
from synth.placement import AlphaIntegral


def _solid(height: int = 20, width: int = 10) -> AlphaIntegral:
    return AlphaIntegral(np.ones((height, width), dtype=np.uint8))


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


class SampleEventPlacementTest(unittest.TestCase):
    def _occluder_frames(self, count: int = 20) -> list[dict]:
        frames = []
        for offset in range(count):
            cf = 0.3 + offset * 0.02  # sweep the occluder horizontally across the victim
            frames.append(
                {
                    "crop_bbox_xywh": [cf * 1000 - 50, 400, 100, 200],
                    "source_image_size": [1000, 1000],
                }
            )
        return frames

    def _integrals(self, count: int = 20) -> list[AlphaIntegral]:
        return [_solid() for _ in range(count)]

    def _victim(self) -> dict[int, list[float]]:
        return {position: [80.0, 20.0, 40.0, 40.0] for position in range(40)}

    def test_finds_a_complete_event(self) -> None:
        best = sample_event_placement(
            self._occluder_frames(),
            self._victim(),
            factor_occluder=1.2,
            factor_victim=1.2,
            integrals=self._integrals(),
            target_size=(200, 100),
            sequence_length=40,
            max_lateral_fraction=1.15,
            rng=random.Random(0),
            attempts=200,
        )
        self.assertIsNotNone(best)
        assert best is not None
        self.assertGreaterEqual(best["achieved_peak"], 0.20)
        self.assertLessEqual(best["achieved_peak"], 1.0)
        self.assertIn(best["band"], {"mild", "moderate", "heavy"})
        self.assertTrue(0 <= best["start_position"] <= 20)
        self.assertEqual(best["rho_series"][0], 0.0)  # event begins un-occluded
        self.assertTrue(abs(best["lateral_offset_fraction"]) <= 1.15)

    def test_returns_none_when_victim_never_present_in_window(self) -> None:
        victim = {0: [80.0, 20.0, 40.0, 40.0]}  # one frame, cannot host a full event
        best = sample_event_placement(
            self._occluder_frames(),
            victim,
            factor_occluder=1.2,
            factor_victim=1.2,
            integrals=self._integrals(),
            target_size=(200, 100),
            sequence_length=40,
            max_lateral_fraction=1.15,
            rng=random.Random(0),
            attempts=50,
        )
        self.assertIsNone(best)

    def test_empty_exposure_returns_none(self) -> None:
        self.assertIsNone(
            sample_event_placement(
                [], {0: [0.0, 0.0, 10.0, 10.0]},
                factor_occluder=1.0, factor_victim=1.0, integrals=[],
                target_size=(200, 100), sequence_length=40,
                max_lateral_fraction=1.0, rng=random.Random(0),
            )
        )

    def test_sparse_mask_scores_lower_than_a_solid_one(self) -> None:
        """A narrow mask inside a wide crop must not be scored as a full cover."""
        frames = self._occluder_frames()
        sparse = np.zeros((20, 10), dtype=np.uint8)
        sparse[:, :2] = 1  # only 20% of the crop box is really the object
        dense = sample_event_placement(
            frames, self._victim(), factor_occluder=1.2, factor_victim=1.2,
            integrals=self._integrals(), target_size=(200, 100), sequence_length=40,
            max_lateral_fraction=0.0, rng=random.Random(3), attempts=200,
        )
        thin = sample_event_placement(
            frames, self._victim(), factor_occluder=1.2, factor_victim=1.2,
            integrals=[AlphaIntegral(sparse) for _ in frames],
            target_size=(200, 100), sequence_length=40,
            max_lateral_fraction=0.0, rng=random.Random(3), attempts=200,
        )
        self.assertIsNotNone(dense)
        assert dense is not None
        if thin is not None:
            self.assertLess(thin["achieved_peak"], dense["achieved_peak"])

    def test_zero_lateral_offset_keeps_the_aligned_placement(self) -> None:
        aligned = sample_event_placement(
            self._occluder_frames(), self._victim(), factor_occluder=1.2, factor_victim=1.2,
            integrals=self._integrals(), target_size=(200, 100), sequence_length=40,
            max_lateral_fraction=0.0, rng=random.Random(0), attempts=200,
        )
        self.assertIsNotNone(aligned)
        assert aligned is not None
        self.assertAlmostEqual(aligned["lateral_offset_fraction"], 0.0)

    def test_larger_height_factor_makes_a_bigger_occluder(self) -> None:
        small = sample_event_placement(
            self._occluder_frames(), self._victim(), factor_occluder=1.0, factor_victim=1.2,
            integrals=self._integrals(), target_size=(200, 100), sequence_length=40,
            max_lateral_fraction=0.0, rng=random.Random(5), attempts=200,
        )
        large = sample_event_placement(
            self._occluder_frames(), self._victim(), factor_occluder=1.2, factor_victim=1.2,
            integrals=self._integrals(), target_size=(200, 100), sequence_length=40,
            max_lateral_fraction=0.0, rng=random.Random(5), attempts=200,
        )
        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        assert small is not None and large is not None
        self.assertLess(small["reference_height"], large["reference_height"])


if __name__ == "__main__":
    unittest.main()
