import random
import unittest

import numpy as np

from synth.placement import (
    GATE_FLOOR,
    PEAK_BANDS,
    AlphaIntegral,
    accept_event,
    bbox_cover_ratio,
    class_relative_target_height,
    classify_band,
    event_shape,
    mask_cover_ratio,
    mid_height_factor,
    occluder_scale,
    offset_span,
    rho_series,
    sample_height_factor,
    sample_lateral_offset,
)

RANGES = {"car": [0.9, 1.1], "truck": [1.7, 1.9], "person": [1.1, 1.3], "bicycle": [0.8, 1.0]}


class ScaleModelTest(unittest.TestCase):
    def test_class_relative_height(self) -> None:
        # same class at the same depth -> same pixel height
        self.assertAlmostEqual(class_relative_target_height(100, 1.0, 1.0), 100.0)
        # a 1.7 m person next to a 1.5 m car is 1.2x as tall on screen
        self.assertAlmostEqual(class_relative_target_height(100, 1.2, 1.0), 120.0)
        self.assertAlmostEqual(class_relative_target_height(180, 1.0, 1.8), 100.0)

    def test_rejects_non_positive_factors(self) -> None:
        with self.assertRaises(ValueError):
            class_relative_target_height(100, 0.0, 1.0)
        with self.assertRaises(ValueError):
            class_relative_target_height(0.0, 1.0, 1.0)

    def test_occluder_scale(self) -> None:
        self.assertAlmostEqual(occluder_scale(120.0, 60.0), 2.0)
        with self.assertRaises(ValueError):
            occluder_scale(120.0, 0.0)


class HeightFactorSamplingTest(unittest.TestCase):
    def test_samples_inside_the_physical_range(self) -> None:
        rng = random.Random(0)
        for _ in range(500):
            for category, (low, high) in RANGES.items():
                self.assertTrue(low <= sample_height_factor(RANGES, category, rng) <= high)

    def test_midpoint_is_used_for_the_victim(self) -> None:
        self.assertAlmostEqual(mid_height_factor(RANGES, "car"), 1.0)
        self.assertAlmostEqual(mid_height_factor(RANGES, "person"), 1.2)

    def test_invalid_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_height_factor({"car": [1.1, 0.9]}, "car", random.Random(0))


class OffsetTest(unittest.TestCase):
    def test_span_is_the_separation_at_which_boxes_clear(self) -> None:
        # a 150 px occluder clears a 40 px victim at 95 px of centre separation
        self.assertAlmostEqual(offset_span(150, 40), 95.0)

    def test_offset_at_span_gives_zero_overlap(self) -> None:
        occluder_width, victim_width = 150.0, 40.0
        span = offset_span(occluder_width, victim_width)
        victim = (0.0, 0.0, victim_width, 50.0)
        # occluder centred on the victim, then pushed out by exactly one span
        occluder = (-occluder_width / 2.0 + victim_width / 2.0 + span, 0.0, occluder_width, 50.0)
        self.assertAlmostEqual(bbox_cover_ratio(occluder, victim), 0.0)

    def test_sampled_offsets_stay_in_range(self) -> None:
        rng = random.Random(1)
        for _ in range(500):
            self.assertTrue(abs(sample_lateral_offset(1.15, rng)) <= 1.15)


class AlphaIntegralTest(unittest.TestCase):
    def test_counts_rectangles(self) -> None:
        alpha = np.zeros((10, 10), dtype=np.uint8)
        alpha[2:6, 3:7] = 255  # 4x4 = 16 set pixels
        integral = AlphaIntegral(alpha)
        self.assertEqual(integral.count(0, 0, 10, 10), 16)
        self.assertEqual(integral.count(3, 2, 7, 6), 16)
        self.assertEqual(integral.count(0, 0, 3, 10), 0)
        self.assertAlmostEqual(integral.fill_ratio, 0.16)

    def test_clamps_out_of_bounds_queries(self) -> None:
        alpha = np.ones((4, 4), dtype=np.uint8)
        integral = AlphaIntegral(alpha)
        self.assertEqual(integral.count(-20, -20, 40, 40), 16)
        self.assertEqual(integral.count(10, 10, 20, 20), 0)
        self.assertEqual(integral.count(3, 3, 1, 1), 0)  # inverted rect

    def test_rejects_non_2d(self) -> None:
        with self.assertRaises(ValueError):
            AlphaIntegral(np.zeros((4, 4, 3), dtype=np.uint8))


class MaskCoverRatioTest(unittest.TestCase):
    def test_full_mask_matches_the_bbox_proxy(self) -> None:
        integral = AlphaIntegral(np.ones((20, 20), dtype=np.uint8))
        occluder = (0.0, 0.0, 100.0, 100.0)
        victim = (25.0, 25.0, 50.0, 50.0)
        self.assertAlmostEqual(mask_cover_ratio(integral, occluder, victim), 1.0)

    def test_half_filled_mask_covers_half_of_what_the_bbox_claims(self) -> None:
        alpha = np.zeros((20, 20), dtype=np.uint8)
        alpha[:, :10] = 1  # only the left half of the crop is real
        integral = AlphaIntegral(alpha)
        occluder = (0.0, 0.0, 100.0, 100.0)
        victim = (0.0, 0.0, 100.0, 100.0)
        # the bbox proxy would say 1.0; the mask says 0.5
        self.assertAlmostEqual(bbox_cover_ratio(occluder, victim), 1.0)
        self.assertAlmostEqual(mask_cover_ratio(integral, occluder, victim), 0.5, places=6)

    def test_no_overlap_is_zero(self) -> None:
        integral = AlphaIntegral(np.ones((8, 8), dtype=np.uint8))
        self.assertEqual(mask_cover_ratio(integral, (0, 0, 10, 10), (500, 500, 10, 10)), 0.0)

    def test_degenerate_boxes_are_zero(self) -> None:
        integral = AlphaIntegral(np.ones((8, 8), dtype=np.uint8))
        self.assertEqual(mask_cover_ratio(integral, (0, 0, 0, 10), (0, 0, 10, 10)), 0.0)
        self.assertEqual(mask_cover_ratio(integral, (0, 0, 10, 10), (0, 0, 10, 0)), 0.0)


class BboxRhoTest(unittest.TestCase):
    def test_cover_ratio(self) -> None:
        self.assertAlmostEqual(bbox_cover_ratio([0, 0, 5, 10], [0, 0, 10, 10]), 0.5)
        self.assertAlmostEqual(bbox_cover_ratio([100, 100, 5, 5], [0, 0, 10, 10]), 0.0)

    def test_rho_series_handles_absent_boxes(self) -> None:
        occ = [[0, 0, 10, 10], None, [0, 0, 5, 10]]
        vic = [[0, 0, 10, 10], [0, 0, 10, 10], None]
        self.assertEqual(rho_series(occ, vic), [1.0, 0.0, 0.0])

    def test_rho_series_uses_masks_when_given(self) -> None:
        alpha = np.zeros((10, 10), dtype=np.uint8)
        alpha[:, :5] = 1
        integrals = [AlphaIntegral(alpha)]
        boxes = [[0, 0, 10, 10]]
        self.assertAlmostEqual(rho_series(boxes, boxes, integrals)[0], 0.5, places=6)


class BandTest(unittest.TestCase):
    def test_classify(self) -> None:
        self.assertIsNone(classify_band(0.10))
        self.assertEqual(classify_band(0.25), "mild")
        self.assertEqual(classify_band(0.50), "moderate")
        self.assertEqual(classify_band(0.95), "heavy")
        # full occlusion is inside heavy now, not above the ceiling
        self.assertEqual(classify_band(1.00), "heavy")


class EventShapeTest(unittest.TestCase):
    def test_single_bell(self) -> None:
        series = [0, 0, 0.05, 0.12, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.12, 0.05, 0]
        shape = event_shape(series)
        self.assertAlmostEqual(shape["peak"], 0.5)
        self.assertEqual(shape["num_runs"], 1)
        self.assertEqual(shape["effective_len"], 9)  # indices 3..11 with rho>=0.1

    def test_two_runs_detected(self) -> None:
        shape = event_shape([0, 0.2, 0.05, 0.2, 0])
        self.assertEqual(shape["num_runs"], 2)


class AcceptEventTest(unittest.TestCase):
    def _bell(self, peak: float = 0.5) -> list[float]:
        return [0, 0, 0.05, 0.12, 0.2, 0.3, 0.4, peak, 0.45, 0.4, 0.3, 0.2, 0.12, 0.05, 0]

    def test_gate_mode_accepts_any_band(self) -> None:
        for peak in (0.25, 0.50, 0.99):
            ok, reason = accept_event(self._bell(peak))
            self.assertTrue(ok, f"{peak}: {reason}")

    def test_gate_mode_rejects_below_the_floor(self) -> None:
        series = [0, 0, 0.05, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.05, 0]
        ok, reason = accept_event(series)
        self.assertFalse(ok)
        self.assertIn("gate_floor", reason)

    def test_full_occlusion_passes_the_ceiling(self) -> None:
        ok, reason = accept_event(self._bell(1.0))
        self.assertTrue(ok, reason)

    def test_band_mode_still_restricts(self) -> None:
        ok, reason = accept_event(self._bell(0.5), PEAK_BANDS["heavy"])
        self.assertFalse(ok)
        self.assertIn("outside band", reason)

    def test_rejects_double_peak(self) -> None:
        series = [0, 0.3, 0.3, 0.3, 0.3, 0.05, 0.3, 0.3, 0.3, 0.3, 0]
        ok, reason = accept_event(series)
        self.assertFalse(ok)
        self.assertIn("num_runs", reason)

    def test_rejects_too_short_event(self) -> None:
        series = [0, 0, 0.3, 0.4, 0.3, 0, 0]  # only 3 frames >= 0.1
        ok, reason = accept_event(series)
        self.assertFalse(ok)
        self.assertIn("effective_len", reason)

    def test_rejects_high_end_rho(self) -> None:
        series = [0.3] * 12  # never returns to ~0, one long run
        ok, reason = accept_event(series)
        self.assertFalse(ok)
        self.assertIn("rho", reason)

    def test_gate_floor_matches_the_mild_band_floor(self) -> None:
        self.assertAlmostEqual(GATE_FLOOR, PEAK_BANDS["mild"][0])


if __name__ == "__main__":
    unittest.main()
