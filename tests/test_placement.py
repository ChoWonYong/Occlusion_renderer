import random
import unittest
from collections import Counter

from synth.placement import (
    PEAK_BANDS,
    accept_event,
    bbox_cover_ratio,
    candidate_peak_error,
    class_relative_target_height,
    event_shape,
    occluder_scale,
    rho_series,
    sample_target_peak,
    select_best_placement,
)

FACTORS = {"car": 1.0, "truck": 1.8, "person": 1.2, "bicycle": 0.9}


class ScaleModelTest(unittest.TestCase):
    def test_class_relative_height(self) -> None:
        self.assertAlmostEqual(class_relative_target_height(100, "car", "car", FACTORS), 100.0)
        self.assertAlmostEqual(class_relative_target_height(100, "person", "car", FACTORS), 120.0)
        self.assertAlmostEqual(class_relative_target_height(100, "bicycle", "car", FACTORS), 90.0)
        self.assertAlmostEqual(class_relative_target_height(120, "car", "person", FACTORS), 100.0)
        self.assertAlmostEqual(class_relative_target_height(180, "car", "truck", FACTORS), 100.0)

    def test_occluder_scale(self) -> None:
        self.assertAlmostEqual(occluder_scale(120.0, 60.0), 2.0)
        with self.assertRaises(ValueError):
            occluder_scale(120.0, 0.0)


class TargetPeakTest(unittest.TestCase):
    def test_peaks_land_in_sampled_band(self) -> None:
        rng = random.Random(0)
        dist = {"mild": 0.20, "moderate": 0.55, "heavy": 0.25}
        seen = Counter()
        for _ in range(2000):
            peak, band, (low, high) = sample_target_peak(dist, rng)
            self.assertTrue(low <= peak <= high)
            self.assertEqual((low, high), PEAK_BANDS[band])
            seen[band] += 1
        # moderate should dominate given its weight
        self.assertGreater(seen["moderate"], seen["mild"])
        self.assertGreater(seen["moderate"], seen["heavy"])


class BboxRhoTest(unittest.TestCase):
    def test_cover_ratio(self) -> None:
        # occluder covers left half of a 10x10 victim
        self.assertAlmostEqual(bbox_cover_ratio([0, 0, 5, 10], [0, 0, 10, 10]), 0.5)
        self.assertAlmostEqual(bbox_cover_ratio([100, 100, 5, 5], [0, 0, 10, 10]), 0.0)

    def test_rho_series_handles_absent_boxes(self) -> None:
        occ = [[0, 0, 10, 10], None, [0, 0, 5, 10]]
        vic = [[0, 0, 10, 10], [0, 0, 10, 10], None]
        self.assertEqual(rho_series(occ, vic), [1.0, 0.0, 0.0])


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
    def _bell(self) -> list[float]:
        return [0, 0, 0.05, 0.12, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4, 0.3, 0.2, 0.12, 0.05, 0]

    def test_accepts_complete_moderate_event(self) -> None:
        ok, reason = accept_event(self._bell(), PEAK_BANDS["moderate"])
        self.assertTrue(ok, reason)

    def test_rejects_peak_out_of_band(self) -> None:
        ok, reason = accept_event(self._bell(), PEAK_BANDS["heavy"])
        self.assertFalse(ok)
        self.assertIn("outside band", reason)

    def test_rejects_double_peak(self) -> None:
        series = [0, 0.3, 0.3, 0.3, 0.3, 0.05, 0.3, 0.3, 0.3, 0.3, 0]
        ok, reason = accept_event(series, PEAK_BANDS["mild"])
        self.assertFalse(ok)
        self.assertIn("num_runs", reason)

    def test_rejects_too_short_event(self) -> None:
        series = [0, 0, 0.3, 0.4, 0.3, 0, 0]  # only 3 frames >= 0.1
        ok, reason = accept_event(series, PEAK_BANDS["moderate"])
        self.assertFalse(ok)
        self.assertIn("effective_len", reason)

    def test_rejects_high_end_rho(self) -> None:
        series = [0.3] * 12  # never returns to ~0, one long run
        ok, reason = accept_event(series, PEAK_BANDS["mild"])
        self.assertFalse(ok)
        self.assertIn("rho", reason)


class SelectBestTest(unittest.TestCase):
    def _bell(self, peak: float) -> list[float]:
        # ramp has 4 frames >= 0.1 so the >=0.1 stretch is 4 + peak + 4 = 9 frames.
        ramp = [0, 0, 0.05, 0.12, 0.15, 0.2, 0.28]
        return ramp + [peak] + list(reversed(ramp))

    def test_picks_closest_to_target_among_accepted(self) -> None:
        candidates = [
            {"params": {"scale": 0.9}, "rho_series": self._bell(0.40)},
            {"params": {"scale": 1.0}, "rho_series": self._bell(0.52)},
            {"params": {"scale": 1.3}, "rho_series": self._bell(0.90)},  # exceeds peak_max -> rejected
        ]
        best = select_best_placement(candidates, target_peak=0.50, band=PEAK_BANDS["moderate"])
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best["params"]["scale"], 1.0)
        self.assertAlmostEqual(best["achieved_peak"], 0.52)

    def test_returns_none_when_nothing_accepts(self) -> None:
        candidates = [{"params": {}, "rho_series": [0.9] * 12}]
        self.assertIsNone(select_best_placement(candidates, 0.5, PEAK_BANDS["moderate"]))


if __name__ == "__main__":
    unittest.main()
