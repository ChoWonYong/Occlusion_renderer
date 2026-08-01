import random
import unittest
from unittest import mock

import numpy as np

from train.run import AUG_LEVELS, PHOTOMETRIC_STEPS
from train.yolox_x_kitti import SelectiveTransform, _distort_subset
from yolox.data import TrainTransform, data_augment


def _sample():
    """A noise image plus one off-centre box, so flips and colour shifts show."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    targets = np.array([[5.0, 10.0, 25.0, 30.0, 0.0, 1.0]], dtype=np.float32)
    return image, targets


def _apply(transform, seed):
    image, targets = _sample()
    random.seed(seed)
    return transform(image, targets.copy(), (40, 60))


def _selective(flip, photometric):
    return SelectiveTransform(p=0.5, max_labels=10, flip=flip, photometric=photometric)


class SelectiveTransformTest(unittest.TestCase):
    def test_everything_off_is_deterministic(self) -> None:
        """No flip and no photometric step means nothing is drawn from the RNG."""
        transform = _selective(False, ())
        first_image, first_targets = _apply(transform, 0)
        for seed in range(1, 8):
            image, targets = _apply(transform, seed)
            np.testing.assert_array_equal(image, first_image)
            np.testing.assert_array_equal(targets, first_targets)

    def test_stock_transform_still_flips_and_distorts(self) -> None:
        """The reference path must stay stochastic, or the comparison is empty."""
        transform = TrainTransform(p=0.5, max_labels=10)
        outputs = {_apply(transform, seed)[0].tobytes() for seed in range(8)}
        self.assertGreater(len(outputs), 1)

    def test_hue_off_never_draws_the_hue_offset(self) -> None:
        """random.randint is the hue step's own draw; no other step uses it."""
        steps = frozenset(AUG_LEVELS["nofliphue"][1].split(","))
        self.assertEqual(steps, {"brightness", "contrast", "saturation"})
        transform = _selective(False, steps)
        with mock.patch("random.randint", side_effect=AssertionError("hue ran")):
            for seed in range(20):
                _apply(transform, seed)

    def test_hue_on_does_draw_the_hue_offset(self) -> None:
        transform = _selective(False, {"hue"})
        with mock.patch("random.randint", wraps=random.randint) as randint:
            for seed in range(20):
                _apply(transform, seed)
        self.assertTrue(randint.called)

    def test_remaining_photometric_steps_still_vary(self) -> None:
        transform = _selective(False, {"brightness", "contrast", "saturation"})
        outputs = {_apply(transform, seed)[0].tobytes() for seed in range(8)}
        self.assertGreater(len(outputs), 1)

    def test_boxes_are_not_mirrored_when_flip_is_off(self) -> None:
        """A flipped box would land at width - x2, which no RNG state may produce."""
        image, raw = _sample()
        ratio = min(40 / image.shape[0], 60 / image.shape[1])
        expected = (raw[0][0] + raw[0][2]) / 2.0 * ratio
        transform = _selective(False, {"brightness", "contrast", "saturation"})
        for seed in range(20):
            _, targets = _apply(transform, seed)
            self.assertAlmostEqual(float(targets[0][1]), float(expected), places=4)

    def test_module_functions_are_restored(self) -> None:
        distort, mirror = data_augment._distort, data_augment._mirror
        _apply(_selective(False, {"hue"}), 0)
        self.assertIs(data_augment._distort, distort)
        self.assertIs(data_augment._mirror, mirror)

    def test_restores_after_an_exception(self) -> None:
        distort, mirror = data_augment._distort, data_augment._mirror
        with self.assertRaises(Exception):
            _selective(False, ())(None, _sample()[1], (40, 60))
        self.assertIs(data_augment._distort, distort)
        self.assertIs(data_augment._mirror, mirror)

    def test_unknown_step_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _selective(True, {"gamma"})

    def test_full_step_set_matches_the_stock_distort(self) -> None:
        """The transcription must agree with ByteTrack's _distort step for step."""
        image, _ = _sample()
        for seed in range(8):
            random.seed(seed)
            mine = _distort_subset(image, frozenset(PHOTOMETRIC_STEPS))
            random.seed(seed)
            theirs = data_augment._distort(image)
            np.testing.assert_array_equal(mine, theirs)


class AugLevelTest(unittest.TestCase):
    def test_levels_declare_flip_photometric_and_mosaic(self) -> None:
        for name, value in AUG_LEVELS.items():
            flip, photometric, mosaic = value
            self.assertIn(flip, {"0", "1"}, name)
            self.assertIn(mosaic, {"0", "1"}, name)
            steps = {step for step in photometric.split(",") if step}
            self.assertLessEqual(steps, set(PHOTOMETRIC_STEPS), name)

    def test_jitter_and_full_keep_every_photometric_step(self) -> None:
        for name in ("jitter", "full"):
            self.assertEqual(AUG_LEVELS[name][0], "1")
            self.assertEqual(AUG_LEVELS[name][1], ",".join(PHOTOMETRIC_STEPS))


if __name__ == "__main__":
    unittest.main()
