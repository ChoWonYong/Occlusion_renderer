import random
import unittest

import numpy as np
from PIL import Image

from synth.geometry import AlphaIntegralCache, map_occluder_box
from synth.paste_jitter import (
    IDENTITY,
    PRESETS,
    FrameJitter,
    expand_ratios,
    jitter_alpha,
    jitter_box,
    jitter_policy_from_config,
    jitter_rgba,
    real_policy_from_config,
    ranges_from_config,
    resolve_preset,
    sample_sequence,
)


def _patch(width=40, height=24):
    """RGBA crop with an off-centre opaque blob, so flips and rotations show."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[4:height - 2, 3:width // 2] = 255
    return np.dstack([rgb, alpha])


class ExpandRatioTest(unittest.TestCase):
    def test_matches_pil_expand_for_every_angle(self) -> None:
        """The target box grows by exactly what the patch grows by, or the
        rotated crop gets squashed back into an unrotated aspect ratio."""
        for width, height in ((40, 24), (13, 61), (7, 7)):
            image = Image.new("L", (width, height))
            for rotation in range(-30, 31, 3):
                ratio_width, ratio_height = expand_ratios(width, height, rotation)
                rotated = image.rotate(rotation, expand=True)
                self.assertEqual(
                    (round(width * ratio_width), round(height * ratio_height)),
                    rotated.size,
                    f"{width}x{height} @ {rotation} deg",
                )

    def test_no_rotation_is_identity(self) -> None:
        self.assertEqual(expand_ratios(40, 24, 0.0), (1.0, 1.0))
        self.assertEqual(expand_ratios(40, 24, 360.0), (1.0, 1.0))


class JitterBoxTest(unittest.TestCase):
    def test_scale_keeps_the_bottom_centre_anchor(self) -> None:
        box = (100.0, 50.0, 40.0, 24.0)
        jittered = jitter_box(box, (40, 24), FrameJitter(scale=1.5))
        self.assertAlmostEqual(jittered[0] + jittered[2] / 2, box[0] + box[2] / 2)
        self.assertAlmostEqual(jittered[1] + jittered[3], box[1] + box[3])
        self.assertAlmostEqual(jittered[2], 60.0)
        self.assertAlmostEqual(jittered[3], 36.0)

    def test_rotation_keeps_the_bottom_centre_anchor(self) -> None:
        box = (100.0, 50.0, 40.0, 24.0)
        jittered = jitter_box(box, (40, 24), FrameJitter(rotation=15.0))
        self.assertAlmostEqual(jittered[0] + jittered[2] / 2, box[0] + box[2] / 2)
        self.assertAlmostEqual(jittered[1] + jittered[3], box[1] + box[3])
        self.assertGreater(jittered[2], box[2])
        self.assertGreater(jittered[3], box[3])

    def test_identity_leaves_the_box_alone(self) -> None:
        box = (100.0, 50.0, 40.0, 24.0)
        self.assertEqual(jitter_box(box, (40, 24), IDENTITY), box)


class PatchJitterTest(unittest.TestCase):
    def test_photometric_leaves_alpha_untouched(self) -> None:
        """rho is computed from alpha, so a colour-only draw must not move it."""
        patch = _patch()
        jitter = FrameJitter(color=1.4, brightness=0.7, contrast=1.3, sharpness=2.0)
        out = jitter_rgba(patch, jitter)
        np.testing.assert_array_equal(out[..., 3], patch[..., 3])
        self.assertFalse(np.array_equal(out[..., :3], patch[..., :3]))

    def test_flip_mirrors_both_channels(self) -> None:
        patch = _patch()
        out = jitter_rgba(patch, FrameJitter(flip=True))
        np.testing.assert_array_equal(out[..., 3], patch[..., 3][:, ::-1])
        np.testing.assert_array_equal(out[..., :3], patch[..., :3][:, ::-1])

    def test_alpha_stays_binary_through_rotation(self) -> None:
        """A bilinear rotation would feather the edge and the compositor's
        ``alpha > 0`` test would silently grow the mask."""
        out = jitter_rgba(_patch(), FrameJitter(rotation=12.0))
        self.assertEqual(set(np.unique(out[..., 3])) - {0, 255}, set())

    def test_rgba_and_alpha_paths_agree(self) -> None:
        """The renderer jitters the patch and the search jitters the alpha alone;
        the two must produce the same mask or rho would not match."""
        patch = _patch()
        for jitter in (
            FrameJitter(flip=True),
            FrameJitter(rotation=-9.0),
            FrameJitter(flip=True, rotation=17.0, color=1.2, brightness=0.8),
        ):
            np.testing.assert_array_equal(
                jitter_rgba(patch, jitter)[..., 3], jitter_alpha(patch[..., 3], jitter)
            )

    def test_identity_returns_the_patch_unchanged(self) -> None:
        patch = _patch()
        np.testing.assert_array_equal(jitter_rgba(patch, IDENTITY), patch)

    def test_rejects_non_rgba(self) -> None:
        with self.assertRaises(ValueError):
            jitter_rgba(_patch()[..., :3], FrameJitter(flip=True))

    def test_rain_elasticity_changes_rgb_but_never_alpha(self) -> None:
        patch = _patch()
        out = jitter_rgba(
            patch,
            FrameJitter(
                real_scenario="rain",
                effect_seed=3,
                elastic_fraction=0.5,
                elastic_block_size=2,
                elastic_displacement=1,
            ),
        )
        np.testing.assert_array_equal(out[..., 3], patch[..., 3])
        self.assertFalse(np.array_equal(out[..., :3], patch[..., :3]))

    def test_snow_whitens_only_rgb_and_preserves_geometry(self) -> None:
        patch = _patch()
        out = jitter_rgba(
            patch,
            FrameJitter(
                real_scenario="snow",
                effect_seed=9,
                snow_fraction=0.5,
                snow_block_size=1,
            ),
        )
        np.testing.assert_array_equal(out[..., 3], patch[..., 3])
        opaque_white = np.all(out[..., :3] == 255, axis=2) & (patch[..., 3] > 0)
        self.assertTrue(opaque_white.any())


class PresetTest(unittest.TestCase):
    def test_off_disables_jitter(self) -> None:
        for value in (None, "off", "none", "", False):
            self.assertIsNone(resolve_preset(value))

    def test_unknown_preset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_preset("wild")

    def test_presets_nest_by_strength(self) -> None:
        """low subset of mid subset of high, so "stronger" means strictly wider."""
        for weaker, stronger in (("low", "mid"), ("mid", "high")):
            for field in ("color", "brightness", "contrast", "sharpness", "scale", "rotation"):
                inner, outer = getattr(PRESETS[weaker], field), getattr(PRESETS[stronger], field)
                self.assertGreaterEqual(inner[0], outer[0], f"{weaker}<{stronger}.{field}")
                self.assertLessEqual(inner[1], outer[1], f"{weaker}<{stronger}.{field}")

    def test_every_range_brackets_the_identity(self) -> None:
        for name, ranges in PRESETS.items():
            for field in ("color", "brightness", "contrast", "sharpness", "scale"):
                low, high = getattr(ranges, field)
                self.assertLessEqual(low, 1.0, f"{name}.{field}")
                self.assertGreaterEqual(high, 1.0, f"{name}.{field}")

    def test_config_section_selects_and_overrides(self) -> None:
        self.assertIsNone(ranges_from_config(None))
        self.assertIsNone(ranges_from_config({"preset": "off"}))
        overridden = ranges_from_config({"preset": "low", "rotation": [-2, 2], "flip_prob": 0.0})
        self.assertEqual(overridden.rotation, (-2.0, 2.0))
        self.assertEqual(overridden.flip_prob, 0.0)
        self.assertEqual(overridden.color, PRESETS["low"].color)


class SampleSequenceTest(unittest.TestCase):
    def test_off_yields_identity_for_every_frame(self) -> None:
        self.assertEqual(sample_sequence(None, 5, random.Random(0)), [IDENTITY] * 5)

    def test_each_frame_is_drawn_independently(self) -> None:
        sequence = sample_sequence(PRESETS["mid"], 30, random.Random(0))
        self.assertEqual(len(sequence), 30)
        self.assertGreater(len({j.brightness for j in sequence}), 25)
        self.assertGreater(len({j.rotation for j in sequence}), 25)
        self.assertEqual({j.flip for j in sequence}, {True, False})

    def test_draws_stay_inside_the_preset(self) -> None:
        ranges = PRESETS["mid"]
        for jitter in sample_sequence(ranges, 200, random.Random(1)):
            self.assertTrue(ranges.color[0] <= jitter.color <= ranges.color[1])
            self.assertTrue(ranges.scale[0] <= jitter.scale <= ranges.scale[1])
            self.assertTrue(ranges.rotation[0] <= jitter.rotation <= ranges.rotation[1])

    def test_same_seed_reproduces_the_sequence(self) -> None:
        first = sample_sequence(PRESETS["low"], 12, random.Random(7))
        second = sample_sequence(PRESETS["low"], 12, random.Random(7))
        self.assertEqual(first, second)


class RealJitterTest(unittest.TestCase):
    CONFIG = {
        "scenarios": ["day", "night", "rain", "snow"],
        "weights": {"day": 1, "night": 1, "rain": 1, "snow": 1},
        "definitions": {
            "day": {"brightness": [0.95, 1.15]},
            "night": {"brightness": [0.35, 0.55]},
            "rain": {
                "brightness": [0.72, 0.92],
                "elastic_fraction": 0.012,
                "elastic_block_size": 2,
            },
            "snow": {
                "brightness": [0.82, 1.0],
                "elastic_fraction": 0.008,
                "snow_fraction": 0.018,
            },
        },
    }

    def test_default_config_mode_is_random(self) -> None:
        policy = jitter_policy_from_config({"preset": "mid"})
        self.assertEqual(policy, PRESETS["mid"])

    def test_real_scenario_is_constant_within_an_event(self) -> None:
        policy = real_policy_from_config(self.CONFIG)
        sequence = sample_sequence(policy, 40, random.Random(5), scenario="night")
        self.assertEqual({item.real_scenario for item in sequence}, {"night"})
        self.assertTrue(all(0.0 < item.brightness < 0.6 for item in sequence))
        self.assertGreater(len({item.brightness for item in sequence}), 30)

    def test_real_sequence_is_reproducible(self) -> None:
        policy = real_policy_from_config(self.CONFIG)
        first = sample_sequence(policy, 12, random.Random(7), scenario="snow")
        second = sample_sequence(policy, 12, random.Random(7), scenario="snow")
        self.assertEqual(first, second)

    def test_invalid_fraction_is_rejected(self) -> None:
        bad = {
            "scenarios": ["rain"],
            "definitions": {"rain": {"elastic_fraction": 1.1}},
        }
        with self.assertRaises(ValueError):
            real_policy_from_config(bad)


class MapOccluderBoxTest(unittest.TestCase):
    FRAME = {
        "crop_bbox_xywh": [100.0, 80.0, 40.0, 24.0],
        "source_image_size": [1242.0, 375.0],
    }

    def _box(self, jitter):
        return map_occluder_box(
            self.FRAME,
            (1242, 375),
            reference_height=48.0,
            source_reference_height=24.0,
            jitter=jitter,
        )

    def test_identity_matches_the_unjittered_box(self) -> None:
        self.assertEqual(self._box(IDENTITY), self._box(FrameJitter()))

    def test_photometric_only_does_not_move_the_box(self) -> None:
        self.assertEqual(self._box(FrameJitter(color=1.3, brightness=0.7)), self._box(IDENTITY))

    def test_scale_and_rotation_grow_the_box_in_place(self) -> None:
        base = self._box(IDENTITY)
        grown = self._box(FrameJitter(scale=1.1, rotation=8.0))
        self.assertGreater(grown[2], base[2])
        self.assertAlmostEqual(grown[0] + grown[2] / 2, base[0] + base[2] / 2)
        self.assertAlmostEqual(grown[1] + grown[3], base[1] + base[3])


class IntegralCacheTest(unittest.TestCase):
    def test_photometric_only_reuses_the_shared_table(self) -> None:
        """Colour never touches alpha, so those frames must not pay for a rebuild."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.png"
            Image.fromarray(_patch()).save(path)
            cache = AlphaIntegralCache()
            shared = cache.get(path)
            self.assertIs(cache.get_jittered(path, FrameJitter(brightness=0.5)), shared)
            self.assertIs(cache.get_jittered(path, FrameJitter(scale=1.2)), shared)
            self.assertIsNot(cache.get_jittered(path, FrameJitter(flip=True)), shared)

    def test_jittered_table_sees_the_flipped_mask(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.png"
            patch = _patch()
            Image.fromarray(patch).save(path)
            height, width = patch.shape[:2]
            cache = AlphaIntegralCache()
            plain = cache.get(path)
            flipped = cache.get_jittered(path, FrameJitter(flip=True))
            # Same total coverage, mirrored: the blob in the left half moves right.
            self.assertEqual(plain.count(0, 0, width, height), flipped.count(0, 0, width, height))
            self.assertGreater(
                plain.count(0, 0, width // 2, height), flipped.count(0, 0, width // 2, height)
            )


if __name__ == "__main__":
    unittest.main()
