"""Per-frame appearance jitter for pasted occluder tracklets.

Every frame of a pasted tracklet is drawn its own jitter — colour, brightness,
contrast, sharpness, horizontal flip, scale and rotation — instead of the crop
being composited verbatim. The synthetic frames train the *detector* only
(ByteTrack is never trained on them), so a jitter that changes frame to frame
costs no temporal realism the training actually uses, and it stops the detector
from memorising the ~62 pool identities pixel for pixel.

Three of the seven change the alpha mask, and therefore rho:

- **flip** mirrors the mask inside the same box;
- **rotation** rotates it and grows the canvas (``expand=True``);
- **scale** resizes the target box.

``synth/geometry.py`` promises that "the value the search accepts is the value
the renderer produces", so those three cannot be applied at render time only.
The jitter sequence is drawn once per candidate placement, before the search, and
the same sequence feeds both the alpha integral images the search scores and the
patches the renderer pastes.

:func:`expand_ratios` replicates PIL's own ``rotate(expand=True)`` sizing rather
than the plain ``w|cos| + h|sin|``: PIL rounds the rotated corner bounds outward
with ceil/floor, and the target box has to grow by exactly what the patch grew by
or the rotated crop would be squashed back into an unrotated aspect ratio.

Strength is chosen by preset — ``low``/``mid``/``high``, or ``off`` for no
jitter at all. ``mid`` is the measured default.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JitterRanges:
    """Sampling ranges for one preset. Photometric factors are PIL
    ``ImageEnhance`` factors, where 1.0 is the original image."""

    color: tuple[float, float]
    brightness: tuple[float, float]
    contrast: tuple[float, float]
    sharpness: tuple[float, float]
    flip_prob: float
    scale: tuple[float, float]
    rotation: tuple[float, float]


PRESETS: dict[str, JitterRanges] = {
    "low": JitterRanges(
        color=(0.90, 1.10),
        brightness=(0.90, 1.10),
        contrast=(0.90, 1.10),
        sharpness=(0.75, 1.50),
        flip_prob=0.5,
        scale=(0.95, 1.05),
        rotation=(-5.0, 5.0),
    ),
    "mid": JitterRanges(
        color=(0.80, 1.20),
        brightness=(0.80, 1.20),
        contrast=(0.80, 1.20),
        sharpness=(0.50, 2.00),
        flip_prob=0.5,
        scale=(0.90, 1.10),
        rotation=(-10.0, 10.0),
    ),
    "high": JitterRanges(
        color=(0.60, 1.40),
        brightness=(0.60, 1.40),
        contrast=(0.60, 1.40),
        sharpness=(0.25, 4.00),
        flip_prob=0.5,
        scale=(0.80, 1.20),
        rotation=(-20.0, 20.0),
    ),
}

# Measured on KITTI eval, ep60, 3 seeds, HOTA against the same pipeline with
# jitter off (55.78 +- 0.38): low 56.20 +- 0.34, mid 56.69 +- 0.15. The gain is
# monotone in strength so far, which is why "high" exists — and why "mid" is the
# configured default.



@dataclass(frozen=True)
class FrameJitter:
    """One frame's draw. The default is the identity transform."""

    color: float = 1.0
    brightness: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0
    flip: bool = False
    scale: float = 1.0
    rotation: float = 0.0

    @property
    def changes_geometry(self) -> bool:
        return self.flip or self.rotation != 0.0 or self.scale != 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "flip": self.flip,
            "scale": self.scale,
            "rotation": self.rotation,
        }


IDENTITY = FrameJitter()


def resolve_preset(name: str | None) -> JitterRanges | None:
    """``None``/``"off"`` disables jitter; anything else must name a preset.

    ``False`` is accepted because YAML 1.1 turns a bare ``off`` into a boolean.
    """
    if name is None or name is False or str(name).lower() in {"off", "none", "false", ""}:
        return None
    key = str(name).lower()
    if key not in PRESETS:
        raise ValueError(f"unknown paste-jitter preset {name!r}; expected one of {sorted(PRESETS)}")
    return PRESETS[key]


def ranges_from_config(section: Mapping[str, Any] | None) -> JitterRanges | None:
    """Read ``tracklet_synthesis.paste_jitter`` — a preset name, optionally with
    individual ranges overridden."""
    if not section:
        return None
    ranges = resolve_preset(section.get("preset", "off"))
    if ranges is None:
        return None
    overrides = {
        field: tuple(float(value) for value in section[field])
        for field in ("color", "brightness", "contrast", "sharpness", "scale", "rotation")
        if field in section
    }
    if "flip_prob" in section:
        overrides["flip_prob"] = float(section["flip_prob"])
    if not overrides:
        return ranges
    from dataclasses import replace

    return replace(ranges, **overrides)


def jitter_policy_from_config(section: Mapping[str, Any] | None) -> JitterRanges | None:
    """Resolve the configured random (default) or disabled policy."""
    if not section:
        return None
    mode = str(section.get("mode", "random")).lower()
    if mode == "random":
        return ranges_from_config(section)
    if mode in {"off", "none", "false"}:
        return None
    raise ValueError("paste_jitter.mode must be one of: random, off")


def sample_frame(ranges: JitterRanges, rng: random.Random) -> FrameJitter:
    return FrameJitter(
        color=rng.uniform(*ranges.color),
        brightness=rng.uniform(*ranges.brightness),
        contrast=rng.uniform(*ranges.contrast),
        sharpness=rng.uniform(*ranges.sharpness),
        flip=rng.random() < ranges.flip_prob,
        scale=rng.uniform(*ranges.scale),
        rotation=rng.uniform(*ranges.rotation),
    )


def sample_sequence(
    ranges: JitterRanges | None, length: int, rng: random.Random
) -> list[FrameJitter]:
    """Draw one event sequence: an independent jitter for every exposure frame."""
    if ranges is None:
        return [IDENTITY] * int(length)
    return [sample_frame(ranges, rng) for _ in range(int(length))]


def expand_ratios(width: float, height: float, rotation: float) -> tuple[float, float]:
    """How much ``Image.rotate(rotation, expand=True)`` grows a ``width x height``
    patch, as (width ratio, height ratio).

    Mirrors PIL's own computation rather than the textbook
    ``w|cos| + h|sin|``: PIL rotates about the patch centre and only *then*
    rounds the corner bounds outward with ceil/floor, so the constant centre
    offset decides which side of an integer each bound lands on. Dropping it is
    off by a pixel at some angles (40x24 at -30 deg gives 47x41 instead of
    48x42), and a box one pixel off in one axis stretches the pasted crop.
    """
    if rotation % 360.0 == 0 or width <= 0 or height <= 0:
        return (1.0, 1.0)
    angle = -math.radians(rotation % 360.0)
    cosine, sine = round(math.cos(angle), 15), round(math.sin(angle), 15)
    center_x, center_y = width / 2, height / 2
    # PIL's reverse affine: rotate about the origin, then put the centre back.
    offset_x = cosine * -center_x + sine * -center_y + center_x
    offset_y = -sine * -center_x + cosine * -center_y + center_y
    corners = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    xs = [cosine * x + sine * y + offset_x for x, y in corners]
    ys = [-sine * x + cosine * y + offset_y for x, y in corners]
    new_width = math.ceil(max(xs)) - math.floor(min(xs))
    new_height = math.ceil(max(ys)) - math.floor(min(ys))
    return (new_width / width, new_height / height)


def jitter_box(
    box: Sequence[float], patch_size: tuple[float, float], jitter: FrameJitter
) -> tuple[float, float, float, float]:
    """Grow/shrink a placed box for one frame's rotation and scale.

    Anchored bottom-centre, the same anchor the unjittered placement uses, so a
    rotated or rescaled occluder stays grounded where it was put.
    """
    x, y, width, height = (float(value) for value in box)
    ratio_width, ratio_height = expand_ratios(patch_size[0], patch_size[1], jitter.rotation)
    new_width = width * ratio_width * jitter.scale
    new_height = height * ratio_height * jitter.scale
    center_x = x + width / 2.0
    bottom_y = y + height
    return (center_x - new_width / 2.0, bottom_y - new_height, new_width, new_height)


def _photometric(image: Any, jitter: FrameJitter) -> Any:
    from PIL import ImageEnhance

    for enhancer, factor in (
        (ImageEnhance.Color, jitter.color),
        (ImageEnhance.Brightness, jitter.brightness),
        (ImageEnhance.Contrast, jitter.contrast),
        (ImageEnhance.Sharpness, jitter.sharpness),
    ):
        if factor != 1.0:
            image = enhancer(image).enhance(factor)
    return image


def _geometric(image: Any, jitter: FrameJitter, resample: Any) -> Any:
    from PIL import Image

    if jitter.flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if jitter.rotation % 360 != 0:
        image = image.rotate(jitter.rotation, resample=resample, expand=True)
    return image


def jitter_alpha(alpha: np.ndarray, jitter: FrameJitter) -> np.ndarray:
    """Geometric half only — what the rho integral image is built from.

    NEAREST keeps the mask binary; a bilinear rotation would feather the edge and
    the compositor's ``alpha > 0`` test would silently grow the mask.
    """
    if not (jitter.flip or jitter.rotation % 360 != 0):
        return alpha
    from PIL import Image

    image = _geometric(
        Image.fromarray(np.asarray(alpha, dtype=np.uint8)), jitter, Image.Resampling.NEAREST
    )
    return np.asarray(image).copy()


def jitter_rgba(rgba: np.ndarray, jitter: FrameJitter) -> np.ndarray:
    """Full jitter for the pasted patch. Scale is not applied here — it rides on
    the target box (see :func:`jitter_box`), which is what the crop is resized to."""
    patch = np.asarray(rgba, dtype=np.uint8)
    if patch.ndim != 3 or patch.shape[2] != 4:
        raise ValueError("paste jitter expects an RGBA patch")
    if jitter == IDENTITY:
        return patch
    from PIL import Image

    rgb = _photometric(Image.fromarray(patch[..., :3]), jitter)
    rgb = _geometric(rgb, jitter, Image.Resampling.BILINEAR)
    alpha = _geometric(
        Image.fromarray(patch[..., 3]), jitter, Image.Resampling.NEAREST
    )
    return np.dstack([np.asarray(rgb), np.asarray(alpha)]).copy()
