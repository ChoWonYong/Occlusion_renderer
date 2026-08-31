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

Two policies are available:

``random`` (the default)
    The original independent ``low``/``mid``/``high`` draws.
``real``
    One physically named condition is held for the whole event: day, night,
    tunnel, rain, or snow. Day/night are exposure changes, tunnel adds warm
    centre-weighted lighting, rain locally displaces a sparse set of 2x2 RGB
    blocks, and snow adds the same weak precipitation distortion plus sparse
    white RGB pixels. These effects never alter alpha, so the placement search
    and rendered occlusion ratio remain identical.
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


@dataclass(frozen=True)
class RealScenario:
    """Parameters for one event-level real-world appearance condition."""

    name: str
    weight: float
    brightness: tuple[float, float]
    temporal_variation: float = 0.02
    warmth: float = 0.0
    vignette: float = 0.0
    elastic_fraction: float = 0.0
    elastic_block_size: int = 2
    elastic_displacement: int = 1
    snow_fraction: float = 0.0
    snow_block_size: int = 1


@dataclass(frozen=True)
class RealJitterPolicy:
    """Validated collection of real scenarios sampled once per event."""

    scenarios: tuple[RealScenario, ...]


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
    real_scenario: str = "none"
    effect_seed: int = 0
    warmth: float = 0.0
    vignette: float = 0.0
    elastic_fraction: float = 0.0
    elastic_block_size: int = 2
    elastic_displacement: int = 1
    snow_fraction: float = 0.0
    snow_block_size: int = 1

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
            "real_scenario": self.real_scenario,
            "effect_seed": self.effect_seed,
            "warmth": self.warmth,
            "vignette": self.vignette,
            "elastic_fraction": self.elastic_fraction,
            "elastic_block_size": self.elastic_block_size,
            "elastic_displacement": self.elastic_displacement,
            "snow_fraction": self.snow_fraction,
            "snow_block_size": self.snow_block_size,
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


def _pair(value: Any, *, field: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two numbers")
    pair = (float(value[0]), float(value[1]))
    if pair[0] > pair[1]:
        raise ValueError(f"{field} lower bound must not exceed upper bound")
    return pair


def real_policy_from_config(section: Mapping[str, Any] | None) -> RealJitterPolicy:
    """Parse ``paste_jitter.real`` and validate every configured scenario."""
    if not section:
        raise ValueError("real jitter mode requires a non-empty 'real' section")
    names = section.get("scenarios", ["day", "night", "tunnel", "rain", "snow"])
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
        raise ValueError("real.scenarios must be a non-empty sequence")
    weights = section.get("weights", {})
    definitions = section.get("definitions", section)
    scenarios: list[RealScenario] = []
    for raw_name in names:
        name = str(raw_name).lower()
        raw = definitions.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"real jitter scenario {name!r} has no definition")
        brightness = _pair(raw.get("brightness", [1.0, 1.0]), field=f"real.{name}.brightness")
        weight = float(weights.get(name, raw.get("weight", 1.0)))
        if weight < 0:
            raise ValueError(f"real jitter weight for {name!r} must be non-negative")
        scenario = RealScenario(
            name=name,
            weight=weight,
            brightness=brightness,
            temporal_variation=float(raw.get("temporal_variation", 0.02)),
            warmth=float(raw.get("warmth", 0.0)),
            vignette=float(raw.get("vignette", 0.0)),
            elastic_fraction=float(raw.get("elastic_fraction", 0.0)),
            elastic_block_size=int(raw.get("elastic_block_size", 2)),
            elastic_displacement=int(raw.get("elastic_displacement", 1)),
            snow_fraction=float(raw.get("snow_fraction", 0.0)),
            snow_block_size=int(raw.get("snow_block_size", 1)),
        )
        for field in ("temporal_variation", "vignette", "elastic_fraction", "snow_fraction"):
            value = float(getattr(scenario, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"real.{name}.{field} must be in [0, 1]")
        if scenario.elastic_block_size < 1 or scenario.snow_block_size < 1:
            raise ValueError(f"real.{name} block sizes must be positive")
        if scenario.elastic_displacement < 0:
            raise ValueError(f"real.{name}.elastic_displacement must be non-negative")
        scenarios.append(scenario)
    if not any(item.weight > 0 for item in scenarios):
        raise ValueError("at least one real jitter scenario must have positive weight")
    return RealJitterPolicy(tuple(scenarios))


def jitter_policy_from_config(
    section: Mapping[str, Any] | None,
) -> JitterRanges | RealJitterPolicy | None:
    """Resolve the configured random (default), real, or disabled policy."""
    if not section:
        return None
    mode = str(section.get("mode", "random")).lower()
    if mode == "random":
        return ranges_from_config(section)
    if mode == "real":
        return real_policy_from_config(section.get("real"))
    if mode in {"off", "none", "false"}:
        return None
    raise ValueError("paste_jitter.mode must be one of: random, real, off")


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
    ranges: JitterRanges | RealJitterPolicy | None,
    length: int,
    rng: random.Random,
    *,
    scenario: str | None = None,
) -> list[FrameJitter]:
    """Draw one event sequence; real mode holds one scenario for every frame."""
    if ranges is None:
        return [IDENTITY] * int(length)
    if isinstance(ranges, RealJitterPolicy):
        return sample_real_sequence(ranges, length, rng, scenario=scenario)
    return [sample_frame(ranges, rng) for _ in range(int(length))]


def sample_real_sequence(
    policy: RealJitterPolicy,
    length: int,
    rng: random.Random,
    *,
    scenario: str | None = None,
) -> list[FrameJitter]:
    """Sample an event-level condition with small frame-to-frame exposure drift."""
    if scenario is None:
        selected = rng.choices(
            policy.scenarios,
            weights=[item.weight for item in policy.scenarios],
            k=1,
        )[0]
    else:
        matches = [item for item in policy.scenarios if item.name == str(scenario).lower()]
        if not matches:
            raise ValueError(f"real jitter scenario {scenario!r} is not configured")
        selected = matches[0]
    base_brightness = rng.uniform(*selected.brightness)
    output: list[FrameJitter] = []
    for _ in range(int(length)):
        drift = rng.uniform(-selected.temporal_variation, selected.temporal_variation)
        output.append(
            FrameJitter(
                brightness=max(0.0, base_brightness * (1.0 + drift)),
                real_scenario=selected.name,
                effect_seed=rng.randrange(0, 2**32),
                warmth=selected.warmth,
                vignette=selected.vignette,
                elastic_fraction=selected.elastic_fraction,
                elastic_block_size=selected.elastic_block_size,
                elastic_displacement=selected.elastic_displacement,
                snow_fraction=selected.snow_fraction,
                snow_block_size=selected.snow_block_size,
            )
        )
    return output


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


def _eligible_origins(alpha: np.ndarray, block_size: int) -> np.ndarray:
    """Return top-left block coordinates whose block intersects the object."""
    height, width = alpha.shape
    if height < block_size or width < block_size:
        return np.empty((0, 2), dtype=np.int64)
    ys, xs = np.nonzero(alpha[: height - block_size + 1, : width - block_size + 1] > 0)
    return np.column_stack([ys, xs])


def _sparse_elastic(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    fraction: float,
    block_size: int,
    displacement: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Locally displace sparse RGB blocks without changing the alpha mask."""
    if fraction <= 0 or displacement <= 0:
        return rgb
    origins = _eligible_origins(alpha, block_size)
    if not len(origins):
        return rgb
    opaque = max(1, int(np.count_nonzero(alpha)))
    count = max(1, round(opaque * fraction / (block_size * block_size)))
    chosen = origins[rng.integers(0, len(origins), size=count)]
    source = rgb.copy()
    output = rgb.copy()
    height, width = alpha.shape
    for y, x in chosen:
        dy = int(rng.integers(-displacement, displacement + 1))
        dx = int(rng.integers(-displacement, displacement + 1))
        if dx == 0 and dy == 0:
            dx = 1
        source_y = min(max(0, int(y) + dy), height - block_size)
        source_x = min(max(0, int(x) + dx), width - block_size)
        output[y : y + block_size, x : x + block_size] = source[
            source_y : source_y + block_size,
            source_x : source_x + block_size,
        ]
    return output


def _sparse_snow(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    fraction: float,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Turn a sparse set of object pixels white; alpha and geometry stay fixed."""
    if fraction <= 0:
        return rgb
    origins = _eligible_origins(alpha, block_size)
    if not len(origins):
        return rgb
    opaque = max(1, int(np.count_nonzero(alpha)))
    count = max(1, round(opaque * fraction / (block_size * block_size)))
    chosen = origins[rng.integers(0, len(origins), size=count)]
    output = rgb.copy()
    for y, x in chosen:
        mask = alpha[y : y + block_size, x : x + block_size] > 0
        block = output[y : y + block_size, x : x + block_size]
        block[mask] = 255
    return output


def apply_real_effect_rgb(
    rgb: np.ndarray, alpha: np.ndarray, jitter: FrameJitter
) -> np.ndarray:
    """Apply deterministic RGB-only tunnel/precipitation effects."""
    if jitter.real_scenario == "none":
        return rgb
    output = np.asarray(rgb, dtype=np.float32).copy()
    if jitter.warmth != 0.0:
        # Positive warmth raises red and gently suppresses blue.
        output[..., 0] *= 1.0 + jitter.warmth
        output[..., 2] *= max(0.0, 1.0 - jitter.warmth)
    if jitter.vignette > 0.0:
        height, width = alpha.shape
        yy, xx = np.ogrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
        radius = np.clip((xx * xx + yy * yy) / 2.0, 0.0, 1.0)
        output *= (1.0 - jitter.vignette * radius)[..., None]
    output_u8 = np.clip(output, 0, 255).astype(np.uint8)
    generator = np.random.default_rng(jitter.effect_seed)
    output_u8 = _sparse_elastic(
        output_u8,
        alpha,
        fraction=jitter.elastic_fraction,
        block_size=jitter.elastic_block_size,
        displacement=jitter.elastic_displacement,
        rng=generator,
    )
    return _sparse_snow(
        output_u8,
        alpha,
        fraction=jitter.snow_fraction,
        block_size=jitter.snow_block_size,
        rng=generator,
    )


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


def jitter_rgba(
    rgba: np.ndarray, jitter: FrameJitter, *, apply_real_effects: bool = True
) -> np.ndarray:
    """Full jitter for the pasted patch. Scale is not applied here — it rides on
    the target box (see :func:`jitter_box`), which is what the crop is resized to."""
    patch = np.asarray(rgba, dtype=np.uint8)
    if patch.ndim != 3 or patch.shape[2] != 4:
        raise ValueError("paste jitter expects an RGBA patch")
    if jitter == IDENTITY:
        return patch
    from PIL import Image

    rgb = _photometric(Image.fromarray(patch[..., :3]), jitter)
    if apply_real_effects:
        rgb = Image.fromarray(apply_real_effect_rgb(np.asarray(rgb), patch[..., 3], jitter))
    rgb = _geometric(rgb, jitter, Image.Resampling.BILINEAR)
    alpha = _geometric(
        Image.fromarray(patch[..., 3]), jitter, Image.Resampling.NEAREST
    )
    return np.dstack([np.asarray(rgb), np.asarray(alpha)]).copy()
