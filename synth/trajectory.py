from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

from common.schema import Motion


@dataclass(frozen=True)
class FramePlacement:
    frame_index: int
    center_x: float
    center_y: float
    scale: float

    def xywh(self, patch_width: int, patch_height: int) -> tuple[int, int, int, int]:
        width = max(2, int(round(patch_width * self.scale)))
        height = max(2, int(round(patch_height * self.scale)))
        return (
            int(round(self.center_x - width / 2)),
            int(round(self.center_y - height / 2)),
            width,
            height,
        )


@dataclass(frozen=True)
class Trajectory:
    start_frame: int
    end_frame: int
    peak_frame: int
    motion: Motion
    placements: tuple[FramePlacement, ...]
    entry_side: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "peak_frame": self.peak_frame,
            "entry_side": self.entry_side,
            "motion": {
                "model": self.motion.model,
                "p0": list(self.motion.p0),
                "v0": list(self.motion.v0),
                "a": list(self.motion.acceleration),
                "scale0": self.motion.scale0,
                "scale_rate": self.motion.scale_rate,
            },
            "frames": [
                {
                    "frame_index": placement.frame_index,
                    "center": [placement.center_x, placement.center_y],
                    "scale": placement.scale,
                }
                for placement in self.placements
            ],
        }


def sample_crossing_trajectory(
    image_size: tuple[int, int],
    victim_bbox: Sequence[float],
    patch_size: tuple[int, int],
    start_frame: int,
    duration: int,
    motion_model: str,
    scale_range: tuple[float, float],
    scale_rate_range: tuple[float, float],
    rng: random.Random,
    patch_mask_area: int | None = None,
    peak_rho: float | None = None,
) -> Trajectory:
    if motion_model not in {"const_vel", "const_accel"}:
        raise ValueError(f"unsupported motion model: {motion_model}")
    image_width, image_height = image_size
    target_x, target_y, target_width, target_height = (float(value) for value in victim_bbox)
    peak_frame = start_frame + max(1, duration // 2)
    end_frame = start_frame + duration - 1
    peak_time = float(peak_frame - start_frame)
    patch_width, patch_height = patch_size
    base_scale = target_height / max(float(patch_height), 1.0)
    min_scale, max_scale = base_scale * scale_range[0], base_scale * scale_range[1]
    if patch_mask_area and peak_rho:
        desired_scale = (peak_rho * target_width * target_height / patch_mask_area) ** 0.5
        scale0 = min(max_scale, max(min_scale, desired_scale))
    else:
        scale0 = rng.uniform(min_scale, max_scale)
    scale_rate = scale0 * rng.uniform(*scale_rate_range)
    target_center_x = target_x + target_width / 2
    target_bottom = target_y + target_height
    target_center_y = min(image_height - 1.0, target_bottom - patch_height * scale0 / 2)
    entry_side = rng.choice(["left", "right"])
    margin = patch_width * scale0 / 2 + 2
    start_x = -margin if entry_side == "left" else image_width + margin
    acceleration_x = 0.0
    if motion_model == "const_accel":
        direction = 1.0 if entry_side == "left" else -1.0
        acceleration_x = direction * rng.uniform(0.0, max(0.01, image_width / max(duration * duration, 1)))
    velocity_x = (target_center_x - start_x - 0.5 * acceleration_x * peak_time**2) / peak_time
    velocity_y = rng.uniform(-0.15, 0.15)
    acceleration_y = rng.uniform(-0.01, 0.01) if motion_model == "const_accel" else 0.0
    motion = Motion(
        model=motion_model,
        p0=(start_x, target_center_y),
        v0=(velocity_x, velocity_y),
        acceleration=(acceleration_x, acceleration_y),
        scale0=scale0,
        scale_rate=scale_rate,
    )
    placements = []
    for frame_index in range(start_frame, end_frame + 1):
        time = float(frame_index - start_frame)
        center_x, center_y = motion.position(time)
        placements.append(
            FramePlacement(
                frame_index=frame_index,
                center_x=center_x,
                center_y=center_y,
                scale=motion.scale(time),
            )
        )
    return Trajectory(
        start_frame=start_frame,
        end_frame=end_frame,
        peak_frame=peak_frame,
        motion=motion,
        placements=tuple(placements),
        entry_side=entry_side,
    )
