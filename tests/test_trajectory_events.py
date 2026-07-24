import random
import unittest

from label.events import derive_events
from synth.trajectory import sample_crossing_trajectory


class TrajectoryAndEventTest(unittest.TestCase):
    def test_crossing_trajectory_peaks_at_victim_x(self) -> None:
        trajectory = sample_crossing_trajectory(
            image_size=(320, 180),
            victim_bbox=[120, 70, 80, 60],
            patch_size=(40, 50),
            start_frame=10,
            duration=20,
            motion_model="const_vel",
            scale_range=(0.6, 1.4),
            scale_rate_range=(0.0, 0.0),
            rng=random.Random(2),
            patch_mask_area=1500,
            peak_rho=0.4,
        )
        peak = next(item for item in trajectory.placements if item.frame_index == trajectory.peak_frame)
        self.assertAlmostEqual(peak.center_x, 160.0)
        self.assertEqual(len(trajectory.placements), 20)

    def test_event_derivation_splits_non_contiguous_runs(self) -> None:
        histories = [
            {"frame_index": frame, "victim_track": 1, "occluder_track": 9, "occlusion_ratio": ratio}
            for frame, ratio in [(2, 0.2), (3, 0.5), (7, 0.3)]
        ]
        events = derive_events(histories, video_id=4, entry_speed=3.0)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].frame_peak, 3)
        self.assertEqual(events[0].duration, 2)


if __name__ == "__main__":
    unittest.main()

