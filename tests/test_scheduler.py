import random
import unittest
from collections import Counter

from synth.scheduler import (
    EventPlan,
    PoolTracklet,
    identity_available,
    merge_pools,
    paste_exposure_length,
    plan_schedule,
    sample_categories,
    sample_occluders,
    select_eligible_victims,
    victim_event_count,
)


def _tracklet(tid: int, category: str, source: str, seq: str, track: int, length: int = 30) -> PoolTracklet:
    frames = [{"offset": i, "file_name": f"t{tid}/{i:02d}.png"} for i in range(length)]
    return PoolTracklet(
        tracklet_id=tid,
        category=category,
        source=source,
        length=length,
        frames=frames,
        identity_key=(source, seq, track),
    )


def _victim_frame(track_id: int, area: float = 1000.0, occluded: int = 0, category_id: int = 1) -> dict:
    return {
        "track_id": track_id,
        "category_id": category_id,
        "area": area,
        "bbox": [0, 0, 30, 30],
        "kitti": {"occluded": occluded},
    }


class MergePoolsTest(unittest.TestCase):
    def test_merges_and_assigns_global_ids_and_categories(self) -> None:
        mot = {"tracklets": [{"category": "person", "source": "MOT17", "sequence": "M-02",
                              "source_track_id": 7, "length": 2, "frames": [{}, {}]}]}
        kitti = {"tracklets": [{"category": "car", "source": "KITTI", "sequence": "0000",
                                "source_track_id": 3, "length": 1, "frames": [{}]}]}
        pool = merge_pools([mot, kitti])
        self.assertEqual([t.tracklet_id for t in pool], [1, 2])
        self.assertEqual([t.category for t in pool], ["person", "car"])
        self.assertEqual(pool[0].identity_key, ("MOT17", "M-02", 7))

    def test_length_mismatch_is_rejected(self) -> None:
        bad = {"tracklets": [{"category": "car", "length": 5, "frames": [{}]}]}
        with self.assertRaisesRegex(ValueError, "length"):
            merge_pools([bad])


class VictimEventCountTest(unittest.TestCase):
    def test_scales_with_length(self) -> None:
        self.assertEqual(victim_event_count(100, 3.0), 3)
        self.assertEqual(victim_event_count(300, 3.0), 9)
        self.assertEqual(victim_event_count(800, 3.0), 24)
        self.assertEqual(victim_event_count(10, 3.0), 1)  # floor at 1


class SampleCategoriesTest(unittest.TestCase):
    def test_apportions_by_ratio(self) -> None:
        counts = Counter(sample_categories(10, {"car": 0.6, "person": 0.3, "bicycle": 0.1}, random.Random(0)))
        self.assertEqual(counts, Counter({"car": 6, "person": 3, "bicycle": 1}))

    def test_largest_remainder_for_count_four(self) -> None:
        counts = Counter(sample_categories(4, {"car": 0.6, "person": 0.3, "bicycle": 0.1}, random.Random(1)))
        self.assertEqual(counts, Counter({"car": 3, "person": 1}))


class SelectVictimsTest(unittest.TestCase):
    def test_filters_area_occlusion_and_short_presence(self) -> None:
        frames = []
        for position in range(40):
            row = [_victim_frame(1)]                       # clean, large, present everywhere
            row.append(_victim_frame(2, area=100.0))       # too small
            row.append(_victim_frame(3, occluded=2))       # occluded
            if position < 3:
                row.append(_victim_frame(4))               # present too briefly
            frames.append(row)
        victims = select_eligible_victims(frames, min_area=400.0, max_base_occlusion=0, min_presence_frames=8)
        self.assertEqual([v.track_id for v in victims], [1])
        self.assertEqual(victims[0].presence_frames, 40)


class SampleOccludersTest(unittest.TestCase):
    def _grouped(self) -> dict[str, list[PoolTracklet]]:
        return {
            "car": [_tracklet(1, "car", "KITTI", "0000", 1), _tracklet(2, "car", "KITTI", "0000", 2)]
        }

    def test_no_identity_repeats_within_one_sequence(self) -> None:
        chosen, over_cap = sample_occluders(
            ["car"] * 2, self._grouped(), random.Random(0), max_per_identity=2
        )
        self.assertEqual(len(chosen), 2)
        self.assertEqual(over_cap, 0)
        self.assertEqual(len({t.identity_key for t in chosen}), 2)

    def test_overflow_reported_when_the_pool_is_exhausted(self) -> None:
        # only two identities but five picks: the same-sequence rule forces the
        # last three over, and they are reported rather than silently allowed
        chosen, over_cap = sample_occluders(
            ["car"] * 5, self._grouped(), random.Random(0), max_per_identity=2
        )
        self.assertEqual(len(chosen), 5)
        self.assertEqual(over_cap, 3)

    def test_global_cap_carries_across_sequences(self) -> None:
        grouped = self._grouped()
        usage: dict = {}
        for _ in range(2):  # two "sequences", fresh exclusion set each time
            sample_occluders(
                ["car"] * 2, grouped, random.Random(0), max_per_identity=2, usage=usage
            )
        self.assertEqual(sorted(usage.values()), [2, 2])
        # a third sequence has nothing left within the cap
        chosen, over_cap = sample_occluders(
            ["car"] * 2, grouped, random.Random(0), max_per_identity=2, usage=usage
        )
        self.assertEqual(over_cap, 2)


class IdentityAvailabilityTest(unittest.TestCase):
    def test_blocked_by_either_rule(self) -> None:
        tracklet = _tracklet(1, "car", "KITTI", "0000", 1)
        self.assertTrue(identity_available(tracklet, {}, set(), 2))
        # already used in this sequence
        self.assertFalse(identity_available(tracklet, {}, {tracklet.identity_key}, 2))
        # global cap reached
        self.assertFalse(identity_available(tracklet, {tracklet.identity_key: 2}, set(), 2))


class PasteExposureTest(unittest.TestCase):
    def test_within_range(self) -> None:
        rng = random.Random(0)
        for _ in range(200):
            value = paste_exposure_length(80, rng, min_frames=30)
            self.assertTrue(30 <= value <= 80)

    def test_too_short_raises(self) -> None:
        with self.assertRaises(ValueError):
            paste_exposure_length(20, random.Random(0), min_frames=30)


class PlanScheduleTest(unittest.TestCase):
    def _pool(self) -> list[PoolTracklet]:
        return [
            _tracklet(1, "car", "KITTI", "0000", 10),
            _tracklet(2, "car", "KITTI", "0000", 11),
            _tracklet(3, "car", "KITTI", "0001", 12),
            _tracklet(4, "person", "MOT17", "M-02", 20),
            _tracklet(5, "person", "MOT17", "M-04", 21),
            _tracklet(6, "bicycle", "KITTI", "0000", 30),
        ]

    def _frames(self, victim_ids: list[int], frame_count: int) -> list[list[dict]]:
        return [[_victim_frame(tid) for tid in victim_ids] for _ in range(frame_count)]

    def _synthesis(self) -> dict:
        return {
            "class_ratio": {"car": 0.6, "person": 0.3, "bicycle": 0.1},
            "victim_events_per_100_frames": 3.0,
            "tracklet_min_frames": 30,
            "max_per_identity": 2,
            "effective_event_frames": [8, 20],
            "victim_min_area": 400.0,
            "victim_max_base_occlusion": 0,
        }

    def test_events_capped_by_distinct_victims_one_per_victim(self) -> None:
        frames = self._frames([1, 2, 3, 4], frame_count=200)  # target 6, victims 4
        result = plan_schedule(frames, self._pool(), self._synthesis(), random.Random(7))
        summary = result["summary"]
        self.assertEqual(summary["target_victim_events"], 6)
        self.assertEqual(summary["distinct_eligible_victims"], 4)
        self.assertEqual(summary["victim_events"], 4)  # capped by victims

        victim_ids = [e.victim_track_id for e in result["events"]]
        self.assertEqual(len(set(victim_ids)), 4)  # one event per victim, all distinct
        for event in result["events"]:
            self.assertEqual(event.kind, "victim")
            self.assertTrue(30 <= event.exposure_length <= event.occluder.length)

    def test_no_boundary_events_are_planned(self) -> None:
        """Boundary/truncation events were removed: they were never rendered."""
        frames = self._frames([1, 2, 3, 4], frame_count=200)
        result = plan_schedule(frames, self._pool(), self._synthesis(), random.Random(7))
        self.assertTrue(all(event.kind == "victim" for event in result["events"]))
        self.assertTrue(all(event.victim_track_id is not None for event in result["events"]))
        self.assertNotIn("boundary_events", result["summary"])

    def test_identities_are_unique_within_a_sequence(self) -> None:
        frames = self._frames([1, 2, 3, 4], frame_count=200)
        result = plan_schedule(frames, self._pool(), self._synthesis(), random.Random(7))
        identities = [event.occluder.identity_key for event in result["events"]]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(result["used_identities"], set(identities))

    def test_global_usage_map_is_shared_across_calls(self) -> None:
        frames = self._frames([1, 2, 3, 4], frame_count=200)
        usage: dict = {}
        plan_schedule(frames, self._pool(), self._synthesis(), random.Random(7), usage=usage)
        first = sum(usage.values())
        plan_schedule(frames, self._pool(), self._synthesis(), random.Random(8), usage=usage)
        self.assertGreater(sum(usage.values()), first)


if __name__ == "__main__":
    unittest.main()
