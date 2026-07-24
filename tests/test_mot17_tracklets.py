import tempfile
import unittest
from pathlib import Path

from data.mot17 import Mot17Object, resample_run, strict_tracklets, variable_fps_tracklets


def _object(frame: int, visibility: float = 0.9, track_id: int = 7) -> Mot17Object:
    return Mot17Object(
        frame_index=frame,
        track_id=track_id,
        bbox=(100.0, 200.0, 40.0, 80.0),
        mark=1,
        category_id=1,
        visibility=visibility,
    )


class Mot17TrackletTest(unittest.TestCase):
    def _make_sequence(self, rows: list[str], frame_rate: int | None = None) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "MOT17-02-FRCNN"
        (root / "gt").mkdir(parents=True)
        (root / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        rate_line = f"frameRate={frame_rate}\n" if frame_rate is not None else ""
        (root / "seqinfo.ini").write_text(
            f"[Sequence]\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n{rate_line}",
            encoding="utf-8",
        )
        return root

    @staticmethod
    def _row(frame: int, visibility: float = 0.8, track_id: int = 7) -> str:
        return f"{frame},{track_id},100,200,40,80,1,1,{visibility}"

    def test_requires_all_30_consecutive_visible_frames(self) -> None:
        sequence = self._make_sequence([self._row(frame) for frame in range(1, 31)])
        tracklets = strict_tracklets(sequence, length=30, visibility_min=0.8)
        self.assertEqual(len(tracklets), 1)
        self.assertEqual(tracklets[0].start_frame, 1)
        self.assertEqual(tracklets[0].end_frame, 30)

    def test_low_visibility_splits_the_run(self) -> None:
        rows = [self._row(frame, 0.79 if frame == 15 else 0.9) for frame in range(1, 45)]
        sequence = self._make_sequence(rows)
        self.assertEqual(strict_tracklets(sequence, length=30, visibility_min=0.8), [])

    def test_missing_frame_splits_the_run(self) -> None:
        rows = [self._row(frame) for frame in range(1, 32) if frame != 16]
        sequence = self._make_sequence(rows)
        self.assertEqual(strict_tracklets(sequence, length=30, visibility_min=0.8), [])


class ResampleRunTest(unittest.TestCase):
    def test_30fps_downsamples_to_30_target_frames(self) -> None:
        run = [_object(frame) for frame in range(1, 91)]  # 3s at 30fps
        sampled = resample_run(run, 30.0, 10.0, min_frames=30, visibility_min=0.8)
        assert sampled is not None
        self.assertEqual(len(sampled), 30)
        # target 10fps over 30fps source => every 3rd source frame, starting at 1
        self.assertEqual([obj.frame_index for obj in sampled[:4]], [1, 4, 7, 10])

    def test_length_grows_with_longer_run(self) -> None:
        run = [_object(frame) for frame in range(1, 121)]  # 4s at 30fps
        sampled = resample_run(run, 30.0, 10.0, min_frames=30, visibility_min=0.8)
        assert sampled is not None
        self.assertEqual(len(sampled), 40)

    def test_10fps_is_one_to_one(self) -> None:
        run = [_object(frame) for frame in range(1, 31)]
        sampled = resample_run(run, 10.0, 10.0, min_frames=30, visibility_min=0.8)
        assert sampled is not None
        self.assertEqual([obj.frame_index for obj in sampled], list(range(1, 31)))

    def test_low_visibility_sample_is_substituted_by_neighbor(self) -> None:
        objects = [_object(frame) for frame in range(1, 91)]
        objects[3] = _object(4, visibility=0.5)  # frame 4 is a sampled position
        sampled = resample_run(objects, 30.0, 10.0, min_frames=30, visibility_min=0.8)
        assert sampled is not None
        self.assertEqual(len(sampled), 30)
        frames = [obj.frame_index for obj in sampled]
        self.assertNotIn(4, frames)
        self.assertIn(3, frames)  # nearest valid neighbor

    def test_short_run_is_rejected(self) -> None:
        run = [_object(frame) for frame in range(1, 61)]  # 2s at 30fps -> 20 samples
        self.assertIsNone(resample_run(run, 30.0, 10.0, min_frames=30, visibility_min=0.8))

    def test_unrecoverable_low_visibility_cuts_the_span(self) -> None:
        objects = [_object(frame) for frame in range(1, 91)]
        for index in (2, 3, 4):  # frames 3,4,5 all fail -> sample near 4 has no neighbor
            objects[index] = _object(index + 1, visibility=0.1)
        self.assertIsNone(resample_run(objects, 30.0, 10.0, min_frames=30, visibility_min=0.8))


class VariableFpsTrackletTest(Mot17TrackletTest):
    def test_builds_variable_length_from_seqinfo_fps(self) -> None:
        rows = [self._row(frame, visibility=0.9) for frame in range(1, 121)]
        sequence = self._make_sequence(rows, frame_rate=30)
        tracklets = variable_fps_tracklets(sequence, target_fps=10.0, min_frames=30)
        self.assertEqual(len(tracklets), 1)
        self.assertEqual(tracklets[0].length, 40)
        self.assertEqual(tracklets[0].source_fps, 30.0)
        self.assertEqual(tracklets[0].target_fps, 10.0)

    def test_missing_frame_splits_run_but_keeps_longest_valid_span(self) -> None:
        rows = [self._row(frame, visibility=0.9) for frame in range(1, 200) if frame != 100]
        sequence = self._make_sequence(rows, frame_rate=30)
        tracklets = variable_fps_tracklets(sequence, target_fps=10.0, min_frames=30)
        # frames 101..199 form a 99-frame run -> 33 target frames; 1..99 -> 33 too
        self.assertEqual(len(tracklets), 2)
        self.assertTrue(all(tracklet.length >= 30 for tracklet in tracklets))


if __name__ == "__main__":
    unittest.main()
