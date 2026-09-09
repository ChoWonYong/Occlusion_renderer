import tempfile
import unittest
from pathlib import Path

from common.io import save_json
from synth.event_pipeline import _load_pool


class EventPoolSourcesTest(unittest.TestCase):
    def test_only_the_configured_pool_source_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = root / "detector"
            other = root / "other"
            record = {
                "id": 1,
                "category": "car",
                "source": "Co-DETR:KITTI",
                "sequence": "0000",
                "source_track_id": 7,
                "length": 1,
                "frames": [{"file_name": "00.png"}],
            }
            save_json(detector / "tracklets.json", {"tracklets": [record]})
            save_json(other / "tracklets.json", {"tracklets": [{**record, "id": 2}]})
            config = {"tracklet_synthesis": {"pool_sources": [str(detector)]}}
            pool = _load_pool(config, root)
            self.assertEqual(len(pool), 1)
            self.assertEqual(pool[0].source, "Co-DETR:KITTI")
            self.assertEqual(pool[0].frames[0]["rgba_path"], str(detector / "00.png"))

    def test_missing_pool_sources_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            _load_pool({"tracklet_synthesis": {}}, Path("."))


if __name__ == "__main__":
    unittest.main()
