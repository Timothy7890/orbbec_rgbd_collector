from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rgbd_collector.models import RGBDFrame
from rgbd_collector.storage import DatasetSession, list_sessions


def make_frame(index: int = 1) -> RGBDFrame:
    color = np.zeros((8, 10, 3), dtype=np.uint8)
    color[..., 0] = 20 * index
    color[..., 1] = np.arange(10, dtype=np.uint8)
    raw = np.arange(42, dtype=np.uint16).reshape(6, 7) + index * 100
    aligned = np.zeros((8, 10), dtype=np.uint16)
    aligned[1:7, 2:9] = 1000 + index
    return RGBDFrame(
        color_bgr=color,
        depth_raw=raw,
        depth_aligned=aligned,
        host_time_ns=1_700_000_000_000_000_000 + index,
        color_timestamp_ms=100.0 + index,
        depth_timestamp_ms=99.8 + index,
        color_frame_index=index,
        depth_frame_index=index,
        depth_scale_mm=1.0,
    )


def camera_metadata() -> dict:
    return {
        "device": {"name": "Fake Orbbec", "serial": "TEST123"},
        "color": {
            "width": 10,
            "height": 8,
            "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 4.5, "cy": 3.5},
        },
        "depth": {
            "width": 7,
            "height": 6,
            "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 3.0, "cy": 2.5},
        },
        "depth_to_color": {
            "rotation_row_major": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
            "translation_unit": "mm",
        },
        "depth_scale": {"value": 1.0, "unit": "mm_per_raw_unit"},
    }


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_rgbd_frame_set_is_saved_losslessly(self) -> None:
        session = DatasetSession(
            self.root, "switch sample", camera_metadata(), queue_size=4
        )
        frame = make_frame(1)
        frame_id = session.enqueue(frame, "manual")
        session.close()

        frame_dir = session.path / "frames" / frame_id
        self.assertTrue((frame_dir / "color.jpg").is_file())
        self.assertTrue((frame_dir / "depth_raw.png").is_file())
        self.assertTrue((frame_dir / "depth_aligned.png").is_file())
        self.assertTrue((frame_dir / "frame.json").is_file())

        raw = cv2.imread(
            str(frame_dir / "depth_raw.png"), cv2.IMREAD_UNCHANGED
        )
        aligned = cv2.imread(
            str(frame_dir / "depth_aligned.png"), cv2.IMREAD_UNCHANGED
        )
        self.assertEqual(raw.dtype, np.uint16)
        self.assertEqual(aligned.dtype, np.uint16)
        np.testing.assert_array_equal(raw, frame.depth_raw)
        np.testing.assert_array_equal(aligned, frame.depth_aligned)

        record = json.loads(
            (frame_dir / "frame.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["frame_id"], frame_id)
        self.assertEqual(record["trigger"], "manual")
        self.assertEqual(record["depth_scale"]["value"], 1.0)
        self.assertAlmostEqual(record["timestamp_delta_ms"], 0.2)

        manifest_lines = (session.path / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(manifest_lines), 1)
        self.assertEqual(json.loads(manifest_lines[0])["frame_id"], frame_id)
        self.assertFalse(list((session.path / "frames").glob(".tmp-*")))

    def test_multiple_frames_and_session_index(self) -> None:
        session = DatasetSession(self.root, "连续 采集", camera_metadata())
        ids = [
            session.enqueue(make_frame(index), "interval")
            for index in range(1, 4)
        ]
        session.wait_idle()
        status = session.status()
        self.assertEqual(status["saved"], 3)
        self.assertEqual(status["failed"], 0)
        session.close()

        manifest = [
            json.loads(line)
            for line in (session.path / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            [record["frame_id"] for record in manifest], ids
        )
        indexed = list_sessions(self.root)
        self.assertEqual(indexed[0]["id"], session.session_id)
        self.assertEqual(indexed[0]["frames"], 3)

    def test_frame_validation_rejects_misaligned_depth(self) -> None:
        frame = make_frame()
        invalid = RGBDFrame(
            color_bgr=frame.color_bgr,
            depth_raw=frame.depth_raw,
            depth_aligned=np.zeros((4, 4), dtype=np.uint16),
            host_time_ns=frame.host_time_ns,
            color_timestamp_ms=None,
            depth_timestamp_ms=None,
            color_frame_index=None,
            depth_frame_index=None,
            depth_scale_mm=1.0,
        )
        with self.assertRaisesRegex(ValueError, "match color resolution"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
