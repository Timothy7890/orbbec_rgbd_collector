from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_collector.pointcloud import (
    POINT_DTYPE,
    encode_point_cloud,
    frame_summaries,
    reconstruct_frame,
    session_summaries,
)
from rgbd_collector.storage import DatasetSession

from test_storage import camera_metadata, make_frame


class PointCloudTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.session = DatasetSession(
            self.root, "point cloud", camera_metadata()
        )
        self.frame = make_frame(1)
        self.frame_id = self.session.enqueue(self.frame, "manual")
        self.session.close()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reconstruct_aligned_depth_with_color_intrinsics(self) -> None:
        points, metadata = reconstruct_frame(
            self.root,
            self.session.session_id,
            self.frame_id,
            stride=1,
            min_depth_m=0.1,
            max_depth_m=2.0,
        )
        self.assertEqual(points.dtype, POINT_DTYPE)
        self.assertEqual(points.size, 42)
        self.assertEqual(metadata["point_count"], 42)

        z = 1.001
        expected_first = np.array(
            [(2.0 - 4.5) * z / 5.0, (1.0 - 3.5) * z / 5.0, z],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            points["xyz"][0], expected_first, rtol=1e-5, atol=1e-6
        )
        encoded = encode_point_cloud(points)
        self.assertEqual(encoded[:4], b"PCD1")
        self.assertEqual(
            int.from_bytes(encoded[4:8], "little"), points.size
        )
        self.assertEqual(len(encoded), 8 + points.size * 16)

    def test_dataset_browser_lists_sessions_and_frames(self) -> None:
        sessions = session_summaries(self.root)
        self.assertEqual(sessions[0]["id"], self.session.session_id)
        self.assertEqual(sessions[0]["frame_count"], 1)
        frames = frame_summaries(self.root, self.session.session_id)
        self.assertEqual(frames[0]["id"], self.frame_id)
        self.assertEqual(frames[0]["trigger"], "manual")


if __name__ == "__main__":
    unittest.main()
