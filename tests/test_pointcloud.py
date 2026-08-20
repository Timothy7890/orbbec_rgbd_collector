from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_collector.pointcloud import (
    POINT_DTYPE,
    detection_pixel_mask,
    encode_point_cloud,
    frame_summaries,
    pixel_to_point,
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

    def test_dataset_browser_skips_incomplete_manifest_frame(self) -> None:
        frame_dir = self.session.path / "frames" / self.frame_id
        (frame_dir / "frame.json").unlink()

        frames = frame_summaries(self.root, self.session.session_id)
        sessions = session_summaries(self.root)

        self.assertEqual(frames, [])
        self.assertEqual(sessions[0]["frame_count"], 0)

    def test_semantic_boxes_recolor_matching_points(self) -> None:
        points, metadata = reconstruct_frame(
            self.root,
            self.session.session_id,
            self.frame_id,
            stride=1,
            min_depth_m=0.1,
            max_depth_m=2.0,
            boxes=[
                {
                    "cls": 2,
                    "name": "switch",
                    "conf": 0.9,
                    "xyxy": [0, 0, 4, 4],
                }
            ],
        )
        self.assertTrue(metadata["semantic"]["enabled"])
        self.assertGreater(
            metadata["semantic"]["class_point_counts"]["2"], 0
        )
        self.assertTrue(
            np.any(np.all(points["rgba"][:, :3] == [102, 187, 106], axis=1))
        )

    def test_instance_polygon_takes_priority_over_box(self) -> None:
        inside = detection_pixel_mask(
            np.array([1.0, 3.0, 3.0]),
            np.array([1.0, 1.0, 3.0]),
            {
                "xyxy": [0, 0, 4, 4],
                "polygon": [[0, 0], [4, 0], [0, 4]],
            },
            image_shape=(6, 6),
        )
        np.testing.assert_array_equal(inside, [True, True, False])

    def test_instance_polygon_limits_semantic_point_coloring(self) -> None:
        _, box_metadata = reconstruct_frame(
            self.root,
            self.session.session_id,
            self.frame_id,
            stride=1,
            min_depth_m=0.1,
            max_depth_m=2.0,
            boxes=[{"cls": 1, "conf": 0.9, "xyxy": [0, 0, 9, 7]}],
        )
        _, mask_metadata = reconstruct_frame(
            self.root,
            self.session.session_id,
            self.frame_id,
            stride=1,
            min_depth_m=0.1,
            max_depth_m=2.0,
            boxes=[
                {
                    "cls": 1,
                    "conf": 0.9,
                    "xyxy": [0, 0, 9, 7],
                    "polygon": [[2, 1], [5, 1], [2, 4]],
                }
            ],
        )
        box_count = box_metadata["semantic"]["class_point_counts"]["1"]
        mask_count = mask_metadata["semantic"]["class_point_counts"]["1"]
        self.assertGreater(mask_count, 0)
        self.assertLess(mask_count, box_count)

    def test_rgb_pixel_projects_to_same_point_cloud_coordinates(self) -> None:
        result = pixel_to_point(
            self.root,
            self.session.session_id,
            self.frame_id,
            u=4,
            v=3,
            search_radius=0,
            min_depth_m=0.1,
            max_depth_m=2.0,
        )
        expected = [
            (4 - 4.5) * 1.001 / 5.0,
            (3 - 3.5) * 1.001 / 5.0,
            1.001,
        ]
        np.testing.assert_allclose(
            result["point_camera_m"], expected, atol=1e-7
        )
        self.assertEqual(result["used_pixel"], {"u": 4, "v": 3})

    def test_rgb_pixel_uses_nearest_valid_aligned_depth(self) -> None:
        result = pixel_to_point(
            self.root,
            self.session.session_id,
            self.frame_id,
            u=0,
            v=0,
            search_radius=3,
            min_depth_m=0.1,
            max_depth_m=2.0,
        )
        self.assertEqual(result["used_pixel"], {"u": 2, "v": 1})
        self.assertAlmostEqual(result["search_distance_px"], np.sqrt(5))


if __name__ == "__main__":
    unittest.main()
