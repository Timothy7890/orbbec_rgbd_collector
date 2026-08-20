from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_collector.analysis import (
    fit_dominant_plane,
    load_annotations,
    save_annotation,
    semantic_clusters,
    target_plane_coordinates,
)
from rgbd_collector.storage import DatasetSession

from test_storage import camera_metadata, make_frame


class AnalysisTests(unittest.TestCase):
    def test_ransac_finds_plane_with_outliers(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.uniform(-0.5, 0.5, 4_000)
        y = rng.uniform(-0.4, 0.4, 4_000)
        z = 1.2 + rng.normal(0.0, 0.0015, 4_000)
        plane_points = np.column_stack((x, y, z))
        outliers = rng.uniform(
            [-0.6, -0.5, 0.4], [0.6, 0.5, 2.0], size=(700, 3)
        )
        fitted = fit_dominant_plane(
            np.vstack((plane_points, outliers)),
            threshold_m=0.006,
            iterations=160,
        )
        self.assertGreater(fitted["inlier_ratio"], 0.80)
        self.assertLess(fitted["rms_m"], 0.003)
        np.testing.assert_allclose(
            fitted["normal_camera"], [0.0, 0.0, -1.0], atol=0.01
        )

    def test_annotation_is_replaced_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DatasetSession(root, "annotations", camera_metadata())
            frame_id = session.enqueue(make_frame(1), "manual")
            session.close()
            plane = {
                "origin_camera_m": [0.0, 0.0, 1.0],
                "normal_camera": [0.0, 0.0, -1.0],
                "horizontal_axis_camera": [1.0, 0.0, 0.0],
                "vertical_axis_camera": [0.0, 1.0, 0.0],
            }
            first = save_annotation(
                root,
                session.session_id,
                frame_id,
                target_camera_m=[0.1, 0.2, 0.95],
                plane=plane,
                target_pixel={"u": 320, "v": 240},
                selection_source="rgb",
                target_adjustment_camera_m=[0.001, -0.002, 0.003],
                yolo={
                    "available": True,
                    "boxes": [],
                    "clusters": [
                        {
                            "name": "switch",
                            "centroid_camera_m": [0.08, 0.18, 0.97],
                        }
                    ],
                },
            )
            local = first["target_plane_coordinates_m"]
            self.assertAlmostEqual(local["horizontal_m"], 0.1)
            self.assertAlmostEqual(local["vertical_m"], 0.2)
            self.assertAlmostEqual(local["normal_m"], 0.05)
            relative = first["target_relative_to_semantic_m"]
            self.assertAlmostEqual(relative["horizontal_m"], 0.02)
            self.assertAlmostEqual(relative["vertical_m"], 0.02)
            self.assertAlmostEqual(relative["normal_m"], 0.02)
            self.assertEqual(first["target_pixel"], {"u": 320, "v": 240})
            self.assertEqual(first["selection_source"], "rgb")
            self.assertEqual(
                first["target_adjustment_camera_m"],
                [0.001, -0.002, 0.003],
            )

            save_annotation(
                root,
                session.session_id,
                frame_id,
                target_camera_m=[0.2, 0.2, 0.95],
                plane=plane,
            )
            records = load_annotations(root, session.session_id)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["target_camera_m"], [0.2, 0.2, 0.95])

    def test_target_plane_coordinates_validates_target(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "normal_camera": [0.0, 0.0, -1.0],
            "horizontal_axis_camera": [1.0, 0.0, 0.0],
            "vertical_axis_camera": [0.0, 1.0, 0.0],
        }
        with self.assertRaises(ValueError):
            target_plane_coordinates([0.0, float("nan"), 1.0], plane)

    def test_semantic_cluster_prefers_points_in_front_of_plane(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "normal_camera": [0.0, 0.0, -1.0],
            "horizontal_axis_camera": [1.0, 0.0, 0.0],
            "vertical_axis_camera": [0.0, 1.0, 0.0],
        }
        background = np.column_stack(
            (
                np.linspace(0.04, 0.16, 30),
                np.linspace(0.04, 0.16, 30),
                np.ones(30),
            )
        )
        foreground = np.column_stack(
            (
                np.linspace(0.08, 0.12, 30),
                np.linspace(0.08, 0.12, 30),
                np.full(30, 0.95),
            )
        )
        clusters = semantic_clusters(
            np.vstack((background, foreground)),
            {"intrinsics": {"fx": 100, "fy": 100, "cx": 100, "cy": 100}},
            [
                {
                    "cls": 1,
                    "name": "switch",
                    "conf": 0.9,
                    "xyxy": [100, 100, 120, 120],
                }
            ],
            plane,
        )
        self.assertEqual(len(clusters), 1)
        self.assertTrue(clusters[0]["foreground_only"])
        self.assertAlmostEqual(
            clusters[0]["centroid_camera_m"][2], 0.95, places=3
        )

    def test_semantic_cluster_uses_instance_polygon(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "normal_camera": [0.0, 0.0, -1.0],
            "horizontal_axis_camera": [1.0, 0.0, 0.0],
            "vertical_axis_camera": [0.0, 1.0, 0.0],
        }
        pixels = np.array(
            [[102, 102], [103, 102], [102, 103], [116, 116], [117, 116], [116, 117]],
            dtype=np.float64,
        )
        z = np.full(pixels.shape[0], 0.95)
        points = np.column_stack(
            (
                (pixels[:, 0] - 100) * z / 100,
                (pixels[:, 1] - 100) * z / 100,
                z,
            )
        )
        clusters = semantic_clusters(
            points,
            {
                "intrinsics": {"fx": 100, "fy": 100, "cx": 100, "cy": 100},
                "image_shape": [200, 200],
            },
            [
                {
                    "cls": 1,
                    "name": "switch",
                    "conf": 0.9,
                    "xyxy": [100, 100, 120, 120],
                    "polygon": [[100, 100], [106, 100], [100, 106]],
                }
            ],
            plane,
        )
        self.assertEqual(clusters[0]["point_count"], 3)
        self.assertLess(clusters[0]["centroid_camera_m"][0], 0.04)


if __name__ == "__main__":
    unittest.main()
