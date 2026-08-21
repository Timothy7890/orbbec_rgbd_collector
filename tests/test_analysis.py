from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_collector.analysis import (
    apply_wall_calibration,
    build_wall_calibration,
    fit_dominant_plane,
    load_annotations,
    load_wall_calibration,
    save_annotation,
    save_wall_calibration,
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
            fitted["normal_camera"], [0.0, 0.0, 1.0], atol=0.01
        )
        wall_x = np.asarray(fitted["x_axis_camera"])
        wall_y = np.asarray(fitted["y_axis_camera"])
        wall_z = np.asarray(fitted["z_axis_camera"])
        np.testing.assert_allclose(np.cross(wall_x, wall_y), wall_z, atol=1e-8)
        self.assertLess(wall_z[1], -0.99)
        self.assertGreater(float(wall_y @ fitted["origin_camera_m"]), 0.0)

    def test_tilted_wall_frame_is_right_handed_and_points_up(self) -> None:
        rng = np.random.default_rng(8)
        expected_y = np.array([0.2, 0.1, 0.974679])
        expected_y /= np.linalg.norm(expected_y)
        camera_up = np.array([0.0, -1.0, 0.0])
        expected_z = camera_up - (camera_up @ expected_y) * expected_y
        expected_z /= np.linalg.norm(expected_z)
        expected_x = np.cross(expected_y, expected_z)
        origin = expected_y * 1.2
        along_x = rng.uniform(-0.5, 0.5, 4_000)
        along_z = rng.uniform(-0.4, 0.4, 4_000)
        noise = rng.normal(0.0, 0.001, 4_000)
        points = (
            origin
            + along_x[:, None] * expected_x
            + along_z[:, None] * expected_z
            + noise[:, None] * expected_y
        )

        fitted = fit_dominant_plane(points, threshold_m=0.005, iterations=120)
        wall_x = np.asarray(fitted["x_axis_camera"])
        wall_y = np.asarray(fitted["y_axis_camera"])
        wall_z = np.asarray(fitted["z_axis_camera"])
        self.assertGreater(float(wall_y @ expected_y), 0.999)
        self.assertGreater(float(wall_z @ camera_up), 0.99)
        np.testing.assert_allclose(np.cross(wall_x, wall_y), wall_z, atol=1e-8)
        fitted_origin = np.asarray(fitted["origin_camera_m"])
        self.assertAlmostEqual(float(fitted_origin @ wall_x), 0.0, places=8)
        self.assertAlmostEqual(float(fitted_origin @ wall_z), 0.0, places=8)

    def test_annotation_is_replaced_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DatasetSession(root, "annotations", camera_metadata())
            frame_id = session.enqueue(make_frame(1), "manual")
            session.close()
            plane = {
                "origin_camera_m": [0.0, 0.0, 1.0],
                "normal_camera": [0.0, 0.0, 1.0],
                "x_axis_camera": [1.0, 0.0, 0.0],
                "y_axis_camera": [0.0, 0.0, 1.0],
                "z_axis_camera": [0.0, -1.0, 0.0],
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
            self.assertAlmostEqual(local["x_m"], 0.1)
            self.assertAlmostEqual(local["y_m"], -0.05)
            self.assertAlmostEqual(local["z_m"], -0.2)
            relative = first["target_relative_to_semantic_m"]
            self.assertAlmostEqual(relative["x_m"], 0.02)
            self.assertAlmostEqual(relative["y_m"], -0.02)
            self.assertAlmostEqual(relative["z_m"], -0.02)
            self.assertEqual(first["schema"], "rgbd-target-annotation/v2")
            self.assertEqual(
                first["target_adjustment_wall_m"],
                {"x_m": 0.001, "y_m": 0.003, "z_m": 0.002},
            )
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
            "normal_camera": [0.0, 0.0, 1.0],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
        }
        with self.assertRaises(ValueError):
            target_plane_coordinates([0.0, float("nan"), 1.0], plane)

    def test_two_points_calibrate_and_persist_wall_frame(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "center_camera_m": [0.05, 0.04, 1.0],
            "normal_camera": [0.0, 0.0, 1.0],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
        }
        calibration = build_wall_calibration(
            plane,
            [0.1, 0.2, 0.98],
            [0.6, 0.2, 1.02],
        )
        np.testing.assert_allclose(
            calibration["origin_camera_m"], [0.1, 0.2, 1.0], atol=1e-9
        )
        np.testing.assert_allclose(
            calibration["x_axis_camera"], [1.0, 0.0, 0.0], atol=1e-9
        )
        np.testing.assert_allclose(
            calibration["y_axis_camera"], [0.0, 0.0, 1.0], atol=1e-9
        )
        np.testing.assert_allclose(
            calibration["z_axis_camera"], [0.0, -1.0, 0.0], atol=1e-9
        )
        self.assertAlmostEqual(calibration["x_baseline_m"], 0.5)
        self.assertEqual(
            calibration["schema"], "rgbd-wall-coordinate-calibration/v2"
        )

        applied = apply_wall_calibration(plane, calibration)
        self.assertTrue(applied["calibrated"])
        self.assertEqual(applied["center_camera_m"], plane["center_camera_m"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DatasetSession(root, "calibration", camera_metadata())
            frame_id = session.enqueue(make_frame(1), "manual")
            other_frame_id = session.enqueue(make_frame(2), "manual")
            session.close()
            path = save_wall_calibration(
                root, session.session_id, frame_id, calibration
            )
            self.assertEqual(path.name, f"{frame_id}.json")
            self.assertEqual(
                path.parent.parent.name, "wall_coordinate_calibrations"
            )
            self.assertEqual(
                load_wall_calibration(root, session.session_id, frame_id),
                calibration,
            )
            self.assertIsNone(
                load_wall_calibration(
                    root, session.session_id, other_frame_id
                )
            )

    def test_wall_calibration_rejects_short_x_baseline(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "normal_camera": [0.0, 0.0, 1.0],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
        }
        with self.assertRaisesRegex(ValueError, "至少为 5 cm"):
            build_wall_calibration(
                plane,
                [0.0, 0.0, 1.0],
                [0.0, -0.02, 1.0],
            )

    def test_semantic_cluster_prefers_points_in_front_of_plane(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "normal_camera": [0.0, 0.0, 1.0],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
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
            "normal_camera": [0.0, 0.0, 1.0],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
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
