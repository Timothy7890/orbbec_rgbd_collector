from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_collector.analysis import (
    analyze_frame,
    analyze_yolo_mask_panel,
    apply_wall_calibration,
    build_accepted_wall_calibration,
    build_wall_calibration,
    camera_plane_pose_metrics,
    describe_p0_boundary_lines,
    describe_segmented_plane_axes,
    estimate_wall_x_from_p0_boundary_lines,
    estimate_wall_x_from_plane_intersections,
    estimate_wall_x_from_secondary_plane_shape,
    fit_dominant_plane,
    fit_yolo_panel_rectangle,
    highest_confidence_semantic_pointcloud,
    load_annotations,
    load_wall_calibration,
    save_annotation,
    save_camera_plane_pose_measurements,
    save_highest_confidence_semantic_pointcloud,
    save_wall_calibration,
    segment_dominant_planes,
    semantic_clusters,
    split_plane_labels_by_connectivity,
    target_plane_coordinates,
)
from rgbd_collector.storage import DatasetSession

from test_storage import camera_metadata, make_frame


class AnalysisTests(unittest.TestCase):
    def test_camera_plane_pose_matches_ik_perpendicular_convention(self) -> None:
        front_facing = camera_plane_pose_metrics(
            {
                "normal_camera": [0.0, 0.0, 1.0],
                "center_camera_m": [0.1, -0.1, 0.8],
            }
        )
        self.assertAlmostEqual(front_facing["distance_m"], 0.8)
        self.assertAlmostEqual(front_facing["yaw_err_deg"], 0.0)
        self.assertAlmostEqual(front_facing["pitch_err_deg"], 0.0)
        self.assertAlmostEqual(front_facing["tilt_deg"], 0.0)
        np.testing.assert_allclose(
            front_facing["normal_cam"], [0.0, 0.0, -1.0]
        )

        angle = np.radians(10.0)
        tilted = camera_plane_pose_metrics(
            {
                "normal_camera": [np.sin(angle), 0.0, np.cos(angle)],
                "center_camera_m": [0.0, 0.0, 0.8],
            }
        )
        self.assertAlmostEqual(tilted["yaw_err_deg"], -10.0)
        self.assertAlmostEqual(tilted["pitch_err_deg"], 0.0)
        self.assertAlmostEqual(tilted["tilt_deg"], 10.0)

    def test_camera_plane_pose_batch_payload_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DatasetSession(root, "pose", camera_metadata())
            session.close()
            path, payload = save_camera_plane_pose_measurements(
                root,
                session.session_id,
                [
                    {
                        "ok": True,
                        "frame_id": "frame-1",
                        "distance_m": 0.8,
                    },
                    {
                        "ok": False,
                        "frame_id": "frame-2",
                        "error": "fit failed",
                    },
                ],
                options={"stride": 3, "max_points": 60_000},
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["success_count"], 1)
        self.assertEqual(payload["failure_count"], 1)
        self.assertEqual(saved["schema"], "rgbd-camera-plane-pose/v1")
        self.assertEqual(saved["frame_count"], 2)

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

    def test_multi_plane_segmentation_finds_only_large_planes(self) -> None:
        rng = np.random.default_rng(12)
        count = 2_500
        wall = np.column_stack(
            (
                rng.uniform(-0.6, 0.6, count),
                rng.uniform(-0.5, 0.5, count),
                1.2 + rng.normal(0.0, 0.001, count),
            )
        )
        side = np.column_stack(
            (
                0.65 + rng.normal(0.0, 0.001, count),
                rng.uniform(-0.5, 0.5, count),
                rng.uniform(0.5, 1.8, count),
            )
        )
        tiny = np.column_stack(
            (
                rng.uniform(-0.2, 0.2, 180),
                -0.4 + rng.normal(0.0, 0.001, 180),
                rng.uniform(0.7, 1.5, 180),
            )
        )
        outliers = rng.uniform(
            [-0.8, -0.7, 0.4], [0.8, 0.7, 2.0], size=(300, 3)
        )
        labels, planes = segment_dominant_planes(
            np.vstack((wall, side, tiny, outliers)),
            threshold_m=0.005,
            iterations=120,
            max_planes=4,
            min_inlier_ratio=0.05,
            min_inlier_count=500,
            seed=3,
        )

        self.assertEqual(len(planes), 2)
        wall_labels = labels[:count]
        side_labels = labels[count : count * 2]
        wall_label = int(np.bincount(wall_labels[wall_labels >= 0]).argmax())
        side_label = int(np.bincount(side_labels[side_labels >= 0]).argmax())
        self.assertNotEqual(wall_label, side_label)
        self.assertGreater(int(np.count_nonzero(wall_labels == wall_label)), 2400)
        self.assertGreater(int(np.count_nonzero(side_labels == side_label)), 2400)
        self.assertGreater(
            int(np.count_nonzero(labels[count * 2 : count * 2 + 180] == -1)),
            160,
        )
        descriptions = describe_segmented_plane_axes(
            np.vstack((wall, side, tiny, outliers)), labels, planes
        )
        self.assertEqual(len(descriptions), 2)
        for description in descriptions:
            self.assertGreater(
                description["long_length_m"],
                description["short_length_m"],
            )
            long_axis = np.asarray(description["long_axis_camera"])
            short_axis = np.asarray(description["short_axis_camera"])
            normal = np.asarray(description["normal_camera"])
            self.assertAlmostEqual(float(long_axis @ short_axis), 0.0, places=7)
            self.assertAlmostEqual(float(long_axis @ normal), 0.0, places=7)
            self.assertAlmostEqual(float(short_axis @ normal), 0.0, places=7)

    def test_two_cuts_create_three_independent_plane_patches(self) -> None:
        pixel_blocks = []
        for start_u in (0, 25, 50):
            uu, vv = np.meshgrid(
                np.arange(start_u, start_u + 15),
                np.arange(0, 25),
            )
            pixel_blocks.append(np.column_stack((uu.ravel(), vv.ravel())))
        pixels = np.vstack(pixel_blocks).astype(np.int32)
        points = np.column_stack(
            (
                pixels[:, 0] / 100.0,
                pixels[:, 1] / 100.0,
                np.ones(pixels.shape[0]),
            )
        )
        parent_labels = np.zeros(points.shape[0], dtype=np.int32)
        patch_labels, patches = split_plane_labels_by_connectivity(
            points,
            parent_labels,
            [
                {
                    "index": 0,
                    "origin_camera_m": [0.3, 0.12, 1.0],
                    "normal_camera": [0.0, 0.0, 1.0],
                    "inlier_count": points.shape[0],
                    "inlier_ratio": 1.0,
                    "rms_m": 0.0,
                }
            ],
            pixels,
            [40, 80],
            stride=1,
            source_point_count=points.shape[0],
            min_component_count=100,
        )

        self.assertEqual(len(patches), 3)
        self.assertEqual(
            sorted(np.bincount(patch_labels).tolist()), [375, 375, 375]
        )
        self.assertTrue(
            all(patch["parent_plane_index"] == 0 for patch in patches)
        )

    def test_disconnected_small_patches_are_not_merged_back_together(
        self,
    ) -> None:
        pixel_groups = []
        for start_u in (0, 30):
            uu, vv = np.meshgrid(
                np.arange(start_u, start_u + 10),
                np.arange(0, 10),
            )
            pixel_groups.append(np.column_stack((uu.ravel(), vv.ravel())))
        pixels = np.vstack(pixel_groups).astype(np.int32)
        points = np.column_stack(
            (
                pixels[:, 0] / 100.0,
                pixels[:, 1] / 100.0,
                np.ones(pixels.shape[0]),
            )
        )
        labels, patches = split_plane_labels_by_connectivity(
            points,
            np.zeros(points.shape[0], dtype=np.int32),
            [
                {
                    "index": 0,
                    "origin_camera_m": [0.0, 0.0, 1.0],
                    "normal_camera": [0.0, 0.0, 1.0],
                }
            ],
            pixels,
            [20, 50],
            stride=1,
            source_point_count=points.shape[0],
            min_component_count=150,
            min_component_ratio=0.0,
        )

        self.assertEqual(patches, [])
        self.assertTrue(np.all(labels == -1))

    def test_farthest_plane_is_preserved_as_single_p0(self) -> None:
        pixel_groups = []
        point_groups = []
        label_groups = []
        for depth, parent, start_v in ((2.0, 1, 0), (1.0, 0, 40)):
            for start_u in (0, 25):
                uu, vv = np.meshgrid(
                    np.arange(start_u, start_u + 15),
                    np.arange(start_v, start_v + 20),
                )
                pixels = np.column_stack((uu.ravel(), vv.ravel()))
                pixel_groups.append(pixels)
                point_groups.append(
                    np.column_stack(
                        (
                            pixels[:, 0] / 100.0,
                            pixels[:, 1] / 100.0,
                            np.full(pixels.shape[0], depth),
                        )
                    )
                )
                label_groups.append(
                    np.full(pixels.shape[0], parent, dtype=np.int32)
                )
        pixels = np.vstack(pixel_groups).astype(np.int32)
        points = np.vstack(point_groups)
        parent_labels = np.concatenate(label_groups)
        patch_labels, patches = split_plane_labels_by_connectivity(
            points,
            parent_labels,
            [
                {"index": 0, "normal_camera": [0.0, 0.0, 1.0]},
                {"index": 1, "normal_camera": [0.0, 0.0, 1.0]},
            ],
            pixels,
            [80, 60],
            stride=1,
            source_point_count=points.shape[0],
            min_component_count=100,
            preserve_farthest_plane=True,
        )

        self.assertEqual(len(patches), 3)
        self.assertEqual(patches[0]["index"], 0)
        self.assertEqual(patches[0]["parent_plane_index"], 1)
        self.assertTrue(patches[0]["is_farthest_plane"])
        self.assertEqual(int(np.count_nonzero(patch_labels == 0)), 600)
        self.assertTrue(
            all(not patch["is_farthest_plane"] for patch in patches[1:])
        )

        filtered_labels, filtered_patches = split_plane_labels_by_connectivity(
            points,
            parent_labels,
            [
                {"index": 0, "normal_camera": [0.0, 0.0, 1.0]},
                {"index": 1, "normal_camera": [0.0, 0.0, 1.0]},
            ],
            pixels,
            [80, 60],
            stride=1,
            source_point_count=points.shape[0],
            min_component_count=100,
            preserve_farthest_plane=True,
            max_planar_point_distance_from_farthest_plane_m=0.010,
        )

        self.assertEqual(len(filtered_patches), 1)
        self.assertTrue(filtered_patches[0]["is_farthest_plane"])
        self.assertAlmostEqual(
            filtered_patches[0]["nearest_p0_xz_distance_m"], 0.0
        )
        self.assertTrue(np.all(filtered_labels[parent_labels == 0] == -1))

    def test_multiple_long_boundaries_are_fitted_next_to_p0(self) -> None:
        positions = np.linspace(-0.15, 0.15, 301)
        p0_points = np.vstack(
            (
                np.column_stack(
                    (positions, np.zeros_like(positions), np.zeros_like(positions))
                ),
                np.column_stack(
                    (np.zeros_like(positions), np.zeros_like(positions), positions)
                ),
            )
        )
        neighboring_points = np.vstack(
            (
                np.column_stack(
                    (
                        positions,
                        np.full_like(positions, -0.02),
                        np.full_like(positions, 0.006),
                    )
                ),
                np.column_stack(
                    (
                        np.full_like(positions, 0.006),
                        np.full_like(positions, -0.02),
                        positions,
                    )
                ),
            )
        )
        points = np.vstack((p0_points, neighboring_points))
        labels = np.concatenate(
            (
                np.zeros(p0_points.shape[0], dtype=np.int32),
                np.ones(neighboring_points.shape[0], dtype=np.int32),
            )
        )
        described = describe_p0_boundary_lines(
            points,
            labels,
            [
                {
                    "index": 0,
                    "is_farthest_plane": True,
                    "origin_camera_m": [0.0, 0.0, 0.0],
                    "normal_camera": [0.0, 1.0, 0.0],
                },
                {
                    "index": 1,
                    "is_farthest_plane": False,
                    "origin_camera_m": [0.0, -0.02, 0.0],
                    "normal_camera": [0.0, 1.0, 0.0],
                },
            ],
        )

        lines = described[1]["boundary_lines"]
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line["length_m"] > 0.25 for line in lines))
        directions = [
            np.asarray(line["direction_camera"], dtype=np.float64)
            for line in lines
        ]
        self.assertLess(abs(float(directions[0] @ directions[1])), 0.1)

    def test_isolated_longest_boundary_is_rejected_for_direction_group(
        self,
    ) -> None:
        def boundary(
            length: float, direction: list[float]
        ) -> dict[str, object]:
            vector = np.asarray(direction, dtype=np.float64)
            vector /= np.linalg.norm(vector)
            return {
                "index": 0,
                "start_camera_m": (-0.5 * length * vector).tolist(),
                "end_camera_m": (0.5 * length * vector).tolist(),
                "direction_camera": vector.tolist(),
                "length_m": length,
                "fit_method": "ransac",
            }

        segments = [
            {"index": 0, "boundary_lines": []},
            {"index": 1, "boundary_lines": [boundary(0.80, [1.0, 0.0, 0.0])]},
            {
                "index": 2,
                "boundary_lines": [boundary(0.70, [1.0, 0.0, 0.005])],
            },
            {
                "index": 3,
                "boundary_lines": [boundary(0.60, [1.0, 0.0, -0.006])],
            },
            {"index": 4, "boundary_lines": [boundary(2.00, [0.0, 0.0, 1.0])]},
        ]
        fitted = estimate_wall_x_from_p0_boundary_lines(
            {
                "x_axis_camera": [1.0, 0.0, 0.0],
                "y_axis_camera": [0.0, 1.0, 0.0],
                "z_axis_camera": [0.0, 0.0, 1.0],
                "axis_estimation": "camera-up-projection",
            },
            segments,
        )
        self.assertEqual(
            fitted["axis_estimation"], "p0-nearest-boundary-line"
        )
        self.assertEqual(fitted["axis_reference_plane_index"], 1)
        self.assertEqual(fitted["axis_reference_group_size"], 3)
        self.assertEqual(
            fitted["axis_reference_group_angle_tolerance_deg"], 1.0
        )
        self.assertEqual(
            sum(
                bool(line.get("selected_for_x"))
                for segment in segments
                for line in segment["boundary_lines"]
            ),
            1,
        )
        self.assertFalse(
            segments[4]["boundary_lines"][0]["accepted_for_x_group"]
        )

    def test_direction_group_rejects_lines_close_to_camera_up(self) -> None:
        def boundary(
            length: float, direction: list[float]
        ) -> dict[str, object]:
            vector = np.asarray(direction, dtype=np.float64)
            vector /= np.linalg.norm(vector)
            return {
                "index": 0,
                "start_camera_m": (-0.5 * length * vector).tolist(),
                "end_camera_m": (0.5 * length * vector).tolist(),
                "direction_camera": vector.tolist(),
                "length_m": length,
                "fit_method": "ransac",
            }

        segments = [
            {"index": 0, "boundary_lines": []},
            {"index": 1, "boundary_lines": [boundary(2.0, [0.0, 1.0, 0.0])]},
            {"index": 2, "boundary_lines": [boundary(1.8, [0.0, 1.0, 0.0])]},
            {"index": 3, "boundary_lines": [boundary(0.8, [1.0, 0.0, 0.0])]},
            {
                "index": 4,
                "boundary_lines": [boundary(0.7, [1.0, 0.01, 0.0])],
            },
        ]
        fitted = estimate_wall_x_from_p0_boundary_lines(
            {
                "x_axis_camera": [1.0, 0.0, 0.0],
                "y_axis_camera": [0.0, 0.0, 1.0],
                "z_axis_camera": [0.0, -1.0, 0.0],
                "axis_estimation": "camera-up-projection",
            },
            segments,
        )

        self.assertEqual(fitted["axis_reference_plane_index"], 3)
        self.assertGreaterEqual(
            fitted["axis_reference_camera_up_angle_deg"], 45.0
        )
        self.assertFalse(
            segments[1]["boundary_lines"][0]["passes_camera_up_angle"]
        )
        self.assertFalse(
            segments[2]["boundary_lines"][0]["accepted_for_x_group"]
        )

    def test_plane_intersection_defines_automatic_wall_x(self) -> None:
        fitted = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "x_axis_camera": [0.98, 0.2, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.2, -0.98, 0.0],
            "axis_estimation": "camera-up-projection",
        }
        angle = np.radians(5.0)
        result = estimate_wall_x_from_plane_intersections(
            fitted,
            [
                {
                    "index": 1,
                    "normal_camera": [0.0, np.sin(angle), np.cos(angle)],
                    "inlier_ratio": 0.2,
                }
            ],
        )

        np.testing.assert_allclose(
            result["x_axis_camera"], [1.0, 0.0, 0.0], atol=1e-8
        )
        np.testing.assert_allclose(
            result["z_axis_camera"], [0.0, -1.0, 0.0], atol=1e-8
        )
        self.assertEqual(result["axis_estimation"], "multi-plane-intersection")
        self.assertAlmostEqual(result["axis_reference_angle_deg"], 5.0)

    def test_secondary_narrow_plane_defines_automatic_wall_x(self) -> None:
        rng = np.random.default_rng(21)
        angle = np.radians(10.0)
        expected_x = np.array([np.cos(angle), -np.sin(angle), 0.0])
        wall_y = np.array([0.0, 0.0, 1.0])
        tilt = np.radians(5.0)
        secondary_normal = (
            wall_y * np.cos(tilt)
            + np.cross(expected_x, wall_y) * np.sin(tilt)
        )
        secondary_width = np.cross(secondary_normal, expected_x)
        secondary_width /= np.linalg.norm(secondary_width)
        main_points = np.column_stack(
            (
                rng.uniform(-0.8, 0.8, 2_000),
                rng.uniform(-0.5, 0.5, 2_000),
                np.full(2_000, 1.2),
            )
        )
        secondary_points = (
            np.array([0.0, 0.0, 1.15])
            + rng.uniform(-0.7, 0.7, (800, 1)) * expected_x
            + rng.uniform(-0.08, 0.08, (800, 1)) * secondary_width
        )
        points = np.vstack((main_points, secondary_points))
        labels = np.concatenate(
            (
                np.zeros(main_points.shape[0], dtype=np.int32),
                np.ones(secondary_points.shape[0], dtype=np.int32),
            )
        )
        fitted = {
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": wall_y.tolist(),
            "z_axis_camera": [0.0, -1.0, 0.0],
            "axis_estimation": "camera-up-projection",
        }
        result = estimate_wall_x_from_secondary_plane_shape(
            fitted,
            points,
            labels,
            [
                {"index": 0, "normal_camera": wall_y, "inlier_ratio": 0.7},
                {
                    "index": 1,
                    "normal_camera": secondary_normal,
                    "inlier_ratio": 0.3,
                },
            ],
        )

        np.testing.assert_allclose(
            result["x_axis_camera"], expected_x, atol=0.01
        )
        self.assertEqual(
            result["axis_estimation"], "secondary-plane-principal-axis"
        )
        self.assertEqual(result["axis_reference_plane_index"], 1)

    def test_automatic_wall_x_falls_back_without_reliable_intersection(
        self,
    ) -> None:
        fitted = {
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
            "axis_estimation": "camera-up-projection",
        }
        result = estimate_wall_x_from_plane_intersections(
            fitted,
            [
                {
                    "index": 1,
                    "normal_camera": [0.0, 0.01, 0.99995],
                    "inlier_ratio": 0.3,
                }
            ],
        )
        self.assertEqual(result, fitted)

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
            self.assertEqual(first["schema"], "rgbd-target-annotation/v3")
            self.assertEqual(first["points"]["1"]["target_camera_m"], [0.1, 0.2, 0.95])
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
                point_slot=2,
            )
            records = load_annotations(root, session.session_id)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["active_point_slot"], 2)
            self.assertEqual(
                records[0]["points"]["1"]["target_camera_m"],
                [0.1, 0.2, 0.95],
            )
            self.assertEqual(
                records[0]["points"]["2"]["target_camera_m"],
                [0.2, 0.2, 0.95],
            )
            save_annotation(
                root,
                session.session_id,
                frame_id,
                target_camera_m=[0.3, 0.2, 0.95],
                plane=plane,
            )
            records = load_annotations(root, session.session_id)
            self.assertEqual(
                records[0]["points"]["1"]["target_camera_m"],
                [0.3, 0.2, 0.95],
            )
            self.assertIn("2", records[0]["points"])
            saved = save_annotation(
                root,
                session.session_id,
                frame_id,
                target_camera_m=[0.4, 0.2, 0.95],
                plane=plane,
                point_slot=3,
                selection_source="target-finder/0.1.0",
                target_finder={"model": {"version": "0.1.0"}},
            )
            self.assertEqual(
                saved["points"]["3"]["selection_source"],
                "target-finder/0.1.0",
            )
            self.assertEqual(
                saved["points"]["3"]["target_finder"]["model"]["version"],
                "0.1.0",
            )

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

    def test_legacy_annotation_becomes_point_slot_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DatasetSession(root, "legacy points", camera_metadata())
            frame_id = session.enqueue(make_frame(1), "manual")
            session.close()
            plane = {
                "origin_camera_m": [0.0, 0.0, 1.0],
                "normal_camera": [0.0, 0.0, 1.0],
                "x_axis_camera": [1.0, 0.0, 0.0],
                "y_axis_camera": [0.0, 0.0, 1.0],
                "z_axis_camera": [0.0, -1.0, 0.0],
            }
            legacy = {
                "schema": "rgbd-target-annotation/v2",
                "session_id": session.session_id,
                "frame_id": frame_id,
                "target_camera_m": [0.1, 0.2, 0.95],
                "selection_source": "pointcloud",
                "plane": plane,
                "yolo": {"available": False, "boxes": []},
            }
            (session.path / "annotations.jsonl").write_text(
                json.dumps(legacy) + "\n", encoding="utf-8"
            )
            saved = save_annotation(
                root,
                session.session_id,
                frame_id,
                target_camera_m=[0.3, 0.2, 0.95],
                plane=plane,
                point_slot=2,
            )

        self.assertEqual(
            saved["points"]["1"]["target_camera_m"], [0.1, 0.2, 0.95]
        )
        self.assertEqual(
            saved["points"]["2"]["target_camera_m"], [0.3, 0.2, 0.95]
        )

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
        self.assertEqual(
            applied["axis_estimation"], "manual-two-point-calibration"
        )
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

    def test_current_automatic_coordinate_can_be_accepted(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "normal_camera": [0.0, 0.0, 1.0],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
            "axis_estimation": "p0-nearest-boundary-line",
            "axis_reference_plane_index": 5,
            "axis_reference_boundary_index": 0,
            "axis_reference_line_length_m": 1.74,
        }
        calibration = build_accepted_wall_calibration(plane)
        self.assertEqual(
            calibration["calibration_method"], "accepted-automatic"
        )
        self.assertEqual(
            calibration["accepted_axis_estimation"],
            "p0-nearest-boundary-line",
        )
        applied = apply_wall_calibration(plane, calibration)
        self.assertTrue(applied["calibrated"])
        self.assertEqual(
            applied["axis_estimation"], "saved-accepted-coordinate"
        )
        self.assertEqual(applied["axis_reference_plane_index"], 5)

    def test_saved_coordinate_skips_plane_analysis(self) -> None:
        plane = {
            "origin_camera_m": [0.0, 0.0, 1.0],
            "center_camera_m": [0.0, 0.0, 1.0],
            "normal_camera": [0.0, 0.0, 1.0],
            "x_axis_camera": [1.0, 0.0, 0.0],
            "y_axis_camera": [0.0, 0.0, 1.0],
            "z_axis_camera": [0.0, -1.0, 0.0],
            "axis_estimation": "p0-nearest-boundary-line",
            "threshold_m": 0.008,
            "inlier_count": 42,
            "inlier_ratio": 1.0,
            "rms_m": 0.001,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DatasetSession(root, "saved", camera_metadata())
            frame_id = session.enqueue(make_frame(1), "manual")
            session.close()
            calibration = build_accepted_wall_calibration(plane)
            save_wall_calibration(
                root, session.session_id, frame_id, calibration
            )

            result = analyze_frame(
                root,
                session.session_id,
                frame_id,
                min_plane_points=3,
            )
            panel_enabled_result = analyze_frame(
                root,
                session.session_id,
                frame_id,
                min_plane_points=3,
                include_yolo_panel_fit=True,
            )

        self.assertTrue(result["plane"]["plane_analysis_skipped"])
        self.assertEqual(result["plane_segments"], [])
        self.assertEqual(result["plane"]["inlier_count"], 42)
        self.assertNotIn("yolo_panel_fit", result)
        self.assertFalse(panel_enabled_result["yolo_panel_fit"]["available"])
        self.assertIn(
            "没有 YOLO", panel_enabled_result["yolo_panel_fit"]["reason"]
        )

    def test_yolo_panel_rectangle_rejects_knob_and_handles_occlusion(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        long_positions = rng.uniform(-0.14, 0.14, 18_000)
        short_positions = rng.uniform(-0.10, 0.10, 18_000)
        visible = ~(
            (long_positions > 0.03) & (short_positions < -0.01)
        )
        long_positions = long_positions[visible]
        short_positions = short_positions[visible]
        angle = np.radians(17.0)
        panel = np.column_stack(
            (
                long_positions * np.cos(angle)
                - short_positions * np.sin(angle),
                long_positions * np.sin(angle)
                + short_positions * np.cos(angle),
                0.8
                + rng.normal(0.0, 0.0012, long_positions.shape[0]),
            )
        )
        knob = np.column_stack(
            (
                rng.normal(-0.03, 0.025, 2_500),
                rng.normal(0.02, 0.025, 2_500),
                rng.normal(0.72, 0.008, 2_500),
            )
        )
        outliers = rng.uniform(
            [-0.18, -0.14, 0.65],
            [0.18, 0.14, 0.95],
            size=(500, 3),
        )

        fitted = fit_yolo_panel_rectangle(
            np.vstack((panel, knob, outliers))
        )

        self.assertTrue(fitted["available"])
        self.assertAlmostEqual(fitted["long_length_m"], 0.28, delta=0.015)
        self.assertAlmostEqual(
            fitted["short_length_m"], 0.20, delta=0.015
        )
        self.assertGreater(fitted["excluded_point_count"], 2_500)
        self.assertEqual(
            [edge["role"] for edge in fitted["edges"]],
            ["long", "short"],
        )
        np.testing.assert_allclose(
            fitted["edges"][0]["start_camera_m"],
            fitted["edges"][1]["start_camera_m"],
            atol=1e-9,
        )
        self.assertAlmostEqual(
            abs(float(np.asarray(fitted["normal_camera"]) @ [0, 0, 1])),
            1.0,
            delta=0.01,
        )

    def test_yolo_panel_ignores_coplanar_stragglers_outside_panel(
        self,
    ) -> None:
        rng = np.random.default_rng(11)
        count = 14_000
        panel = np.column_stack(
            (
                rng.uniform(-0.027, 0.027, count),
                rng.uniform(-0.0255, 0.0255, count),
                0.5 + rng.normal(0.0, 0.0008, count),
            )
        )
        # Coplanar mask-bleed points detached from the panel, up and to the
        # right. Before the connected-component fix these dragged the
        # measured edges off the panel.
        stragglers = np.column_stack(
            (
                rng.uniform(0.045, 0.065, 400),
                rng.uniform(-0.05, -0.035, 400),
                0.5 + rng.normal(0.0, 0.0008, 400),
            )
        )

        fitted = fit_yolo_panel_rectangle(np.vstack((panel, stragglers)))

        self.assertTrue(fitted["available"])
        self.assertAlmostEqual(fitted["long_length_m"], 0.054, delta=0.006)
        self.assertAlmostEqual(fitted["short_length_m"], 0.051, delta=0.006)
        corners = np.asarray(fitted["rectangle_corners_camera_m"])
        self.assertEqual(corners.shape, (4, 3))
        self.assertLess(float(np.abs(corners[:, 0]).max()), 0.033)
        self.assertLess(float(np.abs(corners[:, 1]).max()), 0.032)
        for edge in fitted["edges"]:
            endpoints = np.asarray(
                [edge["start_camera_m"], edge["end_camera_m"]]
            )
            self.assertLess(float(np.abs(endpoints[:, 0]).max()), 0.033)
            self.assertLess(float(np.abs(endpoints[:, 1]).max()), 0.032)

    def test_yolo_panel_trims_attached_sparse_bleed_strip(self) -> None:
        rng = np.random.default_rng(17)
        count = 14_000
        panel = np.column_stack(
            (
                rng.uniform(-0.027, 0.027, count),
                rng.uniform(-0.0255, 0.0255, count),
                0.5 + rng.normal(0.0, 0.0008, count),
            )
        )
        # Sparse coplanar bleed attached to the right edge of the panel:
        # connected in the raster, so component filtering alone keeps it.
        bleed = np.column_stack(
            (
                rng.uniform(0.027, 0.035, 180),
                rng.uniform(-0.0255, 0.0255, 180),
                0.5 + rng.normal(0.0, 0.0008, 180),
            )
        )

        fitted = fit_yolo_panel_rectangle(np.vstack((panel, bleed)))

        self.assertTrue(fitted["available"])
        self.assertAlmostEqual(fitted["long_length_m"], 0.054, delta=0.005)
        self.assertAlmostEqual(fitted["short_length_m"], 0.051, delta=0.005)
        corners = np.asarray(fitted["rectangle_corners_camera_m"])
        self.assertLess(float(corners[:, 0].max()), 0.031)

    def test_yolo_panel_uses_wall_frame_axes_when_provided(self) -> None:
        rng = np.random.default_rng(23)
        count = 16_000
        angle = np.radians(17.0)
        long_positions = rng.uniform(-0.14, 0.14, count)
        short_positions = rng.uniform(-0.10, 0.10, count)
        panel = np.column_stack(
            (
                long_positions * np.cos(angle)
                - short_positions * np.sin(angle),
                long_positions * np.sin(angle)
                + short_positions * np.cos(angle),
                0.8 + rng.normal(0.0, 0.0012, count),
            )
        )
        wall_x = [np.cos(angle), np.sin(angle), 0.0]
        wall_z = [-np.sin(angle), np.cos(angle), 0.0]

        fitted = fit_yolo_panel_rectangle(
            panel, preferred_axes_camera=(wall_x, wall_z)
        )

        self.assertTrue(fitted["available"])
        self.assertEqual(fitted["orientation_source"], "wall-frame")
        self.assertAlmostEqual(fitted["long_length_m"], 0.28, delta=0.012)
        self.assertAlmostEqual(fitted["short_length_m"], 0.20, delta=0.012)
        alignment = abs(
            float(np.asarray(fitted["long_axis_camera"]) @ wall_x)
        )
        self.assertGreater(alignment, 0.999)

        degenerate = fit_yolo_panel_rectangle(
            panel,
            preferred_axes_camera=([0.0, 0.0, 1.0], wall_z),
        )
        self.assertEqual(
            degenerate["orientation_source"], "boundary-hough"
        )

    def test_yolo_panel_color_filter_removes_dark_overrun(self) -> None:
        rng = np.random.default_rng(29)
        count = 12_000
        panel = np.column_stack(
            (
                rng.uniform(-0.027, 0.027, count),
                rng.uniform(-0.0255, 0.0255, count),
                0.5 + rng.normal(0.0, 0.0008, count),
            )
        )
        # Dense, coplanar, attached dark region (e.g. black knob/bleed);
        # geometry alone cannot separate it from the panel.
        dark_count = 4_000
        dark = np.column_stack(
            (
                rng.uniform(0.027, 0.045, dark_count),
                rng.uniform(-0.0255, 0.0255, dark_count),
                0.5 + rng.normal(0.0, 0.0008, dark_count),
            )
        )
        points = np.vstack((panel, dark))
        pixels = np.column_stack(
            (
                (points[:, 0] + 0.1) * 4000,
                (points[:, 1] + 0.1) * 4000,
            )
        )
        rgba = np.zeros((points.shape[0], 4), dtype=np.uint8)
        rgba[:count, :3] = (0x68, 0x6F, 0x71)
        rgba[count:, :3] = (0x0D, 0x13, 0x13)
        rgba[:, 3] = 255
        boxes = [
            {"cls": 1, "name": "panel", "conf": 0.9, "xyxy": [0, 0, 900, 900]}
        ]

        unfiltered = analyze_yolo_mask_panel(
            points, pixels, boxes, [900, 900]
        )
        filtered = analyze_yolo_mask_panel(
            points, pixels, boxes, [900, 900], point_rgba=rgba
        )

        self.assertTrue(unfiltered["available"])
        self.assertGreater(unfiltered["long_length_m"], 0.065)
        self.assertTrue(filtered["available"])
        self.assertTrue(filtered["color_filter"]["enabled"])
        self.assertGreater(
            filtered["color_filter"]["removed_point_count"], 3_500
        )
        self.assertAlmostEqual(
            filtered["long_length_m"], 0.054, delta=0.005
        )
        self.assertAlmostEqual(
            filtered["short_length_m"], 0.051, delta=0.005
        )

    def test_yolo_panel_uses_highest_confidence_polygon(self) -> None:
        x_values, y_values = np.meshgrid(
            np.linspace(-0.12, 0.12, 100),
            np.linspace(-0.08, 0.08, 70),
        )
        panel = np.column_stack(
            (
                x_values.ravel(),
                y_values.ravel(),
                np.full(x_values.size, 0.8),
            )
        )
        pixels = np.column_stack(
            (
                100 + (x_values.ravel() + 0.12) * 400,
                100 + (y_values.ravel() + 0.08) * 400,
            )
        )
        boxes = [
            {
                "cls": 3,
                "name": "lower",
                "conf": 0.4,
                "xyxy": [0, 0, 10, 10],
            },
            {
                "cls": 4,
                "name": "panel",
                "conf": 0.95,
                "xyxy": [95, 95, 205, 170],
                "polygon": [
                    [95, 95],
                    [205, 95],
                    [205, 170],
                    [95, 170],
                ],
            },
        ]

        fitted = analyze_yolo_mask_panel(
            panel, pixels, boxes, [300, 300]
        )

        self.assertTrue(fitted["available"])
        self.assertEqual(fitted["detection"]["box_index"], 1)
        self.assertEqual(fitted["detection"]["name"], "panel")
        self.assertTrue(fitted["detection"]["used_polygon_mask"])
        self.assertGreater(fitted["mask_point_count"], 6_000)

    def test_yolo_panel_reports_too_few_mask_points(self) -> None:
        result = analyze_yolo_mask_panel(
            np.zeros((20, 3)),
            np.column_stack((np.arange(20), np.arange(20))),
            [
                {
                    "name": "panel",
                    "conf": 0.9,
                    "xyxy": [0, 0, 30, 30],
                }
            ],
            [40, 40],
        )
        self.assertFalse(result["available"])
        self.assertIn("有效点不足", result["reason"])

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

    def test_only_highest_confidence_yolo_pointcloud_is_saved(self) -> None:
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
        boxes = [
            {
                "cls": 2,
                "name": "lower",
                "conf": 0.4,
                "xyxy": [114, 114, 120, 120],
            },
            {
                "cls": 1,
                "name": "highest",
                "conf": 0.95,
                "xyxy": [100, 100, 106, 106],
            },
        ]
        semantic_cloud = highest_confidence_semantic_pointcloud(
            points,
            np.tile(np.array([[10, 20, 30, 255]], dtype=np.uint8), (6, 1)),
            {
                "intrinsics": {"fx": 100, "fy": 100, "cx": 100, "cy": 100},
                "image_shape": [200, 200],
            },
            boxes,
            plane,
        )
        self.assertIsNotNone(semantic_cloud)
        assert semantic_cloud is not None
        self.assertEqual(semantic_cloud["detection"]["name"], "highest")
        self.assertEqual(semantic_cloud["xyz_camera_m"].shape, (3, 3))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DatasetSession(root, "yolo cloud", camera_metadata())
            frame_id = session.enqueue(make_frame(1), "manual")
            session.close()
            summary = save_highest_confidence_semantic_pointcloud(
                root,
                session.session_id,
                frame_id,
                semantic_cloud,
                target_camera_m=[0.1, 0.2, 1.0],
            )
            data_path = root / summary["data_file"]
            metadata_path = data_path.with_suffix(".json")
            with np.load(data_path) as saved:
                self.assertEqual(saved["xyz_camera_m"].shape, (3, 3))
                self.assertEqual(saved["xyz_wall_m"].shape, (3, 3))
                self.assertEqual(saved["rgb"].dtype, np.uint8)
            data_mtime_ns = data_path.stat().st_mtime_ns
            save_highest_confidence_semantic_pointcloud(
                root,
                session.session_id,
                frame_id,
                semantic_cloud,
                target_camera_m=[0.2, 0.2, 1.0],
                point_slot=2,
            )
            self.assertEqual(data_path.stat().st_mtime_ns, data_mtime_ns)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["selection"], "highest-confidence-only")
        self.assertEqual(metadata["detection"]["conf"], 0.95)
        self.assertEqual(metadata["point_count"], 3)
        self.assertIn("target_minus_semantic_centroid_wall_m", metadata)
        self.assertEqual(set(metadata["targets"]), {"1", "2"})


if __name__ == "__main__":
    unittest.main()
