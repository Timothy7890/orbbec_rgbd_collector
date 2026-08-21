from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_collector.analysis import (
    analyze_frame,
    apply_wall_calibration,
    build_accepted_wall_calibration,
    build_wall_calibration,
    describe_p0_boundary_lines,
    describe_segmented_plane_axes,
    estimate_wall_x_from_p0_boundary_lines,
    estimate_wall_x_from_plane_intersections,
    estimate_wall_x_from_secondary_plane_shape,
    fit_dominant_plane,
    load_annotations,
    load_wall_calibration,
    save_annotation,
    save_wall_calibration,
    segment_dominant_planes,
    semantic_clusters,
    split_plane_labels_by_connectivity,
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
                "boundary_lines": [boundary(0.70, [1.0, 0.0, 0.01])],
            },
            {
                "index": 3,
                "boundary_lines": [boundary(0.60, [1.0, 0.0, -0.015])],
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

        self.assertTrue(result["plane"]["plane_analysis_skipped"])
        self.assertEqual(result["plane_segments"], [])
        self.assertEqual(result["plane"]["inlier_count"], 42)

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
