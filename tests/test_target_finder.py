from __future__ import annotations

import unittest

import numpy as np

from rgbd_collector.target_finder import (
    predict_target_one,
    target_finder_models,
)


class TargetFinderTests(unittest.TestCase):
    def semantic_cloud(self) -> dict:
        center = np.array([0.12, -0.01, 0.20])
        return {
            "xyz_wall_m": np.array(
                [
                    center + [-0.01, 0.00, 0.00],
                    center,
                    center + [0.01, 0.00, 0.00],
                ],
                dtype=np.float32,
            ),
            "coordinate_origin_camera_m": [1.0, 2.0, 3.0],
            "coordinate_axes_camera": np.eye(3).tolist(),
            "detection": {"name": "fixture", "conf": 0.9},
        }

    def test_version_0_1_0_predicts_point_one_from_yolo_median(self) -> None:
        model = target_finder_models()[0]
        prediction = predict_target_one(
            self.semantic_cloud(), version="0.1.0"
        )
        expected_wall = np.array([0.12, -0.01, 0.20]) + np.asarray(
            model["offset_wall_m"]
        )
        np.testing.assert_allclose(
            prediction["target_wall_m"], expected_wall, atol=1e-7
        )
        np.testing.assert_allclose(
            prediction["target_camera_m"],
            np.array([1.0, 2.0, 3.0]) + expected_wall,
            atol=1e-7,
        )
        self.assertEqual(prediction["selection_source"], "target-finder/0.1.0")
        self.assertEqual(prediction["semantic_point_count"], 3)

    def test_existing_point_one_is_used_for_validation(self) -> None:
        model = target_finder_models()[0]
        expected_wall = np.array([0.12, -0.01, 0.20]) + np.asarray(
            model["offset_wall_m"]
        )
        reference_camera = np.array([1.0, 2.0, 3.0]) + expected_wall
        reference_camera[0] -= 0.004
        prediction = predict_target_one(
            self.semantic_cloud(),
            version="0.1.0",
            reference_target_camera_m=reference_camera.tolist(),
        )
        validation = prediction["validation"]
        np.testing.assert_allclose(
            validation["prediction_minus_reference_wall_m"],
            [0.004, 0.0, 0.0],
            atol=1e-7,
        )
        self.assertAlmostEqual(validation["error_distance_m"], 0.004)

    def test_version_0_2_0_uses_34_frame_fixed_offset(self) -> None:
        models = {model["version"]: model for model in target_finder_models()}
        prediction = predict_target_one(
            self.semantic_cloud(), version="0.2.0"
        )
        expected_wall = np.array([0.12, -0.01, 0.20]) + np.asarray(
            models["0.2.0"]["offset_wall_m"]
        )
        np.testing.assert_allclose(
            prediction["target_wall_m"], expected_wall, atol=1e-7
        )
        self.assertEqual(prediction["model"]["training_frame_count"], 34)
        self.assertFalse(prediction["model"]["uses_camera_plane_distance"])
        self.assertFalse(prediction["model"]["uses_camera_plane_angles"])

    def test_unknown_model_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "未知找点算法版本"):
            predict_target_one(self.semantic_cloud(), version="9.9.9")


if __name__ == "__main__":
    unittest.main()
