from __future__ import annotations

from typing import Any

import numpy as np


_MODELS: dict[str, dict[str, Any]] = {
    "0.1.0": {
        "version": "0.1.0",
        "name": "15帧实验版",
        "algorithm": "highest-confidence-yolo-median-plus-wall-offset",
        "target_point_slot": 1,
        "training_session_id": "20260819_153209_capture",
        "training_frame_range": ["000210", "000224"],
        "training_frame_count": 15,
        "offset_wall_m": [
            0.0460673609043963,
            0.009026730625795555,
            -0.014690418704627113,
        ],
        "residual_std_wall_m": [
            0.003221649027852145,
            0.0017061004713092802,
            0.0009210491848090602,
        ],
        "validation_note": "训练集内统计；增加独立标注帧后需要重新验证",
    }
}


def target_finder_models() -> list[dict[str, Any]]:
    return [dict(model) for model in _MODELS.values()]


def predict_target_one(
    semantic_cloud: dict[str, Any],
    *,
    version: str,
    reference_target_camera_m: list[float] | None = None,
) -> dict[str, Any]:
    if version not in _MODELS:
        raise ValueError(f"未知找点算法版本: {version}")
    model = _MODELS[version]
    xyz_wall = np.asarray(semantic_cloud["xyz_wall_m"], dtype=np.float64)
    origin = np.asarray(
        semantic_cloud["coordinate_origin_camera_m"], dtype=np.float64
    )
    axes = np.asarray(
        semantic_cloud["coordinate_axes_camera"], dtype=np.float64
    )
    if (
        xyz_wall.ndim != 2
        or xyz_wall.shape[1] != 3
        or xyz_wall.shape[0] < 3
        or origin.shape != (3,)
        or axes.shape != (3, 3)
        or not np.isfinite(xyz_wall).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(axes).all()
    ):
        raise ValueError("YOLO 语义点云或墙面坐标系无效")

    semantic_center_wall = np.median(xyz_wall, axis=0)
    offset_wall = np.asarray(model["offset_wall_m"], dtype=np.float64)
    target_wall = semantic_center_wall + offset_wall
    target_camera = origin + target_wall @ axes
    prediction: dict[str, Any] = {
        "model": dict(model),
        "selection_source": f"target-finder/{version}",
        "target_camera_m": target_camera.tolist(),
        "target_wall_m": target_wall.tolist(),
        "semantic_center_wall_m": semantic_center_wall.tolist(),
        "semantic_point_count": int(xyz_wall.shape[0]),
        "semantic_detection": semantic_cloud["detection"],
    }

    if reference_target_camera_m is not None:
        reference_camera = np.asarray(
            reference_target_camera_m, dtype=np.float64
        )
        if reference_camera.shape != (3,) or not np.isfinite(
            reference_camera
        ).all():
            raise ValueError("验证点1必须是三个有限数值")
        reference_wall = (reference_camera - origin) @ axes.T
        error_wall = target_wall - reference_wall
        prediction["validation"] = {
            "reference_target_camera_m": reference_camera.tolist(),
            "reference_target_wall_m": reference_wall.tolist(),
            "prediction_minus_reference_wall_m": error_wall.tolist(),
            "error_distance_m": float(np.linalg.norm(error_wall)),
        }
    return prediction
