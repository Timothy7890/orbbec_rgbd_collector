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
    },
    "0.2.0": {
        "version": "0.2.0",
        "name": "34帧固定偏移实验版",
        "algorithm": "highest-confidence-yolo-median-plus-wall-offset",
        "target_point_slot": 1,
        "training_session_id": "20260819_153209_capture",
        "training_frame_range": ["000191", "000224"],
        "training_frame_count": 34,
        "offset_wall_m": [
            0.0433026551380823,
            0.008997318925539144,
            -0.015043360249992858,
        ],
        "residual_std_wall_m": [
            0.0036573874529999893,
            0.001322549069567777,
            0.0011233245713225825,
        ],
        "uses_camera_plane_distance": False,
        "uses_camera_plane_angles": False,
        "validation_note": "34帧均经人工调整确认；未使用距离或角度补偿",
    },
    "0.1.0-s": {
        "version": "0.1.0-s",
        "name": "33帧面板中心版",
        "algorithm": "yolo-panel-rectangle-center-plus-wall-offset",
        "reference_source": "yolo-panel-rectangle-center",
        "target_point_slot": 1,
        "training_session_id": "20260819_153209_capture",
        "training_frame_range": ["000178", "000220"],
        "training_frame_count": 33,
        "excluded_inaccurate_frames": ["000221", "000222", "000223", "000224"],
        "offset_wall_m": [
            0.04793951829,
            0.00586060655,
            -0.01953248751,
        ],
        "residual_std_wall_m": [
            0.0017275515,
            0.0014768564,
            0.00105212902,
        ],
        "fit_rmse_3d_m": 0.002504498851513549,
        "leave_one_out_rmse_3d_m": 0.002582764440623347,
        "max_residual_3d_m": 0.0046597965170179405,
        "same_frame_yolo_reference_leave_one_out_rmse_3d_m": (
            0.003251866782679898
        ),
        "requires_saved_wall_coordinate": True,
        "uses_camera_plane_distance": False,
        "uses_camera_plane_angles": False,
        "validation_note": (
            "33帧固定面板中心偏移；已排除人工确认不准确的221–224；"
            "仍需新增帧独立验证"
        ),
    },
    "0.2.0-s": {
        "version": "0.2.0-s",
        "name": "59帧面板中心双类别版",
        "algorithm": "yolo-panel-rectangle-center-plus-wall-offset",
        "reference_source": "yolo-panel-rectangle-center",
        "target_point_slot": "detection-dependent",
        "training_session_id": "20260819_153209_capture",
        "training_frame_ranges": [
            ["000104", "000126"],
            ["000178", "000224"],
        ],
        "training_frame_count": 59,
        "detection_rules": {
            "远方": {
                "target_point_slot": 1,
                "offset_wall_m": [
                    0.04793951829,
                    0.00586060655,
                    -0.01953248751,
                ],
            },
            "就地": {
                "target_point_slot": 3,
                "offset_wall_m": [
                    -0.04793951829,
                    0.00586060655,
                    -0.01953248751,
                ],
            },
        },
        "residual_std_wall_m": [
            0.0014533,
            0.0014432,
            0.0009459,
        ],
        "fit_rmse_3d_m": 0.002256,
        "leave_one_out_rmse_3d_m": 0.0022949,
        "max_residual_3d_m": 0.0046788,
        "requires_saved_wall_coordinate": True,
        "uses_camera_plane_distance": False,
        "uses_camera_plane_angles": False,
        "validation_note": (
            "远方沿用0.1.0-s生成点1；就地镜像X偏移生成点3；"
            "点3规则是几何对称定义，尚无独立点3标注验证"
        ),
    },
    "0.3.0-s": {
        "version": "0.3.0-s",
        "name": "面板中心三点位版",
        "algorithm": "yolo-panel-rectangle-center-plus-wall-offset",
        "reference_source": "yolo-panel-rectangle-center",
        "target_point_slot": "requested-or-detection-dependent",
        "allowed_requested_point_slots": [1, 2, 3],
        "classification_requested_point_slots": [1, 3],
        "training_session_id": "20260819_153209_capture",
        "training_frame_count": 59,
        "detection_rules": {
            "远方": {
                "target_point_slot": 1,
                "offset_wall_m": [
                    0.04793951829,
                    0.00586060655,
                    -0.01953248751,
                ],
            },
            "就地": {
                "target_point_slot": 3,
                "offset_wall_m": [
                    -0.04793951829,
                    0.00586060655,
                    -0.01953248751,
                ],
            },
        },
        "requested_point_rules": {
            "2": {
                "target_point_slot": 2,
                "offset_wall_m": [
                    0.3955939570123764,
                    -0.0019938704444295305,
                    0.20214201389586847,
                ],
                "training_frame_count": 56,
                "leave_one_out_rmse_3d_m": 0.006261501812479578,
            },
        },
        "residual_std_wall_m": [
            0.0014533,
            0.0014432,
            0.0009459,
        ],
        "fit_rmse_3d_m": 0.002256,
        "leave_one_out_rmse_3d_m": 0.0022949,
        "max_residual_3d_m": 0.0046788,
        "requires_saved_wall_coordinate": True,
        "uses_camera_plane_distance": False,
        "uses_camera_plane_angles": False,
        "validation_note": (
            "当前槽为2时生成点2；当前槽为1或3时按远方/就地"
            "自动切换并生成点1/点3"
        ),
    },
}


def target_finder_models() -> list[dict[str, Any]]:
    return [dict(model) for model in _MODELS.values()]


def predict_target_one(
    semantic_cloud: dict[str, Any] | None,
    *,
    version: str,
    reference_target_camera_m: list[float] | None = None,
    reference_targets_camera_m: dict[str, list[float]] | None = None,
    requested_point_slot: int | None = None,
    panel_fit: dict[str, Any] | None = None,
    plane: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if version not in _MODELS:
        raise ValueError(f"未知找点算法版本: {version}")
    model = _MODELS[version]
    panel_reference = (
        model["algorithm"]
        == "yolo-panel-rectangle-center-plus-wall-offset"
    )
    prediction_details: dict[str, Any]
    if panel_reference:
        if (
            model.get("requires_saved_wall_coordinate")
            and not bool(plane and plane.get("calibrated"))
        ):
            raise ValueError(f"{version} 仅支持已保存坐标系的帧")
        if not isinstance(panel_fit, dict) or not panel_fit.get("available"):
            reason = (
                panel_fit.get("reason")
                if isinstance(panel_fit, dict)
                else None
            )
            raise ValueError(
                f"当前帧面板中心不可用{f'：{reason}' if reason else ''}"
            )
        assert plane is not None
        origin = np.asarray(plane.get("origin_camera_m"), dtype=np.float64)
        axes = np.asarray(
            [
                plane.get("x_axis_camera"),
                plane.get("y_axis_camera"),
                plane.get("z_axis_camera"),
            ],
            dtype=np.float64,
        )
        panel_center_camera = np.asarray(
            panel_fit.get("rectangle_center_camera_m"), dtype=np.float64
        )
        if (
            origin.shape != (3,)
            or axes.shape != (3, 3)
            or panel_center_camera.shape != (3,)
            or not np.isfinite(origin).all()
            or not np.isfinite(axes).all()
            or not np.isfinite(panel_center_camera).all()
        ):
            raise ValueError("面板中心或墙面坐标系无效")
        reference_center_wall = (panel_center_camera - origin) @ axes.T
        prediction_details = {
            "reference_source": "yolo-panel-rectangle-center",
            "reference_center_wall_m": reference_center_wall.tolist(),
            "panel_center_camera_m": panel_center_camera.tolist(),
            "panel_center_wall_m": reference_center_wall.tolist(),
            "panel_detection": panel_fit.get("detection"),
            "panel_fit_quality": {
                key: panel_fit.get(key)
                for key in (
                    "inlier_count",
                    "inlier_ratio",
                    "rms_m",
                    "long_length_m",
                    "short_length_m",
                    "orientation_source",
                )
            },
        }
    else:
        if not isinstance(semantic_cloud, dict):
            raise ValueError("当前帧没有可用于找点的 YOLO 语义点云")
        xyz_wall = np.asarray(
            semantic_cloud["xyz_wall_m"], dtype=np.float64
        )
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
        reference_center_wall = np.median(xyz_wall, axis=0)
        prediction_details = {
            "reference_source": "highest-confidence-yolo-median",
            "reference_center_wall_m": reference_center_wall.tolist(),
            "semantic_center_wall_m": reference_center_wall.tolist(),
            "semantic_point_count": int(xyz_wall.shape[0]),
            "semantic_detection": semantic_cloud["detection"],
        }
    prediction_model = dict(model)
    target_point_slot = model["target_point_slot"]
    offset_source = model
    allowed_requested_slots = model.get("allowed_requested_point_slots")
    if isinstance(allowed_requested_slots, list):
        if requested_point_slot is None:
            requested_point_slot = 1
        if requested_point_slot not in allowed_requested_slots:
            supported = "、".join(str(slot) for slot in allowed_requested_slots)
            raise ValueError(
                f"模型 {version} 只响应当前点位槽 {supported}"
            )
        requested_rules = model.get("requested_point_rules")
        requested_rule = (
            requested_rules.get(str(requested_point_slot))
            if isinstance(requested_rules, dict)
            else None
        )
        if isinstance(requested_rule, dict):
            target_point_slot = requested_rule["target_point_slot"]
            offset_source = requested_rule
    detection_rules = model.get("detection_rules")
    if isinstance(detection_rules, dict) and offset_source is model:
        detection = panel_fit.get("detection") if panel_reference else None
        detection_name = (
            str(detection.get("name"))
            if isinstance(detection, dict) and detection.get("name") is not None
            else ""
        )
        rule = detection_rules.get(detection_name)
        if not isinstance(rule, dict):
            supported = "、".join(str(name) for name in detection_rules)
            raise ValueError(
                f"模型 {version} 不支持检测类别“{detection_name or '未知'}”"
                f"（仅支持：{supported}）"
            )
        target_point_slot = rule["target_point_slot"]
        offset_source = rule
        prediction_model["matched_detection_name"] = detection_name
        prediction_model["target_point_slot"] = target_point_slot
        prediction_model["offset_wall_m"] = list(rule["offset_wall_m"])
        prediction_details["matched_detection_name"] = detection_name
    target_point_slot = int(target_point_slot)
    offset_wall = np.asarray(
        offset_source["offset_wall_m"], dtype=np.float64
    )
    prediction_model["target_point_slot"] = target_point_slot
    prediction_model["offset_wall_m"] = offset_wall.tolist()
    target_wall = reference_center_wall + offset_wall
    target_camera = origin + target_wall @ axes
    prediction: dict[str, Any] = {
        "model": prediction_model,
        "selection_source": f"target-finder/{version}",
        "target_point_slot": target_point_slot,
        "requested_point_slot": requested_point_slot,
        "offset_wall_m": offset_wall.tolist(),
        "target_camera_m": target_camera.tolist(),
        "target_wall_m": target_wall.tolist(),
        **prediction_details,
    }

    reference_camera_value = reference_target_camera_m
    if isinstance(reference_targets_camera_m, dict):
        reference_camera_value = reference_targets_camera_m.get(
            str(target_point_slot)
        )
    if reference_camera_value is not None:
        reference_camera = np.asarray(
            reference_camera_value, dtype=np.float64
        )
        if reference_camera.shape != (3,) or not np.isfinite(
            reference_camera
        ).all():
            raise ValueError(
                f"验证点{target_point_slot}必须是三个有限数值"
            )
        reference_wall = (reference_camera - origin) @ axes.T
        error_wall = target_wall - reference_wall
        prediction["validation"] = {
            "reference_point_slot": target_point_slot,
            "reference_target_camera_m": reference_camera.tolist(),
            "reference_target_wall_m": reference_wall.tolist(),
            "prediction_minus_reference_wall_m": error_wall.tolist(),
            "error_distance_m": float(np.linalg.norm(error_wall)),
        }
    return prediction
