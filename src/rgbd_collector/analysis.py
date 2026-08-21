from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .pointcloud import (
    detection_pixel_mask,
    reconstruct_frame,
    resolve_frame_paths,
)


def fit_dominant_plane(
    points_xyz: np.ndarray,
    *,
    threshold_m: float = 0.008,
    iterations: int = 240,
    min_inlier_ratio: float = 0.20,
    seed: int = 0,
) -> dict[str, Any]:
    points = np.asarray(points_xyz, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] < 100:
        raise ValueError("有效点太少，无法拟合柜面")
    if not 0.001 <= threshold_m <= 0.05:
        raise ValueError("平面阈值必须在 1–50 mm 之间")

    rng = np.random.default_rng(seed)
    if points.shape[0] > 60_000:
        sample_indices = rng.choice(points.shape[0], 60_000, replace=False)
        fitting_points = points[sample_indices]
    else:
        fitting_points = points

    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        triangle = fitting_points[
            rng.choice(fitting_points.shape[0], 3, replace=False)
        ]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        distances = np.abs((fitting_points - triangle[0]) @ normal)
        mask = distances <= threshold_m
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None:
        raise ValueError("未找到稳定平面")
    inlier_ratio = best_count / fitting_points.shape[0]
    if inlier_ratio < min_inlier_ratio:
        raise ValueError(
            f"最大平面内点比例仅 {inlier_ratio:.1%}，请调整视角或阈值"
        )

    inliers = fitting_points[best_mask]
    origin = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - origin, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    # The wall +Y axis points from the camera through the visible wall surface,
    # i.e. into the wall rather than back toward the camera.
    if np.dot(normal, origin) < 0:
        normal = -normal

    distances = np.abs((fitting_points - origin) @ normal)
    refined_mask = distances <= threshold_m
    inliers = fitting_points[refined_mask]
    origin = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - origin, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if np.dot(normal, origin) < 0:
        normal = -normal

    wall_y = normal
    wall_center = origin.copy()
    camera_up = np.array([0.0, -1.0, 0.0])
    wall_z = camera_up - np.dot(camera_up, wall_y) * wall_y
    wall_z_length = np.linalg.norm(wall_z)
    if wall_z_length < 1e-6:
        raise ValueError("墙面法向与相机向上方向平行，无法确定墙面 Z 轴")
    wall_z /= wall_z_length
    wall_x = np.cross(wall_y, wall_z)
    wall_x /= np.linalg.norm(wall_x)
    # Recompute Z to remove numerical non-orthogonality. X × Y = Z.
    wall_z = np.cross(wall_x, wall_y)
    wall_z /= np.linalg.norm(wall_z)

    # Use the camera-origin projection onto the wall as a repeatable frame
    # origin. The inlier centroid would drift when the visible wall area changes.
    origin = np.dot(origin, wall_y) * wall_y
    residuals = (inliers - origin) @ wall_y
    return {
        "origin_camera_m": origin.tolist(),
        "center_camera_m": wall_center.tolist(),
        "normal_camera": wall_y.tolist(),
        "x_axis_camera": wall_x.tolist(),
        "y_axis_camera": wall_y.tolist(),
        "z_axis_camera": wall_z.tolist(),
        "coordinate_system": "wall-right-handed-x-right-y-inward-z-up",
        "origin_definition": "camera-origin-projection-on-wall",
        "threshold_m": threshold_m,
        "inlier_count": int(inliers.shape[0]),
        "sample_count": int(fitting_points.shape[0]),
        "inlier_ratio": float(inliers.shape[0] / fitting_points.shape[0]),
        "rms_m": float(np.sqrt(np.mean(residuals**2))),
    }


def segment_dominant_planes(
    points_xyz: np.ndarray,
    *,
    threshold_m: float = 0.008,
    iterations: int = 160,
    max_planes: int = 6,
    min_inlier_ratio: float = 0.03,
    min_inlier_count: int = 800,
    sample_limit: int = 60_000,
    seed: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云必须是 N×3 数组")
    if not 0.001 <= threshold_m <= 0.05:
        raise ValueError("平面阈值必须在 1–50 mm 之间")
    if not 1 <= max_planes <= 12:
        raise ValueError("最大平面数量必须在 1–12 之间")

    valid = np.isfinite(points).all(axis=1)
    labels = np.full(points.shape[0], -1, dtype=np.int32)
    valid_count = int(valid.sum())
    required_count = max(
        min_inlier_count, int(np.ceil(valid_count * min_inlier_ratio))
    )
    if valid_count < max(3, required_count):
        return labels, []

    rng = np.random.default_rng(seed)
    planes: list[dict[str, Any]] = []
    remaining = valid.copy()
    for plane_index in range(max_planes):
        remaining_indices = np.flatnonzero(remaining)
        if remaining_indices.size < required_count:
            break
        if remaining_indices.size > sample_limit:
            sample_indices = rng.choice(
                remaining_indices, sample_limit, replace=False
            )
        else:
            sample_indices = remaining_indices
        sample = points[sample_indices]

        best_mask: np.ndarray | None = None
        best_count = 0
        for _ in range(iterations):
            triangle = sample[rng.choice(sample.shape[0], 3, replace=False)]
            normal = np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0]
            )
            length = np.linalg.norm(normal)
            if length < 1e-8:
                continue
            normal /= length
            mask = np.abs((sample - triangle[0]) @ normal) <= threshold_m
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_mask = mask
        if best_mask is None or best_count < 3:
            break

        fitting_inliers = sample[best_mask]
        origin = fitting_inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(
            fitting_inliers - origin, full_matrices=False
        )
        normal = vh[-1]
        normal /= np.linalg.norm(normal)

        candidate_points = points[remaining_indices]
        candidate_mask = (
            np.abs((candidate_points - origin) @ normal) <= threshold_m
        )
        if int(candidate_mask.sum()) < required_count:
            break

        full_inliers = candidate_points[candidate_mask]
        if full_inliers.shape[0] > sample_limit:
            refinement = full_inliers[
                rng.choice(full_inliers.shape[0], sample_limit, replace=False)
            ]
        else:
            refinement = full_inliers
        origin = refinement.mean(axis=0)
        _, _, vh = np.linalg.svd(refinement - origin, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        if np.dot(normal, origin) < 0:
            normal = -normal

        distances = np.abs((candidate_points - origin) @ normal)
        candidate_mask = distances <= threshold_m
        inlier_count = int(candidate_mask.sum())
        if inlier_count < required_count:
            break

        selected_indices = remaining_indices[candidate_mask]
        labels[selected_indices] = plane_index
        remaining[selected_indices] = False
        planes.append(
            {
                "index": plane_index,
                "origin_camera_m": origin.tolist(),
                "normal_camera": normal.tolist(),
                "inlier_count": inlier_count,
                "inlier_ratio": float(inlier_count / valid_count),
                "rms_m": float(
                    np.sqrt(np.mean(distances[candidate_mask] ** 2))
                ),
            }
        )
    return labels, planes


def analyze_frame(
    data_root: Path,
    session_id: str,
    frame_id: str,
    *,
    boxes: list[dict[str, Any]] | None = None,
    plane_threshold_m: float = 0.008,
    min_depth_m: float = 0.15,
    max_depth_m: float = 3.0,
    use_saved_wall_calibration: bool = True,
) -> dict[str, Any]:
    cloud, metadata = reconstruct_frame(
        data_root,
        session_id,
        frame_id,
        stride=3,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        max_points=120_000,
    )
    plane = fit_dominant_plane(
        cloud["xyz"], threshold_m=plane_threshold_m
    )
    if use_saved_wall_calibration:
        calibration = load_wall_calibration(data_root, session_id, frame_id)
        if calibration is not None:
            plane = apply_wall_calibration(plane, calibration)
    clusters = semantic_clusters(
        cloud["xyz"], metadata, boxes or [], plane
    )
    return {
        "session_id": session_id,
        "frame_id": frame_id,
        "point_count": int(cloud.shape[0]),
        "plane": plane,
        "yolo": {
            "available": boxes is not None,
            "boxes": boxes or [],
            "clusters": clusters,
        },
        "source": metadata,
    }


def semantic_clusters(
    points_xyz: np.ndarray,
    metadata: dict[str, Any],
    boxes: list[dict[str, Any]],
    plane: dict[str, Any],
) -> list[dict[str, Any]]:
    if not boxes:
        return []
    points = np.asarray(points_xyz, dtype=np.float64)
    intrinsics = metadata["intrinsics"]
    z = points[:, 2]
    u = float(intrinsics["fx"]) * points[:, 0] / z + float(
        intrinsics["cx"]
    )
    v = float(intrinsics["fy"]) * points[:, 1] / z + float(
        intrinsics["cy"]
    )
    origin = np.asarray(plane["origin_camera_m"], dtype=np.float64)
    _, wall_y, _ = _wall_axes(plane)
    wall_depth = (points - origin) @ wall_y
    protrusion = -wall_depth
    shape_value = metadata.get("image_shape")
    image_shape = (
        (int(shape_value[0]), int(shape_value[1]))
        if isinstance(shape_value, (list, tuple)) and len(shape_value) == 2
        else None
    )
    clusters: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = [float(value) for value in box["xyxy"]]
        except (KeyError, TypeError, ValueError):
            continue
        inside = detection_pixel_mask(
            u, v, box, image_shape=image_shape
        )
        object_mask = inside & (protrusion >= 0.003) & (protrusion <= 0.15)
        if int(object_mask.sum()) < 12:
            object_mask = inside
        cluster_points = points[object_mask]
        if cluster_points.shape[0] < 3:
            continue
        centroid = np.median(cluster_points, axis=0)
        clusters.append(
            {
                "box_index": index,
                "cls": int(box.get("cls", -1)),
                "name": str(box.get("name", box.get("cls", "unknown"))),
                "conf": float(box.get("conf", 0.0)),
                "xyxy": [x1, y1, x2, y2],
                "point_count": int(cluster_points.shape[0]),
                "foreground_only": bool(
                    int((inside & (protrusion >= 0.003) & (protrusion <= 0.15)).sum())
                    >= 12
                ),
                "centroid_camera_m": centroid.tolist(),
                "centroid_plane_coordinates_m": target_plane_coordinates(
                    centroid.tolist(), plane
                ),
            }
        )
    return clusters


def target_plane_coordinates(
    target_camera_m: list[float], plane: dict[str, Any]
) -> dict[str, float]:
    target = np.asarray(target_camera_m, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("目标点必须是三个有限数值")
    origin = np.asarray(plane["origin_camera_m"], dtype=np.float64)
    delta = target - origin
    return _vector_wall_coordinates(delta, plane)


def _wall_axes(
    plane: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(
        key in plane
        for key in ("x_axis_camera", "y_axis_camera", "z_axis_camera")
    ):
        return tuple(
            np.asarray(plane[key], dtype=np.float64)
            for key in ("x_axis_camera", "y_axis_camera", "z_axis_camera")
        )

    # Read legacy v1 annotations as X=H, Y=-N, Z=-V.
    horizontal = np.asarray(plane["horizontal_axis_camera"], dtype=np.float64)
    vertical = np.asarray(plane["vertical_axis_camera"], dtype=np.float64)
    outward_normal = np.asarray(plane["normal_camera"], dtype=np.float64)
    return horizontal, -outward_normal, -vertical


def _vector_wall_coordinates(
    vector_camera_m: np.ndarray, plane: dict[str, Any]
) -> dict[str, float]:
    wall_x, wall_y, wall_z = _wall_axes(plane)
    return {
        "x_m": float(vector_camera_m @ wall_x),
        "y_m": float(vector_camera_m @ wall_y),
        "z_m": float(vector_camera_m @ wall_z),
    }


def build_wall_calibration(
    plane: dict[str, Any],
    origin_point_camera_m: list[float],
    x_point_camera_m: list[float],
    *,
    min_separation_m: float = 0.05,
) -> dict[str, Any]:
    selected_origin = np.asarray(origin_point_camera_m, dtype=np.float64)
    selected_x = np.asarray(x_point_camera_m, dtype=np.float64)
    if (
        selected_origin.shape != (3,)
        or selected_x.shape != (3,)
        or not np.isfinite(selected_origin).all()
        or not np.isfinite(selected_x).all()
    ):
        raise ValueError("坐标系标定点必须是三个有限数值")

    plane_origin = np.asarray(plane["origin_camera_m"], dtype=np.float64)
    _, wall_y, _ = _wall_axes(plane)
    wall_y = wall_y / np.linalg.norm(wall_y)

    calibrated_origin = selected_origin - (
        (selected_origin - plane_origin) @ wall_y
    ) * wall_y
    projected_x_point = selected_x - (
        (selected_x - plane_origin) @ wall_y
    ) * wall_y
    x_direction = projected_x_point - calibrated_origin
    separation = float(np.linalg.norm(x_direction))
    if separation < min_separation_m:
        raise ValueError(
            f"两个标定点在墙面上的距离必须至少为 {min_separation_m * 100:.0f} cm"
        )

    wall_x = x_direction / separation
    wall_z = np.cross(wall_x, wall_y)
    z_length = np.linalg.norm(wall_z)
    if z_length < 1e-6:
        raise ValueError("两个标定点方向不能与墙面法向平行")
    wall_z /= z_length
    wall_x = np.cross(wall_y, wall_z)
    wall_x /= np.linalg.norm(wall_x)

    return {
        "schema": "rgbd-wall-coordinate-calibration/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "origin_camera_m": calibrated_origin.tolist(),
        "x_axis_camera": wall_x.tolist(),
        "y_axis_camera": wall_y.tolist(),
        "z_axis_camera": wall_z.tolist(),
        "normal_camera": wall_y.tolist(),
        "coordinate_system": "wall-right-handed-x-right-y-inward-z-up",
        "origin_definition": "first-selected-point-projected-on-wall",
        "selected_origin_point_camera_m": selected_origin.tolist(),
        "selected_x_point_camera_m": selected_x.tolist(),
        "projected_x_point_camera_m": projected_x_point.tolist(),
        "x_baseline_m": separation,
    }


def _wall_calibration_path(
    data_root: Path, session_id: str, frame_id: str
) -> Path:
    root = data_root.expanduser().resolve()
    _, frame_dir = resolve_frame_paths(root, session_id, frame_id)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"帧不存在: {frame_id}")
    return (
        root
        / "wall_coordinate_calibrations"
        / session_id
        / f"{frame_id}.json"
    )


def load_wall_calibration(
    data_root: Path, session_id: str, frame_id: str
) -> dict[str, Any] | None:
    path = _wall_calibration_path(data_root, session_id, frame_id)
    if not path.exists():
        return None
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("schema") not in {
        "rgbd-wall-coordinate-calibration/v1",
        "rgbd-wall-coordinate-calibration/v2",
    }:
        raise ValueError("不支持的墙面坐标系标定文件版本")
    origin = np.asarray(calibration.get("origin_camera_m"), dtype=np.float64)
    axes = np.asarray(
        [
            calibration.get("x_axis_camera"),
            calibration.get("y_axis_camera"),
            calibration.get("z_axis_camera"),
        ],
        dtype=np.float64,
    )
    if (
        origin.shape != (3,)
        or axes.shape != (3, 3)
        or not np.isfinite(origin).all()
        or not np.isfinite(axes).all()
    ):
        raise ValueError("墙面坐标系标定文件包含无效数值")
    gram = axes @ axes.T
    if not np.allclose(gram, np.eye(3), atol=1e-4) or float(
        np.linalg.det(axes)
    ) < 0.999:
        raise ValueError("墙面坐标系标定轴不是有效右手正交坐标系")
    return calibration


def save_wall_calibration(
    data_root: Path,
    session_id: str,
    frame_id: str,
    calibration: dict[str, Any],
) -> Path:
    root = data_root.expanduser().resolve()
    path = _wall_calibration_path(root, session_id, frame_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        calibration, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=".wall-coordinate-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def apply_wall_calibration(
    fitted_plane: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    calibrated = dict(fitted_plane)
    for key in (
        "origin_camera_m",
        "x_axis_camera",
        "y_axis_camera",
        "z_axis_camera",
        "normal_camera",
        "coordinate_system",
        "origin_definition",
    ):
        calibrated[key] = calibration[key]
    calibrated["calibrated"] = True
    calibrated["calibration_created_at"] = calibration["created_at"]
    calibrated["calibration_baseline_m"] = calibration.get(
        "x_baseline_m", calibration.get("z_baseline_m")
    )
    calibrated["calibration_source"] = calibration.get("source")
    return calibrated


def _annotation_path(data_root: Path, session_id: str) -> Path:
    session_dir, _ = resolve_frame_paths(data_root, session_id, "_validate_")
    if not session_dir.is_dir():
        raise FileNotFoundError(f"会话不存在: {session_id}")
    return session_dir / "annotations.jsonl"


def load_annotations(
    data_root: Path, session_id: str
) -> list[dict[str, Any]]:
    path = _annotation_path(data_root, session_id)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def save_annotation(
    data_root: Path,
    session_id: str,
    frame_id: str,
    *,
    target_camera_m: list[float],
    plane: dict[str, Any],
    yolo: dict[str, Any] | None = None,
    target_pixel: dict[str, Any] | None = None,
    selection_source: str = "pointcloud",
    target_adjustment_camera_m: list[float] | None = None,
) -> dict[str, Any]:
    _, frame_dir = resolve_frame_paths(data_root, session_id, frame_id)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"帧不存在: {frame_id}")
    target = np.asarray(target_camera_m, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("目标点必须是三个有限数值")
    if target[2] <= 0:
        raise ValueError("目标点深度必须大于零")
    if selection_source not in {"pointcloud", "rgb"}:
        raise ValueError("selection_source 必须是 pointcloud 或 rgb")
    adjustment = np.asarray(
        target_adjustment_camera_m or [0.0, 0.0, 0.0], dtype=np.float64
    )
    if adjustment.shape != (3,) or not np.isfinite(adjustment).all():
        raise ValueError("目标微调量必须是三个有限数值")

    record = {
        "schema": "rgbd-target-annotation/v2",
        "session_id": session_id,
        "frame_id": frame_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target_camera_m": target.tolist(),
        "target_adjustment_camera_m": adjustment.tolist(),
        "target_adjustment_wall_m": _vector_wall_coordinates(adjustment, plane),
        "selection_source": selection_source,
        "target_plane_coordinates_m": target_plane_coordinates(
            target.tolist(), plane
        ),
        "plane": plane,
        "yolo": yolo or {"available": False, "boxes": []},
    }
    if target_pixel is not None:
        record["target_pixel"] = {
            "u": int(target_pixel["u"]),
            "v": int(target_pixel["v"]),
        }
    clusters = record["yolo"].get("clusters", [])
    if clusters:
        reference = min(
            clusters,
            key=lambda item: float(
                np.linalg.norm(
                    target
                    - np.asarray(item["centroid_camera_m"], dtype=np.float64)
                )
            ),
        )
        reference_point = np.asarray(
            reference["centroid_camera_m"], dtype=np.float64
        )
        delta = target - reference_point
        record["semantic_reference"] = reference
        record["target_relative_to_semantic_m"] = _vector_wall_coordinates(
            delta, plane
        )

    path = _annotation_path(data_root, session_id)
    records = load_annotations(data_root, session_id)
    records = [item for item in records if item.get("frame_id") != frame_id]
    records.append(record)
    records.sort(key=lambda item: str(item.get("frame_id", "")))
    payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in records
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=".annotations-", suffix=".jsonl", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return record
