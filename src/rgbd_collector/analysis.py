from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
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
        "axis_estimation": "camera-up-projection",
        "threshold_m": threshold_m,
        "inlier_count": int(inliers.shape[0]),
        "sample_count": int(fitting_points.shape[0]),
        "inlier_ratio": float(inliers.shape[0] / fitting_points.shape[0]),
        "rms_m": float(np.sqrt(np.mean(residuals**2))),
    }


def camera_plane_pose_metrics(plane: dict[str, Any]) -> dict[str, Any]:
    """Match IK_replay's perpendicular-debug camera/plane conventions."""
    normal_into_plane = np.asarray(
        plane.get("normal_camera", plane.get("y_axis_camera")),
        dtype=np.float64,
    )
    center = np.asarray(
        plane.get("center_camera_m", plane.get("origin_camera_m")),
        dtype=np.float64,
    )
    if (
        normal_into_plane.shape != (3,)
        or center.shape != (3,)
        or not np.isfinite(normal_into_plane).all()
        or not np.isfinite(center).all()
    ):
        raise ValueError("平面法向量或中心无效")
    normal_length = float(np.linalg.norm(normal_into_plane))
    if normal_length < 1e-9:
        raise ValueError("平面法向量长度为零")
    normal_into_plane /= normal_length
    if float(np.dot(normal_into_plane, center)) < 0:
        normal_into_plane = -normal_into_plane

    # IK_replay uses a normal pointing back toward the camera. In camera
    # coordinates x=right, y=down, z=forward; a front-facing plane is
    # therefore normal_cam=(0, 0, -1), with all three angles equal to zero.
    normal_toward_camera = -normal_into_plane
    yaw_err_deg = float(
        np.degrees(
            np.arctan2(
                normal_toward_camera[0], -normal_toward_camera[2]
            )
        )
    )
    pitch_err_deg = float(
        np.degrees(
            np.arctan2(
                normal_toward_camera[1], -normal_toward_camera[2]
            )
        )
    )
    tilt_deg = float(
        np.degrees(
            np.arccos(np.clip(-normal_toward_camera[2], -1.0, 1.0))
        )
    )
    return {
        "yaw_err_deg": yaw_err_deg,
        "pitch_err_deg": pitch_err_deg,
        "tilt_deg": tilt_deg,
        "distance_m": float(abs(np.dot(normal_toward_camera, center))),
        "normal_cam": normal_toward_camera.tolist(),
        "normal_into_plane_camera": normal_into_plane.tolist(),
        "plane_center_camera_m": center.tolist(),
        "convention": "IK_replay-perpendicular-debug",
    }


def measure_frame_camera_plane_pose(
    data_root: Path,
    session_id: str,
    frame_id: str,
    *,
    plane_threshold_m: float = 0.008,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    stride: int = 3,
    max_points: int = 60_000,
) -> dict[str, Any]:
    cloud, metadata = reconstruct_frame(
        data_root,
        session_id,
        frame_id,
        stride=stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        max_points=max_points,
    )
    plane = fit_dominant_plane(
        cloud["xyz"],
        threshold_m=plane_threshold_m,
        iterations=100,
    )
    return {
        "session_id": session_id,
        "frame_id": frame_id,
        **camera_plane_pose_metrics(plane),
        "plane_rms_m": plane["rms_m"],
        "plane_inlier_count": plane["inlier_count"],
        "plane_inlier_ratio": plane["inlier_ratio"],
        "fit_point_count": int(cloud.shape[0]),
        "source_point_count": int(metadata["source_point_count"]),
        "stride": int(metadata["stride"]),
        "depth_range_m": metadata["depth_range_m"],
        "plane_threshold_m": plane_threshold_m,
    }


def save_camera_plane_pose_measurements(
    data_root: Path,
    session_id: str,
    measurements: list[dict[str, Any]],
    *,
    options: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = data_root.expanduser().resolve()
    session_dir, _ = resolve_frame_paths(root, session_id, "_validate_")
    if not session_dir.is_dir():
        raise FileNotFoundError(f"会话不存在: {session_id}")
    path = session_dir / "camera_plane_pose_measurements.json"
    payload = {
        "schema": "rgbd-camera-plane-pose/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "definition": {
            "distance_m": "camera optical center perpendicular distance to plane",
            "yaw_pitch_tilt": "IK_replay perpendicular-debug convention; zero means camera faces plane normally",
            "camera_coordinates": "x-right-y-down-z-forward",
        },
        "options": options,
        "frame_count": len(measurements),
        "success_count": sum(bool(item.get("ok")) for item in measurements),
        "failure_count": sum(not bool(item.get("ok")) for item in measurements),
        "frames": measurements,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    fd, temporary_path = tempfile.mkstemp(
        prefix=".camera-plane-pose-", suffix=".json", dir=session_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path, payload


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


def split_plane_labels_by_connectivity(
    points_xyz: np.ndarray,
    plane_labels: np.ndarray,
    segmented_planes: list[dict[str, Any]],
    pixel_coordinates: np.ndarray,
    image_shape: tuple[int, int] | list[int],
    *,
    stride: int,
    source_point_count: int,
    min_component_count: int = 300,
    min_component_ratio: float = 0.002,
    max_patches: int = 12,
    preserve_farthest_plane: bool = False,
    max_planar_point_distance_from_farthest_plane_m: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = np.asarray(points_xyz, dtype=np.float64)
    labels = np.asarray(plane_labels, dtype=np.int32)
    pixels = np.asarray(pixel_coordinates, dtype=np.int32)
    if labels.shape != (points.shape[0],) or pixels.shape != (
        points.shape[0],
        2,
    ):
        raise ValueError("平面标签和像素坐标必须与点云数量一致")
    if (
        max_planar_point_distance_from_farthest_plane_m is not None
        and max_planar_point_distance_from_farthest_plane_m <= 0
    ):
        raise ValueError("P0 平面内点距离阈值必须大于零")
    height, width = int(image_shape[0]), int(image_shape[1])
    sampling_ratio = max(source_point_count / max(points.shape[0], 1), 1.0)
    sample_factor = (
        1 if sampling_ratio < 1.5 else int(np.ceil(np.sqrt(sampling_ratio)))
    )
    cell_size = max(1, stride * sample_factor)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid_u = np.clip(pixels[:, 0] // cell_size, 0, grid_width - 1)
    grid_v = np.clip(pixels[:, 1] // cell_size, 0, grid_height - 1)
    required_count = max(
        min_component_count,
        int(np.ceil(points.shape[0] * min_component_ratio)),
    )

    farthest_parent_index: int | None = None
    if preserve_farthest_plane and segmented_planes:
        farthest_parent_index = max(
            (int(plane["index"]) for plane in segmented_planes),
            key=lambda index: float(
                np.median(points[labels == index, 2])
            ),
        )
    p0_origin: np.ndarray | None = None
    p0_normal: np.ndarray | None = None
    p0_planar_axes: np.ndarray | None = None
    p0_spatial_buckets: dict[tuple[int, int], np.ndarray] = {}
    if farthest_parent_index is not None:
        p0_plane = next(
            plane
            for plane in segmented_planes
            if int(plane["index"]) == farthest_parent_index
        )
        p0_origin = np.asarray(
            p0_plane.get(
                "origin_camera_m",
                points[labels == farthest_parent_index].mean(axis=0),
            ),
            dtype=np.float64,
        )
        p0_normal = np.asarray(p0_plane["normal_camera"], dtype=np.float64)
        p0_normal /= np.linalg.norm(p0_normal)

        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(reference @ p0_normal)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        planar_x = reference - (reference @ p0_normal) * p0_normal
        planar_x /= np.linalg.norm(planar_x)
        planar_z = np.cross(p0_normal, planar_x)
        planar_z /= np.linalg.norm(planar_z)
        p0_planar_axes = np.column_stack((planar_x, planar_z))

        if max_planar_point_distance_from_farthest_plane_m is not None:
            radius = max_planar_point_distance_from_farthest_plane_m
            p0_coordinates = (
                points[labels == farthest_parent_index] - p0_origin
            ) @ p0_planar_axes
            buckets: dict[tuple[int, int], list[np.ndarray]] = {}
            cells = np.floor(p0_coordinates / radius).astype(np.int64)
            for coordinate, cell in zip(p0_coordinates, cells):
                key = (int(cell[0]), int(cell[1]))
                buckets.setdefault(key, []).append(coordinate)
            p0_spatial_buckets = {
                key: np.asarray(coordinates, dtype=np.float64)
                for key, coordinates in buckets.items()
            }

    planar_distance_cache: dict[int, float] = {}

    def nearest_planar_distance_to_p0(indices: np.ndarray) -> float:
        cache_key = id(indices)
        cached = planar_distance_cache.get(cache_key)
        if cached is not None:
            return cached
        if (
            p0_origin is None
            or p0_planar_axes is None
            or max_planar_point_distance_from_farthest_plane_m is None
        ):
            return 0.0
        if np.all(labels[indices] == farthest_parent_index):
            planar_distance_cache[cache_key] = 0.0
            return 0.0
        radius = max_planar_point_distance_from_farthest_plane_m
        coordinates = (points[indices] - p0_origin) @ p0_planar_axes
        cells = np.floor(coordinates / radius).astype(np.int64)
        nearest = float("inf")
        for coordinate, cell in zip(coordinates, cells):
            for offset_x in (-1, 0, 1):
                for offset_z in (-1, 0, 1):
                    bucket = p0_spatial_buckets.get(
                        (
                            int(cell[0]) + offset_x,
                            int(cell[1]) + offset_z,
                        )
                    )
                    if bucket is None:
                        continue
                    nearest = min(
                        nearest,
                        float(
                            np.min(
                                np.linalg.norm(
                                    bucket - coordinate,
                                    axis=1,
                                )
                            )
                        ),
                    )
        planar_distance_cache[cache_key] = nearest
        return nearest

    preserved_candidate: tuple[np.ndarray, int] | None = None
    component_candidates: list[tuple[np.ndarray, int]] = []
    for plane in segmented_planes:
        parent_index = int(plane["index"])
        parent_indices = np.flatnonzero(labels == parent_index)
        if parent_indices.size < required_count:
            continue
        if parent_index == farthest_parent_index:
            preserved_candidate = (parent_indices, parent_index)
            continue
        mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
        mask[grid_v[parent_indices], grid_u[parent_indices]] = 1
        _, components = cv2.connectedComponents(mask, connectivity=8)
        component_ids = components[
            grid_v[parent_indices], grid_u[parent_indices]
        ]
        counts = np.bincount(component_ids)
        accepted = [
            int(component_id)
            for component_id in np.argsort(counts[1:])[::-1] + 1
            if counts[component_id] >= required_count
        ]
        for component_id in accepted:
            selected = parent_indices[component_ids == component_id]
            component_candidates.append((selected, parent_index))

    if max_planar_point_distance_from_farthest_plane_m is not None:
        component_candidates = [
            candidate
            for candidate in component_candidates
            if nearest_planar_distance_to_p0(candidate[0])
            <= max_planar_point_distance_from_farthest_plane_m
        ]
    component_candidates.sort(key=lambda item: item[0].size, reverse=True)
    if preserved_candidate is not None:
        component_candidates.insert(0, preserved_candidate)
    patch_labels = np.full(points.shape[0], -1, dtype=np.int32)
    patches: list[dict[str, Any]] = []
    valid_count = max(int(np.isfinite(points).all(axis=1).sum()), 1)
    for patch_index, (indices, parent_index) in enumerate(
        component_candidates[:max_patches]
    ):
        patch_points = points[indices]
        origin = patch_points.mean(axis=0)
        _, _, vh = np.linalg.svd(patch_points - origin, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        if np.dot(normal, origin) < 0:
            normal = -normal
        residuals = (patch_points - origin) @ normal
        patch_labels[indices] = patch_index
        patches.append(
            {
                "index": patch_index,
                "parent_plane_index": parent_index,
                "is_farthest_plane": parent_index == farthest_parent_index,
                "median_depth_m": float(np.median(patch_points[:, 2])),
                "nearest_p0_xz_distance_m": nearest_planar_distance_to_p0(
                    indices
                ),
                "origin_camera_m": origin.tolist(),
                "normal_camera": normal.tolist(),
                "inlier_count": int(indices.size),
                "inlier_ratio": float(indices.size / valid_count),
                "rms_m": float(np.sqrt(np.mean(residuals**2))),
                "connectivity_cell_px": cell_size,
            }
        )
    return patch_labels, patches


def describe_segmented_plane_axes(
    points_xyz: np.ndarray,
    plane_labels: np.ndarray,
    segmented_planes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    points = np.asarray(points_xyz, dtype=np.float64)
    labels = np.asarray(plane_labels, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云必须是 N×3 数组")
    if labels.shape != (points.shape[0],):
        raise ValueError("平面标签必须与点云数量一致")

    descriptions: list[dict[str, Any]] = []
    for plane in segmented_planes:
        index = int(plane["index"])
        plane_points = points[labels == index]
        if plane_points.shape[0] < 3:
            continue
        center = plane_points.mean(axis=0)
        centered = plane_points - center
        eigenvalues, eigenvectors = np.linalg.eigh(
            centered.T @ centered / plane_points.shape[0]
        )
        order = np.argsort(eigenvalues)[::-1]
        normal = np.asarray(plane["normal_camera"], dtype=np.float64)
        normal /= np.linalg.norm(normal)

        long_axis = eigenvectors[:, order[0]]
        long_axis -= np.dot(long_axis, normal) * normal
        long_axis /= np.linalg.norm(long_axis)
        camera_right = np.array([1.0, 0.0, 0.0])
        camera_right -= np.dot(camera_right, normal) * normal
        if np.dot(long_axis, camera_right) < 0:
            long_axis = -long_axis
        short_axis = np.cross(normal, long_axis)
        short_axis /= np.linalg.norm(short_axis)

        long_positions = centered @ long_axis
        short_positions = centered @ short_axis
        long_min, long_max = np.quantile(long_positions, [0.02, 0.98])
        short_min, short_max = np.quantile(short_positions, [0.02, 0.98])
        descriptions.append(
            {
                **plane,
                "center_camera_m": center.tolist(),
                "long_axis_camera": long_axis.tolist(),
                "short_axis_camera": short_axis.tolist(),
                "long_start_camera_m": (
                    center + long_min * long_axis
                ).tolist(),
                "long_end_camera_m": (
                    center + long_max * long_axis
                ).tolist(),
                "short_start_camera_m": (
                    center + short_min * short_axis
                ).tolist(),
                "short_end_camera_m": (
                    center + short_max * short_axis
                ).tolist(),
                "long_length_m": float(long_max - long_min),
                "short_length_m": float(short_max - short_min),
                "axis_aspect_ratio": float(
                    (long_max - long_min)
                    / max(short_max - short_min, 1e-9)
                ),
            }
        )
    return descriptions


def describe_p0_boundary_lines(
    points_xyz: np.ndarray,
    plane_labels: np.ndarray,
    segmented_planes: list[dict[str, Any]],
    *,
    max_point_distance_m: float = 0.010,
    max_lines_per_plane: int = 3,
    line_threshold_m: float = 0.003,
    min_line_points: int = 30,
    ransac_trials: int = 250,
) -> list[dict[str, Any]]:
    points = np.asarray(points_xyz, dtype=np.float64)
    labels = np.asarray(plane_labels, dtype=np.int32)
    if not segmented_planes:
        return []
    p0 = next(
        (
            plane
            for plane in segmented_planes
            if bool(plane.get("is_farthest_plane"))
        ),
        segmented_planes[0],
    )
    p0_index = int(p0["index"])
    p0_points = points[labels == p0_index]
    if p0_points.shape[0] < 2:
        return [
            {**plane, "boundary_lines": []} for plane in segmented_planes
        ]

    origin = np.asarray(p0["origin_camera_m"], dtype=np.float64)
    normal = np.asarray(p0["normal_camera"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    planar_x = reference - (reference @ normal) * normal
    planar_x /= np.linalg.norm(planar_x)
    planar_z = np.cross(normal, planar_x)
    planar_z /= np.linalg.norm(planar_z)
    planar_axes = np.column_stack((planar_x, planar_z))
    p0_coordinates = (p0_points - origin) @ planar_axes

    described: list[dict[str, Any]] = []
    for plane in segmented_planes:
        plane_index = int(plane["index"])
        if plane_index == p0_index:
            described.append({**plane, "boundary_lines": []})
            continue
        plane_indices = np.flatnonzero(labels == plane_index)
        plane_points = points[plane_indices]
        plane_coordinates = (plane_points - origin) @ planar_axes
        cells = np.floor(
            plane_coordinates / max_point_distance_m
        ).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = {}
        for local_index, cell in enumerate(cells):
            key = (int(cell[0]), int(cell[1]))
            buckets.setdefault(key, []).append(local_index)

        nearest_local_indices: set[int] = set()
        p0_cells = np.floor(
            p0_coordinates / max_point_distance_m
        ).astype(np.int64)
        for p0_coordinate, cell in zip(p0_coordinates, p0_cells):
            nearest_index: int | None = None
            nearest_distance = float("inf")
            for offset_x in (-1, 0, 1):
                for offset_z in (-1, 0, 1):
                    candidates = buckets.get(
                        (
                            int(cell[0]) + offset_x,
                            int(cell[1]) + offset_z,
                        )
                    )
                    if not candidates:
                        continue
                    candidate_indices = np.asarray(candidates, dtype=np.int64)
                    distances = np.linalg.norm(
                        plane_coordinates[candidate_indices] - p0_coordinate,
                        axis=1,
                    )
                    local_best = int(np.argmin(distances))
                    distance = float(distances[local_best])
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_index = int(candidate_indices[local_best])
            if (
                nearest_index is not None
                and nearest_distance <= max_point_distance_m
            ):
                nearest_local_indices.add(nearest_index)

        boundary_local = np.asarray(
            sorted(nearest_local_indices), dtype=np.int64
        )
        boundary_coordinates = plane_coordinates[boundary_local]
        boundary_points = plane_points[boundary_local]
        lines: list[dict[str, Any]] = []
        remaining = np.arange(boundary_coordinates.shape[0], dtype=np.int64)
        rng = np.random.default_rng(10_000 + plane_index)
        for line_index in range(max_lines_per_plane):
            if remaining.size < min_line_points:
                break
            best_inliers: np.ndarray | None = None
            best_score = 0.0
            for _ in range(ransac_trials):
                sample = rng.choice(remaining, size=2, replace=False)
                start, end = boundary_coordinates[sample]
                direction = end - start
                length = float(np.linalg.norm(direction))
                if length < 1e-6:
                    continue
                direction /= length
                relative = boundary_coordinates[remaining] - start
                distances = np.abs(
                    relative[:, 0] * direction[1]
                    - relative[:, 1] * direction[0]
                )
                inliers = remaining[distances <= line_threshold_m]
                if inliers.size < min_line_points:
                    continue
                center_2d = boundary_coordinates[inliers].mean(axis=0)
                _, _, vh = np.linalg.svd(
                    boundary_coordinates[inliers] - center_2d,
                    full_matrices=False,
                )
                refined_direction = vh[0]
                positions = (
                    boundary_coordinates[inliers] - center_2d
                ) @ refined_direction
                low, high = np.quantile(positions, [0.02, 0.98])
                span = float(high - low)
                score = span * float(inliers.size)
                if span >= 0.02 and score > best_score:
                    best_score = score
                    best_inliers = inliers
            if best_inliers is None:
                break

            center_2d = boundary_coordinates[best_inliers].mean(axis=0)
            _, _, vh = np.linalg.svd(
                boundary_coordinates[best_inliers] - center_2d,
                full_matrices=False,
            )
            direction_2d = vh[0]
            direction_3d = planar_axes @ direction_2d
            direction_3d /= np.linalg.norm(direction_3d)
            inlier_points = boundary_points[best_inliers]
            center_3d = inlier_points.mean(axis=0)
            positions = (inlier_points - center_3d) @ direction_3d
            low, high = np.quantile(positions, [0.02, 0.98])
            lines.append(
                {
                    "index": line_index,
                    "start_camera_m": (
                        center_3d + low * direction_3d
                    ).tolist(),
                    "end_camera_m": (
                        center_3d + high * direction_3d
                    ).tolist(),
                    "direction_camera": direction_3d.tolist(),
                    "length_m": float(high - low),
                    "inlier_count": int(best_inliers.size),
                }
            )
            selected = boundary_coordinates[best_inliers]
            line_origin = selected.mean(axis=0)
            relative = boundary_coordinates[remaining] - line_origin
            distances = np.abs(
                relative[:, 0] * direction_2d[1]
                - relative[:, 1] * direction_2d[0]
            )
            remaining = remaining[distances > line_threshold_m]

        if not lines and boundary_points.shape[0] >= 2:
            center_2d = boundary_coordinates.mean(axis=0)
            _, _, vh = np.linalg.svd(
                boundary_coordinates - center_2d,
                full_matrices=False,
            )
            direction_3d = planar_axes @ vh[0]
            direction_3d /= np.linalg.norm(direction_3d)
            center_3d = boundary_points.mean(axis=0)
            positions = (boundary_points - center_3d) @ direction_3d
            low, high = np.quantile(positions, [0.02, 0.98])
            if high - low >= 0.02:
                lines.append(
                    {
                        "index": 0,
                        "start_camera_m": (
                            center_3d + low * direction_3d
                        ).tolist(),
                        "end_camera_m": (
                            center_3d + high * direction_3d
                        ).tolist(),
                        "direction_camera": direction_3d.tolist(),
                        "length_m": float(high - low),
                        "inlier_count": int(boundary_points.shape[0]),
                        "fit_method": "principal-axis-fallback",
                    }
                )
        lines.sort(key=lambda line: float(line["length_m"]), reverse=True)
        for index, line in enumerate(lines):
            line["index"] = index
            line.setdefault("fit_method", "ransac")
        described.append(
            {
                **plane,
                "p0_near_boundary_point_count": int(boundary_local.size),
                "boundary_lines": lines,
            }
        )
    return described


def estimate_wall_x_from_p0_boundary_lines(
    fitted_plane: dict[str, Any],
    segmented_planes: list[dict[str, Any]],
    *,
    max_direction_angle_deg: float = 1.0,
    min_group_line_length_m: float = 0.10,
    min_group_relative_length: float = 0.25,
    min_distinct_planes: int = 2,
    min_camera_up_line_angle_deg: float = 45.0,
) -> dict[str, Any]:
    result = dict(fitted_plane)
    wall_y = np.asarray(result["y_axis_camera"], dtype=np.float64)
    wall_y /= np.linalg.norm(wall_y)
    camera_up = np.array([0.0, -1.0, 0.0])
    candidates: list[
        tuple[float, np.ndarray, dict[str, Any], dict[str, Any]]
    ] = []
    for segment in segmented_planes:
        for line in segment.get("boundary_lines", []):
            line["selected_for_x"] = False
            line["accepted_for_x_group"] = False
            line["passes_camera_up_angle"] = False
            length = float(line.get("length_m", 0.0))
            direction = np.asarray(
                line.get("direction_camera"), dtype=np.float64
            )
            if direction.shape != (3,) or not np.isfinite(direction).all():
                continue
            direction -= np.dot(direction, wall_y) * wall_y
            direction_length = np.linalg.norm(direction)
            if direction_length >= 1e-6:
                direction /= direction_length
                camera_up_angle_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                abs(float(direction @ camera_up)),
                                0.0,
                                1.0,
                            )
                        )
                    )
                )
                line["camera_up_line_angle_deg"] = camera_up_angle_deg
                line["passes_camera_up_angle"] = (
                    camera_up_angle_deg >= min_camera_up_line_angle_deg
                )
            if (
                length >= min_group_line_length_m
                and direction_length >= 1e-6
                and line["passes_camera_up_angle"]
            ):
                candidates.append(
                    (length, direction, segment, line)
                )
    if not candidates:
        return result

    cosine_threshold = float(
        np.cos(np.radians(max_direction_angle_deg))
    )
    groups: list[
        tuple[
            tuple[int, float, float],
            list[tuple[float, np.ndarray, dict[str, Any], dict[str, Any]]],
        ]
    ] = []
    for reference in candidates:
        nearby = sorted(
            [
                candidate
                for candidate in candidates
                if abs(float(reference[1] @ candidate[1]))
                >= cosine_threshold
            ],
            key=lambda candidate: candidate[0],
            reverse=True,
        )
        members: list[
            tuple[float, np.ndarray, dict[str, Any], dict[str, Any]]
        ] = []
        member_planes: set[int] = set()
        for candidate in nearby:
            plane_index = int(candidate[2]["index"])
            if plane_index in member_planes:
                continue
            if any(
                abs(float(candidate[1] @ member[1])) < cosine_threshold
                for member in members
            ):
                continue
            members.append(candidate)
            member_planes.add(plane_index)
        if members:
            relative_cutoff = (
                max(member[0] for member in members)
                * min_group_relative_length
            )
            members = [
                member for member in members if member[0] >= relative_cutoff
            ]
            member_planes = {
                int(member[2]["index"]) for member in members
            }
        distinct_planes = member_planes
        if len(distinct_planes) < min_distinct_planes:
            continue
        score = (
            len(distinct_planes),
            float(sum(member[0] for member in members)),
            max(member[0] for member in members),
        )
        groups.append((score, members))
    if not groups:
        return result

    _, selected_group = max(groups, key=lambda group: group[0])
    length, _, segment, selected_line = max(
        selected_group, key=lambda candidate: candidate[0]
    )
    for _, _, _, line in selected_group:
        line["accepted_for_x_group"] = True
    line_start = np.asarray(
        selected_line["start_camera_m"], dtype=np.float64
    )
    line_end = np.asarray(selected_line["end_camera_m"], dtype=np.float64)
    wall_x = line_end - line_start
    wall_x -= np.dot(wall_x, wall_y) * wall_y
    wall_x_length = np.linalg.norm(wall_x)
    if wall_x_length < 1e-6:
        return result
    wall_x /= wall_x_length
    wall_z = np.cross(wall_x, wall_y)
    wall_z /= np.linalg.norm(wall_z)
    if np.dot(wall_z, camera_up) < 0:
        wall_x = -wall_x
        wall_z = -wall_z
        line_start, line_end = line_end, line_start

    selected_line["selected_for_x"] = True
    result["x_axis_camera"] = wall_x.tolist()
    result["z_axis_camera"] = wall_z.tolist()
    result["axis_estimation"] = "p0-nearest-boundary-line"
    result["axis_reference_plane_index"] = int(segment["index"])
    result["axis_reference_boundary_index"] = int(selected_line["index"])
    result["axis_reference_line_length_m"] = length
    result["axis_reference_line_fit_method"] = selected_line.get(
        "fit_method", "ransac"
    )
    result["axis_reference_group_size"] = len(
        {int(member[2]["index"]) for member in selected_group}
    )
    result["axis_reference_group_line_count"] = len(selected_group)
    result["axis_reference_group_total_length_m"] = float(
        sum(member[0] for member in selected_group)
    )
    result["axis_reference_group_angle_tolerance_deg"] = (
        max_direction_angle_deg
    )
    result["axis_reference_camera_up_angle_deg"] = float(
        selected_line["camera_up_line_angle_deg"]
    )
    result["axis_reference_min_camera_up_angle_deg"] = (
        min_camera_up_line_angle_deg
    )
    result["automatic_x_line_start_camera_m"] = line_start.tolist()
    result["automatic_x_line_end_camera_m"] = line_end.tolist()
    return result


def estimate_wall_x_from_plane_intersections(
    fitted_plane: dict[str, Any],
    segmented_planes: list[dict[str, Any]],
    *,
    min_angle_deg: float = 2.0,
    max_angle_deg: float = 25.0,
    min_camera_horizontal_alignment: float = 0.65,
) -> dict[str, Any]:
    result = dict(fitted_plane)
    wall_y = np.asarray(result["y_axis_camera"], dtype=np.float64)
    wall_y /= np.linalg.norm(wall_y)
    camera_right = np.array([1.0, 0.0, 0.0])
    camera_right -= np.dot(camera_right, wall_y) * wall_y
    right_length = np.linalg.norm(camera_right)
    if right_length < 1e-6:
        return result
    camera_right /= right_length

    best: tuple[float, np.ndarray, dict[str, Any], float] | None = None
    for candidate in segmented_planes:
        normal = np.asarray(candidate.get("normal_camera"), dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            continue
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-8:
            continue
        normal /= normal_length
        cosine = float(np.clip(abs(wall_y @ normal), 0.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cosine)))
        if not min_angle_deg <= angle_deg <= max_angle_deg:
            continue
        direction = np.cross(wall_y, normal)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction /= length
        alignment = float(abs(direction @ camera_right))
        if alignment < min_camera_horizontal_alignment:
            continue
        score = float(candidate.get("inlier_ratio", 0.0)) * alignment
        if best is None or score > best[0]:
            best = (score, direction, candidate, angle_deg)

    if best is None:
        return result
    _, wall_x, reference, angle_deg = best
    wall_z = np.cross(wall_x, wall_y)
    wall_z /= np.linalg.norm(wall_z)
    camera_up = np.array([0.0, -1.0, 0.0])
    if np.dot(wall_z, camera_up) < 0:
        wall_x = -wall_x
        wall_z = -wall_z

    result["x_axis_camera"] = wall_x.tolist()
    result["z_axis_camera"] = wall_z.tolist()
    result["axis_estimation"] = "multi-plane-intersection"
    result["axis_reference_plane_index"] = int(reference["index"])
    result["axis_reference_angle_deg"] = angle_deg
    return result


def estimate_wall_x_from_secondary_plane_shape(
    fitted_plane: dict[str, Any],
    points_xyz: np.ndarray,
    plane_labels: np.ndarray,
    segmented_planes: list[dict[str, Any]],
    *,
    min_normal_angle_deg: float = 0.8,
    max_normal_angle_deg: float = 25.0,
    min_axis_alignment: float = 0.65,
    min_shape_anisotropy: float = 1.7,
) -> dict[str, Any]:
    result = dict(fitted_plane)
    points = np.asarray(points_xyz, dtype=np.float64)
    labels = np.asarray(plane_labels, dtype=np.int32)
    wall_y = np.asarray(result["y_axis_camera"], dtype=np.float64)
    wall_y /= np.linalg.norm(wall_y)
    camera_right = np.array([1.0, 0.0, 0.0])
    camera_right -= np.dot(camera_right, wall_y) * wall_y
    camera_right /= np.linalg.norm(camera_right)

    best: tuple[
        float, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
    ] | None = None
    for candidate in segmented_planes:
        index = int(candidate["index"])
        if index == 0:
            continue
        candidate_points = points[labels == index]
        if candidate_points.shape[0] < 100:
            continue
        normal = np.asarray(candidate["normal_camera"], dtype=np.float64)
        normal_length = np.linalg.norm(normal)
        if normal.shape != (3,) or normal_length < 1e-8:
            continue
        normal /= normal_length
        angle_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(abs(float(wall_y @ normal)), 0.0, 1.0)
                )
            )
        )
        if not min_normal_angle_deg <= angle_deg <= max_normal_angle_deg:
            continue

        center = candidate_points.mean(axis=0)
        centered = candidate_points - center
        eigenvalues, eigenvectors = np.linalg.eigh(
            centered.T @ centered / candidate_points.shape[0]
        )
        order = np.argsort(eigenvalues)[::-1]
        if eigenvalues[order[1]] <= 0:
            continue
        anisotropy = float(
            np.sqrt(eigenvalues[order[0]] / eigenvalues[order[1]])
        )
        if anisotropy < min_shape_anisotropy:
            continue

        wall_x = eigenvectors[:, order[0]]
        wall_x -= np.dot(wall_x, wall_y) * wall_y
        wall_x_length = np.linalg.norm(wall_x)
        if wall_x_length < 1e-6:
            continue
        wall_x /= wall_x_length
        alignment = float(abs(wall_x @ camera_right))
        if alignment < min_axis_alignment:
            continue
        if best is None or anisotropy > best[0]:
            positions = centered @ wall_x
            axis_start, axis_end = np.quantile(positions, [0.02, 0.98])
            best = (
                anisotropy,
                wall_x,
                center + axis_start * wall_x,
                center + axis_end * wall_x,
                {**candidate, "normal_angle_deg": angle_deg},
            )

    if best is None:
        return result
    anisotropy, wall_x, line_start, line_end, reference = best
    wall_z = np.cross(wall_x, wall_y)
    wall_z /= np.linalg.norm(wall_z)
    camera_up = np.array([0.0, -1.0, 0.0])
    if np.dot(wall_z, camera_up) < 0:
        wall_x = -wall_x
        wall_z = -wall_z
        line_start, line_end = line_end, line_start

    result["x_axis_camera"] = wall_x.tolist()
    result["z_axis_camera"] = wall_z.tolist()
    result["axis_estimation"] = "secondary-plane-principal-axis"
    result["axis_reference_plane_index"] = int(reference["index"])
    result["axis_reference_angle_deg"] = float(
        reference["normal_angle_deg"]
    )
    result["axis_shape_anisotropy"] = anisotropy
    result["automatic_x_line_start_camera_m"] = line_start.tolist()
    result["automatic_x_line_end_camera_m"] = line_end.tolist()
    return result


def fit_yolo_panel_rectangle(
    points_xyz: np.ndarray,
    *,
    threshold_m: float = 0.004,
    min_points: int = 100,
    min_inlier_ratio: float = 0.35,
    grid_cell_m: float = 0.002,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a local panel plane and a supported orthogonal L-shaped outline."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("YOLO Mask 点云必须是 N×3 数组")
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < min_points:
        raise ValueError(
            f"YOLO Mask 内有效点不足：{points.shape[0]} < {min_points}"
        )
    if not 0.001 <= threshold_m <= 0.02:
        raise ValueError("YOLO 面板平面阈值必须在 1–20 mm 之间")

    coarse = fit_dominant_plane(
        points,
        threshold_m=threshold_m,
        iterations=160,
        min_inlier_ratio=0.20,
        seed=seed,
    )
    coarse_center = np.asarray(
        coarse["center_camera_m"], dtype=np.float64
    )
    coarse_normal = np.asarray(
        coarse["normal_camera"], dtype=np.float64
    )
    signed_distances = (points - coarse_center) @ coarse_normal
    plane_mask = np.abs(signed_distances) <= threshold_m
    if int(plane_mask.sum()) < min_points:
        raise ValueError("YOLO Mask 内局部平面内点不足")
    inlier_ratio = float(plane_mask.mean())
    if inlier_ratio < min_inlier_ratio:
        raise ValueError(
            f"YOLO 面板平面内点比例仅 {inlier_ratio:.1%}"
        )

    inliers = points[plane_mask]
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if float(normal @ center) < 0:
        normal = -normal
    final_distances = (points - center) @ normal
    plane_mask = np.abs(final_distances) <= threshold_m
    inliers = points[plane_mask]
    if inliers.shape[0] < min_points:
        raise ValueError("局部平面精化后内点不足")
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if float(normal @ center) < 0:
        normal = -normal
    residuals = (inliers - center) @ normal

    basis_x = np.array([1.0, 0.0, 0.0])
    basis_x -= float(basis_x @ normal) * normal
    if float(np.linalg.norm(basis_x)) < 1e-6:
        basis_x = np.array([0.0, -1.0, 0.0])
        basis_x -= float(basis_x @ normal) * normal
    basis_x /= np.linalg.norm(basis_x)
    basis_z = np.cross(normal, basis_x)
    basis_z /= np.linalg.norm(basis_z)
    planar_basis = np.column_stack((basis_x, basis_z))
    planar = (inliers - center) @ planar_basis

    low = np.quantile(planar, 0.005, axis=0)
    high = np.quantile(planar, 0.995, axis=0)
    extent = high - low
    if float(np.min(extent)) < 0.02:
        raise ValueError("YOLO 面板平面范围太小，无法识别长短边")
    cell_size = max(
        grid_cell_m,
        float(np.max(extent)) / 900.0,
    )
    grid_shape = np.ceil(extent / cell_size).astype(np.int64) + 5
    grid_shape = np.maximum(grid_shape, 8)
    grid = np.zeros((int(grid_shape[1]), int(grid_shape[0])), np.uint8)
    cells = np.rint((planar - low) / cell_size).astype(np.int64) + 2
    valid_cells = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < grid.shape[1])
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < grid.shape[0])
    )
    grid[cells[valid_cells, 1], cells[valid_cells, 0]] = 1
    # On tiny panels the raster is only a few dozen cells wide; a fixed
    # 5x5 close kernel would visibly reshape the outline there.
    close_size = 3 if int(min(grid.shape)) < 64 else 5
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, close_kernel)
    component_count, component_labels, stats, _ = (
        cv2.connectedComponentsWithStats(grid, connectivity=8)
    )
    if component_count <= 1:
        raise ValueError("YOLO 面板平面没有稳定连通区域")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (component_labels == largest_label).astype(np.uint8)
    # Keep only inliers that fall inside the dominant connected component.
    # Coplanar stragglers from mask bleed would otherwise drag the extent
    # (and thus the drawn edges) off the panel.
    in_component = np.zeros(planar.shape[0], dtype=bool)
    in_component[valid_cells] = (
        component_labels[cells[valid_cells, 1], cells[valid_cells, 0]]
        == largest_label
    )
    component_planar = planar[in_component]
    if component_planar.shape[0] < min_points:
        raise ValueError("面板主连通区域内点不足")
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("YOLO 面板平面无法提取外轮廓")
    contour = max(contours, key=cv2.contourArea)
    boundary = np.zeros_like(component)
    cv2.drawContours(boundary, [contour], -1, 255, 1)
    min_line_pixels = max(8, int(min(grid.shape) * 0.10))
    max_gap_m = min(0.012, float(np.max(extent)) * 0.05)
    hough = cv2.HoughLinesP(
        boundary,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(8, min_line_pixels // 2),
        minLineLength=min_line_pixels,
        maxLineGap=max(2, int(round(max_gap_m / cell_size))),
    )
    if hough is None or len(hough) < 2:
        raise ValueError("面板外轮廓没有足够的直线边支持")

    segments: list[dict[str, Any]] = []
    total_segment_length = 0.0
    for raw_line in np.asarray(hough).reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in raw_line]
        start_2d = low + (np.array([x1, y1]) - 2.0) * cell_size
        end_2d = low + (np.array([x2, y2]) - 2.0) * cell_size
        delta = end_2d - start_2d
        length = float(np.linalg.norm(delta))
        if length < 0.012:
            continue
        direction = delta / length
        total_segment_length += length
        segments.append(
            {
                "start": start_2d,
                "end": end_2d,
                "midpoint": (start_2d + end_2d) * 0.5,
                "direction": direction,
                "length": length,
            }
        )
    if len(segments) < 2 or total_segment_length < 0.03:
        raise ValueError("面板边界直线长度不足")
    direction_tolerance_cos = float(np.cos(np.radians(15.0)))
    best_orientation: tuple[float, np.ndarray, np.ndarray] | None = None
    best_orientation_score = -1.0
    for candidate in segments:
        axis_a_candidate = np.asarray(candidate["direction"])
        axis_b_candidate = np.array(
            [-axis_a_candidate[1], axis_a_candidate[0]]
        )
        support_a = 0.0
        support_b = 0.0
        for segment in segments:
            direction = np.asarray(segment["direction"])
            length = float(segment["length"])
            alignment_a = abs(float(direction @ axis_a_candidate))
            alignment_b = abs(float(direction @ axis_b_candidate))
            if max(alignment_a, alignment_b) < direction_tolerance_cos:
                continue
            if alignment_a >= alignment_b:
                support_a += length
            else:
                support_b += length
        aligned_support = support_a + support_b
        balance = min(support_a, support_b) / max(
            support_a, support_b, 1e-9
        )
        score = aligned_support * (0.5 + 0.5 * balance)
        if score > best_orientation_score:
            best_orientation_score = score
            best_orientation = (
                aligned_support,
                axis_a_candidate,
                axis_b_candidate,
            )
    assert best_orientation is not None
    aligned_support, axis_a, axis_b = best_orientation
    orientation_concentration = float(
        aligned_support / total_segment_length
    )
    if orientation_concentration < 0.35:
        raise ValueError("面板边界方向不稳定")
    positions_a = component_planar @ axis_a
    positions_b = component_planar @ axis_b
    a_min, a_max = np.quantile(positions_a, [0.01, 0.99])
    b_min, b_max = np.quantile(positions_b, [0.01, 0.99])
    if a_max - a_min >= b_max - b_min:
        long_axis_2d, short_axis_2d = axis_a, axis_b
        long_min, long_max = float(a_min), float(a_max)
        short_min, short_max = float(b_min), float(b_max)
    else:
        long_axis_2d, short_axis_2d = axis_b, -axis_a
        long_min, long_max = float(b_min), float(b_max)
        short_positions = component_planar @ short_axis_2d
        short_min, short_max = [
            float(value)
            for value in np.quantile(short_positions, [0.01, 0.99])
        ]

    angle_tolerance_cos = direction_tolerance_cos
    # Snapping tolerance scales with panel size so that, on a ~5 cm panel,
    # edges cannot be pulled 15 mm onto a neighbouring structure.
    offset_tolerance = float(
        np.clip(
            0.12 * min(long_max - long_min, short_max - short_min),
            max(0.004, threshold_m + cell_size),
            0.015,
        )
    )

    def supported_offset(
        axis_direction: np.ndarray,
        offset_direction: np.ndarray,
        expected: float,
    ) -> tuple[float, float]:
        candidates: list[tuple[float, float]] = []
        for segment in segments:
            alignment = abs(
                float(np.asarray(segment["direction"]) @ axis_direction)
            )
            offset = float(
                np.asarray(segment["midpoint"]) @ offset_direction
            )
            if (
                alignment >= angle_tolerance_cos
                and abs(offset - expected) <= offset_tolerance
            ):
                candidates.append((offset, float(segment["length"])))
        if not candidates:
            return expected, 0.0
        weights = np.asarray([item[1] for item in candidates])
        values = np.asarray([item[0] for item in candidates])
        return (
            float(np.average(values, weights=weights)),
            float(weights.sum()),
        )

    short_min, long_at_short_min = supported_offset(
        long_axis_2d, short_axis_2d, short_min
    )
    short_max, long_at_short_max = supported_offset(
        long_axis_2d, short_axis_2d, short_max
    )
    long_min, short_at_long_min = supported_offset(
        short_axis_2d, long_axis_2d, long_min
    )
    long_max, short_at_long_max = supported_offset(
        short_axis_2d, long_axis_2d, long_max
    )
    if max(long_at_short_min, long_at_short_max) < 0.012:
        raise ValueError("没有找到受点云支持的完整长边")
    if max(short_at_long_min, short_at_long_max) < 0.012:
        raise ValueError("没有找到受点云支持的完整短边")

    selected_short = (
        short_min
        if long_at_short_min >= long_at_short_max
        else short_max
    )
    selected_long = (
        long_min
        if short_at_long_min >= short_at_long_max
        else long_max
    )
    opposite_long = (
        long_max if selected_long == long_min else long_min
    )
    opposite_short = (
        short_max if selected_short == short_min else short_min
    )

    def camera_point(long_position: float, short_position: float) -> np.ndarray:
        planar_position = (
            long_position * long_axis_2d
            + short_position * short_axis_2d
        )
        return center + planar_basis @ planar_position

    corner = camera_point(selected_long, selected_short)
    long_end = camera_point(opposite_long, selected_short)
    short_end = camera_point(selected_long, opposite_short)
    rectangle_corners = [
        camera_point(long_min, short_min),
        camera_point(long_max, short_min),
        camera_point(long_max, short_max),
        camera_point(long_min, short_max),
    ]
    long_axis_camera = planar_basis @ long_axis_2d
    short_axis_camera = planar_basis @ short_axis_2d
    long_axis_camera /= np.linalg.norm(long_axis_camera)
    short_axis_camera /= np.linalg.norm(short_axis_camera)
    long_length = float(abs(opposite_long - selected_long))
    short_length = float(abs(opposite_short - selected_short))
    return {
        "available": True,
        "fit_method": "local-ransac-supported-orthogonal-rectangle",
        "point_count": int(points.shape[0]),
        "component_point_count": int(component_planar.shape[0]),
        "inlier_count": int(inliers.shape[0]),
        "inlier_ratio": float(inliers.shape[0] / points.shape[0]),
        "excluded_point_count": int(points.shape[0] - inliers.shape[0]),
        "camera_side_protrusion_point_count": int(
            (final_distances < -threshold_m).sum()
        ),
        "threshold_m": float(threshold_m),
        "rms_m": float(np.sqrt(np.mean(residuals**2))),
        "center_camera_m": center.tolist(),
        "normal_camera": normal.tolist(),
        "rectangle_corners_camera_m": [
            corner_point.tolist() for corner_point in rectangle_corners
        ],
        "long_axis_camera": long_axis_camera.tolist(),
        "short_axis_camera": short_axis_camera.tolist(),
        "long_length_m": long_length,
        "short_length_m": short_length,
        "axis_aspect_ratio": float(
            long_length / max(short_length, 1e-9)
        ),
        "orientation_support": orientation_concentration,
        "edges": [
            {
                "role": "long",
                "start_camera_m": corner.tolist(),
                "end_camera_m": long_end.tolist(),
                "length_m": long_length,
                "support_length_m": float(
                    max(long_at_short_min, long_at_short_max)
                ),
            },
            {
                "role": "short",
                "start_camera_m": corner.tolist(),
                "end_camera_m": short_end.tolist(),
                "length_m": short_length,
                "support_length_m": float(
                    max(short_at_long_min, short_at_long_max)
                ),
            },
        ],
    }


def analyze_yolo_mask_panel(
    points_xyz: np.ndarray,
    pixel_coordinates: np.ndarray,
    boxes: list[dict[str, Any]],
    image_shape: tuple[int, int] | list[int] | None,
    *,
    threshold_m: float = 0.004,
    min_points: int = 100,
) -> dict[str, Any]:
    if not boxes:
        return {"available": False, "reason": "当前帧没有 YOLO 实例"}
    valid_boxes = [
        (index, box)
        for index, box in enumerate(boxes)
        if np.isfinite(float(box.get("conf", 0.0)))
    ]
    if not valid_boxes:
        return {"available": False, "reason": "YOLO 实例置信度无效"}
    box_index, box = max(
        valid_boxes, key=lambda item: float(item[1].get("conf", 0.0))
    )
    points = np.asarray(points_xyz, dtype=np.float64)
    pixels = np.asarray(pixel_coordinates, dtype=np.float64)
    if pixels.shape != (points.shape[0], 2):
        raise ValueError("像素坐标必须与点云一一对应")
    shape = (
        (int(image_shape[0]), int(image_shape[1]))
        if image_shape is not None and len(image_shape) == 2
        else None
    )
    inside = detection_pixel_mask(
        pixels[:, 0], pixels[:, 1], box, image_shape=shape
    )
    mask_points = points[inside]
    detection = {
        "box_index": int(box_index),
        "cls": int(box.get("cls", -1)),
        "name": str(box.get("name", box.get("cls", "unknown"))),
        "conf": float(box.get("conf", 0.0)),
        "xyxy": box.get("xyxy"),
        "used_polygon_mask": bool(box.get("polygon") is not None),
    }
    try:
        fitted = fit_yolo_panel_rectangle(
            mask_points,
            threshold_m=threshold_m,
            min_points=min_points,
            seed=20_000 + int(box_index),
        )
    except ValueError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "mask_point_count": int(mask_points.shape[0]),
            "detection": detection,
        }
    fitted["mask_point_count"] = int(mask_points.shape[0])
    fitted["detection"] = detection
    return fitted


def analyze_frame(
    data_root: Path,
    session_id: str,
    frame_id: str,
    *,
    boxes: list[dict[str, Any]] | None = None,
    plane_threshold_m: float = 0.008,
    min_depth_m: float = 0.15,
    max_depth_m: float = 3.0,
    stride: int = 3,
    max_points: int = 1_000_000,
    plane_analysis_max_points: int = 200_000,
    min_plane_points: int = 300,
    use_saved_wall_calibration: bool = True,
    include_plane_debug: bool = False,
    include_highest_confidence_semantic_cloud: bool = False,
    include_yolo_panel_fit: bool = False,
) -> dict[str, Any]:
    if min_plane_points < 3:
        raise ValueError("最少平面点数不能小于 3")
    if not 1_000 <= plane_analysis_max_points <= 1_000_000:
        raise ValueError("X 轴分析最大点数必须在 1000~1000000")
    calibration = (
        load_wall_calibration(data_root, session_id, frame_id)
        if use_saved_wall_calibration
        else None
    )
    run_plane_analysis = calibration is None or include_plane_debug
    cloud, metadata = reconstruct_frame(
        data_root,
        session_id,
        frame_id,
        stride=stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        max_points=max_points,
        include_pixels=run_plane_analysis or include_yolo_panel_fit,
    )
    pixel_coordinates = metadata.pop("_pixel_coordinates", None)
    if run_plane_analysis:
        assert pixel_coordinates is not None
        plane_xyz = cloud["xyz"]
        plane_pixels = pixel_coordinates
        if plane_xyz.shape[0] > plane_analysis_max_points:
            analysis_indices = np.linspace(
                0,
                plane_xyz.shape[0] - 1,
                plane_analysis_max_points,
                dtype=np.int64,
            )
            plane_xyz = plane_xyz[analysis_indices]
            plane_pixels = plane_pixels[analysis_indices]
        plane_sampling_ratio = plane_xyz.shape[0] / max(cloud.shape[0], 1)
        analysis_min_plane_points = max(
            3, int(np.ceil(min_plane_points * plane_sampling_ratio))
        )
        plane = fit_dominant_plane(
            plane_xyz, threshold_m=plane_threshold_m
        )
        plane_labels, segmented_planes = segment_dominant_planes(
            plane_xyz, threshold_m=plane_threshold_m
        )
        plane_labels, segmented_planes = split_plane_labels_by_connectivity(
            plane_xyz,
            plane_labels,
            segmented_planes,
            plane_pixels,
            metadata["image_shape"],
            stride=stride,
            source_point_count=int(metadata["source_point_count"]),
            min_component_count=analysis_min_plane_points,
            min_component_ratio=0.0,
            preserve_farthest_plane=True,
            max_planar_point_distance_from_farthest_plane_m=0.010,
        )
        plane_segments = describe_p0_boundary_lines(
            plane_xyz, plane_labels, segmented_planes
        )
        if calibration is not None:
            plane = apply_wall_calibration(plane, calibration)
        elif use_saved_wall_calibration:
            plane = estimate_wall_x_from_p0_boundary_lines(
                plane, plane_segments
            )
            if plane["axis_estimation"] == "camera-up-projection":
                plane = estimate_wall_x_from_secondary_plane_shape(
                    plane, plane_xyz, plane_labels, segmented_planes
                )
                if plane["axis_estimation"] == "camera-up-projection":
                    plane = estimate_wall_x_from_plane_intersections(
                        plane, segmented_planes
                    )
            plane["automatic_segmented_plane_count"] = len(segmented_planes)
        plane["plane_analysis_point_count"] = int(plane_xyz.shape[0])
        plane["plane_analysis_source_point_count"] = int(cloud.shape[0])
        plane["plane_analysis_downsampled"] = bool(
            plane_xyz.shape[0] < cloud.shape[0]
        )
        plane["plane_analysis_skipped"] = False
    else:
        assert calibration is not None
        source = calibration.get("source") or {}
        plane = apply_wall_calibration(
            {
                "origin_camera_m": calibration["origin_camera_m"],
                "center_camera_m": calibration.get(
                    "center_camera_m", calibration["origin_camera_m"]
                ),
                "normal_camera": calibration["normal_camera"],
                "x_axis_camera": calibration["x_axis_camera"],
                "y_axis_camera": calibration["y_axis_camera"],
                "z_axis_camera": calibration["z_axis_camera"],
                "coordinate_system": calibration["coordinate_system"],
                "origin_definition": calibration["origin_definition"],
                "threshold_m": float(
                    calibration.get("plane_threshold_m", plane_threshold_m)
                ),
                "inlier_count": int(
                    calibration.get("plane_inlier_count", 0)
                ),
                "sample_count": int(cloud.shape[0]),
                "inlier_ratio": float(
                    calibration.get(
                        "plane_inlier_ratio",
                        source.get("plane_inlier_ratio", 0.0),
                    )
                ),
                "rms_m": float(
                    calibration.get(
                        "plane_rms_m", source.get("plane_rms_m", 0.0)
                    )
                ),
            },
            calibration,
        )
        plane["plane_analysis_skipped"] = True
        plane_segments = []
    clusters = semantic_clusters(
        cloud["xyz"], metadata, boxes or [], plane
    )
    result = {
        "session_id": session_id,
        "frame_id": frame_id,
        "point_count": int(cloud.shape[0]),
        "plane": plane,
        "plane_segments": plane_segments,
        "yolo": {
            "available": boxes is not None,
            "boxes": boxes or [],
            "clusters": clusters,
        },
        "source": metadata,
    }
    if include_yolo_panel_fit:
        assert pixel_coordinates is not None
        result["yolo_panel_fit"] = analyze_yolo_mask_panel(
            cloud["xyz"],
            pixel_coordinates,
            boxes or [],
            metadata.get("image_shape"),
        )
    if include_highest_confidence_semantic_cloud:
        result["_highest_confidence_semantic_cloud"] = (
            highest_confidence_semantic_pointcloud(
                cloud["xyz"],
                cloud["rgba"],
                metadata,
                boxes or [],
                plane,
            )
        )
    return result


def semantic_clusters(
    points_xyz: np.ndarray,
    metadata: dict[str, Any],
    boxes: list[dict[str, Any]],
    plane: dict[str, Any],
) -> list[dict[str, Any]]:
    if not boxes:
        return []
    points = np.asarray(points_xyz, dtype=np.float64)
    u, v, protrusion, image_shape = _semantic_projection(
        points, metadata, plane
    )
    clusters: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = [float(value) for value in box["xyxy"]]
        except (KeyError, TypeError, ValueError):
            continue
        object_mask, foreground_only = _semantic_object_mask(
            u, v, protrusion, box, image_shape
        )
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
                "foreground_only": foreground_only,
                "centroid_camera_m": centroid.tolist(),
                "centroid_plane_coordinates_m": target_plane_coordinates(
                    centroid.tolist(), plane
                ),
            }
        )
    return clusters


def highest_confidence_semantic_pointcloud(
    points_xyz: np.ndarray,
    point_rgba: np.ndarray,
    metadata: dict[str, Any],
    boxes: list[dict[str, Any]],
    plane: dict[str, Any],
) -> dict[str, Any] | None:
    if not boxes:
        return None
    valid_boxes = [
        (index, box)
        for index, box in enumerate(boxes)
        if np.isfinite(float(box.get("conf", 0.0)))
    ]
    if not valid_boxes:
        return None
    box_index, box = max(
        valid_boxes, key=lambda item: float(item[1].get("conf", 0.0))
    )
    try:
        xyxy = [float(value) for value in box["xyxy"]]
    except (KeyError, TypeError, ValueError):
        return None

    points = np.asarray(points_xyz, dtype=np.float64)
    rgba = np.asarray(point_rgba, dtype=np.uint8)
    if rgba.shape != (points.shape[0], 4):
        raise ValueError("点云颜色必须是 N×4 RGBA 数组")
    u, v, protrusion, image_shape = _semantic_projection(
        points, metadata, plane
    )
    object_mask, foreground_only = _semantic_object_mask(
        u, v, protrusion, box, image_shape
    )
    selected_points = points[object_mask]
    if selected_points.shape[0] < 3:
        return None

    origin = np.asarray(plane["origin_camera_m"], dtype=np.float64)
    wall_x, wall_y, wall_z = _wall_axes(plane)
    axes = np.asarray((wall_x, wall_y, wall_z), dtype=np.float64)
    wall_points = (selected_points - origin) @ axes.T
    centroid_camera = np.median(selected_points, axis=0)
    centroid_wall = np.median(wall_points, axis=0)
    return {
        "box_index": box_index,
        "detection": {
            "cls": int(box.get("cls", -1)),
            "name": str(box.get("name", box.get("cls", "unknown"))),
            "conf": float(box.get("conf", 0.0)),
            "xyxy": xyxy,
            "polygon": box.get("polygon"),
        },
        "foreground_only": foreground_only,
        "xyz_camera_m": selected_points.astype(np.float32, copy=False),
        "xyz_wall_m": wall_points.astype(np.float32, copy=False),
        "rgb": rgba[object_mask, :3].copy(),
        "centroid_camera_m": centroid_camera.tolist(),
        "centroid_wall_m": centroid_wall.tolist(),
        "coordinate_origin_camera_m": origin.tolist(),
        "coordinate_axes_camera": axes.tolist(),
    }


def reproject_semantic_pointcloud(
    semantic_cloud: dict[str, Any],
    plane: dict[str, Any],
) -> dict[str, Any]:
    xyz_camera = np.asarray(
        semantic_cloud["xyz_camera_m"], dtype=np.float32
    )
    origin = np.asarray(plane["origin_camera_m"], dtype=np.float64)
    wall_x, wall_y, wall_z = _wall_axes(plane)
    axes = np.asarray((wall_x, wall_y, wall_z), dtype=np.float64)
    if (
        xyz_camera.ndim != 2
        or xyz_camera.shape[1] != 3
        or origin.shape != (3,)
        or axes.shape != (3, 3)
        or not np.isfinite(origin).all()
        or not np.isfinite(axes).all()
    ):
        raise ValueError("无法使用当前坐标系转换 YOLO 点云")
    xyz_wall = (xyz_camera - origin) @ axes.T
    projected = dict(semantic_cloud)
    projected["xyz_wall_m"] = xyz_wall.astype(np.float32, copy=False)
    projected["centroid_wall_m"] = np.median(xyz_wall, axis=0).tolist()
    projected["coordinate_origin_camera_m"] = origin.tolist()
    projected["coordinate_axes_camera"] = axes.tolist()
    return projected


def _semantic_projection(
    points: np.ndarray,
    metadata: dict[str, Any],
    plane: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int] | None]:
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
    return u, v, protrusion, image_shape


def _semantic_object_mask(
    u: np.ndarray,
    v: np.ndarray,
    protrusion: np.ndarray,
    box: dict[str, Any],
    image_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, bool]:
    inside = detection_pixel_mask(u, v, box, image_shape=image_shape)
    foreground = inside & (protrusion >= 0.003) & (protrusion <= 0.15)
    foreground_only = int(foreground.sum()) >= 12
    return (foreground if foreground_only else inside), foreground_only


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

    calibration = {
        "schema": "rgbd-wall-coordinate-calibration/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_method": "manual-two-point",
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
    for source_key, calibration_key in (
        ("center_camera_m", "center_camera_m"),
        ("threshold_m", "plane_threshold_m"),
        ("inlier_count", "plane_inlier_count"),
        ("inlier_ratio", "plane_inlier_ratio"),
        ("rms_m", "plane_rms_m"),
    ):
        if source_key in plane:
            calibration[calibration_key] = plane[source_key]
    return calibration


def build_accepted_wall_calibration(
    plane: dict[str, Any],
) -> dict[str, Any]:
    origin = np.asarray(plane.get("origin_camera_m"), dtype=np.float64)
    wall_x, wall_y, wall_z = _wall_axes(plane)
    axes = np.asarray((wall_x, wall_y, wall_z), dtype=np.float64)
    if (
        origin.shape != (3,)
        or axes.shape != (3, 3)
        or not np.isfinite(origin).all()
        or not np.isfinite(axes).all()
    ):
        raise ValueError("当前坐标系包含无效数值，无法保存")
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-4) or float(
        np.linalg.det(axes)
    ) < 0.999:
        raise ValueError("当前坐标轴不是有效的右手正交坐标系")
    calibration = {
        "schema": "rgbd-wall-coordinate-calibration/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_method": "accepted-automatic",
        "accepted_axis_estimation": str(
            plane.get("axis_estimation", "unknown")
        ),
        "origin_camera_m": origin.tolist(),
        "x_axis_camera": wall_x.tolist(),
        "y_axis_camera": wall_y.tolist(),
        "z_axis_camera": wall_z.tolist(),
        "normal_camera": wall_y.tolist(),
        "coordinate_system": "wall-right-handed-x-right-y-inward-z-up",
        "origin_definition": str(
            plane.get(
                "origin_definition",
                "camera-origin-projection-on-wall",
            )
        ),
    }
    for key in (
        "axis_reference_plane_index",
        "axis_reference_boundary_index",
        "axis_reference_line_length_m",
        "axis_reference_line_fit_method",
        "axis_reference_group_size",
        "axis_reference_group_line_count",
        "axis_reference_group_total_length_m",
        "axis_reference_group_angle_tolerance_deg",
        "axis_reference_camera_up_angle_deg",
        "axis_reference_min_camera_up_angle_deg",
    ):
        if key in plane:
            calibration[key] = plane[key]
    for source_key, calibration_key in (
        ("center_camera_m", "center_camera_m"),
        ("threshold_m", "plane_threshold_m"),
        ("inlier_count", "plane_inlier_count"),
        ("inlier_ratio", "plane_inlier_ratio"),
        ("rms_m", "plane_rms_m"),
    ):
        if source_key in plane:
            calibration[calibration_key] = plane[source_key]
    return calibration


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
    calibration_method = str(
        calibration.get("calibration_method", "manual-two-point")
    )
    calibrated["calibration_method"] = calibration_method
    calibrated["axis_estimation"] = (
        "saved-accepted-coordinate"
        if calibration_method == "accepted-automatic"
        else "manual-two-point-calibration"
    )
    calibrated["accepted_axis_estimation"] = calibration.get(
        "accepted_axis_estimation"
    )
    calibrated["calibration_created_at"] = calibration["created_at"]
    calibrated["calibration_baseline_m"] = calibration.get(
        "x_baseline_m", calibration.get("z_baseline_m")
    )
    calibrated["calibration_source"] = calibration.get("source")
    for key in (
        "axis_reference_plane_index",
        "axis_reference_boundary_index",
        "axis_reference_line_length_m",
        "axis_reference_line_fit_method",
        "axis_reference_group_size",
        "axis_reference_group_line_count",
        "axis_reference_group_total_length_m",
        "axis_reference_group_angle_tolerance_deg",
        "axis_reference_camera_up_angle_deg",
        "axis_reference_min_camera_up_angle_deg",
    ):
        if key in calibration:
            calibrated[key] = calibration[key]
    return calibrated


def save_highest_confidence_semantic_pointcloud(
    data_root: Path,
    session_id: str,
    frame_id: str,
    semantic_cloud: dict[str, Any],
    *,
    target_camera_m: list[float] | None = None,
    point_slot: int = 1,
) -> dict[str, Any]:
    root = data_root.expanduser().resolve()
    _, frame_dir = resolve_frame_paths(root, session_id, frame_id)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"帧不存在: {frame_id}")

    xyz_camera = np.asarray(
        semantic_cloud["xyz_camera_m"], dtype=np.float32
    )
    xyz_wall = np.asarray(semantic_cloud["xyz_wall_m"], dtype=np.float32)
    rgb = np.asarray(semantic_cloud["rgb"], dtype=np.uint8)
    if (
        xyz_camera.ndim != 2
        or xyz_camera.shape[1] != 3
        or xyz_wall.shape != xyz_camera.shape
        or rgb.shape != xyz_camera.shape
        or xyz_camera.shape[0] < 3
    ):
        raise ValueError("YOLO 语义点云数组无效")
    if not 1 <= point_slot <= 9:
        raise ValueError("point_slot 必须在 1~9")
    target: np.ndarray | None = None
    if target_camera_m is not None:
        target = np.asarray(target_camera_m, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("目标点必须是三个有限数值")

    output_dir = root / "yolo_semantic_pointclouds" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{frame_id}.npz"
    metadata_path = output_dir / f"{frame_id}.json"
    saved_targets: dict[str, Any] = {}
    existing_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            existing_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if isinstance(existing_metadata.get("targets"), dict):
                saved_targets.update(existing_metadata["targets"])
        except (OSError, ValueError, json.JSONDecodeError):
            saved_targets = {}
            existing_metadata = {}
    point_data_unchanged = (
        data_path.is_file()
        and existing_metadata.get("detection") == semantic_cloud["detection"]
        and int(existing_metadata.get("point_count", -1))
        == int(xyz_camera.shape[0])
        and np.allclose(
            np.asarray(
                existing_metadata.get("coordinate_origin_camera_m"),
                dtype=np.float64,
            ),
            np.asarray(
                semantic_cloud["coordinate_origin_camera_m"],
                dtype=np.float64,
            ),
            atol=1e-6,
        )
        and np.allclose(
            np.asarray(
                existing_metadata.get("coordinate_axes_camera"),
                dtype=np.float64,
            ),
            np.asarray(
                semantic_cloud["coordinate_axes_camera"],
                dtype=np.float64,
            ),
            atol=1e-6,
        )
    )
    if not point_data_unchanged:
        fd, temporary_data = tempfile.mkstemp(
            prefix=".yolo-pointcloud-", suffix=".npz", dir=output_dir
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(
                    handle,
                    xyz_camera_m=xyz_camera,
                    xyz_wall_m=xyz_wall,
                    rgb=rgb,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_data, data_path)
        finally:
            if os.path.exists(temporary_data):
                os.unlink(temporary_data)

    summary: dict[str, Any] = {
        "schema": "rgbd-yolo-semantic-pointcloud/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "frame_id": frame_id,
        "selection": "highest-confidence-only",
        "box_index": int(semantic_cloud["box_index"]),
        "detection": semantic_cloud["detection"],
        "foreground_only": bool(semantic_cloud["foreground_only"]),
        "point_count": int(xyz_camera.shape[0]),
        "centroid_camera_m": semantic_cloud["centroid_camera_m"],
        "centroid_wall_m": semantic_cloud["centroid_wall_m"],
        "coordinate_origin_camera_m": semantic_cloud[
            "coordinate_origin_camera_m"
        ],
        "coordinate_axes_camera": semantic_cloud[
            "coordinate_axes_camera"
        ],
        "data_file": str(data_path.relative_to(root)),
        "arrays": {
            "xyz_camera_m": {"shape": list(xyz_camera.shape), "dtype": "float32"},
            "xyz_wall_m": {"shape": list(xyz_wall.shape), "dtype": "float32"},
            "rgb": {"shape": list(rgb.shape), "dtype": "uint8"},
        },
    }
    origin = np.asarray(
        semantic_cloud["coordinate_origin_camera_m"], dtype=np.float64
    )
    axes = np.asarray(
        semantic_cloud["coordinate_axes_camera"], dtype=np.float64
    )
    centroid_wall = np.asarray(
        semantic_cloud["centroid_wall_m"], dtype=np.float64
    )
    for saved_slot, saved_target in list(saved_targets.items()):
        saved_camera = np.asarray(
            saved_target.get("target_camera_m"), dtype=np.float64
        )
        if saved_camera.shape != (3,) or not np.isfinite(saved_camera).all():
            saved_targets.pop(saved_slot)
            continue
        saved_wall = (saved_camera - origin) @ axes.T
        saved_target["target_wall_m"] = saved_wall.tolist()
        saved_target["target_minus_semantic_centroid_wall_m"] = (
            saved_wall - centroid_wall
        ).tolist()
    if target is not None:
        target_wall = (target - origin) @ axes.T
        summary["target_camera_m"] = target.tolist()
        summary["target_wall_m"] = target_wall.tolist()
        summary["target_minus_semantic_centroid_wall_m"] = (
            target_wall - centroid_wall
        ).tolist()
        saved_targets[str(point_slot)] = {
            "point_slot": point_slot,
            "target_camera_m": target.tolist(),
            "target_wall_m": target_wall.tolist(),
            "target_minus_semantic_centroid_wall_m": (
                target_wall - centroid_wall
            ).tolist(),
        }
    summary["active_point_slot"] = point_slot
    summary["targets"] = saved_targets

    payload = json.dumps(
        summary, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    fd, temporary_metadata = tempfile.mkstemp(
        prefix=".yolo-pointcloud-", suffix=".json", dir=output_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_metadata, metadata_path)
    finally:
        if os.path.exists(temporary_metadata):
            os.unlink(temporary_metadata)
    return summary


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
    point_slot: int = 1,
    target_finder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _, frame_dir = resolve_frame_paths(data_root, session_id, frame_id)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"帧不存在: {frame_id}")
    target = np.asarray(target_camera_m, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("目标点必须是三个有限数值")
    if target[2] <= 0:
        raise ValueError("目标点深度必须大于零")
    if (
        selection_source not in {"pointcloud", "rgb"}
        and not selection_source.startswith("target-finder/")
    ):
        raise ValueError(
            "selection_source 必须是 pointcloud、rgb 或 target-finder/<版本>"
        )
    if not 1 <= point_slot <= 9:
        raise ValueError("point_slot 必须在 1~9")
    adjustment = np.asarray(
        target_adjustment_camera_m or [0.0, 0.0, 0.0], dtype=np.float64
    )
    if adjustment.shape != (3,) or not np.isfinite(adjustment).all():
        raise ValueError("目标微调量必须是三个有限数值")

    updated_at = datetime.now(timezone.utc).isoformat()
    point_record: dict[str, Any] = {
        "point_slot": point_slot,
        "updated_at": updated_at,
        "target_camera_m": target.tolist(),
        "target_adjustment_camera_m": adjustment.tolist(),
        "target_adjustment_wall_m": _vector_wall_coordinates(adjustment, plane),
        "selection_source": selection_source,
        "target_plane_coordinates_m": target_plane_coordinates(
            target.tolist(), plane
        ),
    }
    if target_pixel is not None:
        point_record["target_pixel"] = {
            "u": int(target_pixel["u"]),
            "v": int(target_pixel["v"]),
        }
    if target_finder is not None:
        if not isinstance(target_finder, dict):
            raise ValueError("target_finder 必须是对象")
        point_record["target_finder"] = target_finder
    yolo_record = yolo or {"available": False, "boxes": []}
    clusters = yolo_record.get("clusters", [])
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
        point_record["semantic_reference"] = reference
        point_record["target_relative_to_semantic_m"] = (
            _vector_wall_coordinates(delta, plane)
        )

    path = _annotation_path(data_root, session_id)
    records = load_annotations(data_root, session_id)
    existing = next(
        (item for item in records if item.get("frame_id") == frame_id),
        None,
    )
    points: dict[str, Any] = {}
    if existing is not None:
        existing_points = existing.get("points")
        if isinstance(existing_points, dict):
            points.update(existing_points)
        elif existing.get("target_camera_m") is not None:
            legacy_keys = (
                "updated_at",
                "target_camera_m",
                "target_adjustment_camera_m",
                "target_adjustment_wall_m",
                "selection_source",
                "target_plane_coordinates_m",
                "target_pixel",
                "semantic_reference",
                "target_relative_to_semantic_m",
            )
            legacy_point = {
                key: existing[key] for key in legacy_keys if key in existing
            }
            legacy_point["point_slot"] = 1
            points["1"] = legacy_point
    points[str(point_slot)] = point_record

    record = {
        "schema": "rgbd-target-annotation/v3",
        "session_id": session_id,
        "frame_id": frame_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "active_point_slot": point_slot,
        "points": points,
        "plane": plane,
        "yolo": yolo_record,
    }
    record.update(point_record)
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
