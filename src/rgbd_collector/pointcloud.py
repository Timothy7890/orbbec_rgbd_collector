from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POINT_DTYPE = np.dtype(
    [("xyz", "<f4", (3,)), ("rgba", "u1", (4,))], align=False
)
SEMANTIC_PALETTE = np.array(
    [
        [239, 83, 80],
        [66, 165, 245],
        [102, 187, 106],
        [255, 202, 40],
        [171, 71, 188],
        [255, 112, 67],
        [38, 198, 218],
        [141, 110, 99],
        [236, 64, 122],
        [124, 179, 66],
        [126, 87, 194],
        [255, 167, 38],
    ],
    dtype=np.uint8,
)


def _safe_child(parent: Path, name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("非法目录名称")
    parent = parent.resolve()
    child = (parent / name).resolve()
    if child.parent != parent:
        raise ValueError("目录越界")
    return child


def resolve_frame_paths(
    data_root: Path, session_id: str, frame_id: str
) -> tuple[Path, Path]:
    root = data_root.expanduser().resolve()
    session_dir = _safe_child(root, session_id)
    frames_dir = (session_dir / "frames").resolve()
    frame_dir = _safe_child(frames_dir, frame_id)
    return session_dir, frame_dir


def session_summaries(data_root: Path) -> list[dict[str, Any]]:
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        return []
    output = []
    for session_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()), reverse=True
    ):
        session_file = session_dir / "session.json"
        if not session_file.is_file():
            continue
        try:
            session = json.loads(session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        frames = frame_summaries(root, session_dir.name)
        output.append(
            {
                "id": session_dir.name,
                "name": session.get("session_name", session_dir.name),
                "created_at": session.get("created_at"),
                "frame_count": len(frames),
                "device": session.get("camera", {}).get("device"),
            }
        )
    return output


def frame_summaries(data_root: Path, session_id: str) -> list[dict[str, Any]]:
    session_dir = _safe_child(data_root.expanduser().resolve(), session_id)
    manifest = session_dir / "manifest.jsonl"
    if not manifest.is_file():
        return []
    output = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        output.append(
            {
                "id": record.get("frame_id"),
                "sequence": record.get("sequence"),
                "saved_at": record.get("saved_at"),
                "trigger": record.get("trigger"),
                "valid_ratio": record.get("depth_aligned", {}).get(
                    "valid_ratio"
                ),
            }
        )
    return output


def reconstruct_frame(
    data_root: Path,
    session_id: str,
    frame_id: str,
    *,
    stride: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    max_points: int = 200_000,
    boxes: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 1 <= stride <= 64:
        raise ValueError("stride 必须在 1~64")
    if not 0 <= min_depth_m < max_depth_m <= 100:
        raise ValueError("深度范围非法")
    if not 1_000 <= max_points <= 1_000_000:
        raise ValueError("max_points 必须在 1000~1000000")

    session_dir, frame_dir = resolve_frame_paths(
        data_root, session_id, frame_id
    )
    session_file = session_dir / "session.json"
    frame_file = frame_dir / "frame.json"
    if not session_file.is_file() or not frame_file.is_file():
        raise FileNotFoundError("会话或帧元数据不存在")

    session = json.loads(session_file.read_text(encoding="utf-8"))
    frame_meta = json.loads(frame_file.read_text(encoding="utf-8"))
    color = cv2.imread(str(frame_dir / "color.jpg"), cv2.IMREAD_COLOR)
    depth = cv2.imread(
        str(frame_dir / "depth_aligned.png"), cv2.IMREAD_UNCHANGED
    )
    if color is None or depth is None:
        raise FileNotFoundError("RGB 或对齐深度文件不存在/无法解码")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"对齐深度必须是 uint16 单通道，实际 {depth.dtype}/{depth.shape}")
    if depth.shape != color.shape[:2]:
        raise ValueError(
            f"RGB/对齐深度尺寸不一致: {color.shape[:2]} vs {depth.shape}"
        )

    intrinsics = (
        session.get("camera", {}).get("color", {}).get("intrinsics", {})
    )
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("session.json 缺少有效彩色相机内参") from exc
    if fx <= 0 or fy <= 0:
        raise ValueError("相机焦距必须为正数")
    try:
        scale_mm = float(frame_meta["depth_scale"]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frame.json 缺少有效 depth_scale") from exc

    rows = np.arange(0, depth.shape[0], stride, dtype=np.int32)
    cols = np.arange(0, depth.shape[1], stride, dtype=np.int32)
    uu, vv = np.meshgrid(cols, rows)
    sampled_depth = depth[vv, uu]
    z = sampled_depth.astype(np.float32) * (scale_mm / 1000.0)
    valid = (
        (sampled_depth > 0)
        & np.isfinite(z)
        & (z >= min_depth_m)
        & (z <= max_depth_m)
    )
    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    z = z[valid]
    rgb = color[vv[valid], uu[valid], ::-1]

    if z.size > max_points:
        selected = np.linspace(
            0, z.size - 1, max_points, dtype=np.int64
        )
        u, v, z, rgb = u[selected], v[selected], z[selected], rgb[selected]
    semantic_counts: dict[str, int] = {}
    if boxes:
        semantic_rgb = np.full(rgb.shape, [54, 60, 72], dtype=np.uint8)
        class_ids = np.full(z.size, -1, dtype=np.int32)
        for box in sorted(boxes, key=lambda item: float(item.get("conf", 0.0))):
            try:
                cls = int(box["cls"])
                x1, y1, x2, y2 = [float(value) for value in box["xyxy"]]
            except (KeyError, TypeError, ValueError):
                continue
            inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            class_ids[inside] = cls
            semantic_rgb[inside] = SEMANTIC_PALETTE[
                cls % len(SEMANTIC_PALETTE)
            ]
        rgb = semantic_rgb
        for cls in np.unique(class_ids[class_ids >= 0]):
            semantic_counts[str(int(cls))] = int(np.count_nonzero(class_ids == cls))
    distortion = (
        session.get("camera", {})
        .get("color", {})
        .get("distortion", {})
        .get("coefficients", [])
    )
    try:
        coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        coefficients = np.zeros(0, dtype=np.float64)
    if coefficients.size in {4, 5, 8, 12, 14} and np.any(
        np.abs(coefficients) > 1e-12
    ):
        pixels = np.column_stack((u, v)).astype(np.float64).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(
            pixels,
            np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            coefficients,
        ).reshape(-1, 2)
        x = normalized[:, 0].astype(np.float32) * z
        y = normalized[:, 1].astype(np.float32) * z
    else:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

    points = np.empty(z.size, dtype=POINT_DTYPE)
    points["xyz"][:, 0] = x
    points["xyz"][:, 1] = y
    points["xyz"][:, 2] = z
    points["rgba"][:, :3] = rgb
    points["rgba"][:, 3] = 255

    if z.size:
        minimum = points["xyz"].min(axis=0)
        maximum = points["xyz"].max(axis=0)
        center = ((minimum + maximum) * 0.5).tolist()
        radius = float(np.linalg.norm(maximum - minimum) * 0.5)
    else:
        minimum = maximum = np.zeros(3, dtype=np.float32)
        center, radius = [0.0, 0.0, 0.0], 0.0
    metadata = {
        "session_id": session_id,
        "frame_id": frame_id,
        "point_count": int(points.size),
        "stride": stride,
        "depth_range_m": [min_depth_m, max_depth_m],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "color_distortion_compensated": bool(
            coefficients.size in {4, 5, 8, 12, 14}
            and np.any(np.abs(coefficients) > 1e-12)
        ),
        "semantic": {
            "enabled": bool(boxes),
            "class_point_counts": semantic_counts,
        },
        "bounds": {
            "min": minimum.tolist(),
            "max": maximum.tolist(),
            "center": center,
            "radius": radius,
        },
    }
    return points, metadata


def encode_point_cloud(points: np.ndarray) -> bytes:
    if points.dtype != POINT_DTYPE:
        raise ValueError("unexpected point dtype")
    count = int(points.size)
    return b"PCD1" + count.to_bytes(4, "little") + points.tobytes()


def suggested_stride(
    width: int, height: int, max_points: int = 200_000
) -> int:
    return max(1, int(math.ceil(math.sqrt(width * height / max_points))))
