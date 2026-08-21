from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from .analysis import (
    analyze_frame,
    apply_wall_calibration,
    build_accepted_wall_calibration,
    build_wall_calibration,
    load_annotations,
    load_wall_calibration,
    save_annotation,
    save_wall_calibration,
    segment_dominant_planes,
    split_plane_labels_by_connectivity,
)
from .offline_yolo import OfflineYolo
from .pointcloud import (
    apply_plane_segment_colors,
    encode_point_cloud,
    frame_summaries,
    pixel_to_point,
    reconstruct_frame,
    resolve_frame_paths,
    session_summaries,
)


def create_pointcloud_app(
    data_root: Path,
    web_root: Path,
    *,
    yolo: OfflineYolo | None = None,
) -> FastAPI:
    root = data_root.expanduser().resolve()
    detector = yolo or OfflineYolo()
    app = FastAPI(title="Captured RGB-D Point Cloud Viewer")

    @app.get("/")
    def index():
        return FileResponse(web_root / "pointcloud.html")

    @app.get("/api/status")
    def status():
        return {
            "ok": True,
            "data_root": str(root),
            "sessions": len(session_summaries(root)),
            "yolo": detector.status(),
        }

    @app.get("/api/sessions")
    def sessions():
        return {"ok": True, "sessions": session_summaries(root)}

    @app.get("/api/sessions/{session_id}/frames")
    def frames(session_id: str):
        try:
            items = frame_summaries(root, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "session_id": session_id, "frames": items}

    @app.get("/api/rgb/{session_id}/{frame_id}")
    def rgb_frame(session_id: str, frame_id: str):
        try:
            _, frame_dir = resolve_frame_paths(root, session_id, frame_id)
            color_path = frame_dir / "color.jpg"
            if not color_path.is_file():
                raise FileNotFoundError("RGB 图不存在")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(
            color_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/pixel-to-point/{session_id}/{frame_id}")
    def project_pixel(
        session_id: str,
        frame_id: str,
        body: dict,
    ):
        try:
            result = pixel_to_point(
                root,
                session_id,
                frame_id,
                u=int(body["u"]),
                v=int(body["v"]),
                search_radius=int(body.get("search_radius", 6)),
                min_depth_m=float(body.get("min_depth_m", 0.1)),
                max_depth_m=float(body.get("max_depth_m", 5.0)),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=422, detail=f"缺少字段: {exc.args[0]}"
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, **result}

    @app.get("/api/pointcloud/{session_id}/{frame_id}")
    def pointcloud(
        session_id: str,
        frame_id: str,
        stride: int = Query(default=1, ge=1, le=64),
        min_depth_m: float = Query(default=0.1, ge=0.0, le=99.0),
        max_depth_m: float = Query(default=5.0, gt=0.0, le=100.0),
        max_points: int = Query(default=1_000_000, ge=1_000, le=1_000_000),
        semantic: bool = Query(default=False),
        planes: bool = Query(default=False),
        plane_threshold_m: float = Query(default=0.008, ge=0.001, le=0.05),
        min_plane_points: int = Query(default=300, ge=3, le=200_000),
    ):
        try:
            boxes = None
            if semantic and not planes:
                boxes = detector.infer_frame(root, session_id, frame_id)
            points, metadata = reconstruct_frame(
                root,
                session_id,
                frame_id,
                stride=stride,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                max_points=max_points,
                boxes=boxes,
                include_pixels=planes,
            )
            segmented_planes: list[dict] = []
            if planes:
                pixel_coordinates = metadata.pop("_pixel_coordinates")
                labels, segmented_planes = segment_dominant_planes(
                    points["xyz"], threshold_m=plane_threshold_m
                )
                labels, segmented_planes = split_plane_labels_by_connectivity(
                    points["xyz"],
                    labels,
                    segmented_planes,
                    pixel_coordinates,
                    metadata["image_shape"],
                    stride=stride,
                    source_point_count=int(metadata["source_point_count"]),
                    min_component_count=min_plane_points,
                    min_component_ratio=0.0,
                    preserve_farthest_plane=True,
                    max_planar_point_distance_from_farthest_plane_m=0.010,
                )
                point_counts = apply_plane_segment_colors(points, labels)
                metadata["plane_segmentation"] = {
                    "enabled": True,
                    "planes": segmented_planes,
                    "point_counts": point_counts,
                }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        bounds = metadata["bounds"]
        return Response(
            encode_point_cloud(points),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Point-Count": str(metadata["point_count"]),
                "X-Cloud-Center": ",".join(str(v) for v in bounds["center"]),
                "X-Cloud-Radius": str(bounds["radius"]),
                "X-Cloud-Stride": str(metadata["stride"]),
                "X-Semantic-Enabled": str(
                    metadata["semantic"]["enabled"]
                ).lower(),
                "X-Plane-Segmentation-Enabled": str(planes).lower(),
                "X-Plane-Count": str(len(segmented_planes)),
            },
        )

    @app.post("/api/analyze/{session_id}/{frame_id}")
    def analyze(
        session_id: str,
        frame_id: str,
        body: dict | None = None,
    ):
        options = body or {}
        try:
            boxes = (
                detector.infer_frame(root, session_id, frame_id)
                if detector.enabled
                else None
            )
            result = analyze_frame(
                root,
                session_id,
                frame_id,
                boxes=boxes,
                plane_threshold_m=float(
                    options.get("plane_threshold_m", 0.008)
                ),
                min_depth_m=float(options.get("min_depth_m", 0.15)),
                max_depth_m=float(options.get("max_depth_m", 3.0)),
                stride=int(options.get("stride", 3)),
                max_points=int(options.get("max_points", 1_000_000)),
                min_plane_points=int(options.get("min_plane_points", 300)),
                include_plane_debug=bool(
                    options.get("include_plane_debug", False)
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "analysis": result}

    @app.get("/api/sessions/{session_id}/annotations")
    def annotations(session_id: str):
        try:
            records = load_annotations(root, session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "annotations": records}

    @app.get("/api/wall-calibration/{session_id}/{frame_id}")
    def wall_calibration(session_id: str, frame_id: str):
        try:
            calibration = load_wall_calibration(root, session_id, frame_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "calibration": calibration}

    @app.post("/api/wall-calibration/{session_id}/{frame_id}")
    def calibrate_wall(
        session_id: str,
        frame_id: str,
        body: dict,
    ):
        try:
            analysis = analyze_frame(
                root,
                session_id,
                frame_id,
                plane_threshold_m=float(
                    body.get("plane_threshold_m", 0.008)
                ),
                min_depth_m=float(body.get("min_depth_m", 0.15)),
                max_depth_m=float(body.get("max_depth_m", 3.0)),
                stride=int(body.get("stride", 3)),
                max_points=int(body.get("max_points", 1_000_000)),
                min_plane_points=int(body.get("min_plane_points", 300)),
                use_saved_wall_calibration=False,
            )
            calibration = build_wall_calibration(
                analysis["plane"],
                [float(value) for value in body["origin_point_camera_m"]],
                [float(value) for value in body["x_point_camera_m"]],
            )
            calibration["source"] = {
                "session_id": session_id,
                "frame_id": frame_id,
                "plane_inlier_ratio": analysis["plane"]["inlier_ratio"],
                "plane_rms_m": analysis["plane"]["rms_m"],
            }
            path = save_wall_calibration(
                root, session_id, frame_id, calibration
            )
            plane = apply_wall_calibration(analysis["plane"], calibration)
        except KeyError as exc:
            raise HTTPException(
                status_code=422, detail=f"缺少字段: {exc.args[0]}"
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (
            TypeError,
            ValueError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "calibration": calibration,
            "plane": plane,
            "path": str(path),
        }

    @app.post("/api/wall-calibration/{session_id}/{frame_id}/accept")
    def accept_wall_calibration(
        session_id: str,
        frame_id: str,
        body: dict,
    ):
        try:
            current_plane = dict(body["plane"])
            calibration = build_accepted_wall_calibration(current_plane)
            calibration["source"] = {
                "session_id": session_id,
                "frame_id": frame_id,
                "accepted_from": calibration["accepted_axis_estimation"],
            }
            path = save_wall_calibration(
                root, session_id, frame_id, calibration
            )
            plane = apply_wall_calibration(current_plane, calibration)
        except KeyError as exc:
            raise HTTPException(
                status_code=422, detail=f"缺少字段: {exc.args[0]}"
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "calibration": calibration,
            "plane": plane,
            "path": str(path),
        }

    @app.post("/api/annotations/{session_id}/{frame_id}")
    def annotate(
        session_id: str,
        frame_id: str,
        body: dict,
    ):
        try:
            target = [float(value) for value in body["target_camera_m"]]
            boxes = (
                detector.infer_frame(root, session_id, frame_id)
                if detector.enabled
                else None
            )
            analysis = analyze_frame(
                root,
                session_id,
                frame_id,
                boxes=boxes,
                plane_threshold_m=float(
                    body.get("plane_threshold_m", 0.008)
                ),
                min_depth_m=float(body.get("min_depth_m", 0.15)),
                max_depth_m=float(body.get("max_depth_m", 3.0)),
                stride=int(body.get("stride", 3)),
                max_points=int(body.get("max_points", 1_000_000)),
                min_plane_points=int(body.get("min_plane_points", 300)),
            )
            record = save_annotation(
                root,
                session_id,
                frame_id,
                target_camera_m=target,
                plane=analysis["plane"],
                yolo=analysis["yolo"],
                target_pixel=body.get("target_pixel"),
                selection_source=str(
                    body.get("selection_source", "pointcloud")
                ),
                target_adjustment_camera_m=body.get(
                    "target_adjustment_camera_m"
                ),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=422, detail=f"缺少字段: {exc.args[0]}"
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (
            TypeError,
            ValueError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "annotation": record}

    return app
