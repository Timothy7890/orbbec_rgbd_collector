from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from .analysis import (
    analyze_frame,
    apply_wall_calibration,
    build_accepted_wall_calibration,
    build_wall_calibration,
    delete_frame_and_artifacts,
    measure_frame_camera_plane_pose,
    load_annotations,
    load_wall_calibration,
    reproject_semantic_pointcloud,
    save_annotation,
    save_camera_plane_pose_measurements,
    save_highest_confidence_semantic_pointcloud,
    save_wall_calibration,
    save_yolo_panel_center_measurements,
    segment_dominant_planes,
    split_plane_labels_by_connectivity,
    summarize_panel_center_target_relationships,
    target_plane_coordinates,
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
from .target_finder import predict_target_one, target_finder_models


def create_pointcloud_app(
    data_root: Path,
    web_root: Path,
    *,
    yolo: OfflineYolo | None = None,
) -> FastAPI:
    root = data_root.expanduser().resolve()
    detector = yolo or OfflineYolo()
    app = FastAPI(title="Captured RGB-D Point Cloud Viewer")
    analysis_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()

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

    @app.get("/api/target-finder/models")
    def target_finder_model_list():
        models = target_finder_models()
        return {
            "ok": True,
            "models": models,
            "default_version": models[-1]["version"],
        }

    @app.post("/api/camera-plane-pose/{session_id}/save-all")
    def save_all_camera_plane_poses(
        session_id: str,
        body: dict | None = None,
    ):
        requested = body or {}
        options = {
            "plane_threshold_m": float(
                requested.get("plane_threshold_m", 0.008)
            ),
            "min_depth_m": float(requested.get("min_depth_m", 0.1)),
            "max_depth_m": float(requested.get("max_depth_m", 5.0)),
            "stride": int(requested.get("camera_plane_stride", 3)),
            "max_points": int(
                requested.get("camera_plane_max_points", 60_000)
            ),
        }
        started = time.perf_counter()
        try:
            frames = frame_summaries(root, session_id)
            measurements: list[dict] = []
            for frame in frames:
                frame_id = str(frame["id"])
                try:
                    measurement = measure_frame_camera_plane_pose(
                        root,
                        session_id,
                        frame_id,
                        **options,
                    )
                    measurements.append({"ok": True, **measurement})
                except (
                    FileNotFoundError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as exc:
                    measurements.append(
                        {
                            "ok": False,
                            "session_id": session_id,
                            "frame_id": frame_id,
                            "error": str(exc),
                        }
                    )
            path, payload = save_camera_plane_pose_measurements(
                root,
                session_id,
                measurements,
                options=options,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "session_id": session_id,
            "path": str(path.relative_to(root)),
            "frame_count": payload["frame_count"],
            "success_count": payload["success_count"],
            "failure_count": payload["failure_count"],
            "elapsed_s": time.perf_counter() - started,
        }

    @app.post("/api/yolo-panel-center/{session_id}/save-all")
    def save_all_yolo_panel_centers(
        session_id: str,
        body: dict | None = None,
    ):
        requested = body or {}
        options = {
            "plane_threshold_m": float(
                requested.get("plane_threshold_m", 0.008)
            ),
            "min_depth_m": float(requested.get("min_depth_m", 0.1)),
            "max_depth_m": float(requested.get("max_depth_m", 5.0)),
            "stride": int(requested.get("stride", 1)),
            "max_points": int(requested.get("max_points", 1_000_000)),
            "plane_analysis_max_points": int(
                requested.get("plane_analysis_max_points", 200_000)
            ),
            "min_plane_points": int(
                requested.get("min_plane_points", 300)
            ),
        }
        started = time.perf_counter()
        try:
            frames = frame_summaries(root, session_id)
            measurements: list[dict] = []
            skipped_uncalibrated_count = 0
            for frame in frames:
                frame_id = str(frame["id"])
                try:
                    calibration = load_wall_calibration(
                        root, session_id, frame_id
                    )
                    if calibration is None:
                        skipped_uncalibrated_count += 1
                        continue
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
                        include_yolo_panel_fit=True,
                        **options,
                    )
                    panel = analysis["yolo_panel_fit"]
                    if not panel.get("available"):
                        measurements.append(
                            {
                                "ok": False,
                                "session_id": session_id,
                                "frame_id": frame_id,
                                "error": str(
                                    panel.get("reason")
                                    or "面板拟合不可用"
                                ),
                                "detection": panel.get("detection"),
                                "mask_point_count": panel.get(
                                    "mask_point_count"
                                ),
                            }
                        )
                        continue
                    center_camera = panel.get(
                        "rectangle_center_camera_m"
                    )
                    if center_camera is None:
                        corners = panel.get(
                            "rectangle_corners_camera_m"
                        )
                        if not isinstance(corners, list) or len(corners) != 4:
                            raise ValueError("面板矩形中心无效")
                        center_camera = [
                            sum(float(corner[axis]) for corner in corners) / 4
                            for axis in range(3)
                        ]
                    center_wall_named = target_plane_coordinates(
                        center_camera, analysis["plane"]
                    )
                    measurements.append(
                        {
                            "ok": True,
                            "session_id": session_id,
                            "frame_id": frame_id,
                            "rectangle_center_camera_m": center_camera,
                            "rectangle_center_wall_m": [
                                center_wall_named["x_m"],
                                center_wall_named["y_m"],
                                center_wall_named["z_m"],
                            ],
                            "rectangle_center_wall_named_m": (
                                center_wall_named
                            ),
                            "rectangle_corners_camera_m": panel[
                                "rectangle_corners_camera_m"
                            ],
                            "detection": panel.get("detection"),
                            "panel_normal_camera": panel.get(
                                "normal_camera"
                            ),
                            "long_axis_camera": panel.get(
                                "long_axis_camera"
                            ),
                            "short_axis_camera": panel.get(
                                "short_axis_camera"
                            ),
                            "long_length_m": panel.get("long_length_m"),
                            "short_length_m": panel.get("short_length_m"),
                            "panel_rms_m": panel.get("rms_m"),
                            "panel_inlier_count": panel.get(
                                "inlier_count"
                            ),
                            "panel_inlier_ratio": panel.get(
                                "inlier_ratio"
                            ),
                            "orientation_source": panel.get(
                                "orientation_source"
                            ),
                            "wall_coordinate_calibrated": bool(
                                analysis["plane"].get("calibrated")
                            ),
                        }
                    )
                except (
                    FileNotFoundError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as exc:
                    measurements.append(
                        {
                            "ok": False,
                            "session_id": session_id,
                            "frame_id": frame_id,
                            "error": str(exc),
                        }
                    )
            target_relationships = (
                summarize_panel_center_target_relationships(
                    measurements,
                    load_annotations(root, session_id),
                )
            )
            path, payload = save_yolo_panel_center_measurements(
                root,
                session_id,
                measurements,
                options=options,
                total_session_frame_count=len(frames),
                skipped_uncalibrated_count=(
                    skipped_uncalibrated_count
                ),
                target_relationships=target_relationships,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "session_id": session_id,
            "path": str(path.relative_to(root)),
            "frame_count": payload["frame_count"],
            "total_session_frame_count": payload[
                "total_session_frame_count"
            ],
            "skipped_uncalibrated_count": payload[
                "skipped_uncalibrated_count"
            ],
            "success_count": payload["success_count"],
            "failure_count": payload["failure_count"],
            "target_relationships": payload["target_relationships"][
                "models"
            ],
            "elapsed_s": time.perf_counter() - started,
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

    @app.delete("/api/frames/{session_id}/{frame_id}")
    def delete_frame(session_id: str, frame_id: str):
        try:
            result = delete_frame_and_artifacts(root, session_id, frame_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        analysis_cache.pop((session_id, frame_id), None)
        return {"ok": True, **result}

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
                plane_analysis_max_points=int(
                    options.get("plane_analysis_max_points", 200_000)
                ),
                min_plane_points=int(options.get("min_plane_points", 300)),
                include_plane_debug=bool(
                    options.get("include_plane_debug", False)
                ),
                include_highest_confidence_semantic_cloud=True,
                include_yolo_panel_fit=bool(
                    options.get("include_yolo_panel_fit", False)
                ),
            )
            semantic_cloud = result.pop(
                "_highest_confidence_semantic_cloud", None
            )
            cache_key = (session_id, frame_id)
            analysis_cache[cache_key] = {
                "semantic_cloud": semantic_cloud,
                "yolo": copy.deepcopy(result["yolo"]),
                "plane": copy.deepcopy(result["plane"]),
                "yolo_panel_fit": copy.deepcopy(
                    result.get("yolo_panel_fit")
                ),
            }
            analysis_cache.move_to_end(cache_key)
            while len(analysis_cache) > 32:
                analysis_cache.popitem(last=False)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "analysis": result}

    @app.post("/api/target-finder/{session_id}/{frame_id}")
    def find_target_one(
        session_id: str,
        frame_id: str,
        body: dict | None = None,
    ):
        options = body or {}
        version = str(options.get("version", "0.1.0"))
        try:
            model = next(
                (
                    item
                    for item in target_finder_models()
                    if item["version"] == version
                ),
                None,
            )
            if model is None:
                raise ValueError(f"未知找点算法版本: {version}")
            requires_panel = (
                model["algorithm"]
                == "yolo-panel-rectangle-center-plus-wall-offset"
            )
            cache_key = (session_id, frame_id)
            cached = analysis_cache.get(cache_key)
            submitted_plane = options.get("plane")
            semantic_cloud = (
                cached.get("semantic_cloud")
                if cached is not None
                else None
            )
            panel_fit = (
                cached.get("yolo_panel_fit")
                if cached is not None
                else None
            )
            analyzed_plane = (
                cached.get("plane") if cached is not None else None
            )
            if cached is not None:
                analysis_cache.move_to_end(cache_key)
            if cached is None or (
                requires_panel
                and not bool(panel_fit and panel_fit.get("available"))
            ):
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
                        options.get("plane_threshold_m", 0.008)
                    ),
                    min_depth_m=float(options.get("min_depth_m", 0.15)),
                    max_depth_m=float(options.get("max_depth_m", 3.0)),
                    stride=int(options.get("stride", 3)),
                    max_points=int(options.get("max_points", 1_000_000)),
                    plane_analysis_max_points=int(
                        options.get("plane_analysis_max_points", 200_000)
                    ),
                    min_plane_points=int(
                        options.get("min_plane_points", 300)
                    ),
                    include_highest_confidence_semantic_cloud=True,
                    include_yolo_panel_fit=requires_panel,
                )
                semantic_cloud = analysis.pop(
                    "_highest_confidence_semantic_cloud", None
                )
                panel_fit = analysis.get("yolo_panel_fit")
                analyzed_plane = analysis["plane"]
                analysis_cache[cache_key] = {
                    "semantic_cloud": semantic_cloud,
                    "yolo": copy.deepcopy(analysis["yolo"]),
                    "plane": copy.deepcopy(analyzed_plane),
                    "yolo_panel_fit": copy.deepcopy(panel_fit),
                }
                analysis_cache.move_to_end(cache_key)
                while len(analysis_cache) > 32:
                    analysis_cache.popitem(last=False)
            if (
                not isinstance(submitted_plane, dict)
                or (
                    requires_panel
                    and not submitted_plane.get("calibrated")
                )
            ):
                submitted_plane = analyzed_plane
            if semantic_cloud is None and not requires_panel:
                raise ValueError("当前帧没有可用于找点的 YOLO 语义点云")
            if semantic_cloud is not None and isinstance(
                submitted_plane, dict
            ):
                semantic_cloud = reproject_semantic_pointcloud(
                    semantic_cloud, submitted_plane
                )

            reference_targets: dict[str, list[float]] = {}
            for annotation in load_annotations(root, session_id):
                if annotation.get("frame_id") != frame_id:
                    continue
                points = annotation.get("points")
                if isinstance(points, dict):
                    for slot, point in points.items():
                        if (
                            isinstance(point, dict)
                            and isinstance(point.get("target_camera_m"), list)
                        ):
                            reference_targets[str(slot)] = point[
                                "target_camera_m"
                            ]
                elif annotation.get("target_camera_m") is not None:
                    reference_targets["1"] = annotation["target_camera_m"]
                break
            prediction = predict_target_one(
                semantic_cloud,
                version=version,
                reference_targets_camera_m=reference_targets,
                panel_fit=panel_fit,
                plane=(
                    submitted_plane
                    if isinstance(submitted_plane, dict)
                    else None
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "prediction": prediction,
            "panel_fit": panel_fit if requires_panel else None,
        }

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
                plane_analysis_max_points=int(
                    body.get("plane_analysis_max_points", 200_000)
                ),
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
            point_slot = int(body.get("point_slot", 1))
            cache_key = (session_id, frame_id)
            cached = analysis_cache.get(cache_key)
            submitted_plane = body.get("plane")
            if cached is not None and isinstance(submitted_plane, dict):
                analysis_cache.move_to_end(cache_key)
                analysis = {
                    "plane": dict(submitted_plane),
                    "yolo": copy.deepcopy(cached["yolo"]),
                }
                semantic_cloud = cached["semantic_cloud"]
                if semantic_cloud is not None:
                    semantic_cloud = reproject_semantic_pointcloud(
                        semantic_cloud, analysis["plane"]
                    )
            else:
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
                    plane_analysis_max_points=int(
                        body.get("plane_analysis_max_points", 200_000)
                    ),
                    min_plane_points=int(
                        body.get("min_plane_points", 300)
                    ),
                    include_highest_confidence_semantic_cloud=True,
                )
                semantic_cloud = analysis.pop(
                    "_highest_confidence_semantic_cloud", None
                )
            if semantic_cloud is not None:
                analysis["yolo"]["saved_pointcloud"] = (
                    save_highest_confidence_semantic_pointcloud(
                        root,
                        session_id,
                        frame_id,
                        semantic_cloud,
                        target_camera_m=target,
                        point_slot=point_slot,
                    )
                )
            else:
                analysis["yolo"]["saved_pointcloud"] = None
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
                point_slot=point_slot,
                target_finder=body.get("target_finder"),
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
