from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from .analysis import analyze_frame, load_annotations, save_annotation
from .offline_yolo import OfflineYolo
from .pointcloud import (
    encode_point_cloud,
    frame_summaries,
    reconstruct_frame,
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

    @app.get("/api/pointcloud/{session_id}/{frame_id}")
    def pointcloud(
        session_id: str,
        frame_id: str,
        stride: int = Query(default=2, ge=1, le=64),
        min_depth_m: float = Query(default=0.1, ge=0.0, le=99.0),
        max_depth_m: float = Query(default=5.0, gt=0.0, le=100.0),
        max_points: int = Query(default=200_000, ge=1_000, le=1_000_000),
        semantic: bool = Query(default=False),
    ):
        try:
            boxes = None
            if semantic:
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
            )
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
            )
            record = save_annotation(
                root,
                session_id,
                frame_id,
                target_camera_m=target,
                plane=analysis["plane"],
                yolo=analysis["yolo"],
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
