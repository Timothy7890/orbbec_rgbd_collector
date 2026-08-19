from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from .pointcloud import (
    encode_point_cloud,
    frame_summaries,
    reconstruct_frame,
    session_summaries,
)


def create_pointcloud_app(data_root: Path, web_root: Path) -> FastAPI:
    root = data_root.expanduser().resolve()
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
    ):
        try:
            points, metadata = reconstruct_frame(
                root,
                session_id,
                frame_id,
                stride=stride,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                max_points=max_points,
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
            },
        )

    return app
