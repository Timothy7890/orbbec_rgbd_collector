from __future__ import annotations

import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .camera import CameraConfig, OrbbecCamera
from .storage import CaptureQueueFull, DatasetSession, list_sessions


class CollectorService:
    def __init__(self, camera_config: CameraConfig, output_root: Path) -> None:
        self.camera = OrbbecCamera(camera_config)
        self.output_root = output_root.expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._session: DatasetSession | None = None
        self._record_thread: threading.Thread | None = None
        self._record_stop = threading.Event()
        self._recording = False
        self._record_interval_s = 1.0
        self._record_max_frames: int | None = None
        self._record_captured = 0
        self._record_error: str | None = None
        self._camera_start_error: str | None = None

    def start(self) -> None:
        try:
            self.camera.start()
            self._camera_start_error = None
        except Exception as exc:
            self._camera_start_error = f"{type(exc).__name__}: {exc}"

    def restart_camera(self) -> None:
        self.stop_recording()
        self.camera.stop()
        try:
            self.camera.start()
            self._camera_start_error = None
        except Exception as exc:
            self._camera_start_error = f"{type(exc).__name__}: {exc}"
            raise

    def shutdown(self) -> None:
        self.stop_recording()
        self.close_session()
        self.camera.stop()

    def status(self) -> dict[str, Any]:
        try:
            disk = shutil.disk_usage(self.output_root)
            disk_payload = {
                "free_bytes": disk.free,
                "total_bytes": disk.total,
                "free_gb": round(disk.free / (1024**3), 2),
            }
        except OSError:
            disk_payload = None
        with self._lock:
            return {
                "ok": True,
                "camera": {
                    **self.camera.status(),
                    "start_error": self._camera_start_error,
                },
                "session": (
                    None if self._session is None else self._session.status()
                ),
                "recording": {
                    "active": self._recording,
                    "interval_s": self._record_interval_s,
                    "max_frames": self._record_max_frames,
                    "captured": self._record_captured,
                    "error": self._record_error,
                },
                "output_root": str(self.output_root),
                "disk": disk_payload,
            }

    def capture(self, *, trigger: str, session_name: str = "capture") -> str:
        frame = self.camera.latest()
        session = self._ensure_session(session_name)
        return session.enqueue(frame, trigger)

    def start_recording(
        self, *, interval_s: float, max_frames: int | None, session_name: str
    ) -> None:
        if not 0.1 <= interval_s <= 3600.0:
            raise ValueError("interval_s 必须在 0.1~3600 秒")
        if max_frames is not None and not 1 <= max_frames <= 1_000_000:
            raise ValueError("max_frames 必须在 1~1000000，或留空")
        self._ensure_session(session_name)
        with self._lock:
            if self._recording:
                raise RuntimeError("连续采集已经在运行")
            self._recording = True
            self._record_interval_s = float(interval_s)
            self._record_max_frames = max_frames
            self._record_captured = 0
            self._record_error = None
            self._record_stop.clear()
            self._record_thread = threading.Thread(
                target=self._record_loop,
                name="interval-recorder",
                daemon=True,
            )
            self._record_thread.start()

    def stop_recording(self) -> None:
        self._record_stop.set()
        with self._lock:
            thread = self._record_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            self._recording = False
            self._record_thread = None

    def close_session(self) -> dict[str, Any] | None:
        self.stop_recording()
        with self._lock:
            session, self._session = self._session, None
        if session is None:
            return None
        session.close()
        return session.status()

    def _ensure_session(self, name: str) -> DatasetSession:
        with self._lock:
            if self._session is None:
                self._session = DatasetSession(
                    self.output_root,
                    name.strip() or "capture",
                    self.camera.metadata(),
                )
            return self._session

    def _record_loop(self) -> None:
        last_host_time_ns = -1
        next_capture = time.monotonic()
        try:
            while not self._record_stop.is_set():
                wait = next_capture - time.monotonic()
                if wait > 0 and self._record_stop.wait(wait):
                    break
                frame = self.camera.latest()
                if frame.host_time_ns == last_host_time_ns:
                    self._record_error = "相机没有产生新帧，本次间隔跳过"
                else:
                    session = self._ensure_session("capture")
                    session.enqueue(frame, "interval")
                    last_host_time_ns = frame.host_time_ns
                    with self._lock:
                        self._record_captured += 1
                        captured = self._record_captured
                        maximum = self._record_max_frames
                        self._record_error = None
                    if maximum is not None and captured >= maximum:
                        break
                next_capture = max(
                    next_capture + self._record_interval_s,
                    time.monotonic() + 0.001,
                )
        except CaptureQueueFull as exc:
            self._record_error = str(exc)
        except Exception as exc:
            self._record_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._recording = False
                self._record_thread = None


def _mjpeg_stream(
    service: CollectorService, kind: str, fps: float = 10.0
) -> Iterator[bytes]:
    interval = 1.0 / max(fps, 1.0)
    while True:
        try:
            jpeg = (
                service.camera.color_preview_jpeg()
                if kind == "color"
                else service.camera.depth_preview_jpeg()
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        except Exception:
            time.sleep(0.25)
            continue
        time.sleep(interval)


def create_app(
    camera_config: CameraConfig,
    output_root: Path,
    web_root: Path,
) -> FastAPI:
    service = CollectorService(camera_config, output_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.start()
        yield
        service.shutdown()

    app = FastAPI(title="Orbbec RGB-D Collector", lifespan=lifespan)
    app.state.collector = service

    @app.get("/")
    def index():
        return FileResponse(web_root / "index.html")

    @app.get("/api/status")
    def status():
        return service.status()

    @app.post("/api/camera/restart")
    def restart_camera():
        try:
            service.restart_camera()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/capture")
    def capture(body: dict | None = None):
        body = body or {}
        try:
            frame_id = service.capture(
                trigger="manual",
                session_name=str(body.get("session_name") or "capture"),
            )
        except CaptureQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"ok": True, "frame_id": frame_id}

    @app.post("/api/record/start")
    def record_start(body: dict | None = None):
        body = body or {}
        try:
            raw_max = body.get("max_frames")
            max_frames = (
                None if raw_max in (None, "", 0, "0") else int(raw_max)
            )
            service.start_recording(
                interval_s=float(body.get("interval_s") or 1.0),
                max_frames=max_frames,
                session_name=str(body.get("session_name") or "capture"),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/record/stop")
    def record_stop():
        service.stop_recording()
        return {"ok": True}

    @app.post("/api/session/close")
    def session_close():
        return {"ok": True, "session": service.close_session()}

    @app.get("/api/sessions")
    def sessions():
        return {"ok": True, "sessions": list_sessions(service.output_root)}

    @app.get("/stream/color.mjpg")
    def color_stream():
        return StreamingResponse(
            _mjpeg_stream(service, "color"),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/stream/depth.mjpg")
    def depth_stream():
        return StreamingResponse(
            _mjpeg_stream(service, "depth"),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    return app
