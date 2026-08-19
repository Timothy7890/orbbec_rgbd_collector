from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from .models import RGBDFrame


class CaptureQueueFull(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("._")
    return cleaned[:64] or "session"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class _CaptureItem:
    frame_id: str
    sequence: int
    trigger: str
    queued_at: str
    frame: RGBDFrame


class DatasetSession:
    """Asynchronous, all-or-nothing writer for synchronized RGB-D frame sets."""

    def __init__(
        self,
        output_root: Path,
        session_name: str,
        camera_metadata: dict[str, Any],
        *,
        queue_size: int = 32,
        jpeg_quality: int = 95,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{stamp}_{_safe_name(session_name)}"
        self.path = output_root.expanduser().resolve() / self.session_id
        self.frames_path = self.path / "frames"
        self.frames_path.mkdir(parents=True, exist_ok=False)
        self._manifest_path = self.path / "manifest.jsonl"
        self._queue: queue.Queue[_CaptureItem | None] = queue.Queue(
            maxsize=queue_size
        )
        self._jpeg_quality = int(max(1, min(jpeg_quality, 100)))
        self._lock = threading.Lock()
        self._sequence = 0
        self._saved = 0
        self._failed = 0
        self._dropped = 0
        self._last_saved: dict[str, Any] | None = None
        self._errors: list[str] = []
        self._accepting = True
        self._closed = False
        self.created_at = _utc_now()

        session_payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "session_name": session_name.strip() or "session",
            "created_at": self.created_at,
            "camera": camera_metadata,
            "storage": {
                "color": {
                    "file": "color.jpg",
                    "encoding": "JPEG",
                    "quality": self._jpeg_quality,
                    "pixel_order": "BGR_decoded_as_RGB_by_image_readers",
                },
                "depth_raw": {
                    "file": "depth_raw.png",
                    "encoding": "PNG",
                    "dtype": "uint16",
                    "geometry": "native_depth_sensor",
                },
                "depth_aligned": {
                    "file": "depth_aligned.png",
                    "encoding": "PNG",
                    "dtype": "uint16",
                    "geometry": "aligned_to_color",
                },
                "manifest": "manifest.jsonl",
            },
        }
        _write_json_atomic(self.path / "session.json", session_payload)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"rgbd-writer-{self.session_id}",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, frame: RGBDFrame, trigger: str) -> str:
        frame.validate()
        with self._lock:
            if not self._accepting:
                raise RuntimeError("session is closing; no more captures accepted")
            self._sequence += 1
            sequence = self._sequence
        frame_id = f"{sequence:06d}_{frame.host_time_ns}"
        item = _CaptureItem(
            frame_id=frame_id,
            sequence=sequence,
            trigger=str(trigger)[:32],
            queued_at=_utc_now(),
            frame=frame.detached(),
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            with self._lock:
                self._dropped += 1
            raise CaptureQueueFull(
                f"写盘队列已满（{self._queue.maxsize}），本帧未保存"
            ) from exc
        return frame_id

    def wait_idle(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._accepting = False
        self._queue.put(None)
        self._thread.join()
        with self._lock:
            self._closed = True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.session_id,
                "path": str(self.path),
                "created_at": self.created_at,
                "queued": self._queue.qsize(),
                "saved": self._saved,
                "failed": self._failed,
                "dropped": self._dropped,
                "last_saved": self._last_saved,
                "errors": list(self._errors[-5:]),
                "accepting": self._accepting,
                "closed": self._closed,
            }

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                record = self._write_item(item)
                with self._lock:
                    self._saved += 1
                    self._last_saved = record
            except Exception as exc:
                with self._lock:
                    self._failed += 1
                    self._errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def _write_item(self, item: _CaptureItem) -> dict[str, Any]:
        final_dir = self.frames_path / item.frame_id
        tmp_dir = self.frames_path / f".tmp-{item.frame_id}-{uuid.uuid4().hex}"
        tmp_dir.mkdir()
        try:
            color_path = tmp_dir / "color.jpg"
            raw_path = tmp_dir / "depth_raw.png"
            aligned_path = tmp_dir / "depth_aligned.png"
            if not cv2.imwrite(
                str(color_path),
                item.frame.color_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            ):
                raise OSError(f"failed to write {color_path}")
            if not cv2.imwrite(
                str(raw_path),
                item.frame.depth_raw,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            ):
                raise OSError(f"failed to write {raw_path}")
            if not cv2.imwrite(
                str(aligned_path),
                item.frame.depth_aligned,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            ):
                raise OSError(f"failed to write {aligned_path}")

            record = {
                "schema_version": 1,
                "session_id": self.session_id,
                "frame_id": item.frame_id,
                "sequence": item.sequence,
                "trigger": item.trigger,
                "queued_at": item.queued_at,
                "saved_at": _utc_now(),
                "files": {
                    "color": f"frames/{item.frame_id}/color.jpg",
                    "depth_raw": f"frames/{item.frame_id}/depth_raw.png",
                    "depth_aligned": (
                        f"frames/{item.frame_id}/depth_aligned.png"
                    ),
                    "metadata": f"frames/{item.frame_id}/frame.json",
                },
                **item.frame.metadata(),
            }
            _write_json_atomic(tmp_dir / "frame.json", record)
            os.replace(tmp_dir, final_dir)
            with self._manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                os.fsync(manifest.fileno())
            return record
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise


def list_sessions(output_root: Path) -> list[dict[str, Any]]:
    root = output_root.expanduser().resolve()
    if not root.is_dir():
        return []
    sessions: list[dict[str, Any]] = []
    for path in sorted(
        (entry for entry in root.iterdir() if entry.is_dir()), reverse=True
    ):
        metadata_path = path / "session.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifest = path / "manifest.jsonl"
        count = 0
        if manifest.is_file():
            try:
                with manifest.open(encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
            except OSError:
                pass
        sessions.append(
            {
                "id": metadata.get("session_id", path.name),
                "name": metadata.get("session_name"),
                "created_at": metadata.get("created_at"),
                "path": str(path),
                "frames": count,
            }
        )
    return sessions
