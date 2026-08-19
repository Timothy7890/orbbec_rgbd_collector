from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import RGBDFrame


@dataclass(frozen=True)
class CameraConfig:
    serial: str | None = None
    color_width: int = 1920
    color_height: int = 1080
    depth_width: int = 1280
    depth_height: int = 800
    fps: int = 30
    color_format: str | None = None
    lock_file: Path = Path("/tmp/orbbec_rgbd_collector.lock")
    first_frame_timeout_s: float = 12.0


def _format_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1].upper()


def _optional_call(target: Any, name: str) -> Any:
    method = getattr(target, name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def _intrinsics(profile: Any) -> dict[str, Any]:
    value = profile.get_intrinsic()
    return {
        "width": int(value.width),
        "height": int(value.height),
        "fx": float(value.fx),
        "fy": float(value.fy),
        "cx": float(value.cx),
        "cy": float(value.cy),
    }


def _distortion(profile: Any) -> dict[str, Any]:
    names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    try:
        value = profile.get_distortion()
        coefficients = [float(getattr(value, name)) for name in names]
    except Exception:
        coefficients = [0.0] * len(names)
    return {
        "model": "brown_conrady",
        "coefficients": coefficients,
        "coefficient_order": list(names),
    }


def _profile_metadata(profile: Any) -> dict[str, Any]:
    return {
        "width": int(profile.get_width()),
        "height": int(profile.get_height()),
        "fps": int(profile.get_fps()),
        "format": _format_name(profile.get_format()),
        "intrinsics": _intrinsics(profile),
        "distortion": _distortion(profile),
    }


class OrbbecCamera:
    """Direct, exclusive USB owner of one Orbbec RGB-D camera."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._ob = None
        self._pipeline = None
        self._align = None
        self._lock_handle = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._latest: RGBDFrame | None = None
        self._metadata: dict[str, Any] = {}
        self._error: str | None = None
        self._running = False
        self._frame_count = 0
        self._decode_failures = 0
        self._started_monotonic = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._acquire_process_lock()
        try:
            try:
                import pyorbbecsdk as ob
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 Orbbec SDK v2。请执行 pip install pyorbbecsdk2"
                ) from exc
            self._ob = ob
            context = ob.Context()
            devices = context.query_devices()
            count = int(devices.get_count())
            if count == 0:
                raise RuntimeError("没有发现 Orbbec 相机，请检查 USB 3 连接和 macOS 权限")
            if self.config.serial:
                getter = getattr(devices, "get_device_by_serial_number", None)
                if getter is None:
                    raise RuntimeError("当前 SDK 不支持按序列号打开设备，请升级 pyorbbecsdk2")
                try:
                    device = getter(self.config.serial)
                except Exception as exc:
                    raise RuntimeError(
                        f"无法独占打开 Orbbec {self.config.serial!r}；"
                        "请关闭 Orbbec Viewer、其他采集程序并重新插拔相机"
                    ) from exc
            else:
                if count != 1:
                    raise RuntimeError(
                        f"发现 {count} 台 Orbbec。为避免打开错误设备，请通过 --serial 指定"
                    )
                try:
                    device = devices.get_device_by_index(0)
                except Exception as exc:
                    raise RuntimeError(
                        "Orbbec 已被其他程序占用；请关闭其他相机程序后重试"
                    ) from exc

            self._pipeline = ob.Pipeline(device)
            color_profiles = self._pipeline.get_stream_profile_list(
                ob.OBSensorType.COLOR_SENSOR
            )
            depth_profiles = self._pipeline.get_stream_profile_list(
                ob.OBSensorType.DEPTH_SENSOR
            )
            color_profile = self._select_profile(
                color_profiles,
                width=self.config.color_width,
                height=self.config.color_height,
                fps=self.config.fps,
                requested_format=self.config.color_format,
                preferences=("MJPG", "RGB", "YUYV", "NV12"),
                label="彩色",
            )
            depth_profile = self._select_profile(
                depth_profiles,
                width=self.config.depth_width,
                height=self.config.depth_height,
                fps=self.config.fps,
                requested_format=None,
                preferences=("Y16", "Z16"),
                label="深度",
            )

            sdk_config = ob.Config()
            sdk_config.enable_stream(color_profile)
            sdk_config.enable_stream(depth_profile)
            try:
                sdk_config.set_frame_aggregate_output_mode(
                    ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )
            except Exception:
                pass
            try:
                self._pipeline.enable_frame_sync()
            except Exception:
                pass

            self._align = ob.AlignFilter(
                align_to_stream=ob.OBStreamType.COLOR_STREAM
            )
            self._metadata = self._build_metadata(
                ob, device, color_profile, depth_profile
            )
            self._pipeline.start(sdk_config)
            self._stop_event.clear()
            self._running = True
            self._started_monotonic = time.monotonic()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="orbbec-capture",
                daemon=True,
            )
            self._thread.start()
            self.wait_for_frame(timeout_s=self.config.first_frame_timeout_s)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._align = None
        self._running = False
        self._release_process_lock()

    def wait_for_frame(
        self, *, after_host_time_ns: int = 0, timeout_s: float = 2.0
    ) -> RGBDFrame:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while (
                self._latest is None
                or self._latest.host_time_ns <= after_host_time_ns
            ):
                if self._error and not self._running:
                    raise RuntimeError(self._error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self._error}" if self._error else ""
                    raise TimeoutError(f"{timeout_s:g}s 内没有收到完整 RGB-D 帧{detail}")
                self._condition.wait(remaining)
            return self._latest.detached()

    def latest(self) -> RGBDFrame:
        with self._condition:
            if self._latest is None:
                raise RuntimeError("相机尚未产生完整 RGB-D 帧")
            return self._latest.detached()

    def metadata(self) -> dict[str, Any]:
        with self._condition:
            payload = json.loads(json.dumps(self._metadata))
            if self._latest is not None:
                payload["depth_scale"] = {
                    "value": self._latest.depth_scale_mm,
                    "unit": "mm_per_raw_unit",
                }
            return payload

    def status(self) -> dict[str, Any]:
        with self._condition:
            latest = self._latest
            elapsed = max(time.monotonic() - self._started_monotonic, 1e-6)
            return {
                "running": self._running,
                "error": self._error,
                "frames": self._frame_count,
                "decode_failures": self._decode_failures,
                "average_fps": round(self._frame_count / elapsed, 2),
                "latest_host_time_ns": (
                    None if latest is None else latest.host_time_ns
                ),
                "depth_scale_mm": (
                    None if latest is None else latest.depth_scale_mm
                ),
                "device": self._metadata.get("device"),
                "color": self._metadata.get("color"),
                "depth": self._metadata.get("depth"),
            }

    def color_preview_jpeg(self, max_width: int = 960) -> bytes:
        frame = self.latest()
        image = self._resize_preview(frame.color_bgr, max_width)
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not ok:
            raise RuntimeError("RGB 预览编码失败")
        return encoded.tobytes()

    def depth_preview_jpeg(
        self, max_width: int = 960, min_mm: float = 150.0, max_mm: float = 4000.0
    ) -> bytes:
        frame = self.latest()
        depth_mm = frame.depth_aligned.astype(np.float32) * frame.depth_scale_mm
        valid = (depth_mm >= min_mm) & (depth_mm <= max_mm)
        normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
        normalized[valid] = np.clip(
            (depth_mm[valid] - min_mm) * 255.0 / max(max_mm - min_mm, 1.0),
            0,
            255,
        ).astype(np.uint8)
        colorized = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        colorized[~valid] = 0
        image = self._resize_preview(colorized, max_width)
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            raise RuntimeError("深度预览编码失败")
        return encoded.tobytes()

    @staticmethod
    def _resize_preview(image: np.ndarray, max_width: int) -> np.ndarray:
        if image.shape[1] <= max_width:
            return image
        scale = max_width / image.shape[1]
        return cv2.resize(
            image,
            (max_width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(250)
                if frames is None:
                    continue
                frame_set = self._as_frame_set(frames)
                color = frame_set.get_color_frame()
                raw_depth = frame_set.get_depth_frame()
                if color is None or raw_depth is None:
                    continue
                color_bgr = self._decode_color(color)
                if color_bgr is None:
                    with self._condition:
                        self._decode_failures += 1
                    continue
                raw = np.frombuffer(
                    raw_depth.get_data(), dtype=np.uint16
                ).reshape(
                    raw_depth.get_height(), raw_depth.get_width()
                ).copy()
                depth_scale = float(raw_depth.get_depth_scale())

                aligned_result = self._align.process(frames)
                if not aligned_result:
                    continue
                aligned_frame = self._as_frame_set(
                    aligned_result
                ).get_depth_frame()
                if aligned_frame is None:
                    continue
                aligned = np.frombuffer(
                    aligned_frame.get_data(), dtype=np.uint16
                ).reshape(
                    aligned_frame.get_height(), aligned_frame.get_width()
                ).copy()
                if aligned.shape != color_bgr.shape[:2]:
                    raise RuntimeError(
                        "SDK 对齐深度尺寸与彩色不一致: "
                        f"{aligned.shape} vs {color_bgr.shape[:2]}"
                    )

                sample = RGBDFrame(
                    color_bgr=np.ascontiguousarray(color_bgr),
                    depth_raw=np.ascontiguousarray(raw),
                    depth_aligned=np.ascontiguousarray(aligned),
                    host_time_ns=time.time_ns(),
                    color_timestamp_ms=self._timestamp_ms(color),
                    depth_timestamp_ms=self._timestamp_ms(raw_depth),
                    color_frame_index=self._frame_number(
                        color, "get_index", int
                    ),
                    depth_frame_index=self._frame_number(
                        raw_depth, "get_index", int
                    ),
                    depth_scale_mm=depth_scale,
                )
                sample.validate()
                with self._condition:
                    self._latest = sample
                    self._frame_count += 1
                    self._error = None
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._condition.notify_all()
                if not self._stop_event.is_set():
                    time.sleep(0.1)

    @staticmethod
    def _frame_number(frame: Any, method: str, converter):
        value = _optional_call(frame, method)
        if value is None:
            return None
        try:
            return converter(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _timestamp_ms(frame: Any) -> float | None:
        """Prefer the v2 microsecond clock while keeping the JSON unit stable."""
        timestamp_us = _optional_call(frame, "get_timestamp_us")
        if timestamp_us is not None:
            try:
                return float(timestamp_us) / 1000.0
            except (TypeError, ValueError):
                pass
        return OrbbecCamera._frame_number(frame, "get_timestamp", float)

    @staticmethod
    def _as_frame_set(frame: Any) -> Any:
        """Some v2 releases already return FrameSet; older ones need a cast."""
        if hasattr(frame, "get_color_frame") and hasattr(
            frame, "get_depth_frame"
        ):
            return frame
        converter = getattr(frame, "as_frame_set", None)
        if converter is None:
            raise RuntimeError("Orbbec SDK 返回对象无法转换为 FrameSet")
        return converter()

    def _decode_color(self, frame: Any) -> np.ndarray | None:
        assert self._ob is not None
        fmt = frame.get_format()
        width, height = int(frame.get_width()), int(frame.get_height())
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        ob = self._ob
        if fmt == ob.OBFormat.RGB:
            return cv2.cvtColor(
                data.reshape(height, width, 3), cv2.COLOR_RGB2BGR
            )
        bgr_format = getattr(ob.OBFormat, "BGR", None)
        if bgr_format is not None and fmt == bgr_format:
            return data.reshape(height, width, 3).copy()
        if fmt == ob.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob.OBFormat.YUYV:
            return cv2.cvtColor(
                data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUYV
            )
        if fmt == ob.OBFormat.NV12:
            return cv2.cvtColor(
                data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12
            )
        return None

    @staticmethod
    def _select_profile(
        profiles: Any,
        *,
        width: int,
        height: int,
        fps: int,
        requested_format: str | None,
        preferences: tuple[str, ...],
        label: str,
    ) -> Any:
        candidates: list[tuple[int, Any]] = []
        available: list[str] = []
        for index in range(profiles.get_count()):
            try:
                profile = profiles.get_stream_profile_by_index(
                    index
                ).as_video_stream_profile()
            except Exception:
                continue
            fmt = _format_name(profile.get_format())
            summary = (
                f"{profile.get_width()}x{profile.get_height()}"
                f"@{profile.get_fps()} {fmt}"
            )
            available.append(summary)
            if (
                int(profile.get_width()),
                int(profile.get_height()),
                int(profile.get_fps()),
            ) != (width, height, fps):
                continue
            if requested_format and fmt != requested_format.upper():
                continue
            try:
                rank = preferences.index(fmt)
            except ValueError:
                continue
            candidates.append((rank, profile))
        if not candidates:
            expected = f"{width}x{height}@{fps}"
            if requested_format:
                expected += f" {requested_format}"
            raise RuntimeError(
                f"找不到{label}流 {expected}；设备可用流: {available}"
            )
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    @staticmethod
    def _build_metadata(
        ob: Any, device: Any, color_profile: Any, depth_profile: Any
    ) -> dict[str, Any]:
        info = device.get_device_info()
        extrinsic = depth_profile.get_extrinsic_to(color_profile)
        rotation = np.asarray(
            getattr(extrinsic, "rot"), dtype=np.float64
        ).reshape(3, 3)
        translation_value = getattr(extrinsic, "transform", None)
        if translation_value is None:
            translation_value = getattr(extrinsic, "trans")
        translation = np.asarray(
            translation_value, dtype=np.float64
        ).reshape(3)
        get_version = getattr(ob, "get_version", None)
        return {
            "sdk": {
                "package": "pyorbbecsdk2",
                "version": (
                    str(get_version()) if get_version is not None else None
                ),
            },
            "device": {
                "name": _optional_call(info, "get_name"),
                "serial": _optional_call(info, "get_serial_number"),
                "firmware_version": _optional_call(
                    info, "get_firmware_version"
                ),
            },
            "color": _profile_metadata(color_profile),
            "depth": _profile_metadata(depth_profile),
            "depth_to_color": {
                "rotation_row_major": rotation.tolist(),
                "translation": translation.tolist(),
                "translation_unit": "mm",
                "convention": "p_color_mm = R @ p_depth_mm + t_mm",
            },
            "alignment": {
                "method": "Orbbec AlignFilter",
                "target_stream": "COLOR_STREAM",
            },
        }

    def _acquire_process_lock(self) -> None:
        path = self.config.lock_file.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "未知进程"
            handle.close()
            raise RuntimeError(
                f"采集器已在运行（锁 {path}，持有者 {owner}）"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._lock_handle = handle

    def _release_process_lock(self) -> None:
        if self._lock_handle is None:
            return
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_handle.close()
            self._lock_handle = None
