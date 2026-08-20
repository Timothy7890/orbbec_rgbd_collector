from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from .pointcloud import resolve_frame_paths


class OfflineYolo:
    def __init__(
        self,
        model_path: Path | None = None,
        *,
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.model_path = (
            model_path.expanduser().resolve() if model_path else None
        )
        self.confidence = confidence
        self.device = device
        self._model: Any = None
        self._lock = Lock()
        self._cache: OrderedDict[
            tuple[str, int, float], list[dict[str, Any]]
        ] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.model_path is not None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_exists": bool(
                self.model_path and self.model_path.is_file()
            ),
            "loaded": self._model is not None,
            "task": (
                getattr(self._model, "task", None)
                if self._model is not None
                else None
            ),
            "confidence": self.confidence,
            "device": self.device,
        }

    def _load_model(self) -> Any:
        if not self.enabled:
            raise RuntimeError("未配置 YOLO 模型")
        assert self.model_path is not None
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO 模型不存在: {self.model_path}")
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 ultralytics；请执行 pip install -e '.[yolo]'"
                ) from exc
            self._model = YOLO(str(self.model_path))
        return self._model

    def infer_frame(
        self, data_root: Path, session_id: str, frame_id: str
    ) -> list[dict[str, Any]]:
        _, frame_dir = resolve_frame_paths(data_root, session_id, frame_id)
        color_path = frame_dir / "color.jpg"
        if not color_path.is_file():
            raise FileNotFoundError(f"RGB 图不存在: {color_path}")
        key = (
            str(color_path),
            color_path.stat().st_mtime_ns,
            self.confidence,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return [dict(item) for item in cached]

            model = self._load_model()
            image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取 RGB 图: {color_path}")
            kwargs: dict[str, Any] = {
                "source": image,
                "conf": self.confidence,
                "verbose": False,
            }
            if self.device:
                kwargs["device"] = self.device
            result = model.predict(**kwargs)[0]
            names = result.names
            polygons = result.masks.xy if result.masks is not None else []
            boxes: list[dict[str, Any]] = []
            if result.boxes is not None:
                for index, (xyxy, confidence, class_id) in enumerate(zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                )):
                    cls = int(class_id)
                    if isinstance(names, dict):
                        name = names.get(cls, cls)
                    elif 0 <= cls < len(names):
                        name = names[cls]
                    else:
                        name = cls
                    detection = {
                        "cls": cls,
                        "name": str(name),
                        "conf": float(confidence),
                        "xyxy": [float(value) for value in xyxy],
                    }
                    if index < len(polygons):
                        polygon = np.asarray(polygons[index], dtype=np.float32)
                        if (
                            polygon.ndim == 2
                            and polygon.shape[0] >= 3
                            and polygon.shape[1] == 2
                            and np.isfinite(polygon).all()
                        ):
                            detection["polygon"] = polygon.tolist()
                    boxes.append(detection)
            self._cache[key] = boxes
            while len(self._cache) > 64:
                self._cache.popitem(last=False)
            return [dict(item) for item in boxes]
