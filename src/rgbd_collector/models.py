from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    """One synchronized color/raw-depth/aligned-depth capture."""

    color_bgr: np.ndarray
    depth_raw: np.ndarray
    depth_aligned: np.ndarray
    host_time_ns: int
    color_timestamp_ms: float | None
    depth_timestamp_ms: float | None
    color_frame_index: int | None
    depth_frame_index: int | None
    depth_scale_mm: float

    def detached(self) -> "RGBDFrame":
        """Return a writer-owned copy that camera updates cannot mutate."""
        return RGBDFrame(
            color_bgr=np.ascontiguousarray(self.color_bgr.copy()),
            depth_raw=np.ascontiguousarray(self.depth_raw.copy()),
            depth_aligned=np.ascontiguousarray(self.depth_aligned.copy()),
            host_time_ns=self.host_time_ns,
            color_timestamp_ms=self.color_timestamp_ms,
            depth_timestamp_ms=self.depth_timestamp_ms,
            color_frame_index=self.color_frame_index,
            depth_frame_index=self.depth_frame_index,
            depth_scale_mm=self.depth_scale_mm,
        )

    def validate(self) -> None:
        if self.color_bgr.ndim != 3 or self.color_bgr.shape[2] != 3:
            raise ValueError(f"color_bgr must be HxWx3, got {self.color_bgr.shape}")
        if self.color_bgr.dtype != np.uint8:
            raise ValueError(f"color_bgr must be uint8, got {self.color_bgr.dtype}")
        for name, depth in (
            ("depth_raw", self.depth_raw),
            ("depth_aligned", self.depth_aligned),
        ):
            if depth.ndim != 2 or depth.dtype != np.uint16:
                raise ValueError(
                    f"{name} must be a uint16 HxW image, got "
                    f"{depth.shape}/{depth.dtype}"
                )
        if self.depth_aligned.shape != self.color_bgr.shape[:2]:
            raise ValueError(
                "depth_aligned must match color resolution: "
                f"{self.depth_aligned.shape} vs {self.color_bgr.shape[:2]}"
            )
        if not np.isfinite(self.depth_scale_mm) or self.depth_scale_mm <= 0:
            raise ValueError(f"invalid depth scale: {self.depth_scale_mm}")

    def metadata(self) -> dict[str, Any]:
        valid_raw = self.depth_raw[self.depth_raw > 0]
        valid_aligned = self.depth_aligned[self.depth_aligned > 0]

        def depth_stats(depth: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
            return {
                "width": int(depth.shape[1]),
                "height": int(depth.shape[0]),
                "dtype": str(depth.dtype),
                "valid_pixels": int(valid.size),
                "valid_ratio": round(float(valid.size / depth.size), 6),
                "min_raw": int(valid.min()) if valid.size else None,
                "max_raw": int(valid.max()) if valid.size else None,
            }

        return {
            "host_time_ns": self.host_time_ns,
            "color_timestamp_ms": self.color_timestamp_ms,
            "depth_timestamp_ms": self.depth_timestamp_ms,
            "timestamp_delta_ms": (
                None
                if self.color_timestamp_ms is None or self.depth_timestamp_ms is None
                else round(
                    float(self.color_timestamp_ms - self.depth_timestamp_ms), 6
                )
            ),
            "color_frame_index": self.color_frame_index,
            "depth_frame_index": self.depth_frame_index,
            "depth_scale": {
                "value": self.depth_scale_mm,
                "unit": "mm_per_raw_unit",
            },
            "color": {
                "width": int(self.color_bgr.shape[1]),
                "height": int(self.color_bgr.shape[0]),
                "dtype": str(self.color_bgr.dtype),
            },
            "depth_raw": depth_stats(self.depth_raw, valid_raw),
            "depth_aligned": depth_stats(self.depth_aligned, valid_aligned),
        }
