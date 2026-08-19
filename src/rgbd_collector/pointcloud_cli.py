from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .offline_yolo import OfflineYolo
from .pointcloud_app import create_pointcloud_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="浏览已采集 RGB-D 数据的点云（不占用相机）"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17002)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "datasets"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="可选 YOLO .pt 模型；用于对已保存 RGB 离线推理",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--device",
        help="可选 Ultralytics 设备，例如 mps、cpu 或 0",
    )
    args = parser.parse_args()
    app = create_pointcloud_app(
        args.data_dir,
        web_root=PROJECT_ROOT / "web",
        yolo=OfflineYolo(
            args.model, confidence=args.conf, device=args.device
        ),
    )
    print(f"[pointcloud] 数据目录: {args.data_dir.expanduser().resolve()}")
    print(f"[pointcloud] 前端: http://{args.host}:{args.port}/")
    if args.model:
        print(f"[pointcloud] YOLO 模型: {args.model.expanduser().resolve()}")
    else:
        print("[pointcloud] YOLO 未启用；仍可点选和保存三维目标")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
