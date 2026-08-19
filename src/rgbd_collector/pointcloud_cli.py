from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

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
    args = parser.parse_args()
    app = create_pointcloud_app(
        args.data_dir, web_root=PROJECT_ROOT / "web"
    )
    print(f"[pointcloud] 数据目录: {args.data_dir.expanduser().resolve()}")
    print(f"[pointcloud] 前端: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
