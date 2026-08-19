from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .camera import CameraConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apple Silicon Mac 上的 Orbbec RGB-D 独占采集服务"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7003)
    parser.add_argument("--serial", default=None, help="Orbbec 序列号；多相机时必填")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--color-width", type=int, default=1920)
    parser.add_argument("--color-height", type=int, default=1080)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--color-format",
        choices=["MJPG", "RGB", "YUYV", "NV12"],
        default=None,
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/tmp/orbbec_rgbd_collector.lock"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    camera_config = CameraConfig(
        serial=args.serial,
        color_width=args.color_width,
        color_height=args.color_height,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        fps=args.fps,
        color_format=args.color_format,
        lock_file=args.lock_file,
    )
    app = create_app(
        camera_config,
        output_root=args.data_dir,
        web_root=PROJECT_ROOT / "web",
    )
    print(f"[collector] 数据目录: {args.data_dir.expanduser().resolve()}")
    print(
        "[collector] 独占 Orbbec USB；运行期间请勿打开 Orbbec Viewer "
        "或其他相机程序"
    )
    print(f"[collector] 前端: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
