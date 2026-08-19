#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "未找到虚拟环境：$ROOT/.venv"
    echo "请先按 README 执行安装："
    echo "  python3.11 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  python -m pip install -U pip"
    echo "  python -m pip install -e ."
    exit 1
fi

cd "$ROOT"
exec "$PYTHON" -m rgbd_collector.cli "$@"
