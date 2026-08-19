#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
else
    echo "没有找到可用 Python。请先激活 Conda 环境："
    echo "  conda activate fastapi"
    echo "  python -m pip install -e ."
    exit 1
fi

cd "$ROOT"
if ! "$PYTHON" -c "import rgbd_collector" >/dev/null 2>&1; then
    echo "当前 Python 尚未安装本项目：$PYTHON"
    echo "请在当前环境执行："
    echo "  python -m pip install -e ."
    exit 1
fi

echo "使用 Python: $PYTHON"
exec "$PYTHON" -m rgbd_collector.cli "$@"
