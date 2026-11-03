#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
python_bin="${VFD_VIEWER_PYTHON:-$repo/.venv-vfd-viewer/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Create the CPU viewer environment first:" >&2
    echo "uv venv --python python3.10 .venv-vfd-viewer" >&2
    echo "uv pip install --python .venv-vfd-viewer/bin/python -r requirements/vfd_viewer.txt" >&2
    exit 2
fi
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
exec "$python_bin" -m streamlit run scripts/view_libero_vfd.py \
    --server.address 127.0.0.1 --server.headless true "$@"
