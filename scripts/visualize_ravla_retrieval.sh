#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
viewer_python="${VIEWER_PYTHON:-$PWD/.venv-visualizer/bin/python}"
if ! "$viewer_python" -c 'import streamlit, torch, av' >/dev/null 2>&1; then
  echo "Missing viewer dependencies. See scripts/rag/RETRIEVAL_VIEWER.md for setup." >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "$viewer_python" -m streamlit run scripts/rag/retrieval_viewer.py \
  --server.address 127.0.0.1 --server.headless true \
  --browser.gatherUsageStats false "$@"
