#!/bin/bash
# Run inside a GPU allocation. See --help for checkpoint discovery and options.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${GR00T_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    python_bin=python
fi
exec "$python_bin" "$repo/scripts/eval_libero_rollout.py" "$@"
