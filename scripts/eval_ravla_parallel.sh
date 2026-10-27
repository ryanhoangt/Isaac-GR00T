#!/usr/bin/env bash
# One policy server with a bounded pool of isolated LIBERO task processes.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${POLICY_PYTHON:-${HOME}/miniconda3/envs/gr00t/bin/python}"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
exec "$python_bin" -u -m scripts.rag.eval_libero_parallel "$@"
