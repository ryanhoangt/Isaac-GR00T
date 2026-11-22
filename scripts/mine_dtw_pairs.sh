#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
policy_python="${POLICY_PYTHON:-${HOME}/miniconda3/envs/gr00t/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "$policy_python" -u -m scripts.rag.mine_dtw_pairs "$@"
