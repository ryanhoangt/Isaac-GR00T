#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/eval_ravla_libero.sh [options]

Run one LIBERO subset with RA-VLA in separate policy/simulator environments.
The matching released checkpoint, retriever, and OOD memory are selected by
default. All relative paths are resolved against the repository root.

  --subset NAME              goal | spatial | object | long (default: goal)
                             Full names such as libero_goal also work.
  --checkpoint-path DIR      Override the RA-VLA checkpoint directory.
  --retriever-path DIR       Override the retriever checkpoint directory.
  --memory-name NAME         Memory inside the retriever directory (optional .pt).
  --memory-path FILE         Override the memory file at any path.
                             Use either memory-name or memory-path.
  --trials N                 Rollouts per task (default: 50).
  --task-ids ID [ID ...]      Only these task IDs, 0..9 (default: all ten).
  --cuda N                   GPU index within CUDA_VISIBLE_DEVICES (default: 0).
  --seed N                   Policy seed (default: 0; simulator seed stays 0).
  --k N                      Retrieved segments, 1..8 (default: 1).
  --denoising-steps N         Flow-matching steps (default: 4).
  --port N                   Local server port (default: 8071).
  --startup-timeout SECONDS   Wait for server readiness (default: 300).
  --run-name NAME            Run label (default: unique timestamp).
  --output-dir DIR           New directory for logs, commands.json, and rollouts/.
  --assets-dir DIR           Released assets (default: ra-vla-official-assets).
  --policy-python FILE       Policy Python (default: $POLICY_PYTHON or
                             $HOME/miniconda3/envs/gr00t/bin/python).
  --libero-python FILE       Simulator Python (default: $LIBERO_PYTHON or
                             $HOME/miniconda3/envs/libero/bin/python).
  --dry-run                  Validate paths and show commands without using GPU.
  -h, --help                 Show this help.

Examples:
  bash scripts/eval_ravla_libero.sh --subset goal --trials 50
  bash scripts/eval_ravla_libero.sh --subset spatial --task-ids 0 1 --trials 1
  bash scripts/eval_ravla_libero.sh --subset long --dry-run

Outputs default to outputs/ravla_eval/<run-name>/. Existing run directories
are never overwritten. The runner stops its server when evaluation ends or
is interrupted. Run on a GPU allocation with CUDA and EGL available.
EOF
}

policy_python="${POLICY_PYTHON:-${HOME}/miniconda3/envs/gr00t/bin/python}"
libero_python="${LIBERO_PYTHON:-${HOME}/miniconda3/envs/libero/bin/python}"
args=("$@")

# Select the orchestrator's interpreter as well as the server's interpreter.
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        -h|--help)
            usage
            exit 0
            ;;
        --policy-python)
            if ((i + 1 >= ${#args[@]})); then
                echo "Error: --policy-python requires an executable path." >&2
                exit 2
            fi
            policy_python="${args[i+1]}"
            ;;
        --policy-python=*)
            policy_python="${args[i]#*=}"
            ;;
    esac
done

if ! policy_python="$(command -v -- "$policy_python")"; then
    echo "Error: policy Python not found. Set POLICY_PYTHON or pass --policy-python." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
exec "$policy_python" -u -m scripts.rag.eval_libero \
    --policy-python "$policy_python" \
    --libero-python "$libero_python" \
    "${args[@]}"
