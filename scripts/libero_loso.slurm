#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail

# Arguments: suite to hold out, absolute output root for this four-run experiment.
held_out="${1:?Specify spatial, object, goal, or long}"
run_root="${2:?Specify an absolute experiment output root}"
case "$held_out" in spatial|object|goal|long) ;; *) exit 2 ;; esac
repo="${SLURM_SUBMIT_DIR:?}"
cd "$repo"
python_bin="${GR00T_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}"
data_root="${LIBERO_DATA_ROOT:-$repo/demo_data/libero_loso_clean}"
base_model="${GR00T_BASE_MODEL:-/mnt/data/vhoangth2/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e}"
output_dir="$run_root/heldout_$held_out"
mkdir -p "$output_dir"
if compgen -G "$output_dir/checkpoint-*" > /dev/null; then
    echo "Refusing to overwrite an existing training run: $output_dir" >&2
    exit 1
fi

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export USE_TF=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_MODULES_CACHE="$repo/.cache/huggingface/modules"
unset TRANSFORMERS_CACHE

export WANDB_MODE=online
export WANDB_DISABLED=false
export WANDB_PROJECT="${WANDB_PROJECT:-gr00t-libero-loso}"
export WANDB_ENTITY="${WANDB_ENTITY:-ryan-nlp}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$(basename "$run_root")}"
export WANDB_NAME="heldout_${held_out}_job_${SLURM_JOB_ID}"
export WANDB_RUN_ID="loso-${held_out}-${SLURM_JOB_ID}"
export WANDB_RESUME=never
export WANDB_DIR="$output_dir"
export WANDB_LOG_MODEL=false

datasets=()
for suite in spatial object goal long; do
    if [[ "$suite" != "$held_out" ]]; then
        datasets+=("$data_root/$suite")
    fi
done

"$python_bin" - "$output_dir" "$held_out" "$base_model" "${datasets[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

output, held_out, base_model, *datasets = sys.argv[1:]
manifest = {
    "held_out_suite": held_out,
    "training_datasets": datasets,
    "base_model": base_model,
    "data_config": "examples.Libero.custom_data_config:LiberoDataConfig",
    "num_train_epochs": 2,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "report_to": "wandb",
    "wandb_project": os.environ["WANDB_PROJECT"],
    "wandb_entity": os.environ.get("WANDB_ENTITY"),
    "wandb_group": os.environ["WANDB_RUN_GROUP"],
    "wandb_run_id": os.environ["WANDB_RUN_ID"],
}
Path(output, "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
PY

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --unbuffered "$python_bin" scripts/gr00t_finetune.py \
    --base-model-path "$base_model" \
    --dataset-path "${datasets[@]}" \
    --data-config examples.Libero.custom_data_config:LiberoDataConfig \
    --embodiment-tag new_embodiment \
    --num-gpus 1 \
    --batch-size 32 \
    --gradient-accumulation-steps 1 \
    --num-train-epochs 2 \
    --max-steps -1 \
    --save-strategy epoch \
    --save-total-limit 2 \
    --no-save-final-model \
    --dataloader-num-workers 6 \
    --dataloader-prefetch-factor 2 \
    --video-backend torchvision_av \
    --report-to wandb \
    --output-dir "$output_dir"

# Checkpoint directory names use optimizer steps; record their epoch mapping.
"$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoints = []
for path in sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
    state = json.loads((path / "trainer_state.json").read_text())
    checkpoints.append({"epoch": state["epoch"], "global_step": state["global_step"], "path": str(path)})
assert len(checkpoints) == 2, checkpoints
for expected_epoch, checkpoint in enumerate(checkpoints, start=1):
    assert abs(checkpoint["epoch"] - expected_epoch) < 1e-6, checkpoints
(root / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")
print(json.dumps(checkpoints, indent=2))
PY
