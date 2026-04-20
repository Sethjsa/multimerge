#!/bin/bash
# =============================================================================
# Meta-script: builds a job list and submits a SLURM array, capped at 8 GPUs.
# Usage: bash submit_evals.sh
# =============================================================================

# --- Config ------------------------------------------------------------------
HF_MODELS=(
    "HPLT/hplt2c_ara_checkpoints"
    # "HPLT/hplt2c_deu_checkpoints"
    "HPLT/hplt2c_eng_checkpoints"
    "HPLT/hplt2c_fra_checkpoints"
    # "HPLT/hplt2c_ita_checkpoints"
    # "HPLT/hplt2c_nld_checkpoints"
    # "HPLT/hplt2c_rus_checkpoints"
    "HPLT/hplt2c_por_checkpoints"
    # "HPLT/hplt2c_spa_checkpoints"
    # "HPLT/hplt2c_tur_checkpoints"
    "HPLT/hplt2c_zhos_checkpoints"
)

# Local checkpoints: will glob for checkpoint_* subdirs inside each of these
LOCAL_CHECKPOINT_ROOTS=(
    # "models/mixed-10-checkpoints"
    # "models/merged-10-checkpoints"
    # "models/another-run"   # add more roots here
)

TASK="all"
LANG="all"
MAX_CONCURRENT=8   # max GPUs / concurrent jobs
JOB_LIST="job_list.txt"
# -----------------------------------------------------------------------------

mkdir -p logs

# Build the job list (one model path per line)
rm -f "$JOB_LIST"

for model in "${HF_MODELS[@]}"; do
    echo "$model" >> "$JOB_LIST"
done

for root in "${LOCAL_CHECKPOINT_ROOTS[@]}"; do
    if [ ! -d "$root" ]; then
        echo "WARNING: $root not found, skipping."
        continue
    fi
    for ckpt_dir in "$root"/checkpoint_*/; do
        if [ -d "$ckpt_dir" ]; then
            # SLURM needs absolute paths for local models
            echo "$(realpath "$ckpt_dir")" >> "$JOB_LIST"
        fi
    done
done

NUM_JOBS=$(wc -l < "$JOB_LIST")
if [ "$NUM_JOBS" -eq 0 ]; then
    echo "No jobs found. Check your model lists."
    exit 1
fi

echo "Submitting array of $NUM_JOBS jobs (max $MAX_CONCURRENT concurrent)..."
echo "Job list:"
cat "$JOB_LIST"
echo ""

sbatch --array=1-${NUM_JOBS}%${MAX_CONCURRENT} infer_array.sh "$TASK" "$LANG" "$JOB_LIST"