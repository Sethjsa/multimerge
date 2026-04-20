#!/bin/bash
# =============================================================================
# Meta-script: builds a job list and submits a SLURM array, capped at 8 GPUs.
# Usage: bash submit_evals.sh
# =============================================================================

# --- Config ------------------------------------------------------------------
HF_MODELS=(
    "HPLT/hplt2c_ara_checkpoints"
    "HPLT/hplt2c_deu_checkpoints"
    "HPLT/hplt2c_eng_checkpoints"
    "HPLT/hplt2c_fra_checkpoints"
    "HPLT/hplt2c_ita_checkpoints"
    "HPLT/hplt2c_nld_checkpoints"
    "HPLT/hplt2c_rus_checkpoints"
    "HPLT/hplt2c_por_checkpoints"
    "HPLT/hplt2c_spa_checkpoints"
    "HPLT/hplt2c_tur_checkpoints"
    "HPLT/hplt2c_zhos_checkpoints"
)

CHECKPOINTS=(
#   checkpoint_0001000 checkpoint_0002000 checkpoint_0003000 checkpoint_0004000
#   checkpoint_0005000 checkpoint_0006000 checkpoint_0007000 checkpoint_0008000
#   checkpoint_0009000 checkpoint_0010000 checkpoint_0011000 checkpoint_0012000
#   checkpoint_0013000 checkpoint_0014000 checkpoint_0015000 checkpoint_0016000
#   checkpoint_0017000 checkpoint_0018000 checkpoint_0019000 checkpoint_0020000
#   checkpoint_0021000 checkpoint_0022000 checkpoint_0023000 checkpoint_0024000
#   checkpoint_0025000 checkpoint_0026000 checkpoint_0027000 checkpoint_0028000
#   checkpoint_0029000 checkpoint_0030000 checkpoint_0031000 checkpoint_0032000
#   checkpoint_0033000 checkpoint_0034000 checkpoint_0035000 checkpoint_0036000
#   checkpoint_0037000 checkpoint_0038000 checkpoint_0039000 checkpoint_0040000
#   checkpoint_0041000 checkpoint_0042000 checkpoint_0043000 checkpoint_0044000
#   checkpoint_0045000 checkpoint_0046000 checkpoint_0047000
  checkpoint_0047684
)

# The final checkpoint is revision "main" on HF but we save it as 47684
HF_MAIN_CHECKPOINT="checkpoint_0047684"
HF_MAIN_REVISION="main"

# Local checkpoints: will glob for checkpoint_* subdirs inside each of these
LOCAL_CHECKPOINT_ROOTS=(
    # "models/mixed-10-checkpoints"
    # "models/merged-10-checkpoints"
)

TASK="all"
LANG="all"
MAX_CONCURRENT=8
JOB_LIST="hf_job_list.txt"
# -----------------------------------------------------------------------------

mkdir -p logs
rm -f "$JOB_LIST"

# HF models: emit one line per (model, checkpoint) pair
# Format: "hf:<repo>|<revision>|<save_name>"
for model in "${HF_MODELS[@]}"; do
    for ckpt in "${CHECKPOINTS[@]}"; do
        if [ "$ckpt" = "$HF_MAIN_CHECKPOINT" ]; then
            revision="$HF_MAIN_REVISION"
        else
            revision="$ckpt"
        fi
        # save_name is used by infer_array.sh to name the output directory
        # save_name="$model"
        save_name="$ckpt"
        echo "hf:${model}|${revision}|${save_name}" >> "$JOB_LIST"
    done
done

# # Local checkpoints: emit absolute paths as before
# for root in "${LOCAL_CHECKPOINT_ROOTS[@]}"; do
#     if [ ! -d "$root" ]; then
#         echo "WARNING: $root not found, skipping."
#         continue
#     fi
#     for ckpt_dir in "$root"/checkpoint_*/; do
#         [ -d "$ckpt_dir" ] && echo "$(realpath "$ckpt_dir")" >> "$JOB_LIST"
#     done
# done

NUM_JOBS=$(wc -l < "$JOB_LIST")
if [ "$NUM_JOBS" -eq 0 ]; then
    echo "No jobs found. Check your model lists."
    exit 1
fi

echo "Submitting array of $NUM_JOBS jobs (max $MAX_CONCURRENT concurrent)..."
echo "Job list:"
cat "$JOB_LIST"
echo ""

sbatch --array=1-${NUM_JOBS}%${MAX_CONCURRENT} infer_hf_array.sh "$TASK" "$LANG" "$JOB_LIST"