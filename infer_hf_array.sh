#!/bin/bash
#SBATCH --job-name=multimerge_array
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=logs/infer_%A_%a.log
#SBATCH --error=logs/infer_%A_%a.err

source activate gvllm

TASK=$1
LANG=$2
JOB_LIST=$3

ENTRY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_LIST")

if [ -z "$ENTRY" ]; then
    echo "ERROR: Could not read model for task index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID / Entry: $ENTRY"
echo "Task: $TASK | Lang: $LANG"

if [[ "$ENTRY" == hf:* ]]; then
    # Format: hf:<repo>|<revision>|<save_name>
    payload="${ENTRY#hf:}"
    HF_REPO=$(echo "$payload"  | cut -d'|' -f1)
    REVISION=$(echo "$payload" | cut -d'|' -f2)
    SAVE_NAME=$(echo "$payload" | cut -d'|' -f3)
    MODEL="${HF_REPO}"
    echo "HF model: $HF_REPO  revision: $REVISION  save_name: $SAVE_NAME"
    /home/saycock/miniconda3/envs/gvllm/bin/python -m evaluate \
        --task "$TASK" --lang "$LANG" --gpus 1 \
        --model "$MODEL" \
        --revision "$REVISION" \
        --output_dir_suffix "$SAVE_NAME"
else
    # Local path
    echo "Local model: $ENTRY"
    /home/saycock/miniconda3/envs/gvllm/bin/python -m evaluate \
        --task "$TASK" --lang "$LANG" --gpus 1 \
        --model "$ENTRY"
fi

# /home/saycock/miniconda3/envs/gvllm/bin/python -m evaluate \
#         --task all --lang all --gpus 0 \
#         --model HPLT/hplt2c_eng_checkpoints \
#         --revision checkpoint_0001000 \
#         --output_dir_suffix checkpoint_0001000

# /home/saycock/miniconda3/envs/gvllm/bin/python evaluate.py --task all --lang all --gpus 1 --model /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-6-checkpoints/checkpoint_0047684 --output_dir_suffix checkpoint_0047684 --revision checkpoint_0047684
# /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-6-checkpoints/checkpoint_0047684


# /home/saycock/miniconda3/envs/gvllm/bin/python evaluate.py --task all --lang all --gpus 1 --model HPLT/hplt2c_eng_checkpoints --output_dir_suffix checkpoint_0047684 --revision checkpoint_0047684