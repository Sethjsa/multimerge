#!/bin/bash
#SBATCH --job-name=multimerge_array
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=logs/infer_%A_%a.log   # %A = array job ID, %a = task index
#SBATCH --error=logs/infer_%A_%a.err

source activate gvllm

TASK=$1
LANG=$2
JOB_LIST=$3

# Pick this task's model from the job list using the array index
MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_LIST")

if [ -z "$MODEL" ]; then
    echo "ERROR: Could not read model for task index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID / Model: $MODEL"
echo "Task: $TASK | Lang: $LANG"


/home/saycock/miniconda3/envs/gvllm/bin/python -m evaluate --task "$TASK" --lang "$LANG" --gpus 1 --model "$MODEL"