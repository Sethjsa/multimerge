#!/bin/bash
#SBATCH --job-name=multimerge
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --output=logs/infer_%j.log
#SBATCH --error=logs/infer_%j.err

# export UV_CACHE_DIR=/scratch/$USER/.uv_cache
# export UV_PROJECT_ENVIRONMENT_DIR=/scratch/$USER/.uv_envs/multimerge
# export UV_LINK_MODE=copy  # avoids hardlink warning

while getopts "m:t:l:" opt; do
  case $opt in
    m) MODEL="$OPTARG";;
    t) TASK="$OPTARG";;
    l) LANG="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2;;
  esac
done

if [ -z "$MODEL" ]; then
  echo "Model (-m) argument is required"
  exit 1
fi

if [ -z "$TASK" ]; then
  TASK="all"
fi

if [ -z "$LANG" ]; then
  LANG="all"
fi

# first conda deactivate
uv run python -m evaluate --task $TASK --lang $LANG --gpus 1 --model $MODEL