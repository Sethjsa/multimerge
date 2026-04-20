#!/bin/bash
#SBATCH --job-name=download_hf_models
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source .venv/bin/activate # ensure env synced

export HUGGING_FACE_HUB_TOKEN=$(cat token)
export HF_HOME=/fnwi_fs/ivi/irlab/personal/saycock/hf

python download_hf_models.py --model mixed-10-checkpoints