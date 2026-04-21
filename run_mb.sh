#!/bin/bash
#SBATCH --job-name=mblimp_array
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --output=logs/mb_%A_%a.log
#SBATCH --error=logs/mb_%A_%a.err
#SBATCH --array=0-9%5

source activate gvllm

MODEL_LANGS=("eng" "nld" "spa" "fra" "rus" "ita" "por" "tur" "ara" "deu")

LANG=${MODEL_LANGS[$SLURM_ARRAY_TASK_ID]}

echo "Running multiblimp on cluster for language: ${LANG} (task ${SLURM_ARRAY_TASK_ID})"

/home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --model_langs "${LANG}"