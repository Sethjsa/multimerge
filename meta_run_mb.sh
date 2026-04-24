#!/bin/bash
#SBATCH --job-name=mblimp
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=36:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --output=logs/mb_%j.log
#SBATCH --error=logs/mb_%j.err

source activate gvllm

MODEL_LANGS=("eng" "nld" "spa" "fra" "rus" "ita" "por" "tur" "ara" "deu")

echo "$1"
echo "$2"

if [ "$1" == "--baselines" ]; then
    echo "Running multiblimp with baselines"
    /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --baselines_only --stderr_only
fi

if [ "$1" == "--spectrum" ]; then
    echo "Running multiblimp locally with spectrum of checkpoints"
    /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --local --spectrum --stderr_only
fi

if [ "$1" == "--matrix" ]; then
    echo "Running multiblimp locally with matrix of models"
    /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --local --matrix --stderr_only
fi

if [ "$1" == "--mixed" ]; then
    echo "Running multiblimp locally with mixed models"
    /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --local --mixed_models --stderr_only
fi

if [ "$1" == "--merged" ]; then
    echo "Running multiblimp locally with merged models"
    /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --local --stderr_only
fi



# if [ "$1" == "--local" ]; then
#     echo "Running multiblimp locally"
#     /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --local
# else
if [ "$1" == "--hplt" ]; then
    for lang in "${MODEL_LANGS[@]}"; do
        echo "Running multiblimp on cluster"
        /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --model_langs "${lang}" --stderr_only
    done
fi

# /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --local
# /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py



# python multiblimp/scripts/lm_eval/eval_model.py \
# --model_name HPLT/hplt2c_eng_checkpoints \
# --revision main \
# --data_dir "multiblimp/hf_cache/eng/" \
# --src_dir "multiblimp" \
# --results_dir "multiblimp/results/hplt_multiblimp_results/HPLT/hplt2c_eng_checkpoints_main-eng" \
# --hf_token ./token