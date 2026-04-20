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

if [ "$1" == "--local" ]; then
    echo "Running multiblimp locally"
    /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py --local
else
    echo "Running multiblimp on cluster"
    /home/saycock/miniconda3/envs/gvllm/bin/python multiblimp/run_mb.py
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