#!/bin/bash
#SBATCH --job-name=multimerge
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# export UV_CACHE_DIR=/scratch/$USER/.uv_cache
# export UV_PROJECT_ENVIRONMENT_DIR=/scratch/$USER/.uv_envs/multimerge
# export UV_LINK_MODE=copy  # avoids hardlink warning


# optional: ensure env synced
uv sync

# run your job in the uv environment
# uv run python main.py
uv run lm_eval --model hf --model_args pretrained=HPLT/hplt2c_eng_checkpoints --tasks belebele_eng_Latn --device cuda:0 --batch_size 8
uv run lm_eval --model hf --model_args pretrained=HPLT/hplt2c_fra_checkpoints,revision=checkpoint_0047000  --tasks belebele_fra_Latn --device cuda:0 --batch_size 8

uv run lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-1B --tasks belebele_eng_Latn --device cuda:0 --batch_size 8



export CHECKPOINT_PATH=HPLT/hplt2c_eng_checkpoints
time uv run lighteval accelerate "model_name=${CHECKPOINT_PATH},batch_size=8" --custom-tasks "lighteval.tasks.multilingual.tasks" "lighteval|belebele_eng_Latn_cf|0|1"

time uv run lighteval accelerate "model_name=HPLT/hplt2c_eng_checkpoints,batch_size=8" --custom-tasks "lighteval.tasks.multilingual.tasks" "lighteval|belebele_eng_Latn_cf|0|1"

time uv run lighteval accelerate "model_name=meta-llama/Llama-3.2-1B,batch_size=32"\
--custom-tasks "lighteval.tasks.multilingual.tasks" "lighteval|belebele_eng_Latn_cf|0" \
--output_dir ./results \
--results_path_template '{./results}/baselines/{org}/{model}' \
--save-details


lighteval accelerate\
    --model_args vllm,pretrained=model_name,pairwise_tokenization=True \
    --custom_task lighteval.tasks.multilingual.tasks \
    --tasks 'examples/tasks/finetasks/{cf,mcf}/{ara,fra,rus,tur,swa,hin,tel,tha,zho}' \
    --max_samples '1000'

HuggingFaceTB/SmolLM3-3B-Base
HuggingFaceTB/SmolLM2-1.7B
HuggingFaceTB/SmolLM2-4B
meta-llama/Llama-3.2-1B
meta-llama/Llama-3.2-3B
LiquidAI/LFM2-2.6B
LiquidAI/LFM2-1.2B
LiquidAI/LFM2-350M
google/gemma-3-1b
google/gemma-3-4B
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
DeepSeeek-R1-1.5B 
Phi-3.5-Mini-3.8B

  - lighteval|belebele_eng_Latn_hybrid
  - lighteval|belebele_eng_Latn_mcf
  - lighteval|belebele_est_Latn_cf

# https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasksx


langs = [
    "bos", "deu", "ron", "ell", "fin", "cat", "dan", "bul", "eng", "est", "eus",
    "fra", "hrv", "hun", "ita", "lit", "lvs", "mkd", "nld", "nno", "nob", "pol",
    "por", "slk", "slv", "spa", "swe", "ces", "tur", "ukr", "ara", "hin", "kor",
    "rus", "glg", "zhos", "zhot", "isl"
]

time uv run lighteval accelerate "model_name=HPLT/hplt2c_zhos_checkpoints,batch_size=8" --custom-tasks "lighteval.tasks.multilingual.tasks" "lighteval|cmmlu_zho_cf|0"
cmmlu_zho_cf

langs=(bos deu ron ell fin cat dan bul eng est eus fra hrv hun ita lit lvs mkd nld nno nob pol por slk slv spa swe ces tur ukr ara hin kor rus glg zhos zhot isl)





time CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn uv run lighteval vllm "model_name=HPLT/hplt2c_eng_checkpoints,tensor_parallel_size=2" --custom-tasks "lighteval.tasks.multilingual.tasks" "lighteval|global_mmlu_all_eng_cf|0|0"


# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


time CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn uv run lighteval vllm "model_name=HPLT/hplt2c_eng_checkpoints,data_parallel_size=1,max_model_length=2048" --custom-tasks "lighteval.tasks.multilingual.tasks" "lighteval|flores:eng_Latn-deu_Latn|0|0"