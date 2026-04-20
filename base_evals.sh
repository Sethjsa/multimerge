#!/bin/bash
#SBATCH --job-name=base_evals
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

uv sync

langs=(bos deu ron ell fin cat dan bul eng est eus fra hrv hun ita lit lvs mkd nld nno nob pol por slk slv spa swe ces tur ukr ara hin kor rus glg zhos zhot isl)


for lang in "${langs[@]}"
do
lighteval accelerate\
    --model_args model_name=HPLT/hplt2c_${lang}_checkpoints,batch_size=8,pairwise_tokenization=True \
    --custom_task lighteval.tasks.multilingual.tasks \
    --tasks 'examples/tasks/finetasks/cf/{ara,fra,rus,tur,swa,hin,tel,tha,zho}' \
    --max_samples '1000'
done