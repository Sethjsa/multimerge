bash submit_hf_evals.sh
sbatch --array=1-6%6 infer_hf_array.sh all all baseline_job_list.txt
sbatch --array=1-9%8 infer_array.sh all all custom_job_list.txt