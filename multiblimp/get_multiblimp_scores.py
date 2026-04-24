from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import os
import torch

# Set the MKL threading layer to GNU to avoid conflicts
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# checkpoints = [0, 10000, 20000, 30000, 40000, 50000, 64000, 64010, 64020, 64030, 64040, 64050, 64060, 64070, 64080, 64090, 64100, 64110, 64120, 64130, 64140, 64150, 64160, 64170, 64180, 64190, 64200, 64300, 64400, 64500, 64600, 64700, 64800, 64900, 65000, 66000, 67000, 68000, 69000, 70000, 80000, 90000, 100000, 110000, 120000, 128000]
checkpoints = ["checkpoint_0001000", "checkpoint_0002000", "checkpoint_0003000", "checkpoint_0004000",
                "checkpoint_0005000", "checkpoint_0006000", "checkpoint_0007000", "checkpoint_0008000",
                "checkpoint_0009000", "checkpoint_0010000", "checkpoint_0011000", "checkpoint_0012000",
                "checkpoint_0013000", "checkpoint_0014000", "checkpoint_0015000", "checkpoint_0016000",
                "checkpoint_0017000", "checkpoint_0018000", "checkpoint_0019000", "checkpoint_0020000",
                "checkpoint_0021000", "checkpoint_0022000", "checkpoint_0023000", "checkpoint_0024000",
                "checkpoint_0025000", "checkpoint_0026000", "checkpoint_0027000", "checkpoint_0028000",
                "checkpoint_0029000", "checkpoint_0030000", "checkpoint_0031000", "checkpoint_0032000",
                "checkpoint_0033000", "checkpoint_0034000", "checkpoint_0035000", "checkpoint_0036000",
                "checkpoint_0037000", "checkpoint_0038000", "checkpoint_0039000", "checkpoint_0040000",
                "checkpoint_0041000", "checkpoint_0042000", "checkpoint_0043000", "checkpoint_0044000",
                "checkpoint_0045000", "checkpoint_0046000", "checkpoint_0047000", "checkpoint_0047684", "main"]

# checkpoints = ["main", "checkpoint_0047684"]

# checkpoints = ["checkpoint_0047684"]

api = HfApi()
all_models = api.list_models(author="sethjsa")
total_downloads = 0
all_bgts = []

for model in tqdm(all_models):
    model_info = api.model_info(model.modelId, expand=["downloadsAllTime"])  
    model_name = str(model.modelId)
    all_bgts.append(model_name)


# mapping for language codes used for model to language codes used for multiblimp
language_map = {
    'eng': 'eng',
    'nld': 'nld',
    'spa': 'spa',
    'fra': 'fra',
    'rus': 'rus',
    # # 'tur': 'tur',
    'hin': 'hin',
    'ita': 'ita',
    'por': 'por',
    'ara': 'arb',
    # 'zhos': 'zho'
}

# create results dataframe
results = pd.DataFrame(columns=['model', 'checkpoint'] + list(language_map.keys()))
results.to_csv('multiblimp/results/all_multiblimp_results.csv', mode='w', index=False)


models = ["HPLT/hplt2c_eng_checkpoints",
          "HPLT/hplt2c_nld_checkpoints",
          "HPLT/hplt2c_spa_checkpoints",
          "HPLT/hplt2c_fra_checkpoints",
          "HPLT/hplt2c_rus_checkpoints",
          "HPLT/hplt2c_tur_checkpoints",
          "HPLT/hplt2c_zhos_checkpoints",
          "HPLT/hplt2c_ita_checkpoints",
          "HPLT/hplt2c_deu_checkpoints",
          "HPLT/hplt2c_ara_checkpoints"]

local_models = ["/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-2-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-4-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-6-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-8-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/mixed-10-checkpoints"]

for model in models:
    for checkpoint in checkpoints:
        # if number of "/" in checkpoint is greater than 1, then remove the first "/" and everything after it

        if model.count("/") > 1:
            m_str = model.split("/")[-2:]
            m_str = "/".join(m_str) + "/" + checkpoint
            c_str = checkpoint
            name = m_str + "_" + c_str
            results_str = m_str
        else:
            m_str = model
            # if checkpoint == "checkpoint_0047684":
            #     c_str = "main"
            name = m_str
            c_str = checkpoint
            results_str = m_str + "/" + c_str

        lang_accuracy = {}

        # Gather accuracy for all languages for this model/checkpoint
        for model_lang, test_lang in language_map.items():
            l1_iso_str = model_lang
            try:
                subprocess.run([
                    "python", "multiblimp/scripts/lm_eval/eval_model.py",
                    "--model_name", m_str,
                    "--revision", c_str,
                    "--data_dir", f"multiblimp/hf_cache/{l1_iso_str}/",
                    "--src_dir", "multiblimp",
                    "--results_dir", f"multiblimp/results/{results_str}/",
                    "--hf_token", "./token"
                ], check=True, env={**os.environ})
            
            except (subprocess.CalledProcessError, Exception) as e:
                if isinstance(e, subprocess.CalledProcessError):
                    print(f"Error processing model {m_str} at checkpoint {c_str}:")
                    print(f"Command failed with exit code {e.returncode}")
                    print(f"Error details: {e}")
                else:
                    print(f"Unexpected error with model {m_str} at checkpoint {c_str}:")
                    print(f"Error: {e}")
                # Continue with next checkpoint instead of stopping the entire script
                continue
     

            
            results_path = f"multiblimp/results/{results_str}/{test_lang}.tsv"
            df = pd.read_csv(results_path, sep='\t')
            total_samples = len(df)
            correct_predictions = len(df[df['delta'] > 0])
            test_lang_accuracy = correct_predictions / total_samples

            lang_accuracy[model_lang] = test_lang_accuracy

        # Save all language accuracies for this model/checkpoint as a single row in the dataframe
        row_data = {
            'model': m_str,
            'checkpoint': c_str
        }
        # Add all 10 language accuracy columns to the row
        for model_lang in language_map.keys():
            row_data[f"{model_lang}_acc"] = lang_accuracy.get(model_lang, None)
        # One row per model/ckpt with all 10 lang accs
        new_line = pd.DataFrame([row_data])
        new_line.to_csv('multiblimp/results/all_multiblimp_results.csv', mode='a', header=False, index=False)
        print(new_line)


# python multiblimp/scripts/lm_eval/eval_model.py \
# --model_name HPLT/hplt2c_eng_checkpoints \
# --revision main \
# --data_dir "multiblimp/hf_cache/eng/" \
# --src_dir "multiblimp" \
# --results_dir "multiblimp/results/hplt_multiblimp_results/HPLT/hplt2c_eng_checkpoints_main-eng" \
# --hf_token ./token