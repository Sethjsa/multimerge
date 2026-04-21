from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import os
import torch
import argparse

# Set the MKL threading layer to GNU to avoid conflicts
os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true', help='Use local models instead of HuggingFace models')
parser.add_argument('--test_langs', nargs="+", help='List of languages to evaluate', default=[])
parser.add_argument('--model_langs', nargs="+", help='List of languages to evaluate', default=[])
parser.add_argument('--mixed_models', action='store_true', help='Use mixed models instead of merged models')
args = parser.parse_args()

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

# checkpoints = ["checkpoint_0047684"]

api = HfApi()
all_models = api.list_models(author="sethjsa")
total_downloads = 0
all_bgts = []

for model in tqdm(all_models):
    model_info = api.model_info(model.modelId, expand=["downloadsAllTime"])  
    model_name = str(model.modelId)
    all_bgts.append(model_name)

language_map = {
    'eng': 'eng',
    'nld': 'nld',
    'spa': 'spa',
    'fra': 'fra',
    'rus': 'rus',
    'ita': 'ita',
    'por': 'por',
    'tur': 'tur',
    'ara': 'arb',
    'deu': 'deu',
    # 'zhos': 'zho',
}

if args.test_langs:
    language_map = {lang: lang for lang in args.test_langs}
else:
    language_map = language_map


hf_models = ["HPLT/hplt2c_eng_checkpoints",
             "HPLT/hplt2c_nld_checkpoints",
             "HPLT/hplt2c_spa_checkpoints",
             "HPLT/hplt2c_fra_checkpoints",
             "HPLT/hplt2c_rus_checkpoints",
             "HPLT/hplt2c_tur_checkpoints",
             "HPLT/hplt2c_ita_checkpoints",
             "HPLT/hplt2c_por_checkpoints",
             "HPLT/hplt2c_ara_checkpoints",
             "HPLT/hplt2c_deu_checkpoints"]

if args.mixed_models:
    local_models = ["/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/mixed-10-checkpoints"]
else:
    local_models = ["/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints"]

# local_models = ["/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints",
#                 "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/mixed-10-checkpoints",
#                 "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-2-checkpoints",
#                 "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-4-checkpoints",
#                 "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-6-checkpoints",
#                 "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-8-checkpoints"]

if args.model_langs:
    print("Model languages: ", args.model_langs)
    hf_models = [f"HPLT/hplt2c_{lang}_checkpoints" for lang in args.model_langs]
else:
    hf_models = hf_models

# Select model list and results CSV based on flag
if args.local:
    models = local_models
    results_csv = 'multiblimp/results/all_multiblimp_results_local.csv'
else:
    models = hf_models
    results_csv = 'multiblimp/results/all_multiblimp_results.csv'

print("Models: ", models)
print("Test languages: ", language_map.keys())

os.makedirs(f"multiblimp/results", exist_ok=True)


# create results dataframe
results = pd.DataFrame(columns=['model', 'checkpoint'] + list(language_map.keys()))
results.to_csv(results_csv, mode='w', index=False)

for model in models:
    for checkpoint in checkpoints:
        if model.count("/") > 1:
            m_str = model.split("/")[-2:]
            m_str = "/".join(m_str) + "/" + checkpoint
            c_str = checkpoint
            name = m_str + "_" + c_str
            results_str = m_str
        else:
            m_str = model
            name = m_str
            c_str = checkpoint
            results_str = m_str + "/" + c_str

        lang_accuracy = {}

        for model_lang, test_lang in language_map.items():
            l1_iso_str = model_lang
            try:
                os.makedirs(f"multiblimp/results/{results_str}", exist_ok=True)
                subprocess.run([
                    "python", "multiblimp/scripts/lm_eval/eval_model.py",
                    "--model_name", m_str,
                    "--revision", c_str,
                    "--data_dir", f"multiblimp/hf_cache/{test_lang}/",
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
                continue

            results_path = f"multiblimp/results/{results_str}/{test_lang}.tsv"
            df = pd.read_csv(results_path, sep='\t')
            total_samples = len(df)
            correct_predictions = len(df[df['delta'] > 0])
            test_lang_accuracy = correct_predictions / total_samples
            lang_accuracy[model_lang] = test_lang_accuracy

        per_ckpt_results = pd.DataFrame([{
            "model": m_str,
            "checkpoint": c_str,
            **lang_accuracy
        }])
        os.makedirs(f"multiblimp/results/{results_str}", exist_ok=True)
        per_ckpt_results_path = f"multiblimp/results/{results_str}/all.csv"  # E.g., .../eng_checkpoint_0001000.csv
        per_ckpt_results.to_csv(per_ckpt_results_path, index=False)

        row_data = {'model': m_str, 'checkpoint': c_str}
        for model_lang in language_map.keys():
            row_data[f"{model_lang}_acc"] = lang_accuracy.get(model_lang, None)
        new_line = pd.DataFrame([row_data])
        new_line.to_csv(results_csv, mode='a', header=False, index=False)
        print(new_line)