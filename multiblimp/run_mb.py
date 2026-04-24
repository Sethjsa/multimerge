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
os.environ["HF_TOKEN"] = open("./token", "r").read().strip()
os.environ["HF_HOME"] = "/fnwi_fs/ivi/irlab/personal/saycock/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true', help='Use local models instead of HuggingFace models')
parser.add_argument('--test_langs', nargs="+", help='List of languages to evaluate', default=[])
parser.add_argument('--model_langs', nargs="+", help='List of languages to evaluate', default=[])
parser.add_argument('--mixed_models', action='store_true', help='Use mixed models instead of merged models')
parser.add_argument('--main_only', action='store_true', help='Only use the main checkpoint')
parser.add_argument('--baselines_only', action='store_true', help='Only use the baseline models')
parser.add_argument('--spectrum', action='store_true', help='Use spectrum of checkpoints instead of just the final checkpoint')
parser.add_argument('--matrix', action='store_true', help='Use matrix of models instead of just the final checkpoint')
parser.add_argument('--stderr_only', action='store_true', help='Only calculate the stderr of the model')
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

if args.spectrum:
    checkpoints = ["checkpoint_0047684"]
else:
    checkpoints = checkpoints

if args.main_only:
    checkpoints = ["main"]
else:
    checkpoints = checkpoints

if args.matrix:
    checkpoints = ["checkpoint_0047684"]
    

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
    'zhos': 'zho',
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
             "HPLT/hplt2c_ara_checkpoints",
             "HPLT/hplt2c_deu_checkpoints",
             "HPLT/hplt2c_zhos_checkpoints"]

if args.mixed_models:
    local_models = ["/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/mixed-10-checkpoints"]
else:
    local_models = ["/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints"]


if args.spectrum:
    local_models = ["/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-2-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-3-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-5-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-7-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-9-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-4-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-6-checkpoints",
                "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-8-checkpoints"]

if args.matrix:
    with open("multiblimp/pairs.txt", "r") as f:
        pairs = [line.strip().replace(" ", "") for line in f if line.strip()]
    # For each pair like "engnld", generate the corresponding merged checkpoints model directory name
    local_models = [f"/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-{pair}-checkpoints" for pair in pairs]

if args.baselines_only:
    hf_models = ["meta-llama/Llama-3.2-1B",
                "CohereLabs/tiny-aya-base",
                 "google/gemma-2-2b"]
    checkpoints = ["main"]

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
l1, l2 = None, None
for model in models:
    print("Model: ", model)
    for checkpoint in checkpoints:
        if model.count("/") > 1:
            m_str = model.split("/")[-2:]
            m_str = "/".join(m_str) + "/" + checkpoint
            model_path = model + "/" + checkpoint
            c_str = checkpoint
            name = m_str + "_" + c_str
            results_str = m_str
            if args.matrix:
                l1 = model.split("/")[-1].split("-")[1][:3]
                l2 = model.split("/")[-1].split("-")[1][3:]
                print(l1, l2)
        else:
            m_str = model
            model_path = model
            name = m_str
            c_str = checkpoint
            results_str = m_str + "/" + c_str

        lang_accuracy = {}
        lang_se = {}

        for model_lang, test_lang in language_map.items():
            l1_iso_str = model_lang

            # only test the languages that are in the matrix
            if args.matrix:
                if model_lang not in [l1, l2]:
                    print(test_lang, l1, l2)
                    continue
            
            results_path = f"multiblimp/results/{results_str}/{test_lang}.tsv"
            
            # if os.path.exists(results_path):
            #     print(f"Results path {results_path} already exists, skipping")
            #     continue
            try:
                if not args.stderr_only:
                    os.makedirs(f"multiblimp/results/{results_str}", exist_ok=True)
                    subprocess.run([
                        "/home/saycock/miniconda3/envs/gvllm/bin/python", "multiblimp/scripts/lm_eval/eval_model.py",
                        "--model_name", model_path,
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

            
            if not os.path.exists(results_path):
                print(f"Results path {results_path} does not exist")
                continue
            df = pd.read_csv(results_path, sep='\t')
            total_samples = len(df)
            correct_predictions = len(df[df['delta'] > 0])
            test_lang_accuracy = correct_predictions / total_samples
            # TODO: rerun all with this, with
            test_lang_se = (test_lang_accuracy * (1 - test_lang_accuracy) / total_samples) ** 0.5
            lang_accuracy[model_lang] = test_lang_accuracy
            lang_se[model_lang] = test_lang_se

        per_ckpt_results = pd.DataFrame([{
            "model": m_str,
            "checkpoint": c_str,
            # **lang_accuracy
            **{f"{k}_acc": v for k, v in lang_accuracy.items()},
            **{f"{k}_se": v for k, v in lang_se.items()}
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