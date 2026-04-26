"""
export_results_csv.py

Exports per-checkpoint evaluation results to a CSV with mean and CV
(coefficient of variation = std/mean × 100) for each task.

Two model regimes:
  - Multilingual : mean/CV computed across ALL available languages.
  - Monolingual  : each model contributes only its own-language score;
                   the aggregate row averages those expert scores and
                   reports CV across the mono-expert values.

MultiBLIMP results are loaded from a separate CSV (fixed ckpt=47684)
and merged into the matching rows by model name.

Output columns:
  model, ckpt,
  flores200_chrf, flores200_cv,
  xcsqa_acc,      xcsqa_cv,
  hellaswag_acc,  hellaswag_cv,
  belebele_acc,   belebele_cv,
  multiblimp_acc, multiblimp_cv
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.langutils import GENERATIVE_TASKS
from process_evals import load_experiment

# ── Config ────────────────────────────────────────────────────────────────────

MULTILINGUAL_EXPERIMENTS = [
    "mixed-10-checkpoints",
    "merged-10-checkpoints",
    "merged-2-checkpoints",
    "merged-3-checkpoints",
    "merged-4-checkpoints",
    "merged-5-checkpoints",
    "merged-6-checkpoints",
    "merged-7-checkpoints",
    "merged-8-checkpoints",
    "merged-9-checkpoints",
    "gemma-2-2b",
    "tiny-aya-base",
    "EuroLLM-1.7B",
    "tiny-aya-fire",
    "tiny-aya-water",
    "tiny-aya-earth",
    "task-aya-checkpoints",
    "linear-aya-checkpoints",
    "widen-10-checkpoints",
    "dareties-10-checkpoints",
    "ties-10-checkpoints",
]

_HPLT_DIRS = [
    "hplt2c_eng_checkpoints",
    "hplt2c_nld_checkpoints",
    "hplt2c_spa_checkpoints",
    "hplt2c_fra_checkpoints",
    "hplt2c_rus_checkpoints",
    "hplt2c_tur_checkpoints",
    "hplt2c_ita_checkpoints",
    "hplt2c_ara_checkpoints",
    "hplt2c_deu_checkpoints",
    "hplt2c_zhos_checkpoints",
]

def _parse_hplt_lang(exp_name: str) -> str:
    return exp_name.split("_")[1]

MONOLINGUAL_EXPERIMENTS: dict[str, str] = {
    d: _parse_hplt_lang(d) for d in _HPLT_DIRS
}

LANGUAGE_MAP = {
    "eng":  "eng",
    "nld":  "nld",
    "spa":  "spa",
    "fra":  "fra",
    "rus":  "rus",
    "ita":  "ita",
    "tur":  "tur",
    "ara":  "arb",
    "deu":  "deu",
    "zhos": "zho",
}

EXPORT_TASKS = ["flores200", "xcsqa", "hellaswag", "belebele"]

MULTIBLIMP_CSV  = Path("multiblimp/summary_mean_cv.csv")
MULTIBLIMP_CKPT = 47684

# Model names in the multiblimp CSV that differ from our output row names
MULTIBLIMP_MODEL_MAP = {
    "hplt-mono-avg": "mono_experts",
}

OUT_CSV = Path("results_summary.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────

def col_names(task: str) -> tuple[str, str]:
    metric = "chrf" if task in GENERATIVE_TASKS else "acc"
    return f"{task}_{metric}", f"{task}_cv"


def mean_and_cv(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    m = float(np.mean(values))
    cv = float(np.std(values, ddof=1) / m * 100) if m != 0 else math.nan
    return m, cv


def collect_lang_values(lang_data: dict, cp: int) -> list[float]:
    return [
        cp_data[cp]["metric"]
        for cp_data in lang_data.values()
        if cp in cp_data
    ]


def fmt(v: float) -> str | float:
    return "" if math.isnan(v) else round(v, 4)


# ── MultiBLIMP loader ─────────────────────────────────────────────────────────

def load_multiblimp() -> dict[str, dict]:
    """
    Returns {model_name: {"multiblimp_acc": float, "multiblimp_cv": float}}
    keyed by the model name used in our output (after applying MULTIBLIMP_MODEL_MAP).
    """
    if not MULTIBLIMP_CSV.exists():
        print(f"  [warn] MultiBLIMP CSV not found: {MULTIBLIMP_CSV}")
        return {}

    result = {}
    with open(MULTIBLIMP_CSV, newline="") as f:
        for row in csv.DictReader(f):
            raw_model = row["model"].strip()
            model = MULTIBLIMP_MODEL_MAP.get(raw_model, raw_model)
            try:
                result[model] = {
                    "multiblimp_acc": round(float(row["mean_acc"]), 4),
                    "multiblimp_cv":  round(float(row["cv"]) * 100, 4),  # stored as ratio → %
                }
            except (ValueError, KeyError):
                print(f"  [warn] Could not parse MultiBLIMP row for model: {raw_model}")
    print(f"  Loaded MultiBLIMP for {len(result)} models: {list(result)}")
    return result


# ── Row builders ──────────────────────────────────────────────────────────────

def multilingual_rows(exp_name: str) -> list[dict]:
    print(f"[multilingual] Loading: {exp_name}")
    checkpoints, data = load_experiment(exp_name)
    print(f"  Checkpoints: {checkpoints}")

    rows = []
    for cp in checkpoints:
        row = {"model": exp_name, "ckpt": cp}
        for task in EXPORT_TASKS:
            lang_data = data.get(task, {})
            values = collect_lang_values(lang_data, cp)
            mean_val, cv = mean_and_cv(values)
            m_col, cv_col = col_names(task)
            row[m_col]  = fmt(mean_val)
            row[cv_col] = fmt(cv)
        rows.append(row)
    return rows


def monolingual_aggregate_rows() -> list[dict]:
    if not MONOLINGUAL_EXPERIMENTS:
        return []

    mono_scores: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_checkpoints: set[int] = set()

    for exp_name, own_lang in MONOLINGUAL_EXPERIMENTS.items():
        print(f"[monolingual]  Loading: {exp_name}  (lang={own_lang})")
        checkpoints, data = load_experiment(exp_name)
        all_checkpoints.update(checkpoints)

        for cp in checkpoints:
            for task in EXPORT_TASKS:
                lang_data = data.get(task, {})
                task_lang = LANGUAGE_MAP.get(own_lang, own_lang)
                own_lang_cp = lang_data.get(task_lang, {})
                if cp in own_lang_cp:
                    mono_scores[cp][task].append(own_lang_cp[cp]["metric"])

    rows = []
    for cp in sorted(all_checkpoints):
        row = {"model": "mono_experts", "ckpt": cp}
        for task in EXPORT_TASKS:
            values = mono_scores[cp].get(task, [])
            mean_val, cv = mean_and_cv(values)
            m_col, cv_col = col_names(task)
            row[m_col]  = fmt(mean_val)
            row[cv_col] = fmt(cv)
        rows.append(row)
    return rows


# ── Merge MultiBLIMP ──────────────────────────────────────────────────────────

def merge_multiblimp(rows: list[dict], mb: dict[str, dict]) -> list[dict]:
    """
    Attach multiblimp_acc / multiblimp_cv to any row whose (model, ckpt)
    matches a MultiBLIMP entry at MULTIBLIMP_CKPT.
    """
    for row in rows:
        if row["ckpt"] == MULTIBLIMP_CKPT and row["model"] in mb:
            row.update(mb[row["model"]])
        else:
            row.setdefault("multiblimp_acc", "")
            row.setdefault("multiblimp_cv",  "")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []
    for exp_name in MULTILINGUAL_EXPERIMENTS:
        rows.extend(multilingual_rows(exp_name))
    rows.extend(monolingual_aggregate_rows())

    print("\n[multiblimp] Loading MultiBLIMP CSV …")
    mb = load_multiblimp()
    rows = merge_multiblimp(rows, mb)

    if not rows:
        print("No data found.")
        return

    fieldnames = ["model", "ckpt"]
    for task in EXPORT_TASKS:
        fieldnames += list(col_names(task))
    fieldnames += ["multiblimp_acc", "multiblimp_cv"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()


# """
# export_results_csv.py

# Exports per-checkpoint evaluation results to a CSV with mean and CV
# (coefficient of variation = std/mean × 100) for each task.

# Two model regimes:
#   - Multilingual : mean/CV computed across ALL available languages.
#   - Monolingual  : each model contributes only its own-language score;
#                    the aggregate row averages those expert scores and
#                    reports CV across the mono-expert values.

# Output columns:
#   model, ckpt,
#   flores200_chrf, flores200_cv,
#   xcsqa_acc,      xcsqa_cv,
#   hellaswag_acc,  hellaswag_cv,
#   belebele_acc,   belebele_cv
# """

# import csv
# import math
# from collections import defaultdict
# from pathlib import Path

# import numpy as np

# from utils.langutils import GENERATIVE_TASKS
# from process_evals import load_experiment

# # ── Config ────────────────────────────────────────────────────────────────────

# MULTILINGUAL_EXPERIMENTS = [
#     "mixed-10-checkpoints",
#     "merged-10-checkpoints",
#     "merged-2-checkpoints",
#     "merged-3-checkpoints",
#     "merged-4-checkpoints",
#     "merged-5-checkpoints",
#     "merged-6-checkpoints",
#     "merged-7-checkpoints",
#     "merged-8-checkpoints",
#     "merged-9-checkpoints",
#     "gemma-2-2b",
#     "tiny-aya-base",
# ]

# # hplt2c_{lang}_checkpoints → lang extracted automatically
# _HPLT_DIRS = [
#     "hplt2c_eng_checkpoints",
#     "hplt2c_nld_checkpoints",
#     "hplt2c_spa_checkpoints",
#     "hplt2c_fra_checkpoints",
#     "hplt2c_rus_checkpoints",
#     "hplt2c_tur_checkpoints",
#     "hplt2c_ita_checkpoints",
#     "hplt2c_ara_checkpoints",
#     "hplt2c_deu_checkpoints",
#     "hplt2c_zhos_checkpoints",
# ]

# def _parse_hplt_lang(exp_name: str) -> str:
#     """Extract lang code from 'hplt2c_{lang}_checkpoints'."""
#     return exp_name.split("_")[1]

# MONOLINGUAL_EXPERIMENTS: dict[str, str] = {
#     d: _parse_hplt_lang(d) for d in _HPLT_DIRS
# }

# # Maps the lang code parsed from the hplt dir name to the key used in task data.
# # Identity for most languages; handles mismatches like ara→arb, zhos→zho.
# LANGUAGE_MAP = {
#     "eng":  "eng",
#     "nld":  "nld",
#     "spa":  "spa",
#     "fra":  "fra",
#     "rus":  "rus",
#     "ita":  "ita",
#     "tur":  "tur",
#     "ara":  "arb",
#     "deu":  "deu",
#     "zhos": "zho",
# }

# EXPORT_TASKS = ["flores200", "xcsqa", "hellaswag", "belebele"]

# OUT_CSV = Path("results_summary.csv")

# # ── Helpers ───────────────────────────────────────────────────────────────────

# def col_names(task: str) -> tuple[str, str]:
#     metric = "chrf" if task in GENERATIVE_TASKS else "acc"
#     return f"{task}_{metric}", f"{task}_cv"


# def mean_and_cv(values: list[float]) -> tuple[float, float]:
#     """(mean, CV%) from a list of per-language metric values."""
#     if not values:
#         return math.nan, math.nan
#     m = float(np.mean(values))
#     cv = float(np.std(values, ddof=1) / m * 100) if m != 0 else math.nan
#     return m, cv


# def collect_lang_values(lang_data: dict, cp: int) -> list[float]:
#     """All per-language metric values for a task at a given checkpoint."""
#     return [
#         cp_data[cp]["metric"]
#         for cp_data in lang_data.values()
#         if cp in cp_data
#     ]


# def fmt(v: float) -> str | float:
#     return "" if math.isnan(v) else round(v, 4)


# # ── Row builders ──────────────────────────────────────────────────────────────

# def multilingual_rows(exp_name: str) -> list[dict]:
#     print(f"[multilingual] Loading: {exp_name}")
#     checkpoints, data = load_experiment(exp_name)
#     print(f"  Checkpoints: {checkpoints}")

#     rows = []
#     for cp in checkpoints:
#         row = {"model": exp_name, "ckpt": cp}
#         for task in EXPORT_TASKS:
#             lang_data = data.get(task, {})
#             values = collect_lang_values(lang_data, cp)
#             mean_val, cv = mean_and_cv(values)
#             m_col, cv_col = col_names(task)
#             row[m_col]  = fmt(mean_val)
#             row[cv_col] = fmt(cv)
#         rows.append(row)
#     return rows


# def monolingual_aggregate_rows() -> list[dict]:
#     """
#     Load every mono model, extract its own-language score per task/checkpoint,
#     then aggregate across all mono models into a single 'mono_experts' row
#     per checkpoint.
#     """
#     if not MONOLINGUAL_EXPERIMENTS:
#         return []

#     # mono_scores[cp][task] = list of that task's own-lang metric across mono models
#     mono_scores: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

#     all_checkpoints: set[int] = set()

#     for exp_name, own_lang in MONOLINGUAL_EXPERIMENTS.items():
#         print(f"[monolingual]  Loading: {exp_name}  (lang={own_lang})")
#         checkpoints, data = load_experiment(exp_name)
#         all_checkpoints.update(checkpoints)

#         for cp in checkpoints:
#             for task in EXPORT_TASKS:
#                 lang_data = data.get(task, {})
#                 own_lang_cp = lang_data.get(own_lang, {})
#                 task_lang = LANGUAGE_MAP.get(own_lang, own_lang)
#                 own_lang_cp = lang_data.get(task_lang, {})
#                 if cp in own_lang_cp:
#                     mono_scores[cp][task].append(own_lang_cp[cp]["metric"])

#     rows = []
#     for cp in sorted(all_checkpoints):
#         row = {"model": "mono_experts", "ckpt": cp}
#         for task in EXPORT_TASKS:
#             values = mono_scores[cp].get(task, [])
#             mean_val, cv = mean_and_cv(values)
#             m_col, cv_col = col_names(task)
#             row[m_col]  = fmt(mean_val)
#             row[cv_col] = fmt(cv)
#         rows.append(row)
#     return rows


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     rows = []
#     for exp_name in MULTILINGUAL_EXPERIMENTS:
#         rows.extend(multilingual_rows(exp_name))
#     rows.extend(monolingual_aggregate_rows())

#     if not rows:
#         print("No data found.")
#         return

#     fieldnames = ["model", "ckpt"]
#     for task in EXPORT_TASKS:
#         fieldnames += list(col_names(task))

#     with open(OUT_CSV, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)

#     print(f"\nWrote {len(rows)} rows → {OUT_CSV}")


# if __name__ == "__main__":
#     main()