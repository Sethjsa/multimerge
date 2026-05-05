"""
export_results_lang_csv.py

Exports per-checkpoint evaluation results to a CSV with per-language scores
for each task. Each row = one model × one checkpoint.

Output columns:
  model, ckpt,
  flores200_eng, flores200_nld, ...,
  xcsqa_eng,     xcsqa_nld, ...,
  hellaswag_eng, hellaswag_nld, ...,
  belebele_eng,  belebele_nld, ...,
  multiblimp_eng, multiblimp_nld, ...
"""

import csv
import math
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

ALL_LANGS = ["eng", "nld", "spa", "fra", "rus", "ita", "tur", "arb", "deu", "zho"]

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

MULTIBLIMP_CSV = Path("multiblimp/multiblimp_all_results.csv")
OUT_CSV = Path("results_per_lang.csv")

MULTIBLIMP_MODEL_PREFIXES = {
    "models/mixed-10-checkpoints":    "mixed-10-checkpoints",
    "models/merged-10-checkpoints":   "merged-10-checkpoints",
    "models/merged-2-checkpoints":    "merged-2-checkpoints",
    "models/merged-3-checkpoints":    "merged-3-checkpoints",
    "models/merged-4-checkpoints":    "merged-4-checkpoints",
    "models/merged-5-checkpoints":    "merged-5-checkpoints",
    "models/merged-6-checkpoints":    "merged-6-checkpoints",
    "models/merged-7-checkpoints":    "merged-7-checkpoints",
    "models/merged-8-checkpoints":    "merged-8-checkpoints",
    "models/merged-9-checkpoints":    "merged-9-checkpoints",
    "models/task-aya-checkpoints":    "task-aya-checkpoints",
    "models/linear-aya-checkpoints":  "linear-aya-checkpoints",
    "models/widen-10-checkpoints":    "widen-10-checkpoints",
    "models/dareties-10-checkpoints": "dareties-10-checkpoints",
    "models/ties-10-checkpoints":     "ties-10-checkpoints",
    "CohereLabs/tiny-aya-base":       "tiny-aya-base",
    "CohereLabs/tiny-aya-earth":      "tiny-aya-earth",
    "CohereLabs/tiny-aya-fire":       "tiny-aya-fire",
    "CohereLabs/tiny-aya-water":      "tiny-aya-water",
    "google/gemma-2-2b":              "gemma-2-2b",
    "utter-project/EuroLLM-1.7B":     "EuroLLM-1.7B",
    "HPLT/hplt2c_eng_checkpoints":    "hplt2c_eng_checkpoints",
    "HPLT/hplt2c_nld_checkpoints":    "hplt2c_nld_checkpoints",
    "HPLT/hplt2c_spa_checkpoints":    "hplt2c_spa_checkpoints",
    "HPLT/hplt2c_fra_checkpoints":    "hplt2c_fra_checkpoints",
    "HPLT/hplt2c_rus_checkpoints":    "hplt2c_rus_checkpoints",
    "HPLT/hplt2c_tur_checkpoints":    "hplt2c_tur_checkpoints",
    "HPLT/hplt2c_ita_checkpoints":    "hplt2c_ita_checkpoints",
    "HPLT/hplt2c_ara_checkpoints":    "hplt2c_ara_checkpoints",
    "HPLT/hplt2c_deu_checkpoints":    "hplt2c_deu_checkpoints",
    "HPLT/hplt2c_zhos_checkpoints":   "hplt2c_zhos_checkpoints",
}

# multiblimp CSV uses ara; we use arb. por is omitted (not in ALL_LANGS).
MULTIBLIMP_LANG_MAP = {
    "eng": "eng", "nld": "nld", "spa": "spa", "fra": "fra",
    "rus": "rus", "ita": "ita", "tur": "tur", "ara": "arb",
    "deu": "deu", "zhos": "zho",
}

FLORES_LANG_MAP = {
    "eng": "eng", "nld": "nld", "spa": "spa", "fra": "fra",
    "rus": "rus", "ita": "ita", "tur": "tur", "ara": "arb",
    "deu": "deu", "zhos": "zho",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def col_name(task: str, lang: str) -> str:
    return f"{task}_{lang}"

def fmt(v: float) -> str | float:
    return "" if (v is None or math.isnan(v)) else round(v, 4)

def get_lang_score(lang_data: dict, lang: str, cp: int) -> float:
    cp_data = lang_data.get(lang, {}).get(cp)
    if cp_data is None:
        return math.nan
    return cp_data["metric"]

# ── MultiBLIMP helpers ────────────────────────────────────────────────────────

def _parse_ckpt(ckpt_str: str) -> int | None:
    """'checkpoint_0047684' → 47684, 'main' → -1, else None."""
    if ckpt_str == "main":
        return -1
    if ckpt_str.startswith("checkpoint_"):
        try:
            return int(ckpt_str.split("_", 1)[1])
        except ValueError:
            pass
    return None

def _strip_ckpt_from_path(raw_model: str) -> str:
    """Strip a trailing /checkpoint_XXXXXXX or /main segment.

    'models/mixed-10-checkpoints/checkpoint_0047684' → 'models/mixed-10-checkpoints'
    'HPLT/hplt2c_eng_checkpoints'                   → unchanged
    'CohereLabs/tiny-aya-base'                      → unchanged
    """
    parts = raw_model.rsplit("/", 1)
    if len(parts) == 2:
        suffix = parts[1]
        if suffix == "main" or (suffix.startswith("checkpoint_") and suffix[11:].isdigit()):
            return parts[0]
    return raw_model

# ── MultiBLIMP loader ─────────────────────────────────────────────────────────

def load_multiblimp_per_lang() -> dict[tuple[str, int], dict[str, float]]:
    """
    Returns {(model, ckpt_int): {lang: acc}} where lang is in ALL_LANGS.
    Reads multiblimp/multiblimp_all_results.csv which has columns:
      model, checkpoint, eng_acc, nld_acc, spa_acc, fra_acc, rus_acc,
      ita_acc, por_acc, tur_acc, ara_acc, deu_acc, eng_se, ...
    """
    if not MULTIBLIMP_CSV.exists():
        print(f"  [warn] MultiBLIMP CSV not found: {MULTIBLIMP_CSV}")
        return {}

    result: dict[tuple[str, int], dict[str, float]] = {}
    with open(MULTIBLIMP_CSV, newline="") as f:
        for row in csv.DictReader(f):
            raw_model = row["model"].strip()
            model = MULTIBLIMP_MODEL_PREFIXES.get(_strip_ckpt_from_path(raw_model))
            if model is None:
                continue

            ckpt = _parse_ckpt(row["checkpoint"].strip())
            if ckpt is None:
                continue

            lang_scores: dict[str, float] = {}
            for mb_lang, our_lang in MULTIBLIMP_LANG_MAP.items():
                val = row.get(f"{mb_lang}_acc", "").strip()
                if val:
                    try:
                        lang_scores[our_lang] = round(float(val), 4)
                    except ValueError:
                        pass
            if lang_scores:
                result[(model, ckpt)] = lang_scores

    print(f"  Loaded MultiBLIMP per-lang for {len(result)} (model, ckpt) pairs")
    return result

# ── Row builders ──────────────────────────────────────────────────────────────

def build_row(model: str, cp: int, data: dict) -> dict:
    row = {"model": model, "ckpt": cp}
    for task in EXPORT_TASKS:
        lang_data = data.get(task, {})
        for lang in ALL_LANGS:
            row[col_name(task, lang)] = fmt(get_lang_score(lang_data, lang, cp))
    for lang in ALL_LANGS:
        row[col_name("multiblimp", lang)] = ""
    return row

def multilingual_rows(exp_name: str) -> list[dict]:
    print(f"[multilingual] Loading: {exp_name}")
    checkpoints, data = load_experiment(exp_name)
    return [build_row(exp_name, cp, data) for cp in checkpoints]

def monolingual_rows() -> list[dict]:
    """One row per monolingual model × checkpoint (not aggregated)."""
    rows = []
    for exp_name, own_lang in MONOLINGUAL_EXPERIMENTS.items():
        print(f"[monolingual]  Loading: {exp_name}  (lang={own_lang})")
        checkpoints, data = load_experiment(exp_name)
        for cp in checkpoints:
            rows.append(build_row(exp_name, cp, data))
    return rows

# ── Merge MultiBLIMP ──────────────────────────────────────────────────────────

def merge_multiblimp(rows: list[dict], mb: dict[tuple[str, int], dict[str, float]]) -> list[dict]:
    for row in rows:
        scores = mb.get((row["model"], row["ckpt"]))
        if scores:
            for lang, val in scores.items():
                row[col_name("multiblimp", lang)] = val
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []
    for exp_name in MULTILINGUAL_EXPERIMENTS:
        rows.extend(multilingual_rows(exp_name))
    rows.extend(monolingual_rows())

    print("\n[multiblimp] Loading MultiBLIMP per-lang CSV …")
    mb = load_multiblimp_per_lang()
    rows = merge_multiblimp(rows, mb)

    if not rows:
        print("No data found.")
        return

    fieldnames = ["model", "ckpt"]
    for task in [*EXPORT_TASKS, "multiblimp"]:
        for lang in ALL_LANGS:
            fieldnames.append(col_name(task, lang))

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {OUT_CSV}")

if __name__ == "__main__":
    main()