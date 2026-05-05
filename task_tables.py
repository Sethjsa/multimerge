import csv

# Relevant language codes, matching order in main_results_table.py
LANGS = ["eng", "deu", "fra", "ita", "nld", "rus", "spa", "tur", "ara"]

# Tasks and corresponding row prefixes as appear in results_per_lang.csv
# These are the main tasks in main_results_table.py
TASKS = [
    ("belebele",   "Belebele"),
    ("flores200",  "FLORES"),
    ("hellaswag",  "Hellaswag"),
    ("xcsqa",      "XCSQA"),
    ("multiblimp", "MultiBlimp"),
]

# The model groups and CSV key hints used in main_results_table.py
MODEL_TABLES = [
    (r"\textit{Multilingual baselines}", [
        ("tiny-aya-base",  "Tiny-Aya-Base"),
        ("EuroLLM-1.7B",   "EuroLLM-1.7B"),
        ("gemma-2-2b",     "Gemma-2-2B"),
    ]),
    (r"\textit{Monolingual experts (oracle upper bound)}", [
        ("mono_experts",   r"HPLT Mono (avg)"),
    ]),
    (r"\textit{Mixed pre-training (10 langs)}", [
        ("mixed-10-checkpoints", "Mixed"),
    ]),
    (r"\textit{Merged (10 monolingual models)}", [
        ("merged-10-checkpoints",  "Linear"),
        ("ties-10-checkpoints",    "TIES"),
        ("dareties-10-checkpoints","DARE-TIES"),
    ]),
]

# Model identifier to full name mapping
MODEL_DISPLAY = {
    "tiny-aya-base":  "Tiny-Aya-Base",
    "EuroLLM-1.7B":   "EuroLLM-1.7B",
    "gemma-2-2b":     "Gemma-2-2B",
    "mono_experts":   r"HPLT Mono (avg)",
    "mixed-10-checkpoints": "Mixed",
    "merged-10-checkpoints":  "Linear",
    "ties-10-checkpoints":    "TIES",
    "dareties-10-checkpoints":"DARE-TIES",
}

# Map model key to its per-language mono models for the mono_experts row
# e.g. mono_experts: for each language, lookup hplt2c_{lang}_checkpoints
MONO_EXPERT_CSV_PREFIX = {
    lang: f"hplt2c_{lang}_checkpoints" for lang in LANGS
}

# The step number (checkpoint) to pull from, matching main_results_table.py
TARGET_CKPT = "47684"

# Each row in results_per_lang.csv looks something like:
# <model/exp>, <step>, ..., <other info>, <belebele_eng>, <belebele_deu>, ... (12-20), etc.

# Column indices of the scores for each task and language in results_per_lang.csv
SCORE_COL_IDX = {
    "belebele":   {lang: 12+LANGS.index(lang) for lang in LANGS},   # Example indices, may vary for your columns
    "flores200":  {lang: 21+LANGS.index(lang) for lang in LANGS},
    "hellaswag":  {lang: 30+LANGS.index(lang) for lang in LANGS},
    "xcsqa":      {lang: 39+LANGS.index(lang) for lang in LANGS},
    "multiblimp": {lang: 48+LANGS.index(lang) for lang in LANGS},
}
# ^^^ Adjust column indices to match your results_per_lang.csv! (order: task1_eng...task1_ara, task2_eng... etc.)

def read_results(filename):
    # Read all rows from csv
    with open(filename, encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]

def find_best_ckpt_row(rows, model_prefix):
    # Find row for this model/checkpoint, where row[0] startswith model_prefix and row[1] == TARGET_CKPT
    for row in rows:
        if row[0].startswith(model_prefix) and row[1] == TARGET_CKPT:
            return row
    return None

def collect_scores(rows, task_key, model_key):
    """
    Returns a dict of lang -> score, or None if not found
    """
    col_idx = SCORE_COL_IDX[task_key]
    # Handle mono_experts (pull each mono model separately for each lang)
    if model_key == "mono_experts":
        lang_scores = {}
        for lang in LANGS:
            row = find_best_ckpt_row(rows, MONO_EXPERT_CSV_PREFIX[lang])
            if row:
                idx = col_idx[lang]
                x = row[idx].strip()
                lang_scores[lang] = x if x else ""
            else:
                lang_scores[lang] = ""
        return lang_scores
    else:
        # For other rows, get the model row and all lang scores
        row = find_best_ckpt_row(rows, model_key)
        if row:
            return {lang: (row[col_idx[lang]].strip() if row[col_idx[lang]].strip() else "") for lang in LANGS}
        else:
            return {lang: "" for lang in LANGS}

def make_latex_table(task_name, lang_scores_by_model):
    """lang_scores_by_model: list of (model_display, lang->score)"""
    latex = []
    latex.append("\\begin{table}[h!]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{l" + "c" * len(LANGS) + "}")
    latex.append("\\toprule")
    header_row = ["Model"] + [l.upper() for l in LANGS]
    latex.append(" & ".join(header_row) + " \\\\")
    latex.append("\\midrule")
    for model_display, lang_scores in lang_scores_by_model:
        row = [model_display] + [lang_scores.get(lang, "") for lang in LANGS]
        latex.append(" & ".join(row) + " \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(f"\\caption{{Per-language scores for {task_name}}}")
    latex.append("\\end{table}")
    return "\n".join(latex)

def main():
    fname = "results_per_lang.csv"
    rows = read_results(fname)

    for task_key, task_name in TASKS:
        lang_scores_by_model = []
        for section_title, models in MODEL_TABLES:
            for model_key, model_disp in models:
                lang_scores = collect_scores(rows, task_key, model_key)
                lang_scores_by_model.append((model_disp, lang_scores))
        print(make_latex_table(task_name, lang_scores_by_model))
        print()

if __name__ == "__main__":
    main()