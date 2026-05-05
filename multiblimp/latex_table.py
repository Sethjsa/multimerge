"""
generate_multiblimp_latex_table.py

Reads multiblimp/summary_mean_cv.csv and generates a LaTeX results table
for MultiBLIMP, with columns for each language (including zho, even if empty).
Models and formatting (rows/ordering) mirror those of main_results_table.py.
Best per-language value is bolded. Missing models render as --.
"""

import csv
import math
from pathlib import Path

MULTIBLIMP_CSV = Path("multiblimp/summary_mean_cv.csv")
OUT_TEX = Path("multiblimp/latex_table.tex")

# Ordered languages as in the CSV
LANGS = ["eng", "deu", "fra", "ita", "nld", "rus", "spa", "tur", "ara", "zho"]

LANG_DISPLAY = {
    "eng": "eng",
    "deu": "deu",
    "ara": "ara",
    "fra": "fra",
    "spa": "spa",
    "zho": "zho",
    "rus": "rus",
    "tur": "tur",
    "nld": "nld",
    "ita": "ita",
}

# Models and their display names, following main_results_table.py (order/sections)
TABLE_STRUCTURE = [
    (r"\textit{Multilingual baselines}", [
        ("tiny-aya-base",    "Tiny-Aya-Base"),
        ("EuroLLM-1.7B",     "EuroLLM-1.7B"),
        ("gemma-2-2b",       "Gemma-2-2B"),
    ]),
    (r"\textit{Monolingual experts}", [
        ("hplt-mono-avg",   r"HPLT\textsubscript{1} (per-language)"),
    ]),
    (r"\textit{Mixed pre-training }", [
        ("mixed-10-checkpoints",      "Mixed\textsubscript{10}"),
    ]),
    (r"\textit{Merged\textsubscript{10} experts}", [
        ("merged-10-checkpoints",   "Linear"),
        ("ties-10-checkpoints",     "TIES"),
        ("dareties-10-checkpoints", "DARE-TIES"),
    ]),
    # (r"\textit{Other baselines}", [
    #     ("smollm2-1.7b",         "SmallLM2-1.7B"),
    #     ("tiny-aya-global",      "Tiny-Aya-Global"),
    #     ("tiny-aya-fire",        "Tiny-Aya-Fire"),
    #     ("tiny-aya-water",       "Tiny-Aya-Water"),
    #     ("tiny-aya-earth",       "Tiny-Aya-Earth"),
    #     ("task-aya-checkpoints", "Task-Aya"),
    #     ("linear-aya-checkpoints","Linear-Aya"),
    #     ("widen-10-checkpoints", "Widen"),
    #     ("merged-2-checkpoints", "Merged-2"),
    #     ("merged-3-checkpoints", "Merged-3"),
    #     ("merged-4-checkpoints", "Merged-4"),
    #     ("merged-5-checkpoints", "Merged-5"),
    #     ("merged-6-checkpoints", "Merged-6"),
    #     ("merged-7-checkpoints", "Merged-7"),
    #     ("merged-8-checkpoints", "Merged-8"),
    #     ("merged-9-checkpoints", "Merged-9"),
    # ]),
]

NA_S = r"{\na}"   # for siunitx S columns
NA_TEXT = r"--"

def load_multiblimp_data():
    """Load CSV as dict: {model: {lang_acc...}}"""
    data = {}
    with open(MULTIBLIMP_CSV, newline="") as f:
        for row in csv.DictReader(f):
            # normalize model name (CSV cols: model,mean_acc,cv,eng_acc,...)
            model = row["model"].strip()
            # Lowercase to match main table if needed (EuroLLM-1.7B → eurollm-1.7b)
            if model.lower() == "eurollm-1.7b":
                model = "EuroLLM-1.7B"
            if model.lower() == "gemma-2-2b":
                model = "gemma-2-2b"
            if model.lower() == "tiny-aya-base":
                model = "tiny-aya-base"
            # Leave hplt-mono-avg, etc. as is

            # Populate only language columns (acc for each)
            d = {}
            for lang in LANGS:
                key = f"{lang}_acc"
                val = row.get(key, "")
                d[lang] = float(val) if val not in ("", None) else None
            data[model] = d
    return data

def find_best_per_language(data):
    """For each language, find the best value across listed models"""
    best = {lang: -math.inf for lang in LANGS}
    for _, section in TABLE_STRUCTURE:
        for model_key, _ in section:
            row = data.get(model_key)
            if not row:
                continue
            for lang in LANGS:
                v = row.get(lang)
                if v is not None and v > best[lang]:
                    best[lang] = v
    return best

def fmt_acc(v, is_best):
    if v is None:
        return NA_S
    s = f"{v*100:.1f}"
    if is_best:
        return rf"\multicolumn{{1}}{{r}}{{\textbf{{{s}}}}}"
    return s

def build_table(data):
    best = find_best_per_language(data)
    n_cols = 1 + len(LANGS)

    # siunitx S columns for each language
    lang_col_spec = r"S[table-format=3.1,table-number-alignment=right]"
    col_spec = "l " + " ".join(lang_col_spec for _ in LANGS)

    header = rf"\textbf{{Model}}" + "".join(
        rf" & \textbf{{{LANG_DISPLAY[lang]}}}" for lang in LANGS
    ) + r" \\"

    cmidrules = " ".join(
        rf"\cmidrule(lr){{{i+2}-{i+2}}}" for i in range(len(LANGS))
    )

    lines = [
        r"% Required: \usepackage{booktabs, siunitx, xcolor}",
        r"% \newcommand{\best}[1]{\textbf{#1}}",
        r"% \newcommand{\na}{--}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\label{tab:multiblimp}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        cmidrules,
        r"\midrule"
    ]

    for s_idx, (section_title, section_rows) in enumerate(TABLE_STRUCTURE):
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{{section_title}}} \\[2pt]")
        for model_key, display_name in section_rows:
            row = data.get(model_key)
            cells = []
            for lang in LANGS:
                v = row.get(lang) if row else None
                is_best = v is not None and abs(v - best[lang]) < 1e-4
                cells.append(fmt_acc(v, is_best))
            lines.append(f"{display_name} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule" if s_idx == len(TABLE_STRUCTURE) - 1 else r"\midrule")

    lines += [
        r"\end{tabular}",
        r"\caption{MultiBLIMP accuracy ($\uparrow$) for each model and language. " +
        r"Best per-language entry in bold. {\na} = not available.}",
        r"\end{table*}",
    ]
    return "\n".join(lines)

def main():
    data = load_multiblimp_data()
    # Normalize lookup by lowercasing keys, if necessary
    for k in list(data.keys()):
        if k != k.strip():
            data[k.strip()] = data.pop(k)
    missing = [
        display
        for _, rows in TABLE_STRUCTURE
        for key, display in rows
        if key not in data
    ]
    if missing:
        print(f"  [warn] Models missing in CSV: {missing} → will render as --")

    table = build_table(data)
    OUT_TEX.write_text(table)
    print(f"\nWrote → {OUT_TEX}\n")
    print(table)

if __name__ == "__main__":
    main()