"""
generate_latex_table.py

Reads results_summary.csv (rows at ckpt=47684) and generates a LaTeX
results table. Missing models render as --. Best per-task column is bolded.
"""

import csv
import math
from pathlib import Path

RESULTS_CSV  = Path("results_summary.csv")
OUT_TEX      = Path("results_table.tex")
TARGET_CKPT  = 47684

# ── Task config ───────────────────────────────────────────────────────────────

TASKS = ["multiblimp", "belebele", "hellaswag", "xcsqa", "flores200"]

TASK_DISPLAY = {
    "multiblimp": "MultiBlimp",
    "belebele":   "Belebele",
    "hellaswag":  "Hellaswag",
    "xcsqa":      "XCSQA",
    "flores200":  "FLORES",
}

TASK_COLS = {
    "multiblimp": ("multiblimp_acc", "multiblimp_cv"),
    "belebele":   ("belebele_acc",   "belebele_cv"),
    "hellaswag":  ("hellaswag_acc",  "hellaswag_cv"),
    "xcsqa":      ("xcsqa_acc",      "xcsqa_cv"),
    "flores200":  ("flores200_chrf", "flores200_cv"),
}

# Tasks whose mean is stored as 0–1 ratio → multiply by 100 for display
SCALE_100 = {"multiblimp", "belebele", "hellaswag", "xcsqa"}

# ── Model metadata ────────────────────────────────────────────────────────────

parameters_model_map = {
    "tiny-aya-base":          "3.35",
    "EuroLLM-1.7B":           "1.7",
    "gemma-2-2b":             "2.6",
    "smollm2-1.7b":           "1.7",
    "tiny-aya-fire":          "3.35",
    "tiny-aya-water":         "3.35",
    "tiny-aya-earth":         "3.35",
    "task-aya-checkpoints":   "2.15",
    "linear-aya-checkpoints": "3.35",
    "widen-10-checkpoints":   "2.15",
    "dareties-10-checkpoints":"2.15",
    "ties-10-checkpoints":    "2.15",
    "mixed-10-checkpoints":   "2.15",
    "merged-10-checkpoints":  "2.15",
    "merged-2-checkpoints":   "2.15",
    "merged-3-checkpoints":   "2.15",
    "merged-4-checkpoints":   "2.15",
    "merged-5-checkpoints":   "2.15",
    "merged-6-checkpoints":   "2.15",
    "merged-7-checkpoints":   "2.15",
    "merged-8-checkpoints":   "2.15",
    "mono_experts":           "2.15",
}

tokens_seen = {
    "mixed-10-checkpoints":   "100B",
    "merged-10-checkpoints":  "1T",
    "EuroLLM-1.7B":           "4T",
    "gemma-2-2b":             "2T",
    "tiny-aya-base":          "6T",
    "mono_experts":           "100B",
    "ties-10-checkpoints":    "1T",
    "dareties-10-checkpoints":"1T",
}

# ── Table row structure ───────────────────────────────────────────────────────

TABLE_STRUCTURE = [
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

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> dict[str, dict]:
    data = {}
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ckpt = int(float(row["ckpt"]))
            except ValueError:
                continue
            if ckpt == TARGET_CKPT:
                data[row["model"]] = row
    return data


def get_val(row: dict, col: str, scale: bool) -> float | None:
    raw = row.get(col, "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
        return v * 100 if scale else v
    except ValueError:
        return None

# ── Best-per-column for bolding ───────────────────────────────────────────────

def find_best(data: dict) -> dict[str, float]:
    best = {t: -math.inf for t in TASKS}
    for _, rows in TABLE_STRUCTURE:
        for model_key, _ in rows:
            if not model_key or model_key not in data:
                continue
            for task in TASKS:
                mean_col, _ = TASK_COLS[task]
                v = get_val(data[model_key], mean_col, task in SCALE_100)
                if v is not None and v > best[task]:
                    best[task] = v
    return best

# ── Cell helpers ──────────────────────────────────────────────────────────────

NA_S    = r"{\na}"   # inside siunitx S columns (must be wrapped in braces)
NA_TEXT = r"--"      # inside plain text/c columns

def fmt_mean(v: float | None, is_best: bool) -> str:
    """Format a mean value for an S column.
    Bold cells use \\multicolumn{1}{r}{} to override the S column so that
    \\textbf doesn't break siunitx number alignment."""
    if v is None:
        return NA_S
    s = f"{v:.1f}"
    if is_best:
        return rf"\multicolumn{{1}}{{r}}{{\textbf{{{s}}}}}"
    return s

def fmt_cv(v: float | None) -> str:
    if v is None:
        return NA_S
    return f"{v:.2f}"

def fmt_params(model_key: str | None) -> str:
    if model_key is None:
        return NA_S
    val = parameters_model_map.get(model_key)
    return val if val is not None else NA_S

def fmt_tokens(model_key: str | None) -> str:
    if model_key is None:
        return NA_TEXT
    val = tokens_seen.get(model_key)
    return val if val is not None else NA_TEXT

# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(data: dict) -> str:
    best   = find_best(data)
    n_task = len(TASKS)
    # model name | params | tokens | (μ, CV) × tasks
    n_cols = 3 + 2 * n_task

    # siunitx S columns for params and each task pair; plain c for tokens
    params_col_spec = r"S[table-format=1.2,table-number-alignment=right]"
    tokens_col_spec = r"c"
    task_col_spec   = (
        r"S[table-format=3.1,table-number-alignment=right]"
        r" S[table-format=2.2,table-number-alignment=right]"
    )
    col_spec = "l " + params_col_spec + " " + tokens_col_spec + " " + \
               " ".join(task_col_spec for _ in TASKS)

    # Task headers start at column 4
    task_headers = "".join(
        rf" & \multicolumn{{2}}{{c}}{{{TASK_DISPLAY[t]}}}" for t in TASKS
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{4 + 2*i}-{5 + 2*i}}}" for i in range(n_task)
    )
    subheader = "".join(r" & {$\mu$} & {CV\%}" for _ in TASKS)

    lines = [
        r"% Required: \usepackage{booktabs, multirow, siunitx, xcolor}",
        r"% \newcommand{\best}[1]{\textbf{#1}}",
        r"% \newcommand{\na}{--}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\label{tab:main-results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"\textbf{{Model}} & {{\textbf{{Params (B)}}}} & {{\textbf{{Tokens}}}}{task_headers} \\",
        cmidrules,
        rf" & & {subheader} \\",
        r"\midrule",
    ]

    for s_idx, (section_title, section_rows) in enumerate(TABLE_STRUCTURE):
        lines.append(
            rf"\multicolumn{{{n_cols}}}{{l}}{{{section_title}}} \\[2pt]"
        )
        for model_key, display_name in section_rows:
            row = data.get(model_key) if model_key else None
            cells = [fmt_params(model_key), fmt_tokens(model_key)]
            for task in TASKS:
                mean_col, cv_col = TASK_COLS[task]
                mean_v = get_val(row, mean_col, task in SCALE_100) if row else None
                cv_v   = get_val(row, cv_col,   False)             if row else None
                is_best = mean_v is not None and abs(mean_v - best[task]) < 1e-4
                cells += [fmt_mean(mean_v, is_best), fmt_cv(cv_v)]
            lines.append(f"{display_name} & " + " & ".join(cells) + r" \\")

        lines.append(r"\bottomrule" if s_idx == len(TABLE_STRUCTURE) - 1 else r"\midrule")

    acc_label = "acc\\_norm"
    task_notes = "; ".join(
        f"{TASK_DISPLAY[t]} ({'ChrF' if t == 'flores200' else acc_label})"
        for t in TASKS
    )
    lines += [
        r"\end{tabular}",
        r"\caption{%",
        r"    Results across tasks and model conditions at final checkpoint.",
        r"    $\mu$ = mean performance across languages;",
        r"    CV = coefficient of variation (\%) measuring cross-lingual consistency.",
        rf"    {task_notes}.",
        r"    \textbf{Bold} = best result per column.",
        r"}",
        r"\end{table*}",
    ]

    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = load_data()
    print(f"Loaded {len(data)} models at ckpt={TARGET_CKPT}: {sorted(data)}")
    missing = [
        display
        for _, rows in TABLE_STRUCTURE
        for key, display in rows
        if key and key not in data
    ]
    if missing:
        print(f"  [warn] No CSV row found for: {missing} → will render as --")

    table = build_table(data)
    OUT_TEX.write_text(table)
    print(f"\nWrote → {OUT_TEX}\n")
    print(table)

if __name__ == "__main__":
    main()