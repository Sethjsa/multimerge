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

# Display order in the table
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

# ── Table row structure ───────────────────────────────────────────────────────
# Each section: (latex_section_label, [(csv_model_key | None, display_name)])
# None as model key → missing row, all cells render as --

TABLE_STRUCTURE = [
    (r"\textit{Multilingual baselines}", [
        ("tiny-aya-base",  "Tiny-Aya-Base"),
        ("llama-3.2-1b",   "Llama-3.2-1B"),
        ("gemma-2-2b",     "Gemma-2-2B"),
    ]),
    (r"\textit{Monolingual experts (oracle upper bound)}", [
        ("mono_experts",   r"HPLT Mono (avg)"),
    ]),
    (r"\textit{Mixed pre-training}", [
        ("mixed-10-checkpoints", "Mixed (10 langs)"),
    ]),
    (r"\textit{Merged (10 monolingual models)}", [
        ("merged-10-checkpoints", r"Merged -- Linear"),
        (None,                    r"Merged -- TIES"),
        (None,                    r"Merged -- DARE-TIES"),
    ]),
]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> dict[str, dict]:
    """Return {model_name: csv_row_dict} for all rows at TARGET_CKPT."""
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

NA = r"{\na}"

def fmt_mean(v: float | None, is_best: bool) -> str:
    if v is None:
        return NA
    s = f"{v:.1f}"
    return r"{\best{" + s + r"}}" if is_best else s

def fmt_cv(v: float | None) -> str:
    if v is None:
        return NA
    return f"{v:.2f}"

# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(data: dict) -> str:
    best   = find_best(data)
    n_task = len(TASKS)
    n_cols = 1 + 2 * n_task  # model name + (μ, CV) × tasks

    # siunitx S columns: mean right-aligned (up to 3 digits + 1 dec),
    # CV right-aligned (up to 2 digits + 2 dec)
    col_spec = "l " + " ".join(
        r"S[table-format=3.1,table-number-alignment=right]"
        r" S[table-format=2.2,table-number-alignment=right]"
        for _ in TASKS
    )

    task_headers = "".join(
        rf" & \multicolumn{{2}}{{c}}{{{TASK_DISPLAY[t]}}}" for t in TASKS
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(n_task)
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
        rf"\textbf{{Model}}{task_headers} \\",
        cmidrules,
        rf"\textbf{{}}{subheader} \\",
        r"\midrule",
    ]

    for s_idx, (section_title, section_rows) in enumerate(TABLE_STRUCTURE):
        lines.append(
            rf"\multicolumn{{{n_cols}}}{{l}}{{{section_title}}} \\[2pt]"
        )
        for model_key, display_name in section_rows:
            row = data.get(model_key) if model_key else None
            cells = []
            for task in TASKS:
                mean_col, cv_col = TASK_COLS[task]
                mean_v = get_val(row, mean_col, task in SCALE_100) if row else None
                cv_v   = get_val(row, cv_col,   False)             if row else None
                is_best = mean_v is not None and abs(mean_v - best[task]) < 1e-4
                cells += [fmt_mean(mean_v, is_best), fmt_cv(cv_v)]
            lines.append(f"{display_name} & " + " & ".join(cells) + r" \\")

        # \midrule between sections, \bottomrule after the last one
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
        r"    \best{Bold} = best result per column.",
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