#!/usr/bin/env python3
"""
gen_latex_tables.py

Reads lighteval results JSONs and emits one LaTeX table per task.
Edit MODEL_GROUPS, LANGUAGES, and TASK_CONFIGS at the top to match your setup.
"""

import json
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("results")
OUT_DIR = Path("latex_tables")

# Ordered language columns shown in every table.
# Use "–" in the per-task lang list to emit \na for missing ones.
LANGUAGES = ["eng", "deu", "fra", "ita", "nld", "rus", "spa", "tur", "ara", "zho", "hin", "por", "swe"]

# Model groups → rows in the table, in order.
#
# Normal row:   ("Display name", "results_subdir")
# Mono row:     ("Display name", {"lang": "results_subdir", ...})
#   A mono row shows, for each language column, only the result from
#   that language's own monolingual model. Columns with no entry → \na.
#
# For dirs containing multiple checkpoint_XXXXX sub-dirs the *last* checkpoint is used.
MODEL_GROUPS = [
    ("Multilingual baselines", [
        ("EuroLLM-1.7B", "EuroLLM-1.7B"),
        ("Gemma-2-2B",   "gemma-2-2b"),
        ("Tiny-Aya-Base", "tiny-aya-base"),
    ]),
    ("Monolingual experts", [
        ("HPLT\\textsubscript{1} (per-language)", {
            "ara": "hplt2c_ara_checkpoints",
            "deu": "hplt2c_deu_checkpoints",
            "eng": "hplt2c_eng_checkpoints",
            "fra": "hplt2c_fra_checkpoints",
            "ita": "hplt2c_ita_checkpoints",
            "nld": "hplt2c_nld_checkpoints",
            "por": "hplt2c_por_checkpoints",
            "rus": "hplt2c_rus_checkpoints",
            "spa": "hplt2c_spa_checkpoints",
            "tur": "hplt2c_tur_checkpoints",
            "zho": "hplt2c_zhos_checkpoints",
        }),
    ]),
    ("Mixed pre-training ", [
        ("Mixed\\textsubscript{10}", "mixed-10-checkpoints"),
    ]),
    ("Merged\\textsubscript{10} experts", [
        ("Linear", "merged-10-checkpoints"),
        ("TIES", "ties-10-checkpoints"),
        ("DARE-TIES", "dareties-10-checkpoints"),
    ]),
    # ("Other baselines", [
    #     ("SmallLM2-1.7B", "smalllm2-1.7b"),
    #     ("Tiny-Aya-Global", "tiny-aya-global"),
    #     ("Tiny-Aya-Fire", "tiny-aya-fire"),
    #     ("Tiny-Aya-Water", "tiny-aya-water"),
    #     ("Tiny-Aya-Earth", "tiny-aya-earth"),
    #     ("Task-Aya", "task-aya-checkpoints"),
    #     ("Linear-Aya", "linear-aya-checkpoints"),
    #     ("Widen", "widen-10-checkpoints"),
    #     ("Merged-2", "merged-2-checkpoints"),
    #     ("Merged-3", "merged-3-checkpoints"),
    #     ("Merged-4", "merged-4-checkpoints"),
    #     ("Merged-5", "merged-5-checkpoints"),
    #     ("Merged-6", "merged-6-checkpoints"),
    #     ("Merged-7", "merged-7-checkpoints"),
    #     ("Merged-8", "merged-8-checkpoints"),
    #     ("Merged-9", "merged-9-checkpoints"),
    # ]),
]
#!/usr/bin/env python3
"""
gen_latex_tables.py

Reads lighteval results JSONs and emits one LaTeX table per task.
Edit MODEL_GROUPS, LANGUAGES, and TASK_CONFIGS at the top to match your setup.
"""

import json
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("results")
OUT_DIR = Path("latex_tables")

# Ordered language columns shown in every table.
# Use "–" in the per-task lang list to emit \na for missing ones.
LANGUAGES = ["eng", "deu", "fra", "ita", "nld", "rus", "spa", "tur", "ara", "zho"]

# Model groups → rows in the table, in order.
#
# Normal row:   ("Display name", "results_subdir")
# Mono row:     ("Display name", {"lang": "results_subdir", ...})
#   A mono row shows, for each language column, only the result from
#   that language's own monolingual model. Columns with no entry → \na.
#
# # For dirs containing multiple checkpoint_XXXXX sub-dirs the *last* checkpoint is used.
# MODEL_GROUPS = [
#     ("Multilingual baselines", [
#         ("EuroLLM-1.7B", "EuroLLM-1.7B"),
#         ("Gemma-2-2B",   "gemma-2-2b"),
#     ]),
#     ("Monolingual experts (oracle upper bound)", [
#         ("HPLT Mono", {
#             "ara": "hplt2c_ara_checkpoints",
#             "deu": "hplt2c_deu_checkpoints",
#             "eng": "hplt2c_eng_checkpoints",
#             "fra": "hplt2c_fra_checkpoints",
#             "ita": "hplt2c_ita_checkpoints",
#             "nld": "hplt2c_nld_checkpoints",
#             "por": "hplt2c_por_checkpoints",
#             "rus": "hplt2c_rus_checkpoints",
#             "spa": "hplt2c_spa_checkpoints",
#             "tur": "hplt2c_tur_checkpoints",
#             "zho": "hplt2c_zhos_checkpoints",
#         }),
#     ]),
#     ("Mixed pre-training (10 langs)", [
#         ("Mixed", "mixed-10-checkpoints"),
#     ]),
#     ("Merged (10 monolingual models)", [
#         ("DARE-TIES", "dareties-10-checkpoints"),
#     ]),
# ]

# Per-task config: task_id -> {lang -> (json_key, metric_field), ...}
# Languages absent from a task's dict will render as \na.
# Scale: multiply the raw metric by this factor before display (e.g. 100 for %).
TASK_CONFIGS = {
    "belebele": {
        "caption": "Belebele accuracy (\\(\\uparrow\\)) per model and language.",
        "scale": 100,
        "fmt": "1",       # decimal places
        "langs": {
            "eng": ("lighteval|belebele_eng_Latn_cf|5",  "acc_norm"),
            "deu": ("lighteval|belebele_deu_Latn_cf|5",  "acc_norm"),
            "fra": ("lighteval|belebele_fra_Latn_cf|5",  "acc_norm"),
            "ita": ("lighteval|belebele_ita_Latn_cf|5",  "acc_norm"),
            "nld": ("lighteval|belebele_nld_Latn_cf|5",  "acc_norm"),
            "rus": ("lighteval|belebele_rus_Cyrl_cf|5",  "acc_norm"),
            "spa": ("lighteval|belebele_spa_Latn_cf|5",  "acc_norm"),
            "tur": ("lighteval|belebele_tur_Latn_cf|5",  "acc_norm"),
            "ara": ("lighteval|belebele_arb_Arab_cf|5",  "acc_norm"),
            "zho": ("lighteval|belebele_zho_Hans_cf|5",  "acc_norm"),
            "hin": ("lighteval|belebele_hin_Deva_cf|5",  "acc_norm"),
            "swe": ("lighteval|belebele_swe_Latn_cf|5",  "acc_norm"),
        },
    },
    "hellaswag": {
        "caption": "HellaSwag accuracy normalised (\\(\\uparrow\\)) per model and language.",
        "scale": 100,
        "fmt": "1",
        "langs": {
            "eng": ("leaderboard|hellaswag|5",                "acc_norm"),
            "deu": ("lighteval|mlmm_hellaswag_deu_cf|5",      "acc_norm"),
            "fra": ("lighteval|mlmm_hellaswag_fra_cf|5",      "acc_norm"),
            "ita": ("lighteval|mlmm_hellaswag_ita_cf|5",      "acc_norm"),
            "nld": ("lighteval|mlmm_hellaswag_nld_cf|5",      "acc_norm"),
            "rus": ("lighteval|mlmm_hellaswag_rus_cf|5",      "acc_norm"),
            "spa": ("lighteval|mlmm_hellaswag_spa_cf|5",      "acc_norm"),
            "tur": ("lighteval|community_hellaswag_tur_cf|5", "acc_norm"),
            "ara": ("lighteval|mlmm_hellaswag_ara_cf|5",      "acc_norm"),
            "zho": ("lighteval|mlmm_hellaswag_zho_cf|5",      "acc_norm"),
            "hin": ("lighteval|mlmm_hellaswag_hin_cf|5",      "acc_norm"),
            "swe": ("lighteval|mlmm_hellaswag_swe_cf|5",      "acc_norm"),
        },
    },
    "xcsqa": {
        "caption": "XCSQA accuracy normalised (\\(\\uparrow\\)) per model and language.",
        "scale": 100,
        "fmt": "1",
        "langs": {
            "eng": ("lighteval|xcsqa_eng_cf|5", "acc_norm"),
            "deu": ("lighteval|xcsqa_deu_cf|5", "acc_norm"),
            "fra": ("lighteval|xcsqa_fra_cf|5", "acc_norm"),
            "ita": ("lighteval|xcsqa_ita_cf|5", "acc_norm"),
            "nld": ("lighteval|xcsqa_nld_cf|5", "acc_norm"),
            "rus": ("lighteval|xcsqa_rus_cf|5", "acc_norm"),
            "spa": ("lighteval|xcsqa_spa_cf|5", "acc_norm"),
            "ara": ("lighteval|xcsqa_ara_cf|5", "acc_norm"),
            "zho": ("lighteval|xcsqa_zho_cf|5", "acc_norm"),
            "hin": ("lighteval|xcsqa_hin_cf|5", "acc_norm"),
            "tur": ("lighteval|xcsqa_tur_cf|5", "acc_norm"),
            "por": ("lighteval|xcsqa_por_cf|5", "acc_norm"),
        },
    },
    "flores200": {
        "caption": "Flores-200 ChrF++ (eng→X, \\(\\uparrow\\)) per model and language.",
        "scale": 1,
        "fmt": "1",
        "langs": {
            "eng": ("lighteval|flores200:fra_Latn-eng_Latn|5", "chrf++"),
            "deu": ("lighteval|flores200:eng_Latn-deu_Latn|5", "chrf++"),
            "fra": ("lighteval|flores200:eng_Latn-fra_Latn|5", "chrf++"),
            "ita": ("lighteval|flores200:eng_Latn-ita_Latn|5", "chrf++"),
            "nld": ("lighteval|flores200:eng_Latn-nld_Latn|5", "chrf++"),
            "rus": ("lighteval|flores200:eng_Latn-rus_Cyrl|5", "chrf++"),
            "spa": ("lighteval|flores200:eng_Latn-spa_Latn|5", "chrf++"),
            "tur": ("lighteval|flores200:eng_Latn-tur_Latn|5", "chrf++"),
            "ara": ("lighteval|flores200:eng_Latn-arb_Arab|5", "chrf++"),
            "zho": ("lighteval|flores200:eng_Latn-zho_Hans|5", "chrf++"),
            "hin": ("lighteval|flores200:eng_Latn-hin_Deva|5", "chrf++"),
            "swe": ("lighteval|flores200:eng_Latn-swe_Latn|5", "chrf++"),
            "por": ("lighteval|flores200:eng_Latn-por_Latn|5", "chrf++"),
        },
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def checkpoint_num(p: Path) -> int:
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else -1


def find_last_result_json(model_dir: Path) -> Path | None:
    """Return the results JSON from the last checkpoint (or the dir itself)."""
    # Check for checkpoint_XXXXX sub-dirs
    cp_dirs = sorted(
        [d for d in model_dir.rglob("*") if d.is_dir() and re.search(r"checkpoint[_-]\d+$", d.name)],
        key=checkpoint_num,
    )
    search_roots = [cp_dirs[-1]] if cp_dirs else [model_dir]
    for root in search_roots:
        hits = list(root.rglob("results_*.json"))
        if hits:
            return hits[0]
    return None


def load_results(json_path: Path) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    return data.get("results", data)  # some files wrap, some don't


# ── Table generation ──────────────────────────────────────────────────────────

def get_value(raw: dict, json_key: str, metric: str, scale: float) -> float | None:
    """raw is a single model's results dict (never a mono wrapper)."""
    entry = raw.get(json_key)
    if entry is None:
        return None
    v = entry.get(metric)
    return v * scale if v is not None else None


def get_value_for_row(model_entry, lang: str, json_key: str, metric: str, scale: float) -> float | None:
    """Dispatch between normal rows and mono rows."""
    if model_entry is None:
        return None
    if isinstance(model_entry, dict) and "__mono__" in model_entry:
        # Mono row: only use the model trained on *this* language
        lang_raw = model_entry["__mono__"].get(lang)
        if lang_raw is None:
            return None
        return get_value(lang_raw, json_key, metric, scale)
    return get_value(model_entry, json_key, metric, scale)


def col_spec(n_langs: int) -> str:
    fmt = "S[table-format=3.1,table-number-alignment=right]"
    return "l " + " ".join([fmt] * n_langs)


def make_table(task_id: str, cfg: dict, model_data: dict) -> str:
    task_langs = [l for l in LANGUAGES if l in cfg["langs"]]
    n = len(task_langs)
    scale = cfg["scale"]
    fmt = cfg["fmt"]

    # Collect all values to find per-column best
    col_values: dict[str, list[float]] = {l: [] for l in task_langs}
    for _, rows in MODEL_GROUPS:
        for display, _ in rows:
            entry = model_data.get(display)
            for lang in task_langs:
                json_key, metric = cfg["langs"][lang]
                v = get_value_for_row(entry, lang, json_key, metric, scale)
                if v is not None:
                    col_values[lang].append(v)

    col_best: dict[str, float] = {
        l: max(vs) for l, vs in col_values.items() if vs
    }

    def fmt_val(v: float | None, lang: str) -> str:
        if v is None:
            return r"{\na}"
        formatted = f"{v:.{fmt}f}"
        if col_best.get(lang) is not None and abs(v - col_best[lang]) < 1e-9:
            return r"\multicolumn{1}{r}{\best{" + formatted + r"}}"
        return formatted

    # Build header
    col_headers = " & ".join(r"\textbf{" + l + r"}" for l in task_langs)
    # cmidrules = " ".join(
    #     r"\cmidrule(lr){" + str(i + 2) + "-" + str(i + 2) + r"}"
    #     for i in range(n)
    # )

    lines = [
        r"% Required: \usepackage{booktabs, siunitx, xcolor}",
        r"% \newcommand{\best}[1]{\textbf{#1}}",
        r"% \newcommand{\na}{--}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\label{{tab:{task_id}}}",
        rf"\begin{{tabular}}{{{col_spec(n)}}}",
        r"\toprule",
        rf"\textbf{{Model}} & {col_headers} \\",
        # cmidrules,
        r"\midrule",
    ]

    first_group = True
    for section_title, rows in MODEL_GROUPS:
        if not first_group:
            lines.append(r"\midrule")
        first_group = False
        lines.append(
            rf"\multicolumn{{{n + 1}}}{{l}}{{\textit{{{section_title}}}}} \\[2pt]"
        )
        for display, _ in rows:
            entry = model_data.get(display)
            cells = []
            for lang in task_langs:
                json_key, metric = cfg["langs"][lang]
                v = get_value_for_row(entry, lang, json_key, metric, scale)
                cells.append(fmt_val(v, lang))
            row = display + " & " + " & ".join(cells) + r" \\"
            lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{cfg['caption']}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_one(subdir: str) -> dict | None:
    model_dir = RESULTS_ROOT / subdir
    if not model_dir.exists():
        print(f"[warn] directory not found: {model_dir}")
        return None
    json_path = find_last_result_json(model_dir)
    if json_path is None:
        print(f"[warn] no results JSON in {model_dir}")
        return None
    print(f"  {subdir:40s} ← {json_path.relative_to(RESULTS_ROOT)}")
    return load_results(json_path)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # model_data holds either:
    #   str  key → dict          (normal model: one JSON for all langs)
    #   str  key → {lang: dict}  (mono model: one JSON per lang, keyed by lang)
    model_data: dict[str, dict | None] = {}
    for _, rows in MODEL_GROUPS:
        for display, subdir in rows:
            if display in model_data:
                continue
            if isinstance(subdir, dict):
                # Mono row: load each language's own model separately
                per_lang = {}
                for lang, sd in subdir.items():
                    raw = load_one(sd)
                    if raw is not None:
                        per_lang[lang] = raw
                model_data[display] = {"__mono__": per_lang}
            else:
                model_data[display] = load_one(subdir)

    # Generate one table per task
    for task_id, cfg in TASK_CONFIGS.items():
        table = make_table(task_id, cfg, model_data)
        out_path = OUT_DIR / f"table_{task_id}.tex"
        out_path.write_text(table)
        print(f"  → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()