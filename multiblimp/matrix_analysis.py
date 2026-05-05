

import json
import numpy as np
import pandas as pd
from itertools import permutations

df = pd.read_csv("multiblimp/multiblimp_all_results.csv")

def get_row(model, ckpt):
    r = df[(df["model"] == model) & (df["checkpoint"] == ckpt)]
    return r.iloc[0] if not r.empty else None

def get_acc(row, lang):
    col = f"{lang}_acc"
    if row is not None and col in row and not pd.isna(row[col]):
        return float(row[col])
    return None

# ── HPLT monolingual final scores (own language only) ────────────────────────
def hplt_score(lang):
    row = get_row(f"HPLT/hplt2c_{lang}_checkpoints", "main")
    if row is None:
        row = get_row(f"HPLT/hplt2c_{lang}_checkpoints", "checkpoint_0047684")
    return get_acc(row, lang)

# ── Bilingual merge model (try both orderings) ───────────────────────────────
def merged_row(lang1, lang2):
    for a, b in [(lang1, lang2), (lang2, lang1)]:
        r = get_row(f"models/merged-{a}{b}-checkpoints/checkpoint_0047684", "checkpoint_0047684")
        if r is not None:
            return r
    return None

# ── Read pairs ────────────────────────────────────────────────────────────────
with open("multiblimp/pairs.txt") as f:
    pairs = [line.strip().split() for line in f if line.strip()]

# ── Build symmetric dict ──────────────────────────────────────────────────────
result = {}

for lang1, lang2 in pairs:
    h1 = hplt_score(lang1)
    h2 = hplt_score(lang2)

    # Average only over available scores
    hplt_scores = [s for s in [h1, h2] if s is not None]
    hplt_avg = float(np.mean(hplt_scores)) if hplt_scores else None

    mrow = merged_row(lang1, lang2)
    m1 = get_acc(mrow, lang1)
    m2 = get_acc(mrow, lang2)
    merged_scores = [s for s in [m1, m2] if s is not None]
    merged_avg = float(np.mean(merged_scores)) if merged_scores else None

    entry = {
        "hplt_avg":   hplt_avg,
        "merged_avg": merged_avg,
        "diff": merged_avg - hplt_avg,
        "hplt":   {lang1: h1,  lang2: h2},
        "merged": {lang1: m1,  lang2: m2},
    }

    # Symmetric: both [lang1][lang2] and [lang2][lang1]
    for a, b in [(lang1, lang2), (lang2, lang1)]:
        result.setdefault(a, {})[b] = entry

    # Report
    print(f"{lang1}-{lang2}: hplt_avg={hplt_avg:.4f}, merged_avg={merged_avg}")

out_path = "multiblimp/bilingual_pair_results.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {out_path}")


import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "pdf.fonttype": 42,
})

# ── Load data ─────────────────────────────────────────────────────────────────
with open("multiblimp/bilingual_pair_results.json") as f:
    pair_data = json.load(f)

with open("multiblimp/distances.json") as f:
    dist_data = json.load(f)

LANGS = ["eng", "nld", "spa", "fra", "rus", "ita", "tur", "ara", "deu", "zhos"]
LANGUAGE_MAP = {
    "eng": "eng",
    "nld": "nld",
    "spa": "spa",
    "fra": "fra",
    "rus": "rus",
    "ita": "ita",
    "tur": "tur",
    "ara": "arb",
    "deu": "deu",
    "zhos": "zho",
}
LANGS = list(LANGUAGE_MAP.keys())
N = len(LANGS)

# ── Build matrices (upper triangle only; NaN on diagonal and lower) ───────────
diff_mat  = np.full((N, N), np.nan)
typo_mat  = np.full((N, N), np.nan)

for i, l1 in enumerate(LANGS):
    for j, l2 in enumerate(LANGS):
        if j <= i:
            continue  # upper triangle only
        # mblimp diff
        try:
            diff_mat[i, j] = pair_data[l1][l2]["diff"]
        except KeyError:
            pass
        # typology average distance
        try:
            typo_mat[i, j] = dist_data[LANGUAGE_MAP[l1]][LANGUAGE_MAP[l2]]["typology"]["average"]
        except KeyError:
            pass

# ── Plotting helper ───────────────────────────────────────────────────────────
def plot_triangle(ax, mat, langs, cmap, vmin, vmax, title, fmt=".2f", center=None):
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(langs, fontsize=9)

    # Annotate cells
    for i in range(N):
        for j in range(N):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                # choose text colour based on distance from centre
                mid = (vmin + vmax) / 2 if center is None else center
                txt_col = "white" if abs(val - mid) > (vmax - vmin) * 0.3 else "black"
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        fontsize=7.5, color=txt_col)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, pad=10, fontsize=11)
    return im

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: mblimp diff (merged_avg - hplt_avg) — diverging around 0
diff_abs = np.nanmax(np.abs(diff_mat))
plot_triangle(axes[0], diff_mat * 100, LANGS,
              cmap="RdYlGn",
              vmin=-diff_abs * 100, vmax=diff_abs * 100,
              title="Δ MultiBLiMP accuracy\n(%)",
              fmt=".1f", center=0)

# Panel B: typology average distance — sequential
typo_min = np.nanmin(typo_mat)
typo_max = np.nanmax(typo_mat)
plot_triangle(axes[1], typo_mat, LANGS,
              cmap="YlOrRd",
              vmin=typo_min, vmax=typo_max,
              title="Typological distance\n(average)",
              fmt=".2f")

fig.tight_layout(w_pad=4)
path = "multiblimp/plots/pair_matrices.pdf"
fig.savefig(path, bbox_inches="tight")
print(f"Saved → {path}")
plt.show()


"""
Correlations between MultiBLiMP bilingual merge diff and:
  - language distance measures (distances.json)
  - model similarity measures (sim_results.json): cosine_sim, l2_norm, mean_cka
Only measures with n=45 (all pairs) are included.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

os.makedirs("multiblimp/plots/correlations", exist_ok=True)

with open("multiblimp/bilingual_pair_results.json") as f:
    pair_data = json.load(f)
with open("multiblimp/distances.json") as f:
    dist_data = json.load(f)
with open("02-analysis/sim_results_en.json") as f:
    sim_data = json.load(f)

# with open("")

LANGUAGE_MAP = {
    "eng": "eng", "nld": "nld", "spa": "spa", "fra": "fra",
    "rus": "rus", "ita": "ita", "tur": "tur", "ara": "arb",
    "deu": "deu", "zhos": "zho",
}
LANGS = list(LANGUAGE_MAP.keys())

DIST_FIELDS = {
    "metadata":  ["wiki_size", "nlp_state", "speakers", "AES", "loc", "average"],
    "typology":  ["lang2vec", "lang2vec_knn", "phoible", "grambank",
                  "gb_clause", "gb_nominal_domain", "gb_numeral",
                  "gb_pronoun", "gb_verbal_domain", "glot_tree", "scripts", "average"],
    "wordlists": ["asjp", "concepts", "average"],
    "textbased": ["whitespace", "punctuation", "char_JSD", "textcat", "average"],
}

CAT_COLORS = {
    "metadata":  "#56B4E9",
    "typology":  "#E69F00",
    "wordlists": "#009E73",
    "textbased": "#D55E00",
    "model_sim": "#CC79A7",
}

# ── Build flat dataframe ──────────────────────────────────────────────────────
rows = []
seen = set()
for l1 in LANGS:
    for l2 in LANGS:
        if l1 >= l2:
            continue
        if (l1, l2) in seen:
            continue
        seen.add((l1, l2))

        try:
            diff = pair_data[l1][l2]["diff"]
        except KeyError:
            continue
        if diff is None:
            continue

        d1, d2 = LANGUAGE_MAP[l1], LANGUAGE_MAP[l2]
        try:
            dist_entry = dist_data[d1][d2]
        except KeyError:
            try:
                dist_entry = dist_data[d2][d1]
            except KeyError:
                dist_entry = None

        record = {"lang1": l1, "lang2": l2, "diff": diff}

        # Distance measures
        for grp, fields in DIST_FIELDS.items():
            for field in fields:
                val = np.nan
                if dist_entry is not None:
                    try:
                        v = dist_entry[grp][field]
                        val = float(v) if v != -1 else np.nan
                    except (KeyError, TypeError):
                        pass
                record[f"{grp}__{field}"] = val

        # Model similarity measures (try both orderings)
        sim_entry = None
        for a, b in [(l1, l2), (l2, l1)]:
            try:
                sim_entry = sim_data[a][b]
                break
            except KeyError:
                pass
        for col, path in [
            ("model_sim__cosine_sim", ["weight_space", "cosine_sim"]),
            ("model_sim__l2_norm",    ["weight_space", "l2_norm"]),
            ("model_sim__mean_cka",   ["cka",          "mean_cka"]),
            ("model_sim__mean_cka_num",   ["cka_num",          "mean_cka"]),
            ("model_sim__mean_cka_inlang",   ["cka_inlang",          "mean_cka"]),
            ("model_sim__mean_cka_eng",   ["cka_eng",          "mean_cka"]),
            ("model_sim__mean_rank_diff",   ["weight_space",          "mean_rank_diff"]),
        ]:
            try:
                val = sim_entry
                for k in path:
                    val = val[k]
                record[col] = float(val)
            except (KeyError, TypeError):
                record[col] = np.nan

        rows.append(record)

df = pd.DataFrame(rows)
print(f"Total pairs: {len(df)}")

# ── Keep only measures where ALL 45 pairs have data ───────────────────────────
dist_cols = [c for c in df.columns if "__" in c]
full_cols  = [c for c in dist_cols if df[c].notna().sum() == len(df)]
dropped    = [c for c in dist_cols if c not in full_cols]
print(f"Dropped {len(dropped)} measures with missing data: {dropped}")
print(f"Remaining: {len(full_cols)} measures\n")

# ── Spearman + Pearson correlations ───────────────────────────────────────────
results = []
for col in full_cols:
    sub = df[["diff", col]].dropna()
    rho, p_s = stats.spearmanr(sub["diff"], sub[col])
    r,   p_p = stats.pearsonr(sub["diff"],  sub[col])
    grp, field = col.split("__", 1)
    results.append({"measure": col, "group": grp, "field": field,
                    "rho": rho, "p_spearman": p_s,
                    "r":   r,   "p_pearson":  p_p,
                    "n": len(sub)})

res = pd.DataFrame(results).sort_values("rho").reset_index(drop=True)

# remove metadata__AES
res = res[res["measure"] != "metadata__AES"]

print("── All correlations (sorted by Spearman ρ) ──")
print(res[["measure", "rho", "p_spearman", "r", "p_pearson", "n"]].to_string(index=False))

# ── Co-correlations between measures ──────────────────────────────────────────
print("\n── Co-correlations between measures (Spearman rho, lower triangle) ──")
measures_matrix = df[full_cols].copy()
# drop any rows with nans just for the pairwise matrix calculation
measures_matrix_dropna = measures_matrix.dropna()
rho_matrix = measures_matrix_dropna.corr(method="spearman")
save_path = "multiblimp/plots/correlations/co_correlations.csv"
rho_matrix.to_csv(save_path, index=True)
print(f"Saved co-correlations to {save_path}")

# print(rho_matrix.to_string(float_format=lambda x: f"{x:.2f}"))

# 

name_mapping = {
    "mean_cka_eng": "Mean layer-wise CKA",
    "mean_rank_diff": "Mean stable-rank difference",
    "cosine_sim": "Layer-wise cosine similarity",
    "l2_norm": "Layer-wise L2 norm difference",
    "lang2vec_knn": "Lang2Vec (kNN) Distance",
    "glot_tree": "Language Tree Distance",
}

name_mapping = {
    "mean_cka_eng": "Mean CKA",
    "mean_rank_diff": "Mean Rank Δ",
    "cosine_sim": "Cosine Similarity",
    "l2_norm": "Mean L2 Norm Δ",
    "lang2vec_knn": "Lang2Vec (kNN) Distance",
    "glot_tree": "Language Tree Distance",
}

def print_latex_table(res, name_mapping):
    filtered = res[res["field"].isin(name_mapping.keys())].copy()
    filtered["display_name"] = filtered["field"].map(name_mapping)
    filtered = filtered.sort_values("rho", ascending=False)

    def fmt_r(x):
        return f"{x:.2f}"

    def fmt_p(x):
        if x < 0.001:
            return "<0.001"
        return f"{x:.3f}"

    def sig(p):
        if p < 0.01:   return "$^{**}$"
        if p < 0.05:   return "$^{*}$"
        return ""

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Measure & $\rho$ & $p_S$ & $r$ & $p_P$ \\")
    lines.append(r"\midrule")

    for _, row in filtered.iterrows():
        name = row["display_name"]
        rho  = fmt_r(row["rho"])  + sig(row["p_spearman"])
        p_s  = fmt_p(row["p_spearman"])
        r    = fmt_r(row["r"])    + sig(row["p_pearson"])
        p_p  = fmt_p(row["p_pearson"])
        lines.append(rf"{name} & {rho} & {p_s} & {r} & {p_p} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Spearman and Pearson correlations with $\Delta$ MultiBLiMP accuracy. $^{*}p<0.05$, $^{**}p<0.01$.}")
    lines.append(r"\label{tab:correlations}")
    lines.append(r"\end{table}")

    print("\n".join(lines))

print_latex_table(res, name_mapping)

# ── Summary plot function ─────────────────────────────────────────────────────
def summary_plot(res, val_col, p_col, xlabel, filename):

    # Only include rows whose measure's field is in name_mapping
    filtered = res[res["field"].isin(name_mapping.keys())].copy()
    # Update display name
    filtered["display_name"] = filtered["field"].map(name_mapping)
    ordered = filtered.sort_values(val_col).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(ordered))
    ax.barh(y, ordered[val_col],
            color=[CAT_COLORS[g] for g in ordered["group"]],
            edgecolor="none", height=0.7)
    for idx, row in ordered.iterrows():
        p = row[p_col]
        if np.isnan(p):
            continue
        marker = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        if marker:
            xoff = np.sign(row[val_col]) * 0.015
            ax.text(row[val_col] + xoff, idx, marker, va="center",
                    ha="left" if row[val_col] > 0 else "right",
                    fontsize=14, color="black")
    # ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([row['display_name'] for _, row in ordered.iterrows()], fontsize=10)
    
    ax.set_xlabel(xlabel)
    legend_patches = [mpatches.Patch(color=c, label=grp)
                      for grp, c in [("Model-based", "#CC79A7"), ("Language-based", "#E69F00")] ]
    legend_patches.append(mpatches.Patch(color="none", label="* p<0.05   ** p<0.01"))
    ax.legend(handles=legend_patches, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(f"multiblimp/plots/correlations/{filename}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

# Only plot the metrics in name_mapping
summary_plot(res, "rho", "p_spearman",
             "Spearman ρ  (Δ MultiBLiMP accuracy vs predictor)", "summary_spearman.pdf")
summary_plot(res, "r",   "p_pearson",
             "Pearson r  (Δ MultiBLiMP accuracy vs predictor)",  "summary_pearson.pdf")

name_mapping = {
    # "mean_cka_num": "Mean CKA numerical",
    # "mean_cka_inlang": "Mean CKA in language",
    "mean_cka_eng": "Mean layer-wise CKA",
    "mean_rank_diff": "Mean stable-rank difference",
    "cosine_sim": "Layer-wise cosine similarity",
    "l2_norm": "Layer-wise L2 norm difference",
    # "lang2vec": "Lang2Vec",
    "lang2vec_knn": "Lang2Vec (kNN) Distance",
    "glot_tree": "Language Tree Distance",
    # "phoible": "Phoible",
    # "grambank": "Grambank",
    # "gb_clause": "GB Clause",
    # "gb_nominal_domain": "GB Nominal Domain",
    # "gb_numeral": "GB Numeral",
    # "gb_pronoun": "GB Pronoun",
    # "gb_verbal_domain": "GB Verbal Domain",
}



# ── Scatter plots for p < 0.05 on either measure ─────────────────────────────
sig_res = res[(res["p_spearman"] < 0.05) | (res["p_pearson"] < 0.05)]
for _, row in sig_res.iterrows():
    col = row["measure"]

    sub = df[["lang1", "lang2", "diff", col]].dropna().copy()
    sub["diff_pct"] = sub["diff"] * 100

    # Map column names for X-axis, fallback to default
    x_label_base = name_mapping.get(row['field'], f"{row['group']} · {row['field']}")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    # ax.margins(x=0.1)
    ax.scatter(sub[col], sub["diff_pct"],
               color="#20B2AA",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
    m, b, *_ = stats.linregress(sub[col], sub["diff_pct"])
    ax.set_xlim(sub[col].min() - 0.03, sub[col].max() + 0.03)
    x_line = np.linspace(sub[col].min() - 0.03, sub[col].max() + 0.03, 100)
    # add 95% confidence interval
    ci = 1.96 * np.std(sub["diff_pct"]) / np.sqrt(len(sub))
    ax.fill_between(x_line, (m * x_line + b) - ci, (m * x_line + b) + ci,
                    color="black", alpha=0.1, zorder=2)
    ax.plot(x_line, m * x_line + b, color="grey", lw=1.5, ls="--", zorder=2)
    # add legend for the confidence interval and regression line
    legend_patches = [mpatches.Patch(color="black", alpha=0.1, label="95% CI"),
                      mlines.Line2D([], [], color="grey", lw=1.4, ls="--", label=f"y = {m:.1f}x {b:.1f}")]
                 
    ax.legend(handles=legend_patches, fontsize=8, frameon=False)
    for _, pt in sub.iterrows():
        ax.annotate(f"{pt.lang1}-{pt.lang2}", (pt[col], pt["diff_pct"]),
                    fontsize=6, alpha=0.6,
                    textcoords="offset points", xytext=(3, 2),
                    ha="right", va="bottom")
               
               
    # ax.axhline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel(x_label_base, fontsize=11)
    ax.set_ylabel("Δ MultiBLiMP accuracy (%)", fontsize=11)
    ax.set_title(
        f"ρ={row['rho']:.3f} (p={row['p_spearman']:.3f})   "
        f"r={row['r']:.3f} (p={row['p_pearson']:.3f})   n={int(row['n'])}",
        fontsize=8)
    # ax.margins(x=0.08)
    fig.tight_layout()
    safe = col.replace("/", "_").replace(" ", "_")
    fig.savefig(f"multiblimp/plots/correlations/{safe}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {safe}.pdf")

print("\nDone.")


# ################################################################################
# """
# Correlations between MultiBLiMP bilingual merge diff and language distance measures.
# """
# import json
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from scipy import stats
# from statsmodels.stats.multitest import multipletests
# from matplotlib import rcParams

# rcParams.update({
#     "font.family": "sans-serif",
#     "font.size": 10,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "pdf.fonttype": 42,
# })

# os.makedirs("multiblimp/plots/correlations", exist_ok=True)

# with open("multiblimp/bilingual_pair_results.json") as f:
#     pair_data = json.load(f)
# with open("multiblimp/distances.json") as f:
#     dist_data = json.load(f)

# LANGUAGE_MAP = {
#     "eng": "eng", "nld": "nld", "spa": "spa", "fra": "fra",
#     "rus": "rus", "ita": "ita", "tur": "tur", "ara": "arb",
#     "deu": "deu", "zhos": "zho",
# }
# LANGS = list(LANGUAGE_MAP.keys())

# DIST_FIELDS = {
#     "metadata":  ["wiki_size", "nlp_state", "speakers", "AES", "loc", "average"],
#     "typology":  ["lang2vec", "lang2vec_knn", "phoible", "grambank",
#                   "gb_clause", "gb_nominal_domain", "gb_numeral",
#                   "gb_pronoun", "gb_verbal_domain", "glot_tree", "scripts", "average"],
#     "wordlists": ["asjp", "concepts", "average"],
#     "textbased": ["whitespace", "punctuation", "char_JSD", "textcat", "average"],
# }

# CAT_COLORS = {
#     "metadata":  "#56B4E9",
#     "typology":  "#E69F00",
#     "wordlists": "#009E73",
#     "textbased": "#D55E00",
# }

# # ── Build flat dataframe ──────────────────────────────────────────────────────
# rows = []
# seen = set()
# for l1 in LANGS:
#     for l2 in LANGS:
#         if l1 >= l2:
#             continue
#         if (l1, l2) in seen:
#             continue
#         seen.add((l1, l2))

#         try:
#             diff = pair_data[l1][l2]["diff"]
#         except KeyError:
#             continue
#         if diff is None:
#             continue

#         d1, d2 = LANGUAGE_MAP[l1], LANGUAGE_MAP[l2]
#         try:
#             dist_entry = dist_data[d1][d2]
#         except KeyError:
#             try:
#                 dist_entry = dist_data[d2][d1]
#             except KeyError:
#                 continue

#         record = {"lang1": l1, "lang2": l2, "diff": diff}
#         for grp, fields in DIST_FIELDS.items():
#             for field in fields:
#                 try:
#                     val = dist_entry[grp][field]
#                     record[f"{grp}__{field}"] = float(val) if val != -1 else np.nan
#                 except (KeyError, TypeError):
#                     record[f"{grp}__{field}"] = np.nan
#         rows.append(record)

# df = pd.DataFrame(rows)
# print(f"Pairs with data: {len(df)}")

# # ── Spearman correlations ─────────────────────────────────────────────────────
# dist_cols = [c for c in df.columns if "__" in c]
# results = []
# for col in dist_cols:
#     sub = df[["diff", col]].dropna()
#     if len(sub) < 10:
#         continue
#     rho, p_s = stats.spearmanr(sub["diff"], sub[col])
#     r, p_p   = stats.pearsonr(sub["diff"], sub[col])
#     grp, field = col.split("__")
#     results.append({"measure": col, "group": grp, "field": field,
#                     "rho": rho, "p_spearman": p_s,
#                     "r": r,   "p_pearson":  p_p,
#                     "n": len(sub)})

# res = pd.DataFrame(results)

# # BH correction — only on rows with valid p-values
# valid = res["p_spearman"].notna()
# _, padj_s, _, _ = multipletests(res.loc[valid, "p_spearman"], method="fdr_bh")
# _, padj_p, _, _ = multipletests(res.loc[valid, "p_pearson"],  method="fdr_bh")
# res["p_spearman_adj"] = np.nan
# res["p_pearson_adj"]  = np.nan
# res.loc[valid, "p_spearman_adj"] = padj_s
# res.loc[valid, "p_pearson_adj"]  = padj_p
# res["sig_s"] = res["p_spearman_adj"] < 0.05
# res["sig_p"] = res["p_pearson_adj"]  < 0.05
# res = res.sort_values("rho").reset_index(drop=True)

# print("\n── All correlations (sorted by Spearman ρ) ──")
# print(res[["measure", "rho", "p_spearman", "r", "p_pearson", "n"]].to_string(index=False))

# # ── Summary bar plot: Spearman & Pearson side by side ────────────────────────
# def summary_plot(res, val_col, p_col, sig_col, xlabel, filename):
#     ordered = res.sort_values(val_col).reset_index(drop=True)
#     fig, ax = plt.subplots(figsize=(10, max(5, len(ordered) * 0.28)))
#     y = np.arange(len(ordered))
#     ax.barh(y, ordered[val_col],
#             color=[CAT_COLORS[g] for g in ordered["group"]],
#             edgecolor="none", height=0.7)
#     for idx, row in ordered.iterrows():
#         if row[sig_col]:
#             xpos = row[val_col] + np.sign(row[val_col]) * 0.01
#             ha   = "left" if row[val_col] > 0 else "right"
#             ax.text(xpos, idx, "★", va="center", ha=ha, fontsize=9)
#         # raw p annotation
#         p = row[p_col]
#         if not np.isnan(p):
#             marker = "**" if p < 0.01 else ("*" if p < 0.05 else "")
#             if marker:
#                 xpos2 = row[val_col] + np.sign(row[val_col]) * 0.03
#                 ax.text(xpos2, idx, marker, va="center",
#                         ha="left" if row[val_col] > 0 else "right",
#                         fontsize=8, color="dimgrey")
#     ax.axvline(0, color="black", lw=0.8)
#     ax.set_yticks(y)
#     ax.set_yticklabels([f"{row['group']} · {row['field']}"
#                         for _, row in ordered.iterrows()], fontsize=8)
#     ax.set_xlabel(xlabel)
#     legend_patches = [mpatches.Patch(color=c, label=grp)
#                       for grp, c in CAT_COLORS.items()]
#     legend_patches += [
#         mpatches.Patch(color="none", label="* p<0.05,  ** p<0.01"),
#         mpatches.Patch(color="none", label="★ FDR p<0.05"),
#     ]
#     ax.legend(handles=legend_patches, fontsize=8, frameon=False)
#     fig.tight_layout()
#     fig.savefig(f"multiblimp/plots/correlations/{filename}", bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved {filename}")

# summary_plot(res, "rho", "p_spearman", "sig_s",
#              "Spearman ρ  (diff vs distance)", "summary_spearman.pdf")
# summary_plot(res, "r",   "p_pearson",  "sig_p",
#              "Pearson r  (diff vs distance)",  "summary_pearson.pdf")

# # ── Individual scatter plots for significant correlations ─────────────────────
# sig_res = res[res["sig_s"] | res["sig_p"]]
# for _, row in sig_res.iterrows():
#     col = row["measure"]
#     sub = df[["lang1", "lang2", "diff", col]].dropna().copy()
#     sub["diff_pct"] = sub["diff"] * 100

#     fig, ax = plt.subplots(figsize=(5, 4))
#     ax.scatter(sub[col], sub["diff_pct"],
#                color=CAT_COLORS[row["group"]],
#                s=50, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)

#     m, b, *_ = stats.linregress(sub[col], sub["diff_pct"])
#     x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
#     ax.plot(x_line, m * x_line + b, color="black", lw=1.5, ls="--", zorder=2)

#     for _, pt in sub.iterrows():
#         ax.annotate(f"{pt.lang1}-{pt.lang2}", (pt[col], pt["diff_pct"]),
#                     fontsize=6, alpha=0.7,
#                     textcoords="offset points", xytext=(3, 2))

#     ax.axhline(0, color="grey", lw=0.7, ls=":")
#     ax.set_xlabel(f"{row['group']} · {row['field']}")
#     ax.set_ylabel("Δ MultiBLiMP (%)\n(merged − HPLT mono)")
#     ax.set_title(
#         f"ρ={row['rho']:.3f} (p={row['p_spearman']:.3f})   "
#         f"r={row['r']:.3f} (p={row['p_pearson']:.3f})   n={int(row['n'])}",
#         fontsize=8)
#     fig.tight_layout()
#     safe = col.replace("/", "_").replace(" ", "_")
#     path = f"multiblimp/plots/correlations/{safe}.pdf"
#     fig.savefig(path, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved {path}")

# print("\nDone.")



####

