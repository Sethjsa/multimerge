import pandas as pd
from pathlib import Path

results_root = Path("multiblimp/results")
output_path = Path("multiblimp/multiblimp_all_results.csv")

dfs = []
for csv_path in sorted(results_root.rglob("all.csv")):
    try:
        df = pd.read_csv(csv_path)
        df["source_path"] = str(csv_path)
        dfs.append(df)
        print(f"[OK]   {csv_path}")
    except Exception as e:
        print(f"[FAIL] {csv_path}: {e}")

# get mixed-10-checkpoints checkpoint_0000000 from df
mixed_row = df[df["model"] == "models/mixed-10-checkpoints/checkpoint_0000000"]
# add it to df again with model name merged-10-checkpoints
mixed_row["model"] = "merged-10-checkpoints"
dfs.append(mixed_row)

# if not dfs:
#     print("No CSVs found.")
# else:
#     combined = pd.concat(dfs, ignore_index=True)
#     combined.to_csv(output_path, index=False)
#     print(f"\nWrote {len(combined)} rows from {len(dfs)} files → {output_path}")
#     print(combined.head())

########################################################
import pandas as pd
import numpy as np

df = pd.read_csv("multiblimp/multiblimp_all_results.csv")

LANGS = ["eng", "deu", "ara", "fra", "spa", "zhos", "rus", "tur", "nld", "ita"]
ACC_COLS = [f"{l}_acc" for l in LANGS]

# Add any missing acc cols as NaN
for c in ACC_COLS:
    if c not in df.columns:
        df[c] = np.nan

# ── Helper: get a single row for a model at its final checkpoint ──────────────
def get_row(model, preferred_ckpts=("main", "checkpoint_0047684")):
    sub = df[df["model"] == model]
    for ckpt in preferred_ckpts:
        r = sub[sub["checkpoint"] == ckpt]
        if not r.empty:
            return r.iloc[0]
    if len(sub) == 1:
        return sub.iloc[0]
    return None

# ── Models to evaluate individually ──────────────────────────────────────────
named_models = {
    "tiny-aya-base":        "CohereLabs/tiny-aya-base",
    "llama-3.2-1b":         "meta-llama/Llama-3.2-1B",
    "gemma-2-2b":           "google/gemma-2-2b",
    "eurollm-1.7b":         "utter-project/EuroLLM-1.7B",
    "smollm2-1.7b":         "HuggingFaceTB/SmolLM2-1.7B",
    "tiny-aya-global":      "CohereLabs/tiny-aya-global",
    "tiny-aya-fire":        "CohereLabs/tiny-aya-fire",
    "tiny-aya-water":       "CohereLabs/tiny-aya-water",
    "tiny-aya-earth":       "CohereLabs/tiny-aya-earth",
    "task-aya-checkpoints": "models/task-aya-checkpoints/checkpoint_0047684",
    "linear-aya-checkpoints": "models/linear-aya-checkpoints/checkpoint_0047684",
    "widen-10-checkpoints": "models/widen-10-checkpoints/checkpoint_0047684",
    "dareties-10-checkpoints": "models/dareties-10-checkpoints/checkpoint_0047684",
    "ties-10-checkpoints": "models/ties-10-checkpoints/checkpoint_0047684",
    "mixed-10-checkpoints": "models/mixed-10-checkpoints/checkpoint_0047684",
    "merged-10-checkpoints":"models/merged-10-checkpoints/checkpoint_0047684",
    "merged-2-checkpoints": "models/merged-2-checkpoints/checkpoint_0047684",
    "merged-3-checkpoints": "models/merged-3-checkpoints/checkpoint_0047684",
    "merged-4-checkpoints": "models/merged-4-checkpoints/checkpoint_0047684",
    "merged-5-checkpoints": "models/merged-5-checkpoints/checkpoint_0047684",
    "merged-6-checkpoints": "models/merged-6-checkpoints/checkpoint_0047684",
    "merged-7-checkpoints": "models/merged-7-checkpoints/checkpoint_0047684",
    "merged-8-checkpoints": "models/merged-8-checkpoints/checkpoint_0047684",
    "merged-9-checkpoints": "models/merged-9-checkpoints/checkpoint_0047684",

}

HPLT_LANGS = ["eng", "nld", "spa", "fra", "rus", "ita", "tur", "ara", "deu", "zhos"]
hplt_models = {f"HPLT/hplt2c_{l}_checkpoints": l for l in HPLT_LANGS}

def load_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

pairs = load_pairs("multiblimp/pairs.txt")

bilingual_models = {f"{l1}{l2}": f"HPLT/hplt2c_{l1}_checkpoints/HPLT/hplt2c_{l2}_checkpoints" for l1, l2 in pairs}


# ── Compute per-model scores ──────────────────────────────────────────────────
records = []

for label, model_id in named_models.items():
    row = get_row(model_id)
    if row is None:
        print(f"[MISSING] {model_id}")
        continue
    scores = row[ACC_COLS].astype(float)
    valid = scores.dropna()
    mean = valid.mean()
    cv   = valid.std() / mean if mean > 0 else np.nan
    records.append({"model": label, "mean_acc": mean, "cv": cv,
                    **scores.to_dict()})

# ── HPLT: average across all 10 monolingual models ───────────────────────────
hplt_rows = []
for model_id in hplt_models:
    row = get_row(model_id)
    if row is not None:
        hplt_rows.append(row[ACC_COLS].astype(float))
    else:
        print(f"[MISSING] {model_id}")

own_lang_scores = {}  # lang -> that model's score on its own language
for model_id, lang in hplt_models.items():
    row = get_row(model_id)
    col = f"{lang}_acc"
    if row is not None and col in row and not pd.isna(row[col]):
        own_lang_scores[lang] = float(row[col])
    else:
        print(f"[MISSING/NaN] {model_id} → {col}")

if own_lang_scores:
    scores_series = pd.Series(own_lang_scores)  # one score per language
    mean = scores_series.mean()
    cv   = scores_series.std() / mean if mean > 0 else np.nan
    # put each score in its acc column, leave others NaN
    lang_acc_dict = {f"{l}_acc": own_lang_scores.get(l, np.nan) for l in LANGS}
    records.append({"model": "hplt-mono-avg", "mean_acc": mean, "cv": cv,
                    **lang_acc_dict})



# ── Output ────────────────────────────────────────────────────────────────────
results = pd.DataFrame(records).set_index("model")
results["mean_acc"] = results["mean_acc"].round(4)
results["cv"]       = results["cv"].round(4)
for c in ACC_COLS:
    results[c] = results[c].round(4)

print(results[["mean_acc", "cv"] + ACC_COLS].to_string())
results.to_csv("multiblimp/summary_mean_cv.csv")
print("\nSaved → multiblimp/summary_mean_cv.csv")

########################################################
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# from matplotlib import rcParams

# os.makedirs("multiblimp/plots", exist_ok=True)

# # ── Style ─────────────────────────────────────────────────────────────────────
# rcParams.update({
#     "font.family": "sans-serif",
#     "font.size": 10,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "pdf.fonttype": 42,   # editable text in Illustrator/Inkscape
# })

# # Okabe-Ito palette (colorblind-safe), 9 colours for 9 languages
# PALETTE = [
#     "#E69F00", "#56B4E9", "#009E73", "#F0E442",
#     "#0072B2", "#D55E00", "#CC79A7", "#999999", "#000080",
# ]

# LANGS = ["eng", "nld", "spa", "fra", "rus", "ita", "tur", "ara", "deu"]
# LANG_LABELS = {l: l.upper() for l in LANGS}  # ISO-3 codes
# LANG_COLOR = {l: PALETTE[i] for i, l in enumerate(LANGS)}
# CI_SCALE   = 1.96

# LINESTYLES = {"mixed": "-", "merged": (0, (3, 1.5))}   # solid vs dense-dotted
# AVG_LW     = 2.5
# LANG_LW    = 1.4
# ALPHA_BAND = 0.18

# # ── Data loading ──────────────────────────────────────────────────────────────
# df = pd.read_csv("multiblimp/multiblimp_all_results.csv")

# def extract_step(ckpt):
#     if isinstance(ckpt, str) and ckpt.startswith("checkpoint_"):
#         return int(ckpt.split("_")[1])
#     return None

# def load_model(model_prefix):
#     sub = df[df["model"].str.startswith(model_prefix)].copy()
#     sub["step"] = sub["checkpoint"].apply(extract_step)
#     return sub.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)

# data = {
#     "merged": load_model("models/merged-10-checkpoints"),
#     "mixed":  load_model("models/mixed-10-checkpoints"),
# }

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def get_lang_arrays(sub, lang):
#     acc = sub[f"{lang}_acc"].astype(float).values * 100
#     se  = sub.get(f"{lang}_se", pd.Series(np.zeros(len(sub)))).astype(float).values * 100
#     return acc, se

# def avg_across_langs(sub):
#     accs, ses = [], []
#     for l in LANGS:
#         a, s = get_lang_arrays(sub, l)
#         accs.append(a); ses.append(s)
#     avg_acc = np.nanmean(np.stack(accs, axis=1), axis=1)
#     avg_se  = np.sqrt(np.nanmean(np.stack(ses, axis=1) ** 2, axis=1))
#     return avg_acc, avg_se

# def clean_ax(ax):
#     ax.set_ylim(40, 101)
#     ax.set_xlabel("Training step")
#     ax.set_ylabel("Accuracy (%)")
#     ax.grid(axis="y", alpha=0.25, lw=0.6)

# # ── Individual plots (merged & mixed) ────────────────────────────────────────
# for key, sub in data.items():
#     steps = sub["step"].values
#     fig, ax = plt.subplots(figsize=(8, 4.5))

#     for lang in LANGS:
#         acc, se = get_lang_arrays(sub, lang)
#         ci = CI_SCALE * se
#         valid = ~np.isnan(acc)
#         c = LANG_COLOR[lang]
#         ax.plot(steps[valid], acc[valid], color=c, lw=LANG_LW, label=LANG_LABELS[lang])
#         ax.fill_between(steps[valid], acc[valid]-ci[valid], acc[valid]+ci[valid],
#                         color=c, alpha=ALPHA_BAND)

#     avg_acc, avg_se = avg_across_langs(sub)
#     avg_ci = CI_SCALE * avg_se
#     ax.plot(steps, avg_acc, color="black", lw=AVG_LW, ls="--", label="Average", zorder=5)
#     ax.fill_between(steps, avg_acc-avg_ci, avg_acc+avg_ci, color="black", alpha=0.12, zorder=4)

#     clean_ax(ax)
#     # Top legend: langs + avg + CI note
#     import matplotlib.patches as mpatches
#     lang_handles = [mlines.Line2D([], [], color=LANG_COLOR[l], lw=LANG_LW,
#                                    label=LANG_LABELS[l]) for l in LANGS]
#     avg_handle   = mlines.Line2D([], [], color="black", lw=AVG_LW, ls="--", label="Avg")
#     ci_handle    = mpatches.Patch(color="grey", alpha=0.4, label="95% CI")
#     ax.legend(handles=lang_handles + [avg_handle, ci_handle],
#               loc="lower center", bbox_to_anchor=(0.5, 1.01),
#               ncol=12, fontsize=8, frameon=False, columnspacing=0.8, handlelength=1.2)
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.88)
#     path = f"multiblimp/plots/{key}_checkpoints.pdf"
#     fig.savefig(path, bbox_inches="tight")
#     print(f"Saved {path}")
#     plt.close(fig)

# # ── Combined plot ─────────────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(7.5, 5))

# for lang in LANGS:
#     for key, sub in data.items():
#         steps = sub["step"].values
#         acc, se = get_lang_arrays(sub, lang)
#         ci = CI_SCALE * se
#         valid = ~np.isnan(acc)
#         ax.plot(steps[valid], acc[valid], color=LANG_COLOR[lang],
#                 lw=LANG_LW, ls=LINESTYLES[key], alpha=0.85)
#         ax.fill_between(steps[valid], acc[valid]-ci[valid], acc[valid]+ci[valid],
#                         color=LANG_COLOR[lang], alpha=0.08)

# # Average lines (black)
# for key, sub in data.items():
#     steps = sub["step"].values
#     avg_acc, avg_se = avg_across_langs(sub)
#     avg_ci = CI_SCALE * avg_se
#     ax.plot(steps, avg_acc, color="black", lw=AVG_LW, ls=LINESTYLES[key], zorder=5)
#     ax.fill_between(steps, avg_acc-avg_ci, avg_acc+avg_ci, color="black", alpha=0.12, zorder=4)

# clean_ax(ax)

# import matplotlib.patches as mpatches
# lang_handles  = [mlines.Line2D([], [], color=LANG_COLOR[l], lw=LANG_LW,
#                                 label=LANG_LABELS[l]) for l in LANGS]
# style_handles = [
#     mlines.Line2D([], [], color="black", lw=AVG_LW, ls=LINESTYLES["mixed"],
#                   label=r"Mixed$_{10}$"),
#     mlines.Line2D([], [], color="black", lw=AVG_LW, ls=LINESTYLES["merged"],
#                   label=r"Merged$_{10}$"),
# ]
# ci_handle = mpatches.Patch(color="grey", alpha=0.4, label="95% CI")

# ax.legend(handles=lang_handles + style_handles + [ci_handle],
#           loc="lower center", bbox_to_anchor=(0.5, 1.01),
#           ncol=12, fontsize=8, frameon=False, columnspacing=0.8, handlelength=1.2)
# fig.tight_layout()
# fig.subplots_adjust(top=0.88)

# fig.tight_layout()
# path = "multiblimp/plots/merged_vs_mixed_avg.pdf"
# fig.savefig(path, bbox_inches="tight")
# print(f"Saved {path}")
# plt.close(fig)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams

os.makedirs("multiblimp/plots", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family": "sans-serif",
    "font.size": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # editable text in Illustrator/Inkscape
})

# Okabe-Ito palette (colorblind-safe), 9 colours for 9 languages
# PALETTE = [
#     "#E69F00", "#56B4E9", "#009E73", "#F0E442",
#     "#0072B2", "#D55E00", "#CC79A7", "#999999", "#000080",
# ]

PALETTE = [  # Okabe-Ito boosted saturation
    "#FFB000",  # amber        (#E69F00 → +sat)
    "#00AAFF",  # sky blue     (#56B4E9 → +sat)
    "#00C48C",  # green        (#009E73 → +sat)
    "#F0E442",  # yellow       (#F0E442 → +sat)
    "#0088D4",  # blue         (#0072B2 → +sat)
    "#FF6D00",  # vermillion   (#D55E00 → +sat)
    "#E0409A",  # pink-purple  (#CC79A7 → +sat)
    "#999999",  # grey         (unchanged — no hue to boost)
    "#0000CC",  # navy         (#000080 → +sat/lightness)
]

LANGS = ["eng", "deu", "fra", "ita", "nld", "rus", "spa", "tur", "ara"]

LANG_LABELS = {l: l for l in LANGS}  # lowercase ISO-3 codes
LANG_COLOR = {l: PALETTE[i] for i, l in enumerate(LANGS)}
CI_SCALE   = 1.96

LINESTYLES = {"mixed": "-", "merged": (0, (3, 1.5))}   # solid vs dense-dotted
AVG_LW     = 2.5
LANG_LW    = 1.4
ALPHA_BAND = 0.18

# ── Data loading ──────────────────────────────────────────────────────────────
df = pd.read_csv("multiblimp/multiblimp_all_results.csv")

def extract_step(ckpt):
    if isinstance(ckpt, str) and ckpt.startswith("checkpoint_"):
        return int(ckpt.split("_")[1])
    return None

def load_model(model_prefix):
    sub = df[df["model"].str.startswith(model_prefix)].copy()
    sub["step"] = sub["checkpoint"].apply(extract_step)
    return sub.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)

data = {
    "merged": load_model("models/merged-10-checkpoints"),
    "mixed":  load_model("models/mixed-10-checkpoints"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_lang_arrays(sub, lang):
    acc = sub[f"{lang}_acc"].astype(float).values * 100
    se  = sub.get(f"{lang}_se", pd.Series(np.zeros(len(sub)))).astype(float).values * 100
    return acc, se

def avg_across_langs(sub):
    accs, ses = [], []
    for l in LANGS:
        a, s = get_lang_arrays(sub, l)
        accs.append(a); ses.append(s)
    avg_acc = np.nanmean(np.stack(accs, axis=1), axis=1)
    avg_se  = np.sqrt(np.nanmean(np.stack(ses, axis=1) ** 2, axis=1))
    return avg_acc, avg_se

def clean_ax(ax, steps):
    max_step = int(max(steps))
    ax.set_xlim(0, max_step)
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_ylim(40, 101)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("MultiBLiMP accuracy (%)")

# ── Individual plots (merged & mixed) ────────────────────────────────────────
for key, sub in data.items():
    steps = sub["step"].values
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for lang in LANGS:
        acc, se = get_lang_arrays(sub, lang)
        ci = CI_SCALE * se
        valid = ~np.isnan(acc)
        c = LANG_COLOR[lang]
        ax.plot(steps[valid], acc[valid], color=c, lw=LANG_LW, label=LANG_LABELS[lang])
        ax.fill_between(steps[valid], acc[valid]-ci[valid], acc[valid]+ci[valid],
                        color=c, alpha=ALPHA_BAND)

    avg_acc, avg_se = avg_across_langs(sub)
    avg_ci = CI_SCALE * avg_se
    ax.plot(steps, avg_acc, color="black", lw=AVG_LW, ls="--", label="Average", zorder=5)
    ax.fill_between(steps, avg_acc-avg_ci, avg_acc+avg_ci, color="black", alpha=0.12, zorder=4)

    clean_ax(ax, steps)
    import matplotlib.patches as mpatches
    lang_handles = [mlines.Line2D([], [], color=LANG_COLOR[l], lw=LANG_LW,
                                   label=LANG_LABELS[l]) for l in LANGS]
    avg_handle = mlines.Line2D([], [], color="black", lw=AVG_LW, ls="--", label="Avg")
    ci_handle  = mpatches.Patch(color="grey", alpha=0.4, label="95% CI")
    leg_style = ax.legend(handles=[avg_handle, ci_handle],
                          loc="lower center", bbox_to_anchor=(0.5, 1.08),
                          ncol=2, fontsize=20, frameon=False,
                          columnspacing=1.0, handlelength=1.2)
    ax.add_artist(leg_style)
    ax.legend(handles=lang_handles,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=9, fontsize=20, frameon=False,
              columnspacing=0.8, handlelength=1.2)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    path = f"multiblimp/plots/{key}_checkpoints.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

# ── Combined plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6)) 

for lang in LANGS:
    for key, sub in data.items():
        steps = sub["step"].values
        acc, se = get_lang_arrays(sub, lang)
        ci = CI_SCALE * se
        valid = ~np.isnan(acc)
        ax.plot(steps[valid], acc[valid], color=LANG_COLOR[lang],
                lw=1.75, ls=LINESTYLES[key], alpha=0.85)
        ax.fill_between(steps[valid], acc[valid]-ci[valid], acc[valid]+ci[valid],
                        color=LANG_COLOR[lang], alpha=0.08)

# Average lines (black)
for key, sub in data.items():
    steps = sub["step"].values
    avg_acc, avg_se = avg_across_langs(sub)
    avg_ci = CI_SCALE * avg_se
    ax.plot(steps, avg_acc, color="black", lw=AVG_LW, ls=LINESTYLES[key], zorder=5)
    ax.fill_between(steps, avg_acc-avg_ci, avg_acc+avg_ci, color="black", alpha=0.12, zorder=4)

clean_ax(ax, data["mixed"]["step"].values)

import matplotlib.patches as mpatches
lang_handles  = [mlines.Line2D([], [], color=LANG_COLOR[l], lw=2.5,
                                label=LANG_LABELS[l]) for l in LANGS]
style_handles = [
    mlines.Line2D([], [], color="black", lw=2.7, ls=LINESTYLES["mixed"],
                  label=r"Mixed$_{10}\;(μ)$"),
    mlines.Line2D([], [], color="black", lw=2.7, ls=LINESTYLES["merged"],
                  label=r"Merged$_{10}\;(μ)$"),
             
    mpatches.Patch(color="grey", alpha=0.4, label="95% CI"),
]

leg_style = ax.legend(
    handles=style_handles,
    loc="center left", bbox_to_anchor=(0.01, 1.05),  # inside axes, empty middle area
    ncol=3, fontsize=16, frameon=False,
    handlelength=1.5, columnspacing=1.0
)
ax.add_artist(leg_style)

ax.legend(
    handles=lang_handles,
    loc="center left", bbox_to_anchor=(0.99, 0.5),
    ncol=1, fontsize=16, frameon=False,
    handlelength=1.2, labelspacing=0.6
)

ax.set_xticklabels(ax.get_xticks(), fontsize=17)
ax.set_yticklabels([f"{int(y)}" if y == int(y) else f"{y}" for y in ax.get_yticks()], fontsize=17)
# stop y axis at 100
ax.set_ylim(40, 100)

ax.set_xlabel("Training Steps", fontsize=22)
ax.set_ylabel("MultiBLiMP accuracy (%)", fontsize=22)

fig.tight_layout()
fig.subplots_adjust(right=0.78)  # Make extra vertical space for the top legend
fig.subplots_adjust(top=0.81)
# fig.tight_layout()
# fig.subplots_adjust(right=0.78)
path = "multiblimp/plots/merged_vs_mixed_avg.pdf"
fig.savefig(path, bbox_inches="tight", pad_inches=0.5)
print(f"Saved {path}")
plt.close(fig)


########################################################


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import rcParams

os.makedirs("multiblimp/plots", exist_ok=True)

rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

PALETTE = [  # Okabe-Ito boosted saturation
    "#FFB000",  # amber        (#E69F00 → +sat)
    "#00AAFF",  # sky blue     (#56B4E9 → +sat)
    "#00C48C",  # green        (#009E73 → +sat)
    "#F0E442",  # yellow       (#F0E442 → +sat)
    "#0088D4",  # blue         (#0072B2 → +sat)
    "#FF6D00",  # vermillion   (#D55E00 → +sat)
    "#E0409A",  # pink-purple  (#CC79A7 → +sat)
    "#999999",  # grey         (unchanged — no hue to boost)
    "#0000CC",  # navy         (#000080 → +sat/lightness)
]

LANGS = ["eng", "deu", "fra", "ita", "nld", "rus", "spa", "tur", "ara"]
LANG_COLOR = {l: PALETTE[i] for i, l in enumerate(LANGS)}
CI_SCALE = 1.96

# Language added at each merge step (n=1 is the base English model)
MERGE_ORDER = {
    1: "eng", 2: "fra", 3: "deu", 4: "ara", 5: "ita",
    6: "nld", 7: "rus", 8: "spa", 9: "tur", 10: "zhos"
}

df = pd.read_csv("multiblimp/multiblimp_all_results.csv")

def get_row(model, ckpt):
    r = df[(df["model"] == model) & (df["checkpoint"] == ckpt)]
    return r.iloc[0] if not r.empty else None

# ── Build data ────────────────────────────────────────────────────────────────
rows = {}
r1 = get_row("HPLT/hplt2c_eng_checkpoints", "main")
if r1 is not None:
    rows[1] = r1

for n in range(2, 11):
    r = get_row(f"models/merged-{n}-checkpoints/checkpoint_0047684", "checkpoint_0047684")
    if r is not None:
        rows[n] = r
    else:
        print(f"[MISSING] merged-{n}-checkpoints")

ns = sorted(rows.keys())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.75))

for lang in LANGS:
    acc_col = f"{lang}_acc"
    se_col  = f"{lang}_se"
    accs, cis, xs = [], [], []
    for n in ns:
        row = rows[n]
        if acc_col in row and not pd.isna(row[acc_col]):
            accs.append(float(row[acc_col]) * 100)
            se = float(row[se_col]) * 100 if se_col in row and not pd.isna(row[se_col]) else 0
            cis.append(CI_SCALE * se)
            xs.append(n)
    if not xs:
        continue

    xs, accs, cis = np.array(xs), np.array(accs), np.array(cis)
    c = LANG_COLOR[lang]
    lw = 2.2 if lang == "eng" else 1.4
    zorder = 5 if lang == "eng" else 2
    ax.plot(xs, accs, color=c, lw=lw, marker="o", ms=4, zorder=zorder, label=lang)
    ax.fill_between(xs, accs - cis, accs + cis, color=c, alpha=0.18, zorder=zorder - 1)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(list(range(1, 11)))
tick_labels = [f"{n}\n+{MERGE_ORDER[n]}" if n > 1 else f"{n}\n{MERGE_ORDER[n]}" 
               for n in range(1, 11)]
ax.set_xticklabels(tick_labels, fontsize=13)
ax.set_yticklabels(ax.get_yticks(), fontsize=13)
ax.set_xlim(1, 10)
ax.set_ylim(40, 100)
ax.set_xlabel("# HPLT$_{1}$ models merged", fontsize=18)
ax.set_ylabel("MultiBLiMP accuracy (%)", fontsize=18)

# ── Legend ────────────────────────────────────────────────────────────────────
lang_handles = [mlines.Line2D([], [], color=LANG_COLOR[l], lw=1.4,
                               marker="o", ms=4, label=l) for l in LANGS]
note_handles = [
    mpatches.Patch(color="grey", alpha=0.4, label="95% CI"),
]
leg_top = ax.legend(
    handles=note_handles,
    loc="upper right",  # Move inside plot, top right
    fontsize=12,
    frameon=False,
    handlelength=1.2,
)
ax.add_artist(leg_top)
ax.legend(handles=lang_handles,
          loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=9, fontsize=12, frameon=False,
          columnspacing=0.9, handlelength=1.2)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
path = "multiblimp/plots/merge_n_languages.pdf"
fig.savefig(path, bbox_inches="tight", pad_inches=0.5)
print(f"Saved {path}")
plt.show()