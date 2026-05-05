#!/usr/bin/env python3
"""
Visualize activation spaces of monolingual HPLT models.

Collects mean-pooled hidden states from a chosen layer for two input types:
  - FLORES+ parallel sentences (each model sees its own language)
  - Shared numerals (e.g. "1", "2", ..., "100") — language-agnostic anchors

Reduces to 2D via UMAP (default) or PCA, plots one scatter per layer selection,
coloring points by language/model. Overlap between clouds = representational
similarity; separation = incompatible weight spaces.

Two plot modes:
  --mode flores   : one point per sentence per model (density clouds)
  --mode numerals : one point per numeral per model (sparse anchor plot)
  --mode both     : side-by-side subplots

Usage
-----
# Quick UMAP plot, middle layer, all 10 languages:
python visualize_activation_spaces.py --langs eng nld spa fra rus tur ita ara deu zhos

# PCA instead of UMAP:
python visualize_activation_spaces.py --langs eng deu zhos --reducer pca

# Specific layer (0-indexed transformer layers, -1 = last):
python visualize_activation_spaces.py --langs eng deu fra --layer -1

# All three layers (early / middle / late) in one figure:
python visualize_activation_spaces.py --langs eng deu zhos --layers early mid late

# Numerals only, save figure:
python visualize_activation_spaces.py --langs eng ara zhos --mode numerals --output plot.pdf

# Full options:
python visualize_activation_spaces.py \\
    --langs eng nld spa fra rus tur ita ara deu zhos \\
    --revision checkpoint-47684 \\
    --mode both \\
    --layers early mid late \\
    --reducer umap \\
    --flores_n 512 \\
    --numerals_max 50 \\
    --batch_size 16 \\
    --max_length 64 \\
    --device cuda \\
    --output activation_spaces.pdf
"""

import argparse
import gc
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ────────────────────────────────────────────────────────────────────

LANG_TO_MODEL = {
    "eng":  "HPLT/hplt2c_eng_checkpoints",
    "nld":  "HPLT/hplt2c_nld_checkpoints",
    "spa":  "HPLT/hplt2c_spa_checkpoints",
    "fra":  "HPLT/hplt2c_fra_checkpoints",
    "rus":  "HPLT/hplt2c_rus_checkpoints",
    "tur":  "HPLT/hplt2c_tur_checkpoints",
    "ita":  "HPLT/hplt2c_ita_checkpoints",
    "ara":  "HPLT/hplt2c_ara_checkpoints",
    "deu":  "HPLT/hplt2c_deu_checkpoints",
    "zhos": "HPLT/hplt2c_zhos_checkpoints",
    "aya-earth": "CohereLabs/tiny-aya-earth",
    "aya-fire": "CohereLabs/tiny-aya-fire",
    "aya-water": "CohereLabs/tiny-aya-water",
    "aya-global": "CohereLabs/tiny-aya-global",
    "aya-base": "CohereLabs/tiny-aya-base",
}

FLORES_CODE = {
    "eng":  "eng_Latn",
    "nld":  "nld_Latn",
    "spa":  "spa_Latn",
    "fra":  "fra_Latn",
    "rus":  "rus_Cyrl",
    "tur":  "tur_Latn",
    "ita":  "ita_Latn",
    "ara":  "arb_Arab",
    "deu":  "deu_Latn",
    "zhos": "cmn_Hans",
}

# Colour palette — distinct enough for 10 languages
PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_flores(lang: str, n: int, split: str = "devtest") -> list[str]:
    from datasets import load_dataset
    # code = FLORES_CODE[lang]
    code = "eng_Latn"
    ds = load_dataset("openlanguagedata/flores_plus", code, split=split,
                      trust_remote_code=True)
    field = "text" if "text" in ds.column_names else "sentence"
    sents = [ex[field] for ex in ds]
    return sents[:n]


def make_numeral_sentences(n: int = 50) -> list[str]:
    """Arabic numerals 1..n as standalone strings.
    Identical across all languages/scripts — true anchor points.
    Keep n small (≤100) so the scatter is readable.
    """
    return [str(i) for i in range(1, n + 1)]


# ── Activation collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_activations(
    model_name: str,
    revision: str | None,
    sentences: list[str],
    layers: list[int],        # resolved 0-based layer indices
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[int, np.ndarray]:
    """Return {layer_idx: [N × H] float32 array} of mean-pooled hidden states."""
    print(f"  Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    # All indices are already non-negative — resolved before this call
    resolved = layers

    # layer_idx (0-based transformer layer) → list of [B × H] chunks
    buf: dict[int, list[torch.Tensor]] = {l: [] for l in resolved}

    for i in tqdm(range(0, len(sentences), batch_size),
                  desc="  Collecting", leave=False):
        enc = tokenizer(sentences[i : i + batch_size], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length).to(device)
        out  = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()   # [B × T × 1]
        for l in resolved:
            hs     = out.hidden_states[l + 1]                # +1 skips embedding
            pooled = (hs.float() * mask).sum(1) / mask.sum(1)
            buf[l].append(pooled.cpu())

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {l: torch.cat(chunks).numpy() for l, chunks in buf.items()}


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_2d(X: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    """Reduce [N × H] → [N × 2]."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed,
                         metric="cosine").fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed,
                    metric="cosine", perplexity=30).fit_transform(X)
    else:
        raise ValueError(f"Unknown reducer: {method!r}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def resolve_layers(layer_names: list[str], n_layers: int) -> list[int]:
    """Convert symbolic names (early/mid/late) or ints to non-negative 0-based indices."""
    out = []
    for name in layer_names:
        if name == "early":
            idx = n_layers // 4
        elif name == "mid":
            idx = n_layers // 2
        elif name == "late":
            idx = 3 * n_layers // 4
        elif name == "last":
            idx = n_layers - 1
        else:
            idx = int(name)
            if idx < 0:              # resolve negative indices (e.g. -1 → last layer)
                idx = n_layers + idx
        assert 0 <= idx < n_layers, f"Layer {name!r} → {idx} out of range [0, {n_layers-1}]"
        out.append(idx)
    return out


def layer_label(idx: int, n_layers: int) -> str:
    pct = int(100 * idx / (n_layers - 1))
    return f"layer {idx} ({pct}%)"


def plot_clouds(
    ax: plt.Axes,
    embeddings_per_lang: dict[str, np.ndarray],  # lang → [N × 2]
    langs: list[str],
    title: str,
    alpha: float,
    marker: str,
    markersize: float,
    show_centroids: bool = True,
) -> list[mpatches.Patch]:
    """Scatter points per language onto ax. Returns legend handles."""
    handles = []
    for lang, col in zip(langs, cycle(PALETTE)):
        pts = embeddings_per_lang[lang]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=col, alpha=alpha, s=markersize,
                   marker=marker, linewidths=0, rasterized=True)
        if show_centroids:
            c = pts.mean(0)
            ax.scatter(*c, c=col, s=120, marker="*",
                       edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(lang, c, fontsize=7, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")
        handles.append(mpatches.Patch(color=col, label=lang))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return handles


def build_figure(
    flores_acts:   dict[str, dict[int, np.ndarray]] | None,
    numeral_acts:  dict[str, dict[int, np.ndarray]] | None,
    langs:         list[str],
    layer_indices: list[int],
    n_layers:      int,
    reducer:       str,
    mode:          str,
) -> plt.Figure:
    """
    Build a grid figure:
      rows = layers, cols = input type (flores / numerals / both)
    All activations from all languages at a given layer are jointly reduced
    so that the axes are comparable across languages.
    """
    n_types = 2 if mode == "both" else 1
    n_rows  = len(layer_indices)
    fig, axes = plt.subplots(n_rows, n_types,
                             figsize=(5 * n_types, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Activation spaces — {reducer.upper()} — "
        f"{len(langs)} monolingual HPLT models",
        fontsize=12, y=1.01,
    )

    for row, layer_idx in enumerate(layer_indices):
        lbl = layer_label(layer_idx, n_layers)
        tasks = []
        if mode in ("flores", "both"):
            tasks.append(("FLORES+ (in-language)", flores_acts, 0.35, "o", 8))
        if mode in ("numerals", "both"):
            tasks.append(("Numerals 1–N", numeral_acts, 0.9, "D", 30))

        for col, (title, acts_dict, alpha, marker, ms) in enumerate(tasks):
            ax = axes[row][col]

            # Joint reduction: stack all langs, reduce, split back
            all_vecs = np.vstack([acts_dict[l][layer_idx] for l in langs])
            n_per    = [acts_dict[l][layer_idx].shape[0] for l in langs]
            reduced  = reduce_2d(all_vecs, reducer)

            emb_per_lang: dict[str, np.ndarray] = {}
            ptr = 0
            for lang, n in zip(langs, n_per):
                emb_per_lang[lang] = reduced[ptr : ptr + n]
                ptr += n

            handles = plot_clouds(ax, emb_per_lang, langs,
                                  title=f"{title}\n{lbl}",
                                  alpha=alpha, marker=marker, markersize=ms)
            if row == 0 and col == n_types - 1:
                ax.legend(handles=handles, fontsize=7,
                          bbox_to_anchor=(1.02, 1), loc="upper left",
                          borderaxespad=0)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize monolingual HPLT model activation spaces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--langs", nargs="+", required=False, default=list(LANG_TO_MODEL.keys()),
                        choices=list(LANG_TO_MODEL),
                        help="Languages to include (space-separated ISO codes)")
    parser.add_argument("--revision",  default="main",
                        help="HF checkpoint revision, e.g. 'checkpoint-47684'")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",    default="multiblimp/plots/activation_spaces.pdf")

    parser.add_argument("--mode", default="both",
                        choices=["flores", "numerals", "both"],
                        help="Which input type(s) to visualize")
    parser.add_argument("--reducer", default="umap",
                        choices=["umap", "pca", "tsne"],
                        help="Dimensionality reduction method")

    # Layer selection: either symbolic or explicit indices
    layer_grp = parser.add_mutually_exclusive_group()
    layer_grp.add_argument("--layer",  type=int, default=None,
                           help="Single layer index (0-based; -1 = last)")
    layer_grp.add_argument("--layers", nargs="+",
                           default=["early", "mid", "late"],
                           help="Layer(s): early/mid/late/last or int indices")

    parser.add_argument("--flores_n",     type=int, default=512,
                        help="Number of FLORES sentences per language")
    parser.add_argument("--flores_split", default="devtest",
                        choices=["dev", "devtest"])
    parser.add_argument("--numerals_max", type=int, default=50,
                        help="Max numeral N (sentences will be '1', '2', ..., N)")
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--max_length",   type=int, default=64)

    args = parser.parse_args()

    # ── Prepare inputs ────────────────────────────────────────────────────────
    flores_sents:   dict[str, list[str]] = {}
    numeral_sents:  list[str] = make_numeral_sentences(args.numerals_max)

    if args.mode in ("flores", "both"):
        print("Loading FLORES+ sentences...")
        for l in args.langs:
            flores_sents[l] = load_flores(l, args.flores_n, args.flores_split)
        min_n = min(len(s) for s in flores_sents.values())
        flores_sents = {l: s[:min_n] for l, s in flores_sents.items()}
        print(f"  → {min_n} sentences per language")

    # ── Collect activations ───────────────────────────────────────────────────
    # We need to know n_layers to resolve symbolic layer names, so peek at first model.
    first_model = AutoModelForCausalLM.from_pretrained(
        LANG_TO_MODEL[args.langs[0]], revision=args.revision,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    n_layers = first_model.config.num_hidden_layers
    del first_model;  gc.collect();  torch.cuda.empty_cache()

    if args.layer is not None:
        layer_names = [str(args.layer)]
    else:
        layer_names = args.layers
    layer_indices = resolve_layers(layer_names, n_layers)
    print(f"Layers to visualize: {layer_indices} (of {n_layers} total)")

    flores_acts:  dict[str, dict[int, np.ndarray]] = {}
    numeral_acts: dict[str, dict[int, np.ndarray]] = {}

    for lang in args.langs:
        print(f"\n{'='*50}\nLanguage: {lang}")
        # Each model is loaded once; we collect activations for both input
        # types in a single forward pass to avoid loading the model twice.
        combined_sents = []
        n_flores = 0
        if args.mode in ("flores", "both"):
            combined_sents += flores_sents[lang]
            n_flores = len(flores_sents[lang])
        if args.mode in ("numerals", "both"):
            combined_sents += numeral_sents

        acts = collect_activations(
            model_name=LANG_TO_MODEL[lang],
            revision=args.revision,
            sentences=combined_sents,
            layers=layer_indices,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # Split combined activations back into flores / numeral portions
        for l_idx, arr in acts.items():
            if args.mode in ("flores", "both"):
                flores_acts.setdefault(lang, {})[l_idx] = arr[:n_flores]
            if args.mode in ("numerals", "both"):
                numeral_acts.setdefault(lang, {})[l_idx] = arr[n_flores:]

    # ── Plot ──────────────────────────────────────────────────────────────────
    print(f"\nBuilding plot ({args.reducer.upper()})...")
    fig = build_figure(
        flores_acts   = flores_acts  if args.mode in ("flores", "both")   else None,
        numeral_acts  = numeral_acts if args.mode in ("numerals", "both") else None,
        langs         = args.langs,
        layer_indices = layer_indices,
        n_layers      = n_layers,
        reducer       = args.reducer,
        mode          = args.mode,
    )

    out = Path(args.output)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()