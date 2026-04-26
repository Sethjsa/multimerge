#!/usr/bin/env python3
"""
Compute pairwise model similarity metrics for monolingual HPLT models.

Weight-space metrics (no data needed):
  - Cosine similarity of flattened parameters
  - L2 / Frobenius norm of parameter difference
  - Mean absolute stable-rank difference (opt-in via --rank_diff)

CKA (opt-in via --cka): computed under up to THREE input conditions:
  - cka_inlang : each model sees FLORES+ in its own language (parallel input).
  - cka_eng    : all models see the same English FLORES+ sentences.
  - cka_num    : all models see Arabic numerals "0", "1", ..., N (opt-in via --cka_numerals).
                 Numerals are orthographically identical across all scripts,
                 making them true language-agnostic anchors. Useful for probing
                 whether models agree on representations of universal symbols
                 independent of any linguistic content.

Output JSON structure:
  { lang_a: { lang_b: { weight_space: {...}, cka_inlang: {...}, cka_eng: {...}, cka_num: {...} } } }
  Symmetric: results[a][b] == results[b][a] (computed once, written to both).
  Diagonal: self-similarity values (cosine=1, L2=0, mean_cka=1).

Usage
-----
python compute_model_similarity.py --pairs pairs.txt
python compute_model_similarity.py --pairs pairs.txt --revision checkpoint-47684
python compute_model_similarity.py --pairs pairs.txt --cka
python compute_model_similarity.py --pairs pairs.txt --cka --cka_numerals
python compute_model_similarity.py --pairs pairs.txt --cka --cka_numerals --numerals_max 1000
python compute_model_similarity.py --pairs pairs.txt --cka --no_sim
python compute_model_similarity.py \\
    --pairs pairs.txt --output results.json --revision checkpoint-47684 \\
    --cka --cka_numerals --numerals_max 1000 \\
    --cka_n_samples 1012 --cka_batch_size 8 --cka_max_length 128 \\
    --flores_split devtest --device cuda
"""

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

SKIP_KEYS = {"lm_head.weight"}  # tied weights — skip to avoid double-counting


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def dedup_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for a, b in pairs:
        key = canonical(a, b)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def write_symmetric(results: dict, lang_a: str, lang_b: str, entry: dict) -> None:
    for a, b in [(lang_a, lang_b), (lang_b, lang_a)]:
        results.setdefault(a, {}).setdefault(b, {}).update(entry)


def self_similarity(n_layers: int | None = None,
                    include_cka: bool = False,
                    include_cka_num: bool = False) -> dict:
    entry: dict = {"weight_space": {"cosine_sim": 1.0, "l2_norm": 0.0}}
    if include_cka:
        cka_self = {"mean_cka": 1.0, "cka_per_layer": [1.0] * (n_layers or 0)}
        entry["cka_inlang"] = cka_self
        entry["cka_eng"]    = cka_self
    if include_cka_num:
        entry["cka_num"] = {"mean_cka": 1.0, "cka_per_layer": [1.0] * (n_layers or 0)}
    return entry


# ── Model loading ─────────────────────────────────────────────────────────────

def load_state_dict(model_name: str, revision: str | None = None) -> dict:
    print(f"  Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    sd = {k: v.float().cpu() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return sd


# ── Weight-space metrics ──────────────────────────────────────────────────────

def flatten_params(sd: dict) -> torch.Tensor:
    return torch.cat([
        v.flatten()
        for k, v in sorted(sd.items())
        if k not in SKIP_KEYS and v.dtype == torch.float32 and v.ndim >= 1
    ])


def spectral_norm_power_iter(W: torch.Tensor, n_iter: int = 20) -> float:
    """Approximate largest singular value via power iteration (~100× faster than svdvals)."""
    v = torch.randn(W.shape[1], dtype=W.dtype, device=W.device)
    v = v / v.norm()
    for _ in range(n_iter):
        u = W @ v;  u = u / u.norm()
        v = W.T @ u; v = v / v.norm()
    return (u @ (W @ v)).abs().item()


def stable_rank(W: torch.Tensor) -> float:
    fro_sq = torch.sum(W ** 2).item()
    if fro_sq == 0:
        return 0.0
    s_max = spectral_norm_power_iter(W)
    return fro_sq / (s_max ** 2) if s_max > 0 else float("nan")


def compute_weight_metrics(sd_a: dict, sd_b: dict,
                           compute_rank: bool = False) -> dict:
    vec_a = flatten_params(sd_a)
    vec_b = flatten_params(sd_b)
    cos = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
    l2  = torch.norm(vec_a - vec_b, p=2).item()
    ws: dict = {"cosine_sim": round(cos, 6), "l2_norm": round(l2, 4)}

    if compute_rank:
        rank_diffs = []
        for k in sorted(set(sd_a) & set(sd_b)):
            if k in SKIP_KEYS:
                continue
            va, vb = sd_a[k], sd_b[k]
            if va.ndim < 2:
                continue
            rank_diffs.append(abs(stable_rank(va) - stable_rank(vb)))
        ws["mean_rank_diff"] = (
            round(float(np.mean(rank_diffs)), 4) if rank_diffs else None
        )

    return {"weight_space": ws}


# ── CKA ───────────────────────────────────────────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al., 2019) between [n × d_X] and [n × d_Y]."""
    X, Y = X.double(), Y.double()
    n = X.shape[0]

    def gram_and_center(Z: torch.Tensor) -> torch.Tensor:
        G = Z @ Z.T
        return G - G.mean(1, keepdim=True) - G.mean(0, keepdim=True) + G.mean()

    KX, KY = gram_and_center(X), gram_and_center(Y)
    f = 1.0 / (n - 1) ** 2
    hsic_xy = f * (KX * KY).sum()
    hsic_xx = f * (KX * KX).sum()
    hsic_yy = f * (KY * KY).sum()
    return (hsic_xy / ((hsic_xx * hsic_yy).sqrt() + 1e-12)).item()


def load_flores(lang: str, n_samples: int, split: str) -> list[str]:
    from datasets import load_dataset
    code = FLORES_CODE.get(lang)
    if code is None:
        raise ValueError(f"No FLORES+ code for language: {lang!r}")
    ds = load_dataset("openlanguagedata/flores_plus", code, split=split,
                      trust_remote_code=True)
    field = "text" if "text" in ds.column_names else "sentence"
    sents = [ex[field] for ex in ds]
    return sents[:n_samples] if n_samples and n_samples < len(sents) else sents


def make_numeral_sentences(max_n: int) -> list[str]:
    """Arabic numeral strings "0", "1", ..., str(max_n).
    Orthographically identical across all scripts — true language-agnostic anchors.
    Each numeral is its own 'sentence' so each gets one mean-pooled representation.
    """
    return [str(i) for i in range(max_n + 1)]


@torch.no_grad()
def collect_activations(
    model_name: str, revision: str | None, sentences: list[str],
    device: str, batch_size: int, max_length: int,
) -> list[torch.Tensor]:
    """Mean-pooled hidden states per transformer layer → list of [N × H] CPU tensors.
    Embedding layer (index 0) is skipped."""
    print(f"  Loading {model_name} for CKA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    layer_acts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    for i in tqdm(range(0, len(sentences), batch_size),
                  desc="  Activations", leave=False):
        enc = tokenizer(sentences[i : i + batch_size], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length).to(device)
        out  = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for l, hs in enumerate(out.hidden_states[1:]):   # skip embedding
            pooled = (hs.float() * mask).sum(1) / mask.sum(1)
            layer_acts[l].append(pooled.cpu())

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [torch.cat(acts, dim=0) for acts in layer_acts]


def compute_cka_entry(acts_a: list[torch.Tensor],
                      acts_b: list[torch.Tensor],
                      key: str) -> dict:
    per_layer = [linear_cka(a, b) for a, b in zip(acts_a, acts_b)]
    return {key: {"mean_cka":      round(float(np.mean(per_layer)), 6),
                  "cka_per_layer": [round(v, 6) for v in per_layer]}}


def run_cka_pass(
    pairs:     list[tuple[str, str]],
    langs:     list[str],
    act_cache: dict[str, list[torch.Tensor]],
    results:   dict,
    cka_key:   str,
    n_layers:  int,
) -> None:
    for l in langs:
        results[l][l].update({cka_key: {"mean_cka": 1.0,
                                         "cka_per_layer": [1.0] * n_layers}})
    print(f"\nComputing pairwise CKA [{cka_key}]...")
    for lang_a, lang_b in tqdm(pairs, desc=f"CKA [{cka_key}]"):
        t0 = time.time()
        entry = compute_cka_entry(act_cache[lang_a], act_cache[lang_b], cka_key)
        write_symmetric(results, lang_a, lang_b, entry)
        tqdm.write(f"  {lang_a} × {lang_b}: mean_CKA={entry[cka_key]['mean_cka']:.4f}"
                   f"  ({time.time()-t0:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pairwise model similarity for monolingual HPLT models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs",     required=False, default="multiblimp/pairs.txt")
    parser.add_argument("--output",    default="02-analysis/similarity_results.json")
    parser.add_argument("--revision",  default=None)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_sim",    action="store_true",
                        help="Skip weight-space metrics")
    parser.add_argument("--rank_diff", action="store_true",
                        help="Also compute mean stable-rank difference (slow)")

    g = parser.add_argument_group("CKA")
    g.add_argument("--cka",            action="store_true",
                   help="Compute CKA (in-language + English passes)")
    g.add_argument("--cka_numerals",   action="store_true",
                   help="Also compute CKA on Arabic numeral strings (cka_num). "
                        "Numerals are script-identical across all languages, making "
                        "them true language-agnostic anchors. Requires --cka.")
    g.add_argument("--numerals_max",   type=int, default=1000,
                   help="Compute CKA on numerals 0..N (default 0–1000, i.e. 1001 points)")
    g.add_argument("--cka_n_samples",  type=int, default=1012)
    g.add_argument("--cka_batch_size", type=int, default=8)
    g.add_argument("--cka_max_length", type=int, default=128)
    g.add_argument("--flores_split",   default="devtest", choices=["dev", "devtest"])

    args = parser.parse_args()

    if args.cka_numerals and not args.cka:
        parser.error("--cka_numerals requires --cka")

    raw_pairs = load_pairs(args.pairs)
    pairs = dedup_pairs(raw_pairs)

    seen, langs = set(), []
    for a, b in pairs:
        for l in (a, b):
            if l not in seen:
                langs.append(l)
                seen.add(l)

    print(f"Languages ({len(langs)}): {langs}")
    print(f"Unique pairs: {len(pairs)} (from {len(raw_pairs)} in file)")

    results: dict[str, dict] = {}
    for l in langs:
        results.setdefault(l, {})[l] = self_similarity()

    # ── 1. Weight-space metrics ───────────────────────────────────────────────
    if not args.no_sim:
        print("\n=== Weight-space metrics ===")
        anchor_to_partners: dict[str, list[str]] = defaultdict(list)
        for a, b in pairs:
            anchor_to_partners[a].append(b)

        for i, (lang_a, partners) in enumerate(anchor_to_partners.items()):
            print(f"\n[{i+1}/{len(anchor_to_partners)}] Anchor: {lang_a}")
            sd_a = load_state_dict(LANG_TO_MODEL[lang_a], args.revision)

            for lang_b in partners:
                t0 = time.time()
                print(f"  ↳ {lang_a} × {lang_b}", end=" ... ", flush=True)
                sd_b = load_state_dict(LANG_TO_MODEL[lang_b], args.revision)
                m    = compute_weight_metrics(sd_a, sd_b, compute_rank=args.rank_diff)
                write_symmetric(results, lang_a, lang_b, m)
                del sd_b;  gc.collect()
                ws = m["weight_space"]
                msg = f"cos={ws['cosine_sim']:.4f}  L2={ws['l2_norm']:.1f}"
                if args.rank_diff:
                    msg += f"  rank_diff={ws.get('mean_rank_diff'):.4f}"
                print(f"{msg}  ({time.time()-t0:.1f}s)")

            del sd_a;  gc.collect();  torch.cuda.empty_cache()

    # ── 2. CKA ───────────────────────────────────────────────────────────────
    if args.cka:
        print("\n=== CKA ===")
        n_layers = None   # set after first model load

        # ── Pass 1: in-language ───────────────────────────────────────────────
        print("\n[Pass 1] In-language FLORES+...")
        inlang_sents: dict[str, list[str]] = {}
        for l in langs:
            inlang_sents[l] = load_flores(l, args.cka_n_samples, args.flores_split)
        n_inlang = min(len(s) for s in inlang_sents.values())
        inlang_sents = {l: s[:n_inlang] for l, s in inlang_sents.items()}
        print(f"  → {n_inlang} sentences per language")

        act_inlang: dict[str, list[torch.Tensor]] = {}
        for l in langs:
            t0 = time.time()
            act_inlang[l] = collect_activations(
                LANG_TO_MODEL[l], args.revision, inlang_sents[l],
                args.device, args.cka_batch_size, args.cka_max_length,
            )
            print(f"  ✓ {l}: {len(act_inlang[l])} layers, "
                  f"hidden={act_inlang[l][0].shape[1]}  ({time.time()-t0:.1f}s)")

        n_layers = len(act_inlang[langs[0]])
        run_cka_pass(pairs, langs, act_inlang, results, "cka_inlang", n_layers)
        del act_inlang;  gc.collect();  torch.cuda.empty_cache()

        # ── Pass 2: English ───────────────────────────────────────────────────
        print("\n[Pass 2] English FLORES+...")
        eng_sents = load_flores("eng", args.cka_n_samples, args.flores_split)[:n_inlang]
        print(f"  → {len(eng_sents)} English sentences for all models")

        act_eng: dict[str, list[torch.Tensor]] = {}
        for l in langs:
            t0 = time.time()
            act_eng[l] = collect_activations(
                LANG_TO_MODEL[l], args.revision, eng_sents,
                args.device, args.cka_batch_size, args.cka_max_length,
            )
            print(f"  ✓ {l}  ({time.time()-t0:.1f}s)")

        run_cka_pass(pairs, langs, act_eng, results, "cka_eng", n_layers)
        del act_eng;  gc.collect();  torch.cuda.empty_cache()

        # ── Pass 3: numerals (opt-in) ─────────────────────────────────────────
        # Arabic numeral strings are orthographically identical across all scripts,
        # so every model is processing exactly the same byte sequences. Any
        # representational similarity here is purely due to shared inductive biases
        # in the architecture/initialisation, not shared linguistic content.
        if args.cka_numerals:
            num_sents = make_numeral_sentences(args.numerals_max)
            print(f"\n[Pass 3] Numerals 0–{args.numerals_max} "
                  f"({len(num_sents)} sentences, same for all models)...")

            act_num: dict[str, list[torch.Tensor]] = {}
            for l in langs:
                t0 = time.time()
                act_num[l] = collect_activations(
                    LANG_TO_MODEL[l], args.revision, num_sents,
                    args.device, args.cka_batch_size,
                    max_length=8,   # numerals are at most a few tokens
                )
                print(f"  ✓ {l}  ({time.time()-t0:.1f}s)")

            run_cka_pass(pairs, langs, act_num, results, "cka_num", n_layers)
            del act_num;  gc.collect();  torch.cuda.empty_cache()

        # Update diagonals with correct layer count
        for l in langs:
            results[l][l].update(self_similarity(
                n_layers=n_layers,
                include_cka=True,
                include_cka_num=args.cka_numerals,
            ))

    # ── 3. Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    hdr = f"{'Pair':<14} {'cos_sim':>9} {'l2_norm':>10}"
    if args.rank_diff:
        hdr += f" {'rank_diff':>10}"
    if args.cka:
        hdr += f" {'cka_inlang':>10} {'cka_eng':>10}"
    if args.cka_numerals:
        hdr += f" {'cka_num':>10}"
    print(hdr)
    print("─" * len(hdr))
    for a, b in pairs:
        v  = results.get(a, {}).get(b, {})
        ws = v.get("weight_space", {})
        row = (f"{a+' × '+b:<14} "
               f"{ws.get('cosine_sim', float('nan')):>9.4f} "
               f"{ws.get('l2_norm',    float('nan')):>10.1f}")
        if args.rank_diff:
            row += f" {ws.get('mean_rank_diff', float('nan')):>10.4f}"
        if args.cka:
            row += (f" {v.get('cka_inlang', {}).get('mean_cka', float('nan')):>10.4f}"
                    f" {v.get('cka_eng',    {}).get('mean_cka', float('nan')):>10.4f}")
        if args.cka_numerals:
            row += f" {v.get('cka_num', {}).get('mean_cka', float('nan')):>10.4f}"
        print(row)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# THIS VERSION WORKS
# """
# Compute pairwise model similarity metrics for monolingual HPLT models.

# Weight-space metrics (no data needed):
#   - Cosine similarity of flattened parameters
#   - L2 / Frobenius norm of parameter difference
#   - Mean absolute stable-rank difference (opt-in via --rank_diff)

# CKA (opt-in via --cka): computed under TWO input conditions for comparison:
#   - cka_inlang: each model sees FLORES+ in its own language (parallel input).
#     Fairest for monolingual models: same concepts, each model's native script.
#   - cka_eng: all models see the same English FLORES+ sentences.
#     Useful as a controlled reference; note non-English models were never trained
#     on English, so lower scores here are expected and interpretable.

# Output JSON structure:
#   { lang_a: { lang_b: { weight_space: {...}, cka_inlang: {...}, cka_eng: {...} } } }
#   Symmetric: results[a][b] == results[b][a] (computed once, written to both).
#   Diagonal: self-similarity values (cosine=1, L2=0, mean_cka=1).

# Usage
# -----
# python compute_model_similarity.py --pairs pairs.txt
# python compute_model_similarity.py --pairs pairs.txt --revision checkpoint-47684
# python compute_model_similarity.py --pairs pairs.txt --cka
# python compute_model_similarity.py --pairs pairs.txt --cka --no_sim
# python compute_model_similarity.py --pairs pairs.txt --rank_diff
# python compute_model_similarity.py \\
#     --pairs pairs.txt --output results.json --revision checkpoint-47684 \\
#     --cka --cka_n_samples 1012 --cka_batch_size 8 --cka_max_length 128 \\
#     --flores_split devtest --device cuda
# """

# import argparse
# import gc
# import json
# import time
# from collections import defaultdict
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer


# # ── Config ────────────────────────────────────────────────────────────────────

# LANG_TO_MODEL = {
#     "eng":  "HPLT/hplt2c_eng_checkpoints",
#     "nld":  "HPLT/hplt2c_nld_checkpoints",
#     "spa":  "HPLT/hplt2c_spa_checkpoints",
#     "fra":  "HPLT/hplt2c_fra_checkpoints",
#     "rus":  "HPLT/hplt2c_rus_checkpoints",
#     "tur":  "HPLT/hplt2c_tur_checkpoints",
#     "ita":  "HPLT/hplt2c_ita_checkpoints",
#     "ara":  "HPLT/hplt2c_ara_checkpoints",
#     "deu":  "HPLT/hplt2c_deu_checkpoints",
#     "zhos": "HPLT/hplt2c_zhos_checkpoints",
# }

# FLORES_CODE = {
#     "eng":  "eng_Latn",
#     "nld":  "nld_Latn",
#     "spa":  "spa_Latn",
#     "fra":  "fra_Latn",
#     "rus":  "rus_Cyrl",
#     "tur":  "tur_Latn",
#     "ita":  "ita_Latn",
#     "ara":  "arb_Arab",
#     "deu":  "deu_Latn",
#     "zhos": "cmn_Hans",
# }

# SKIP_KEYS = {"lm_head.weight"}  # tied weights — skip to avoid double-counting


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def load_pairs(path: str) -> list[tuple[str, str]]:
#     pairs = []
#     with open(path) as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 pairs.append((parts[0], parts[1]))
#     return pairs


# def canonical(a: str, b: str) -> tuple[str, str]:
#     return (a, b) if a <= b else (b, a)


# def dedup_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
#     """Unique pairs in canonical order, preserving first-occurrence order."""
#     seen: set[tuple[str, str]] = set()
#     out: list[tuple[str, str]] = []
#     for a, b in pairs:
#         key = canonical(a, b)
#         if key not in seen:
#             seen.add(key)
#             out.append(key)
#     return out


# def write_symmetric(results: dict, lang_a: str, lang_b: str, entry: dict) -> None:
#     """Merge entry into results[a][b] AND results[b][a].
#     Uses .update() so successive calls (weight_space, then cka_*) accumulate."""
#     for a, b in [(lang_a, lang_b), (lang_b, lang_a)]:
#         results.setdefault(a, {}).setdefault(b, {}).update(entry)


# def self_similarity(n_layers: int | None = None, include_cka: bool = False) -> dict:
#     entry: dict = {"weight_space": {"cosine_sim": 1.0, "l2_norm": 0.0}}
#     if include_cka:
#         cka_self = {"mean_cka": 1.0, "cka_per_layer": [1.0] * (n_layers or 0)}
#         entry["cka_inlang"] = cka_self
#         entry["cka_eng"]    = cka_self
#     return entry


# # ── Model loading ─────────────────────────────────────────────────────────────

# def load_state_dict(model_name: str, revision: str | None = None) -> dict:
#     print(f"  Loading {model_name}...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, revision=revision,
#         torch_dtype=torch.float16, low_cpu_mem_usage=True,
#     )
#     sd = {k: v.float().cpu() for k, v in model.state_dict().items()}
#     del model
#     gc.collect()
#     torch.cuda.empty_cache()
#     return sd


# # ── Weight-space metrics ──────────────────────────────────────────────────────

# def flatten_params(sd: dict) -> torch.Tensor:
#     return torch.cat([
#         v.flatten()
#         for k, v in sorted(sd.items())
#         if k not in SKIP_KEYS and v.dtype == torch.float32 and v.ndim >= 1
#     ])


# def spectral_norm_power_iter(W: torch.Tensor, n_iter: int = 20) -> float:
#     """Approximate largest singular value via power iteration.

#     ~100× faster than torch.linalg.svdvals on large matrices:
#     O(n_iter·m·n) vs O(m·n·min(m,n)) for full SVD.
#     20 iterations gives <1% error in practice.
#     """
#     v = torch.randn(W.shape[1], dtype=W.dtype, device=W.device)
#     v = v / v.norm()
#     for _ in range(n_iter):
#         u = W @ v;  u = u / u.norm()
#         v = W.T @ u; v = v / v.norm()
#     return (u @ (W @ v)).abs().item()


# def stable_rank(W: torch.Tensor) -> float:
#     """Stable rank = ‖W‖²_F / ‖W‖²_2, spectral norm via power iteration."""
#     fro_sq = torch.sum(W ** 2).item()
#     if fro_sq == 0:
#         return 0.0
#     s_max = spectral_norm_power_iter(W)
#     return fro_sq / (s_max ** 2) if s_max > 0 else float("nan")


# def compute_weight_metrics(sd_a: dict, sd_b: dict,
#                            compute_rank: bool = False) -> dict:
#     vec_a = flatten_params(sd_a)
#     vec_b = flatten_params(sd_b)
#     cos = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
#     l2  = torch.norm(vec_a - vec_b, p=2).item()
#     ws: dict = {"cosine_sim": round(cos, 6), "l2_norm": round(l2, 4)}

#     if compute_rank:
#         rank_diffs = []
#         for k in sorted(set(sd_a) & set(sd_b)):
#             if k in SKIP_KEYS:
#                 continue
#             va, vb = sd_a[k], sd_b[k]
#             if va.ndim < 2:
#                 continue
#             rank_diffs.append(abs(stable_rank(va) - stable_rank(vb)))
#         ws["mean_rank_diff"] = (
#             round(float(np.mean(rank_diffs)), 4) if rank_diffs else None
#         )

#     return {"weight_space": ws}


# # ── CKA ───────────────────────────────────────────────────────────────────────

# def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
#     """Linear CKA (Kornblith et al., 2019) between [n × d_X] and [n × d_Y]."""
#     X, Y = X.double(), Y.double()
#     n = X.shape[0]

#     def gram_and_center(Z: torch.Tensor) -> torch.Tensor:
#         G = Z @ Z.T
#         return G - G.mean(1, keepdim=True) - G.mean(0, keepdim=True) + G.mean()

#     KX, KY = gram_and_center(X), gram_and_center(Y)
#     f = 1.0 / (n - 1) ** 2
#     hsic_xy = f * (KX * KY).sum()
#     hsic_xx = f * (KX * KX).sum()
#     hsic_yy = f * (KY * KY).sum()
#     return (hsic_xy / ((hsic_xx * hsic_yy).sqrt() + 1e-12)).item()


# def load_flores(lang: str, n_samples: int, split: str) -> list[str]:
#     from datasets import load_dataset
#     code = FLORES_CODE.get(lang)
#     if code is None:
#         raise ValueError(f"No FLORES+ code for language: {lang!r}")
#     ds = load_dataset("openlanguagedata/flores_plus", code, split=split,
#                       trust_remote_code=True)
#     field = "text" if "text" in ds.column_names else "sentence"
#     sents = [ex[field] for ex in ds]
#     return sents[:n_samples] if n_samples and n_samples < len(sents) else sents


# @torch.no_grad()
# def collect_activations(
#     model_name: str, revision: str | None, sentences: list[str],
#     device: str, batch_size: int, max_length: int,
# ) -> list[torch.Tensor]:
#     """Mean-pooled hidden states per transformer layer → list of [N × H] CPU tensors.
#     Embedding layer (index 0) is skipped."""
#     print(f"  Loading {model_name} for CKA...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, revision=revision,
#         torch_dtype=torch.float16, low_cpu_mem_usage=True,
#     ).to(device).eval()
#     tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     n_layers = model.config.num_hidden_layers
#     layer_acts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

#     for i in tqdm(range(0, len(sentences), batch_size),
#                   desc="  Activations", leave=False):
#         enc = tokenizer(sentences[i : i + batch_size], return_tensors="pt",
#                         padding=True, truncation=True,
#                         max_length=max_length).to(device)
#         out  = model(**enc, output_hidden_states=True)
#         mask = enc["attention_mask"].unsqueeze(-1).float()
#         for l, hs in enumerate(out.hidden_states[1:]):   # skip embedding
#             pooled = (hs.float() * mask).sum(1) / mask.sum(1)
#             layer_acts[l].append(pooled.cpu())

#     del model
#     gc.collect()
#     torch.cuda.empty_cache()
#     return [torch.cat(acts, dim=0) for acts in layer_acts]


# def compute_cka_entry(acts_a: list[torch.Tensor],
#                       acts_b: list[torch.Tensor],
#                       key: str) -> dict:
#     """Compute per-layer CKA and return as {key: {mean_cka, cka_per_layer}}."""
#     per_layer = [linear_cka(a, b) for a, b in zip(acts_a, acts_b)]
#     return {key: {"mean_cka":      round(float(np.mean(per_layer)), 6),
#                   "cka_per_layer": [round(v, 6) for v in per_layer]}}


# def run_cka_pass(
#     pairs:       list[tuple[str, str]],
#     langs:       list[str],
#     act_cache:   dict[str, list[torch.Tensor]],   # lang → activations
#     results:     dict,
#     cka_key:     str,                              # "cka_inlang" or "cka_eng"
#     n_layers:    int,
# ) -> None:
#     """Compute pairwise CKA from a pre-built activation cache and write into results."""
#     # Update diagonals
#     for l in langs:
#         results[l][l].update({cka_key: {"mean_cka": 1.0,
#                                          "cka_per_layer": [1.0] * n_layers}})
#     print(f"\nComputing pairwise CKA [{cka_key}]...")
#     for lang_a, lang_b in tqdm(pairs, desc=f"CKA [{cka_key}]"):
#         t0 = time.time()
#         entry = compute_cka_entry(act_cache[lang_a], act_cache[lang_b], cka_key)
#         write_symmetric(results, lang_a, lang_b, entry)
#         tqdm.write(f"  {lang_a} × {lang_b}: mean_CKA={entry[cka_key]['mean_cka']:.4f}"
#                    f"  ({time.time()-t0:.1f}s)")


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Pairwise model similarity for monolingual HPLT models.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("--pairs",     required=True)
#     parser.add_argument("--output",    default="similarity_results.json")
#     parser.add_argument("--revision",  default=None,
#                         help="HF checkpoint revision, e.g. 'checkpoint-47684'")
#     parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--no_sim",    action="store_true",
#                         help="Skip weight-space metrics")
#     parser.add_argument("--rank_diff", action="store_true",
#                         help="Also compute mean stable-rank difference (slow; power iteration)")

#     g = parser.add_argument_group("CKA")
#     g.add_argument("--cka",            action="store_true",
#                    help="Compute CKA under two conditions: in-language and English input")
#     g.add_argument("--cka_n_samples",  type=int, default=1012,
#                    help="FLORES sentences per language (full devtest=1012, recommended)")
#     g.add_argument("--cka_batch_size", type=int, default=8)
#     g.add_argument("--cka_max_length", type=int, default=128)
#     g.add_argument("--flores_split",   default="devtest", choices=["dev", "devtest"])

#     args = parser.parse_args()

#     raw_pairs = load_pairs(args.pairs)
#     pairs = dedup_pairs(raw_pairs)   # canonical, no duplicates — computed once each

#     seen, langs = set(), []
#     for a, b in pairs:
#         for l in (a, b):
#             if l not in seen:
#                 langs.append(l)
#                 seen.add(l)

#     print(f"Languages ({len(langs)}): {langs}")
#     print(f"Unique pairs: {len(pairs)} (from {len(raw_pairs)} in file)")

#     results: dict[str, dict] = {}
#     for l in langs:
#         results.setdefault(l, {})[l] = self_similarity()   # diagonals updated later

#     # ── 1. Weight-space metrics ───────────────────────────────────────────────
#     if not args.no_sim:
#         print("\n=== Weight-space metrics ===")
#         # Group by anchor so each anchor model is loaded exactly once
#         anchor_to_partners: dict[str, list[str]] = defaultdict(list)
#         for a, b in pairs:
#             anchor_to_partners[a].append(b)

#         for i, (lang_a, partners) in enumerate(anchor_to_partners.items()):
#             print(f"\n[{i+1}/{len(anchor_to_partners)}] Anchor: {lang_a}")
#             sd_a = load_state_dict(LANG_TO_MODEL[lang_a], args.revision)

#             for lang_b in partners:
#                 t0 = time.time()
#                 print(f"  ↳ {lang_a} × {lang_b}", end=" ... ", flush=True)
#                 sd_b = load_state_dict(LANG_TO_MODEL[lang_b], args.revision)
#                 m    = compute_weight_metrics(sd_a, sd_b, compute_rank=args.rank_diff)
#                 write_symmetric(results, lang_a, lang_b, m)
#                 del sd_b;  gc.collect()
#                 ws = m["weight_space"]
#                 msg = (f"cos={ws['cosine_sim']:.4f}  L2={ws['l2_norm']:.1f}")
#                 if args.rank_diff:
#                     msg += f"  rank_diff={ws.get('mean_rank_diff'):.4f}"
#                 print(f"{msg}  ({time.time()-t0:.1f}s)")

#             del sd_a;  gc.collect();  torch.cuda.empty_cache()

#     # ── 2. CKA ───────────────────────────────────────────────────────────────
#     if args.cka:
#         print("\n=== CKA ===")

#         # ── 2a. In-language pass: each model sees its own language ────────────
#         # Rationale: monolingual models each process text in their native script;
#         # parallel FLORES ensures the same semantic content reaches both models.
#         print("\n[Pass 1/2] In-language: loading FLORES+ per language...")
#         inlang_sents: dict[str, list[str]] = {}
#         for l in langs:
#             inlang_sents[l] = load_flores(l, args.cka_n_samples, args.flores_split)
#         n_inlang = min(len(s) for s in inlang_sents.values())
#         inlang_sents = {l: s[:n_inlang] for l, s in inlang_sents.items()}
#         print(f"  → {n_inlang} sentences per language")

#         print("\nCollecting in-language activations (each model loaded once)...")
#         act_inlang: dict[str, list[torch.Tensor]] = {}
#         for l in langs:
#             t0 = time.time()
#             act_inlang[l] = collect_activations(
#                 LANG_TO_MODEL[l], args.revision, inlang_sents[l],
#                 args.device, args.cka_batch_size, args.cka_max_length,
#             )
#             print(f"  ✓ {l}: {len(act_inlang[l])} layers, "
#                   f"hidden={act_inlang[l][0].shape[1]}  ({time.time()-t0:.1f}s)")

#         n_layers = len(act_inlang[langs[0]])
#         run_cka_pass(pairs, langs, act_inlang, results, "cka_inlang", n_layers)
#         del act_inlang;  gc.collect();  torch.cuda.empty_cache()

#         # ── 2b. English pass: all models see the same English sentences ───────
#         # Rationale: fixed reference input controls for content variation.
#         # Expected: non-English models score lower (never trained on English),
#         # so divergence from cka_inlang is itself interpretable.
#         print("\n[Pass 2/2] English: loading FLORES+ English sentences...")
#         eng_sents = load_flores("eng", args.cka_n_samples, args.flores_split)
#         eng_sents = eng_sents[:n_inlang]   # same N as in-language pass
#         print(f"  → {len(eng_sents)} English sentences for all models")

#         print("\nCollecting English activations (each model loaded once)...")
#         act_eng: dict[str, list[torch.Tensor]] = {}
#         for l in langs:
#             t0 = time.time()
#             act_eng[l] = collect_activations(
#                 LANG_TO_MODEL[l], args.revision, eng_sents,
#                 args.device, args.cka_batch_size, args.cka_max_length,
#             )
#             print(f"  ✓ {l}  ({time.time()-t0:.1f}s)")

#         run_cka_pass(pairs, langs, act_eng, results, "cka_eng", n_layers)
#         del act_eng;  gc.collect();  torch.cuda.empty_cache()

#         # Update diagonals with correct layer count
#         for l in langs:
#             results[l][l].update(self_similarity(n_layers=n_layers, include_cka=True))

#     # ── 3. Save ───────────────────────────────────────────────────────────────
#     out_path = Path(args.output)
#     with open(out_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nResults written to {out_path}")

#     # ── 4. Summary table ──────────────────────────────────────────────────────
#     print("\n── Summary ──────────────────────────────────────────────────────────")
#     hdr = f"{'Pair':<14} {'cos_sim':>9} {'l2_norm':>10}"
#     if args.rank_diff:
#         hdr += f" {'rank_diff':>10}"
#     if args.cka:
#         hdr += f" {'cka_inlang':>10} {'cka_eng':>10}"
#     print(hdr)
#     print("─" * len(hdr))
#     for a, b in pairs:
#         v  = results.get(a, {}).get(b, {})
#         ws = v.get("weight_space", {})
#         row = (f"{a+' × '+b:<14} "
#                f"{ws.get('cosine_sim', float('nan')):>9.4f} "
#                f"{ws.get('l2_norm',    float('nan')):>10.1f}")
#         if args.rank_diff:
#             row += f" {ws.get('mean_rank_diff', float('nan')):>10.4f}"
#         if args.cka:
#             row += (f" {v.get('cka_inlang', {}).get('mean_cka', float('nan')):>10.4f}"
#                     f" {v.get('cka_eng',    {}).get('mean_cka', float('nan')):>10.4f}")
#         print(row)


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# Compute pairwise model similarity metrics for monolingual HPLT models.

# Weight-space metrics (no data needed):
#   - Cosine similarity of flattened parameters
#   - L2 / Frobenius norm of parameter difference

# Optional representation-space metric (requires GPU + data):
#   - Layer-wise linear CKA using parallel FLORES+ sentences
#     Each model sees FLORES in its OWN language (parallel input).
#     This is the fairest comparison for monolingual models:
#     we ask whether independently trained models develop similar
#     internal geometry when processing the same concepts in their
#     respective languages.
#     Full devtest (1012 sentences) is used by default — enough for
#     stable Gram matrix estimates without being excessive.

# Output: nested JSON matching language-distance format:
#   { lang_a: { lang_b: { weight_space: {...}, cka: {...} } } }
#   Symmetry: results[a][b] and results[b][a] always share the same values
#   (computed once, written to both keys).
#   Diagonal entries (lang == lang) contain self-similarity values
#   (cosine=1, L2=0, mean_cka=1).

# Usage
# -----
# # Weight metrics only (final checkpoint):
# python compute_model_similarity.py --pairs pairs.txt

# # Specific checkpoint revision:
# python compute_model_similarity.py --pairs pairs.txt --revision checkpoint-47684

# # Include CKA:
# python compute_model_similarity.py --pairs pairs.txt --cka

# # Skip weight metrics (CKA only):
# python compute_model_similarity.py --pairs pairs.txt --cka --no_sim

# # Full options:
# python compute_model_similarity.py \\
#     --pairs pairs.txt \\
#     --output similarity_results.json \\
#     --revision checkpoint-47684 \\
#     --cka \\
#     --cka_n_samples 1012 \\
#     --cka_batch_size 8 \\
#     --cka_max_length 128 \\
#     --flores_split devtest \\
#     --device cuda
# """

# import argparse
# import gc
# import json
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer


# # ── Config ────────────────────────────────────────────────────────────────────

# LANG_TO_MODEL = {
#     "eng":  "HPLT/hplt2c_eng_checkpoints",
#     "nld":  "HPLT/hplt2c_nld_checkpoints",
#     "spa":  "HPLT/hplt2c_spa_checkpoints",
#     "fra":  "HPLT/hplt2c_fra_checkpoints",
#     "rus":  "HPLT/hplt2c_rus_checkpoints",
#     "tur":  "HPLT/hplt2c_tur_checkpoints",
#     "ita":  "HPLT/hplt2c_ita_checkpoints",
#     "ara":  "HPLT/hplt2c_ara_checkpoints",
#     "deu":  "HPLT/hplt2c_deu_checkpoints",
#     "zhos": "HPLT/hplt2c_zhos_checkpoints",
# }

# FLORES_CODE = {
#     "eng":  "eng_Latn",
#     "nld":  "nld_Latn",
#     "spa":  "spa_Latn",
#     "fra":  "fra_Latn",
#     "rus":  "rus_Cyrl",
#     "tur":  "tur_Latn",
#     "ita":  "ita_Latn",
#     "ara":  "arb_Arab",
#     "deu":  "deu_Latn",
#     "zhos": "cmn_Hans",
# }

# SKIP_KEYS = {"lm_head.weight"}  # tied weights — skip to avoid double-counting


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def load_pairs(path: str) -> list[tuple[str, str]]:
#     pairs = []
#     with open(path) as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 pairs.append((parts[0], parts[1]))
#     return pairs


# def canonical(a: str, b: str) -> tuple[str, str]:
#     """Always return the pair in a consistent (alphabetically first, second) order."""
#     return (a, b) if a <= b else (b, a)


# def dedup_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
#     """Return unique pairs in canonical order, preserving first-occurrence order."""
#     seen: set[tuple[str, str]] = set()
#     out: list[tuple[str, str]] = []
#     for a, b in pairs:
#         key = canonical(a, b)
#         if key not in seen:
#             seen.add(key)
#             out.append(key)
#     return out


# def write_symmetric(results: dict, lang_a: str, lang_b: str, entry: dict) -> None:
#     """Write entry into results[a][b] AND results[b][a] (same dict object, no copy).
#     Uses .update() so existing keys (e.g. weight_space) are preserved when adding cka.
#     """
#     for a, b in [(lang_a, lang_b), (lang_b, lang_a)]:
#         results.setdefault(a, {}).setdefault(b, {}).update(entry)


# def self_similarity(n_layers: int | None = None) -> dict:
#     entry: dict = {"weight_space": {"cosine_sim": 1.0, "l2_norm": 0.0}}
#     if n_layers is not None:
#         entry["cka"] = {"mean_cka": 1.0, "cka_per_layer": [1.0] * n_layers}
#     return entry


# # ── Model loading ─────────────────────────────────────────────────────────────

# def load_state_dict(model_name: str, revision: str | None = None) -> dict:
#     """Load state dict to CPU as float32 (cast from fp16 for numerical stability)."""
#     print(f"  Loading {model_name}...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         revision=revision,
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#     )
#     sd = {k: v.float().cpu() for k, v in model.state_dict().items()}
#     del model
#     gc.collect()
#     torch.cuda.empty_cache()
#     return sd


# # ── Weight-space metrics ──────────────────────────────────────────────────────

# def flatten_params(sd: dict) -> torch.Tensor:
#     return torch.cat([
#         v.flatten()
#         for k, v in sorted(sd.items())
#         if k not in SKIP_KEYS and v.dtype == torch.float32 and v.ndim >= 1
#     ])


# def compute_weight_metrics(sd_a: dict, sd_b: dict) -> dict:
#     vec_a = flatten_params(sd_a)
#     vec_b = flatten_params(sd_b)
#     cos = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
#     l2  = torch.norm(vec_a - vec_b, p=2).item()
#     return {"weight_space": {"cosine_sim": round(cos, 6), "l2_norm": round(l2, 4)}}


# # ── CKA ───────────────────────────────────────────────────────────────────────

# def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
#     """Linear CKA (Kornblith et al., 2019) between [n × d_X] and [n × d_Y]."""
#     X, Y = X.double(), Y.double()
#     n = X.shape[0]

#     def gram_and_center(Z: torch.Tensor) -> torch.Tensor:
#         G = Z @ Z.T
#         return G - G.mean(1, keepdim=True) - G.mean(0, keepdim=True) + G.mean()

#     KX, KY = gram_and_center(X), gram_and_center(Y)
#     f = 1.0 / (n - 1) ** 2
#     hsic_xy = f * (KX * KY).sum()
#     hsic_xx = f * (KX * KX).sum()
#     hsic_yy = f * (KY * KY).sum()
#     return (hsic_xy / ((hsic_xx * hsic_yy).sqrt() + 1e-12)).item()


# def load_flores(lang: str, n_samples: int, split: str) -> list[str]:
#     from datasets import load_dataset
#     code = FLORES_CODE.get(lang)
#     if code is None:
#         raise ValueError(f"No FLORES+ code configured for language: {lang!r}")
#     ds = load_dataset("openlanguagedata/flores_plus", code, split=split,
#                       trust_remote_code=True)
#     field = "text" if "text" in ds.column_names else "sentence"
#     sents = [ex[field] for ex in ds]
#     return sents[:n_samples] if n_samples and n_samples < len(sents) else sents


# @torch.no_grad()
# def collect_activations(
#     model_name: str, revision: str | None, sentences: list[str],
#     device: str, batch_size: int, max_length: int,
# ) -> list[torch.Tensor]:
#     """Mean-pooled hidden states per layer → list of [N × H] CPU tensors."""
#     print(f"  Loading {model_name} for CKA...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, revision=revision, torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#     ).to(device).eval()

#     tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     n_layers = model.config.num_hidden_layers
#     layer_acts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

#     for i in tqdm(range(0, len(sentences), batch_size), desc="  Activations", leave=False):
#         enc = tokenizer(sentences[i : i + batch_size], return_tensors="pt",
#                         padding=True, truncation=True,
#                         max_length=max_length).to(device)
#         out = model(**enc, output_hidden_states=True)
#         mask = enc["attention_mask"].unsqueeze(-1).float()
#         for l, hs in enumerate(out.hidden_states[1:]):   # skip embedding layer
#             pooled = (hs.float() * mask).sum(1) / mask.sum(1)
#             layer_acts[l].append(pooled.cpu())

#     del model
#     gc.collect()
#     torch.cuda.empty_cache()
#     return [torch.cat(acts, dim=0) for acts in layer_acts]


# def compute_cka_entry(acts_a: list[torch.Tensor],
#                       acts_b: list[torch.Tensor]) -> dict:
#     per_layer = [linear_cka(a, b) for a, b in zip(acts_a, acts_b)]
#     return {"cka": {"mean_cka": round(float(np.mean(per_layer)), 6),
#                     "cka_per_layer": [round(v, 6) for v in per_layer]}}


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Pairwise model similarity metrics for monolingual HPLT models.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("--pairs",     required=True)
#     parser.add_argument("--output",    default="similarity_results.json")
#     parser.add_argument("--revision",  default=None)
#     parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--no_sim",    action="store_true",
#                         help="Skip weight-space metrics")
#     parser.add_argument("--rank_diff", action="store_true",
#                         help="Also compute mean stable-rank difference (slow)")

#     g = parser.add_argument_group("CKA")
#     g.add_argument("--cka",            action="store_true")
#     g.add_argument("--cka_n_samples",  type=int, default=1012)
#     g.add_argument("--cka_batch_size", type=int, default=8)
#     g.add_argument("--cka_max_length", type=int, default=128)
#     g.add_argument("--flores_split",   default="devtest", choices=["dev", "devtest"])

#     args = parser.parse_args()

#     raw_pairs = load_pairs(args.pairs)
#     # Deduplicate: treat (a,b) and (b,a) as the same pair — compute only once.
#     pairs = dedup_pairs(raw_pairs)

#     seen, langs = set(), []
#     for a, b in pairs:
#         for l in (a, b):
#             if l not in seen:
#                 langs.append(l)
#                 seen.add(l)

#     print(f"Languages ({len(langs)}): {langs}")
#     print(f"Unique pairs: {len(pairs)} (from {len(raw_pairs)} in file)")

#     results: dict[str, dict] = {}

#     # Diagonal
#     for l in langs:
#         results.setdefault(l, {})[l] = self_similarity()

#     # ── 1. Weight-space metrics ───────────────────────────────────────────────
#     if not args.no_sim:
#         print("\n=== Weight-space metrics ===")
#         # pairs is already deduplicated and canonical, so each (a,b) appears once.
#         # Group by anchor (a) to minimise model loads: load a once, iterate all b's.
#         from collections import defaultdict
#         anchor_to_partners: dict[str, list[str]] = defaultdict(list)
#         for a, b in pairs:
#             anchor_to_partners[a].append(b)

#         for i, (lang_a, partners) in enumerate(anchor_to_partners.items()):
#             print(f"\n[{i+1}/{len(anchor_to_partners)}] Anchor: {lang_a}")
#             sd_a = load_state_dict(LANG_TO_MODEL[lang_a], args.revision)

#             for lang_b in partners:
#                 t0 = time.time()
#                 print(f"  ↳ {lang_a} × {lang_b}", end=" ... ", flush=True)
#                 sd_b = load_state_dict(LANG_TO_MODEL[lang_b], args.revision)
#                 m = compute_weight_metrics(sd_a, sd_b)
#                 # write_symmetric writes to BOTH (a,b) and (b,a) from one computation
#                 write_symmetric(results, lang_a, lang_b, m)
#                 del sd_b
#                 gc.collect()
#                 ws = m["weight_space"]
#                 print(f"cos={ws['cosine_sim']:.4f}  L2={ws['l2_norm']:.1f}  "
#                       f"({time.time()-t0:.1f}s)")

#             del sd_a
#             gc.collect()
#             torch.cuda.empty_cache()

#     # ── 2. CKA ───────────────────────────────────────────────────────────────
#     if args.cka:
#         print("\n=== CKA (parallel FLORES+, each model sees its own language) ===")

#         print("\nLoading FLORES+ sentences...")
#         lang_to_sents: dict[str, list[str]] = {}
#         for l in langs:
#             lang_to_sents[l] = load_flores(l, args.cka_n_samples, args.flores_split)
#         # Truncate to equal length across all languages so Gram matrices are comparable
#         actual_n = min(len(s) for s in lang_to_sents.values())
#         lang_to_sents = {l: s[:actual_n] for l, s in lang_to_sents.items()}
#         print(f"  → {actual_n} sentences per language")

#         print("\nCollecting activations (each model loaded exactly once)...")
#         act_cache: dict[str, list[torch.Tensor]] = {}
#         for l in langs:
#             t0 = time.time()
#             act_cache[l] = collect_activations(
#                 LANG_TO_MODEL[l], args.revision, lang_to_sents[l],
#                 args.device, args.cka_batch_size, args.cka_max_length,
#             )
#             n_lay = len(act_cache[l])
#             print(f"  ✓ {l}: {n_lay} layers, hidden={act_cache[l][0].shape[1]}  "
#                   f"({time.time()-t0:.1f}s)")

#         n_layers = len(act_cache[langs[0]])

#         # Update diagonals with correct layer count now that we know n_layers
#         for l in langs:
#             results[l][l].update({"cka": {"mean_cka": 1.0,
#                                            "cka_per_layer": [1.0] * n_layers}})

#         # pairs is deduplicated — compute CKA once per unordered pair,
#         # write_symmetric mirrors the result to both (a,b) and (b,a).
#         print("\nComputing pairwise CKA...")
#         for lang_a, lang_b in tqdm(pairs, desc="CKA"):
#             t0 = time.time()
#             entry = compute_cka_entry(act_cache[lang_a], act_cache[lang_b])
#             write_symmetric(results, lang_a, lang_b, entry)
#             tqdm.write(f"  {lang_a} × {lang_b}: mean_CKA={entry['cka']['mean_cka']:.4f}  "
#                        f"({time.time()-t0:.1f}s)")

#     # ── 3. Save ───────────────────────────────────────────────────────────────
#     out_path = Path(args.output)
#     with open(out_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nResults written to {out_path}")

#     # ── 4. Summary table ──────────────────────────────────────────────────────
#     print("\n── Summary ─────────────────────────────────────────────────────")
#     hdr = f"{'Pair':<14} {'cos_sim':>9} {'l2_norm':>10}"
#     if args.cka:
#         hdr += f" {'mean_cka':>10}"
#     print(hdr)
#     print("─" * len(hdr))
#     for a, b in pairs:
#         v  = results.get(a, {}).get(b, {})
#         ws = v.get("weight_space", {})
#         row = (f"{a+' × '+b:<14} "
#                f"{ws.get('cosine_sim', float('nan')):>9.4f} "
#                f"{ws.get('l2_norm',    float('nan')):>10.1f}")
#         if args.cka:
#             row += f" {v.get('cka', {}).get('mean_cka', float('nan')):>10.4f}"
#         print(row)


# if __name__ == "__main__":
#     main()



# # #!/usr/bin/env python3
# # """
# # Compute pairwise model similarity metrics for monolingual HPLT models.

# # Weight-space metrics (no data needed):
# #   - Cosine similarity of flattened parameters
# #   - L2 / Frobenius norm of parameter difference
# #   - Mean absolute stable-rank difference across weight matrices

# # Optional representation-space metric (requires GPU + data):
# #   - Layer-wise linear CKA using parallel FLORES+ sentences
# #     Each model sees FLORES in its OWN language (parallel input).
# #     This is the fairest comparison for monolingual models:
# #     we ask whether independently trained models develop similar
# #     internal geometry when processing the same concepts in their
# #     respective languages.
# #     Full devtest (1012 sentences) is used by default — enough for
# #     stable Gram matrix estimates without being excessive.

# # Output: nested JSON matching language-distance format:
# #   { lang_a: { lang_b: { weight_space: {...}, cka: {...} } } }
# #   Diagonal entries (lang == lang) contain self-similarity values
# #   (cosine=1, L2=0, rank_diff=0, mean_cka=1).

# # Usage
# # -----
# # # Weight metrics only (final checkpoint):
# # python compute_model_similarity.py --pairs pairs.txt

# # # Specific checkpoint revision:
# # python compute_model_similarity.py --pairs pairs.txt --revision checkpoint-47684

# # # Include CKA:
# # python compute_model_similarity.py --pairs pairs.txt --cka

# # # Full options:
# # python compute_model_similarity.py \\
# #     --pairs pairs.txt \\
# #     --output similarity_results.json \\
# #     --revision checkpoint-47684 \\
# #     --cka \\
# #     --cka_n_samples 1012 \\
# #     --cka_batch_size 8 \\
# #     --cka_max_length 128 \\
# #     --flores_split devtest \\
# #     --device cuda
# # """

# # import time
# # import argparse
# # import gc
# # import json
# # from pathlib import Path

# # import numpy as np
# # import torch
# # import torch.nn.functional as F
# # from tqdm import tqdm
# # from transformers import AutoModelForCausalLM, AutoTokenizer


# # # ── Config ────────────────────────────────────────────────────────────────────

# # LANG_TO_MODEL = {
# #     "eng":  "HPLT/hplt2c_eng_checkpoints",
# #     "nld":  "HPLT/hplt2c_nld_checkpoints",
# #     "spa":  "HPLT/hplt2c_spa_checkpoints",
# #     "fra":  "HPLT/hplt2c_fra_checkpoints",
# #     "rus":  "HPLT/hplt2c_rus_checkpoints",
# #     "tur":  "HPLT/hplt2c_tur_checkpoints",
# #     "ita":  "HPLT/hplt2c_ita_checkpoints",
# #     "ara":  "HPLT/hplt2c_ara_checkpoints",
# #     "deu":  "HPLT/hplt2c_deu_checkpoints",
# #     "zhos": "HPLT/hplt2c_zhos_checkpoints",
# # }

# # # flores_plus config names (openlanguagedata/flores_plus)
# # FLORES_CODE = {
# #     "eng":  "eng_Latn",
# #     "nld":  "nld_Latn",
# #     "spa":  "spa_Latn",
# #     "fra":  "fra_Latn",
# #     "rus":  "rus_Cyrl",
# #     "tur":  "tur_Latn",
# #     "ita":  "ita_Latn",
# #     "ara":  "arb_Arab",
# #     "deu":  "deu_Latn",
# #     "zhos": "cmn_Hans",
# # }

# # SKIP_KEYS = {"lm_head.weight"}  # tied weights — skip to avoid double-counting


# # # ── I/O ───────────────────────────────────────────────────────────────────────

# # def load_pairs(path: str) -> list[tuple[str, str]]:
# #     pairs = []
# #     with open(path) as f:
# #         for line in f:
# #             parts = line.strip().split()
# #             if len(parts) == 2:
# #                 pairs.append((parts[0], parts[1]))
# #     return pairs


# # def nested_set(d: dict, lang_a: str, lang_b: str, metrics: dict) -> None:
# #     """Write metrics into d[lang_a][lang_b] and d[lang_b][lang_a] (symmetric)."""
# #     for a, b in [(lang_a, lang_b), (lang_b, lang_a)]:
# #         d.setdefault(a, {})[b] = metrics


# # def self_similarity(n_layers: int | None = None) -> dict:
# #     """Diagonal entry: a model compared with itself."""
# #     entry = {
# #         "weight_space": {
# #             "cosine_sim":     1.0,
# #             "l2_norm":        0.0,
# #             "mean_rank_diff": 0.0,
# #         }
# #     }
# #     if n_layers is not None:
# #         entry["cka"] = {
# #             "mean_cka":     1.0,
# #             "cka_per_layer": [1.0] * n_layers,
# #         }
# #     return entry


# # # ── Model loading ─────────────────────────────────────────────────────────────

# # def load_state_dict(model_name: str, revision: str | None = None) -> dict:
# #     """Load state dict to CPU as float32 (cast from fp16 for numerical stability)."""
# #     print(f"Loading {model_name}...")
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         revision=revision,
# #         torch_dtype=torch.float16,
# #         low_cpu_mem_usage=True,
# #     )
# #     sd = {k: v.float().cpu() for k, v in model.state_dict().items()}
# #     print(f"Loaded {len(sd)} parameters")
# #     del model
# #     gc.collect()
# #     torch.cuda.empty_cache()
# #     return sd


# # # ── Weight-space metrics ──────────────────────────────────────────────────────

# # def flatten_params(sd: dict) -> torch.Tensor:
# #     return torch.cat([
# #         v.flatten()
# #         for k, v in sorted(sd.items())
# #         if k not in SKIP_KEYS and v.dtype == torch.float32 and v.ndim >= 1
# #     ])


# # def spectral_norm_power_iter(W: torch.Tensor, n_iter: int = 20) -> float:
# #     """Approximate largest singular value via power iteration.

# #     ~100x faster than torch.linalg.svdvals on large matrices:
# #     O(n_iter * m * n) vs O(m * n * min(m,n)) for full SVD.
# #     20 iterations is sufficient for <1% error in practice.
# #     Only used if --rank_diff is passed.
# #     """
# #     v = torch.randn(W.shape[1], dtype=W.dtype, device=W.device)
# #     v = v / v.norm()
# #     for _ in range(n_iter):
# #         u = W @ v;  u = u / u.norm()
# #         v = W.T @ u; v = v / v.norm()
# #     return (u @ (W @ v)).abs().item()


# # def stable_rank(W: torch.Tensor) -> float:
# #     """Stable rank = ||W||²_F / ||W||²_2 (approximated via power iteration)."""
# #     fro_sq = torch.sum(W ** 2).item()
# #     if fro_sq == 0:
# #         return 0.0
# #     s_max = spectral_norm_power_iter(W)
# #     return fro_sq / (s_max ** 2) if s_max > 0 else float("nan")


# # def compute_weight_metrics(sd_a: dict, sd_b: dict, compute_rank: bool = False) -> dict:
# #     vec_a = flatten_params(sd_a)
# #     vec_b = flatten_params(sd_b)

# #     print("computing cosine similarity...")
# #     cos = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
# #     print("computing L2 norm...")
# #     l2  = torch.norm(vec_a - vec_b, p=2).item()
    

# #     entry: dict = {
# #         "weight_space": {
# #             "cosine_sim": round(cos, 6),
# #             "l2_norm":    round(l2, 4),
# #         }
# #     }

# #     if compute_rank:
# #         print("computing mean rank difference...")
# #         rank_diffs = []
# #         for k in sorted(set(sd_a) & set(sd_b)):
# #             if k in SKIP_KEYS:
# #                 continue
# #             va, vb = sd_a[k], sd_b[k]
# #             if va.ndim < 2:
# #                 continue
# #             rank_diffs.append(abs(stable_rank(va) - stable_rank(vb)))
# #         entry["weight_space"]["mean_rank_diff"] = (
# #             round(float(np.mean(rank_diffs)), 4) if rank_diffs else None
# #         )

# #     return entry


# # # ── CKA ───────────────────────────────────────────────────────────────────────

# # def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
# #     """Linear CKA (Kornblith et al., 2019) between [n × d_X] and [n × d_Y].

# #     Uses the double-centred Gram matrix formulation:
# #       CKA = HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))
# #     where K = XX^T, L = YY^T, both double-centred.
# #     """
# #     X = X.double()
# #     Y = Y.double()
# #     n = X.shape[0]

# #     def gram_and_center(Z: torch.Tensor) -> torch.Tensor:
# #         G = Z @ Z.T
# #         row  = G.mean(1, keepdim=True)
# #         col  = G.mean(0, keepdim=True)
# #         tot  = G.mean()
# #         return G - row - col + tot

# #     KX = gram_and_center(X)
# #     KY = gram_and_center(Y)

# #     factor = 1.0 / (n - 1) ** 2
# #     hsic_xy = factor * (KX * KY).sum()
# #     hsic_xx = factor * (KX * KX).sum()
# #     hsic_yy = factor * (KY * KY).sum()

# #     denom = (hsic_xx * hsic_yy).sqrt()
# #     return (hsic_xy / (denom + 1e-12)).item()


# # def load_flores(lang: str, n_samples: int, split: str) -> list[str]:
# #     """Load FLORES+ sentences for one language from openlanguagedata/flores_plus."""
# #     from datasets import load_dataset

# #     code = FLORES_CODE.get(lang)
# #     if code is None:
# #         raise ValueError(f"No FLORES+ code configured for language: {lang!r}")

# #     ds = load_dataset(
# #         "openlanguagedata/flores_plus",
# #         code,
# #         split=split,
# #         trust_remote_code=True,
# #     )
# #     # Field is 'text' in flores_plus; fall back to 'sentence' for compatibility
# #     field = "text" if "text" in ds.column_names else "sentence"
# #     sentences = [ex[field] for ex in ds]

# #     if n_samples and n_samples < len(sentences):
# #         sentences = sentences[:n_samples]
# #     return sentences


# # @torch.no_grad()
# # def collect_activations(
# #     model_name: str,
# #     revision: str | None,
# #     sentences: list[str],
# #     device: str,
# #     batch_size: int,
# #     max_length: int,
# # ) -> list[torch.Tensor]:
# #     """Run sentences through model; return mean-pooled hidden states per layer
# #     as a list of [n_samples × hidden_size] CPU tensors (layer 0 / embedding skipped)."""
# #     print(f"    Loading {model_name} for CKA...")
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         revision=revision,
# #         torch_dtype=torch.float16,
# #         low_cpu_mem_usage=True,
# #     ).to(device).eval()

# #     tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token

# #     n_layers = model.config.num_hidden_layers
# #     layer_acts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

# #     for i in tqdm(range(0, len(sentences), batch_size),
# #                   desc="    Activations", leave=False):
# #         batch = sentences[i : i + batch_size]
# #         enc = tokenizer(
# #             batch,
# #             return_tensors="pt",
# #             padding=True,
# #             truncation=True,
# #             max_length=max_length,
# #         ).to(device)
# #         out = model(**enc, output_hidden_states=True)
# #         mask = enc["attention_mask"].unsqueeze(-1).float()  # [B × T × 1]
# #         for l, hs in enumerate(out.hidden_states[1:]):      # skip embedding layer (idx 0)
# #             pooled = (hs.float() * mask).sum(1) / mask.sum(1)  # mean pool → [B × H]
# #             layer_acts[l].append(pooled.cpu())

# #     del model
# #     gc.collect()
# #     torch.cuda.empty_cache()

# #     return [torch.cat(acts, dim=0) for acts in layer_acts]  # [N × H] per layer


# # def compute_cka_entry(acts_a: list[torch.Tensor],
# #                       acts_b: list[torch.Tensor]) -> dict:
# #     per_layer = [linear_cka(a, b) for a, b in zip(acts_a, acts_b)]
# #     return {
# #         "cka": {
# #             "mean_cka":      round(float(np.mean(per_layer)), 6),
# #             "cka_per_layer": [round(v, 6) for v in per_layer],
# #         }
# #     }


# # # ── Main ──────────────────────────────────────────────────────────────────────

# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Pairwise model similarity metrics for monolingual HPLT models.",
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# #     )
# #     parser.add_argument("--pairs",    required=True, help="Path to pairs.txt")
# #     parser.add_argument("--output",   default="similarity_results.json")
# #     parser.add_argument("--revision", default=None,
# #                         help="HF checkpoint revision, e.g. 'checkpoint-47684'")
# #     parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
# #     parser.add_argument("--rank_diff", action="store_true",
# #                         help="Also compute mean stable-rank difference (slow; uses power "
# #                              "iteration). Not recommended unless specifically needed.")
# #     parser.add_argument("--no_sim", action="store_true", help="Do not compute similarity metrics")

# #     g = parser.add_argument_group("CKA")
# #     g.add_argument("--cka", action="store_true",
# #                    help="Compute CKA (parallel FLORES+ input, each model sees its own language)")
# #     g.add_argument("--cka_n_samples",  type=int, default=1012,
# #                    help="FLORES sentences per language. Full devtest=1012 (recommended).")
# #     g.add_argument("--cka_batch_size", type=int, default=8)
# #     g.add_argument("--cka_max_length", type=int, default=128)
# #     g.add_argument("--flores_split",   default="devtest", choices=["dev", "devtest"])

# #     args = parser.parse_args()

# #     pairs = load_pairs(args.pairs)
# #     seen, langs = set(), []
# #     for a, b in pairs:
# #         for l in (a, b):
# #             if l not in seen:
# #                 langs.append(l)
# #                 seen.add(l)

# #     print(f"Languages ({len(langs)}): {langs}")
# #     print(f"Pairs: {len(pairs)}")

# #     # results[lang_a][lang_b] = { weight_space: {...}, cka?: {...} }
# #     results: dict[str, dict] = {}

# #     # ── Diagonal (self-similarity) ────────────────────────────────────────────
# #     for l in langs:
# #         results.setdefault(l, {})[l] = self_similarity()  # CKA layer count added later

# #     # ── 1. Weight-space metrics ───────────────────────────────────────────────
# #     # Each anchor model i is loaded once; all j > i partners are loaded
# #     # sequentially and discarded before loading the next. Only 2 state
# #     # dicts reside in memory at any point.
# #     print("\n=== Weight-space metrics ===")
# #     pairs_set = {(a, b) for a, b in pairs} | {(b, a) for a, b in pairs}

# #     # initialize results dictionary with all langs
# #     for lang_a in langs:
# #         for lang_b in langs:
# #             if lang_a == lang_b:
# #                 results[lang_a][lang_b] = self_similarity()
# #             else:
# #                 results[lang_a][lang_b] = {
# #                     "weight_space": {
# #                         "cosine_sim": float('nan'),
# #                         "l2_norm": float('nan'),
# #                         "mean_rank_diff": float('nan'),
# #                     }
# #                 }

# #     if not args.no_sim:
# #         for i, lang_a in enumerate(langs):
# #             needs_b = [l for l in langs[i + 1:] if (lang_a, l) in pairs_set]
# #             if not needs_b:
# #                 continue

# #             print(f"\n[{i+1}/{len(langs)}] Anchor: {lang_a}")
# #             sd_a = load_state_dict(LANG_TO_MODEL[lang_a], args.revision)

# #             for lang_b in needs_b:
# #                 start_time = time.time()
# #                 print(f"  ↳ {lang_a} × {lang_b}", end=" ... ", flush=True)
# #                 sd_b = load_state_dict(LANG_TO_MODEL[lang_b], args.revision)
# #                 m = compute_weight_metrics(sd_a, sd_b, compute_rank=args.rank_diff)
# #                 nested_set(results, lang_a, lang_b, m)
# #                 del sd_b
# #                 gc.collect()
# #                 print(f"cos={m['weight_space']['cosine_sim']:.4f}  "
# #                     f"L2={m['weight_space']['l2_norm']:.1f}", end=" ")
# #                 if args.rank_diff:
# #                     print(f"rank_diff={m['weight_space']['mean_rank_diff']:.4f}")
# #                 end_time = time.time()
# #                 print(f"Time taken: {end_time - start_time:.2f} seconds")
# #             del sd_a
# #             gc.collect()
# #             torch.cuda.empty_cache()

# #     # ── 2. CKA ───────────────────────────────────────────────────────────────
# #     if args.cka:
# #         print("\n=== CKA (parallel FLORES+ input) ===")
# #         print(f"Each model sees its own language ({args.cka_n_samples} sentences, "
# #               f"split={args.flores_split}).\n"
# #               "Rationale: monolingual models cannot fairly process another language;\n"
# #               "parallel FLORES ensures the same semantic content reaches both models.")

# #         # Load FLORES sentences for each language
# #         print("\nLoading FLORES+ sentences...")
# #         lang_to_sents: dict[str, list[str]] = {}
# #         for l in langs:
# #             print(f"  [{l}] {FLORES_CODE[l]}")
# #             lang_to_sents[l] = load_flores(l, args.cka_n_samples, args.flores_split)
# #         actual_n = min(len(s) for s in lang_to_sents.values())
# #         print(f"  → using {actual_n} sentences (min across languages)")
# #         lang_to_sents = {l: s[:actual_n] for l, s in lang_to_sents.items()}

# #         # Collect activations — load each model exactly once, cache tensors
# #         print("\nCollecting activations (one model load per language)...")
# #         start_time = time.time()
# #         act_cache: dict[str, list[torch.Tensor]] = {}
# #         for l in langs:
# #             act_cache[l] = collect_activations(
# #                 model_name=LANG_TO_MODEL[l],
# #                 revision=args.revision,
# #                 sentences=lang_to_sents[l],
# #                 device=args.device,
# #                 batch_size=args.cka_batch_size,
# #                 max_length=args.cka_max_length,
# #             )
# #             print(f"  ✓ {l}: {len(act_cache[l])} layers, "
# #                   f"{act_cache[l][0].shape[0]} samples, "
# #                   f"hidden={act_cache[l][0].shape[1]}")
# #         end_time = time.time()
# #         print(f"Time taken: {end_time - start_time:.2f} seconds")
# #         n_layers = len(act_cache[langs[0]])

# #             # initialize results dictionary with all langs
# #         for lang_a in langs:
# #             for lang_b in langs:
# #                 if lang_a == lang_b:
# #                     results[lang_a][lang_b] = self_similarity()
# #                 else:
# #                     results[lang_a][lang_b] = {
# #                         "cka": {
# #                             "mean_cka": float('nan'),
# #                             "cka_per_layer": [float('nan')] * n_layers,
# #                         }
# #                     }

# #         # Update diagonal with correct layer count
# #         for l in langs:
# #             results[l][l]["cka"] = {
# #                 "mean_cka":      1.0,
# #                 "cka_per_layer": [1.0] * n_layers,
# #             }
        
# #         # Pairwise CKA — pure tensor ops on cached activations, no model loading
# #         print("\nComputing pairwise CKA...")
# #         for lang_a, lang_b in tqdm(pairs, desc="CKA pairs"):
# #             start_time = time.time()
# #             cka_entry = compute_cka_entry(act_cache[lang_a], act_cache[lang_b])
# #             # Merge into existing weight_space entry
# #             for a, b in [(lang_a, lang_b), (lang_b, lang_a)]:
# #                 results[a][b].update(cka_entry)
# #             tqdm.write(f"  {lang_a} × {lang_b}: mean_CKA={cka_entry['cka']['mean_cka']:.4f}")
# #             end_time = time.time()
# #             print(f"Time taken: {end_time - start_time:.2f} seconds")

# #     # ── 3. Save ───────────────────────────────────────────────────────────────
# #     out_path = Path(args.output)
# #     with open(out_path, "w") as f:
# #         json.dump(results, f, indent=2)
# #     print(f"\nResults written to {out_path}")

# #     # ── 4. Summary table ──────────────────────────────────────────────────────
# #     print("\n── Summary ─────────────────────────────────────────────────────")
# #     col = f"{'Pair':<14} {'cos_sim':>9} {'l2_norm':>10} {'rank_diff':>10}"
# #     if args.cka:
# #         col += f" {'mean_cka':>10}"
# #     print(col)
# #     print("─" * len(col))
# #     for lang_a in langs:
# #         for lang_b in langs:
# #             if lang_a >= lang_b:
# #                 continue
# #             v = results.get(lang_a, {}).get(lang_b, {})
# #             ws = v.get("weight_space", {})
# #             row = (f"{lang_a+' × '+lang_b:<14} "
# #                    f"{ws.get('cosine_sim', float('nan')):>9.4f} "
# #                    f"{ws.get('l2_norm', float('nan')):>10.1f}")
# #             if args.rank_diff:
# #                 row += f" {ws.get('mean_rank_diff', float('nan')):>10.4f}"
# #             if args.cka:
# #                 row += f" {v.get('cka', {}).get('mean_cka', float('nan')):>10.4f}"
# #             print(row)


# # if __name__ == "__main__":
# #     main()