import json
import numpy as np
import pandas as pd

with open("multiblimp/bilingual_pair_results.json") as f:
    pair_data = json.load(f)
with open("multiblimp/distances.json") as f:
    dist_data = json.load(f)

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

# ── Build full dataframe with all distance fields ─────────────────────────────
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
            diff = None

        d1, d2 = LANGUAGE_MAP[l1], LANGUAGE_MAP[l2]
        try:
            dist_entry = dist_data[d1][d2]
        except KeyError:
            try:
                dist_entry = dist_data[d2][d1]
            except KeyError:
                dist_entry = None

        record = {"lang1": l1, "lang2": l2, "diff": diff}
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
        rows.append(record)

df = pd.DataFrame(rows)
dist_cols = [c for c in df.columns if "__" in c]

# ── For each measure, show which pairs are missing ────────────────────────────
less_than_45 = {col: df[df[col].isna()][["lang1", "lang2"]]
                for col in dist_cols if df[col].isna().any()}

print(f"{'Measure':<35} {'n_missing':>9}  {'Missing pairs'}")
print("─" * 90)
for col, missing in sorted(less_than_45.items(), key=lambda x: len(x[1]), reverse=True):
    pairs_str = ", ".join(f"{r.lang1}-{r.lang2}" for _, r in missing.iterrows())
    print(f"{col:<35} {len(missing):>9}  {pairs_str}")

# ── Also: AES is constant — show its values ───────────────────────────────────
print("\n── metadata__AES values (constant check) ──")
print(df[["lang1", "lang2", "metadata__AES"]].to_string(index=False))