"""
Sample 100 eng→fra translations from FLORES devtest across multiple models
and save to a JSONL file for qualitative analysis.
"""

import json
import random
import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ["HF_TOKEN"] = open("./token", "r").read().strip()
os.environ["HF_HOME"] = "/fnwi_fs/ivi/irlab/personal/saycock/hf"
SEED = 42
N_SAMPLES = 100
BATCH_SIZE = 50
FLORES_SPLIT = "devtest"
OUTPUT_FILE = Path("qualitative_eng_fra.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=False,
    num_beams=4,
)

# (display_key, hf_model_id_or_local_path, revision_or_None)
MODELS = [
    ("tiny-aya",    "CohereLabs/tiny-aya-global",          None),
    ("hplt2c-eng",  "HPLT/hplt2c_eng_checkpoints",         "main"),
    ("hplt2c-fra",  "HPLT/hplt2c_fra_checkpoints",         "main"),
    ("ties",        "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/ties-10-checkpoints/checkpoint_0047684",    None),
    ("dareties",    "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/dareties-10-checkpoints/checkpoint_0047684", None),
    ("mixed",       "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/mixed-10-checkpoints/checkpoint_0047684",   None),
    ("merged",      "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints/checkpoint_0047684",  None),
]


def make_prompt(src: str) -> str:
    return (
        f"Translate the following sentence from English to French.\n"
        f"English: {src}\n"
        f"French:"
    )


def load_flores_pairs(n: int, seed: int) -> list[dict]:
    src_ds = load_dataset("facebook/flores", "eng_Latn", split=FLORES_SPLIT, trust_remote_code=True)
    tgt_ds = load_dataset("facebook/flores", "fra_Latn", split=FLORES_SPLIT, trust_remote_code=True)
    assert len(src_ds) == len(tgt_ds)
    rng = random.Random(seed)
    indices = rng.sample(range(len(src_ds)), n)
    return [
        {"idx": i, "src": src_ds[i]["sentence"], "tgt": tgt_ds[i]["sentence"]}
        for i in indices
    ]


def generate(model, tokenizer, sentences: list[str]) -> list[str]:
    # Left-pad so all prompts in a batch align on the right (required for causal LMs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = [make_prompt(s) for s in sentences[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                **GEN_KWARGS,
                pad_token_id=tokenizer.eos_token_id,
            )
        for seq in out:
            new_tokens = seq[prompt_len:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        print(f"    {min(i + BATCH_SIZE, len(sentences))}/{len(sentences)}", end="\r")
    print()
    return outputs


def load_model(path: str, revision: str | None):
    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        **kwargs,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


def main():
    print(f"Loading FLORES devtest (eng→fra), sampling {N_SAMPLES} …")
    pairs = load_flores_pairs(N_SAMPLES, SEED)
    src_sentences = [p["src"] for p in pairs]
    records = [{"src": p["src"], "tgt": p["tgt"]} for p in pairs]

    for key, path, revision in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {key}  ({path})" + (f"  revision={revision}" if revision else ""))
        print(f"{'='*60}")
        print("  Loading …")
        model, tokenizer = load_model(path, revision)
        print(f"  Translating {N_SAMPLES} sentences …")
        translations = generate(model, tokenizer, src_sentences)
        for rec, hyp in zip(records, translations):
            rec[key] = hyp
        del model
        torch.cuda.empty_cache()
        print(f"  Done.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {N_SAMPLES} records → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
