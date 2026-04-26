import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATHS = [
    # "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/mixed-10-checkpoints/checkpoint_0047684",
    "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints/checkpoint_0047684",
    # "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/dareties-10-checkpoints/checkpoint_0047684",
    # "/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/ties-10-checkpoints/checkpoint_0047684",
]

LANG_PAIRS = {
    "eng-fra": ("eng_Latn", "fra_Latn", "French"),
    "eng-zho": ("eng_Latn", "zho_Hans", "Chinese"),
    "eng-ara": ("eng_Latn", "ara_Arab", "Arabic"),
}

N_EXAMPLES  = 50
FLORES_SPLIT = "devtest"          # flores200 uses 'devtest'
OUTPUT_DIR  = Path("flores")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Generation settings — tweak as needed
GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=False,          # greedy; set True + temperature for sampling
    num_beams=4,
)

# ── Prompt template ───────────────────────────────────────────────────────────

def make_prompt(src_sentence: str, tgt_lang_name: str) -> str:
    return (
        f"Translate the following sentence from English to {tgt_lang_name}.\n"
        f"English: {src_sentence}\n"
        f"{tgt_lang_name}:"
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def model_name(path: str) -> str:
    if "merged" in path:
        return "Merged"
    elif "mixed" in path:
        return "Mixed"
    elif "dareties" in path:
        return "Dareties"
    elif "ties" in path:
        return "Ties"
    else:
        raise ValueError(f"Unknown model path: {path}")
    return Path(path).name

def load_flores_sentences(flores_lang: str, n: int) -> list[str]:
    ds = load_dataset("facebook/flores", flores_lang, split=FLORES_SPLIT, trust_remote_code=True)
    return [row["sentence"] for row in ds.select(range(n))]

def generate_translations(
    model, tokenizer, sentences: list[str], tgt_lang_name: str
) -> list[str]:
    translations = []
    for sent in sentences:
        prompt = make_prompt(sent, tgt_lang_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KWARGS, pad_token_id=tokenizer.eos_token_id)
        # Decode only the newly generated tokens
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        translations.append(translation)
    return translations

def save_output(path: Path, src_sentences: list[str], translations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (src, tgt) in enumerate(zip(src_sentences, translations), 1):
            f.write(f"[{i}]\n")
            f.write(f"SRC: {src}\n")
            f.write(f"HYP: {tgt}\n\n")
    print(f"  Saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Pre-load all source sentences once per language pair
    print("Loading FLORES-200 source sentences …")
    flores_data = {}
    for lang_key, (src_code, _, _) in LANG_PAIRS.items():
        if src_code not in flores_data:
            flores_data[src_code] = load_flores_sentences(src_code, N_EXAMPLES)
        flores_data[lang_key] = flores_data[src_code]

    for model_path in MODEL_PATHS:
        mname = model_name(model_path)
        print(f"\n{'='*60}")
        print(f"Model: {mname}")
        print(f"{'='*60}")

        print("  Loading tokenizer & model …")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        model.eval()

        for lang_key, (src_code, _tgt_code, tgt_lang_name) in LANG_PAIRS.items():
            print(f"  Translating {lang_key} ({N_EXAMPLES} examples) …")
            src_sentences = flores_data[lang_key]
            translations  = generate_translations(model, tokenizer, src_sentences, tgt_lang_name)

            out_path = OUTPUT_DIR / f"outputs_{mname}_{lang_key}.txt"
            save_output(out_path, src_sentences, translations)

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    print("\nDone.")

if __name__ == "__main__":
    main()