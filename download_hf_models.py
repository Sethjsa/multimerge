"""
Download all checkpoint revisions from HuggingFace to local disk.

Usage:
    python download_checkpoints.py
    python download_checkpoints.py --model merged-10-checkpoints
    python download_checkpoints.py --checkpoint checkpoint_0001000  # single checkpoint
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER    = "sethjsa"
LOCAL_BASE = Path("/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models")
TOKEN      = open("./token").read().strip()
# ─────────────────────────────────────────────────────────────────────────────

def get_all_revisions(api: HfApi, repo_id: str) -> list[str]:
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model", token=TOKEN)
    return sorted(
        b.name for b in refs.branches
        if b.name.startswith("checkpoint_")
    )

def download_checkpoint(repo_id: str, revision: str, local_dir: Path):
    print(f"[DL]  {revision} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        token=TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # skip non-safetensors formats
    )
    print(f"[OK]  {revision}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mixed-10-checkpoints",
                        help="HF repo name (default: mixed-10-checkpoints)")
    parser.add_argument("--checkpoint", default=None,
                        help="Download a single checkpoint revision only")
    args = parser.parse_args()

    repo_id = f"{HF_USER}/{args.model}"
    local_model_dir = LOCAL_BASE / args.model

    api = HfApi()

    if args.checkpoint:
        revisions = [args.checkpoint]
    else:
        print(f"Fetching revision list from {repo_id}...")
        revisions = get_all_revisions(api, repo_id)
        print(f"Found {len(revisions)} checkpoints\n")

    failed = []
    for rev in revisions:
        local_dir = local_model_dir / rev
        if local_dir.exists() and any(local_dir.iterdir()):
            # Check if a checkpoint safetensor or pytorch bin exists in the local_dir
            safetensor_exists = any(f.suffix == '.safetensors' for f in local_dir.iterdir())
            ptbin_exists = any(f.name in ('pytorch_model.bin',) for f in local_dir.iterdir())
            if safetensor_exists or ptbin_exists:
                print(f"[SKIP] {rev} — already has checkpoint file(s)")
                continue
     
        try:
            download_checkpoint(repo_id, rev, local_dir)
        except Exception as e:
            print(f"[FAIL] {rev} — {e}")
            failed.append(rev)

    print(f"\n{'═'*40}")
    print(f"  Done: {len(revisions) - len(failed)}/{len(revisions)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'═'*40}")

if __name__ == "__main__":
    main()