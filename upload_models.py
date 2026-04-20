"""
Upload merged checkpoints to HuggingFace as named revisions/branches.
Each checkpoint becomes its own branch, e.g. checkpoint_0001000.

Usage:
    python upload_models.py --model merged-10-checkpoints
    python upload_models.py --model mixed-10-checkpoints
    python upload_models.py --model merged-10-checkpoints --checkpoint checkpoint_0001000
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER    = "sethjsa"
LOCAL_BASE = Path("/scratch/project_462000963/users/sethayco/checkpoints_hf")
TOKEN      = open("./token").read().strip()
# ─────────────────────────────────────────────────────────────────────────────

def upload_checkpoint(api: HfApi, local_ckpt_dir: Path, repo_id: str):
    ckpt_name = local_ckpt_dir.name  # e.g. checkpoint_0001000
    print(f"\n── Uploading {ckpt_name} as revision '{ckpt_name}' → {repo_id} ──")

    # Create the branch if it doesn't exist
    try:
        api.create_branch(repo_id=repo_id, branch=ckpt_name, repo_type="model", token=TOKEN)
        print(f"   Created branch: {ckpt_name}")
    except Exception:
        print(f"   Branch already exists: {ckpt_name}")

    # Upload the entire checkpoint folder to the root of that branch
    api.upload_folder(
        folder_path=str(local_ckpt_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=ckpt_name,
        commit_message=f"Add {ckpt_name}",
        token=TOKEN,
    )
    print(f"   ✓ Done: {ckpt_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["merged-10-checkpoints", "mixed-10-checkpoints"],
                        help="Which model folder to upload")
    parser.add_argument("--checkpoint", default=None,
                        help="Upload a single checkpoint only (e.g. checkpoint_0001000)")
    args = parser.parse_args()

    repo_id = f"{HF_USER}/{args.model}"
    local_model_dir = LOCAL_BASE / args.model

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=TOKEN)
    print(f"Repo: https://huggingface.co/{repo_id}")

    if args.checkpoint:
        ckpt_dir = local_model_dir / args.checkpoint
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
        upload_checkpoint(api, ckpt_dir, repo_id)
    else:
        ckpt_dirs = sorted(local_model_dir.glob("checkpoint_*"))
        print(f"Found {len(ckpt_dirs)} checkpoints in {local_model_dir}")
        for ckpt_dir in ckpt_dirs:
            upload_checkpoint(api, ckpt_dir, repo_id)

    print(f"\nAll done! → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()