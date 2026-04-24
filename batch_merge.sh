#!/usr/bin/env bash
# submit_merge_jobs.sh — generates all lang-pair configs and submits a SLURM array job

export HUGGING_FACE_HUB_TOKEN=$(cat token)
export HF_HOME=/fnwi_fs/ivi/irlab/personal/saycock/hf

CONFIG_DIR="merging"
OUTPUT_BASE="/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints"
LOG_DIR="merging/logs"
PAIRS_FILE="$LOG_DIR/lang_pairs.txt"   # one "lang1 lang2" per line — used as the array index lookup

mkdir -p "$LOG_DIR"

# ── Build the list of pairs ───────────────────────────────────────────────────
langs=("eng" "nld" "spa" "fra" "rus" "ita" "tur" "ara" "deu" "zhos")
num_langs=${#langs[@]}

> "$PAIRS_FILE"   # truncate / create
for ((i=0; i<num_langs-1; i++)); do
  for ((j=i+1; j<num_langs; j++)); do
    echo "${langs[$i]} ${langs[$j]}" >> "$PAIRS_FILE"
  done
done

NUM_PAIRS=$(wc -l < "$PAIRS_FILE")
echo "Submitting array of $NUM_PAIRS jobs (1–${NUM_PAIRS})"

# ── Submit ────────────────────────────────────────────────────────────────────
sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=merges
#SBATCH --array=1-${NUM_PAIRS}%10
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/merge_%A_%a.log
#SBATCH --error=${LOG_DIR}/merge_%A_%a.err

source activate gvllm

# ── Environment ───────────────────────────────────────────────────────────────
export HUGGING_FACE_HUB_TOKEN=\$(cat token)
export HF_HOME=/fnwi_fs/ivi/irlab/personal/saycock/hf
MERGEKIT_CMD="python -m mergekit.scripts.run_yaml"

# ── Resolve this task's language pair ─────────────────────────────────────────
read lang1 lang2 < <(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$PAIRS_FILE")
ckpt="checkpoint_0047684"
config="${CONFIG_DIR}/linear-\${lang1}\${lang2}-\${ckpt}.yaml"
out_dir="/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-\${lang1}\${lang2}-checkpoints/\${ckpt}"

echo "[TASK \${SLURM_ARRAY_TASK_ID}] \${lang1}+\${lang2} → \${out_dir}"

# ── Skip if already done ───────────────────────────────────────────────────────
if [ -d "\$out_dir" ] && [ -n "\$(ls -A "\$out_dir" 2>/dev/null)" ]; then
  echo "[SKIP] output already exists"
  exit 0
fi

mkdir -p "\$out_dir"

# ── Run merge (CPU only — no --cuda) ─────────────────────────────────────────
if \$MERGEKIT_CMD "\$config" "\$out_dir" --lazy-unpickle --allow-crimes; then
  echo "[OK]"
else
  echo "[FAIL]"
  rm -rf "\$out_dir"
  exit 1
fi
EOF