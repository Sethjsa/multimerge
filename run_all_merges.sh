#!/bin/bash
#SBATCH --job-name=merge_linear10
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# source .venv/bin/activate # ensure env synced

export HUGGING_FACE_HUB_TOKEN=$(cat token)
export HF_HOME=/fnwi_fs/ivi/irlab/personal/saycock/hf
CONFIG_DIR="merging"
OUTPUT_BASE="/fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-10-checkpoints"
LOG_DIR="merging/logs"
mkdir -p "$LOG_DIR"

# Fall back to python -m if mergekit-yaml not on PATH
if command -v mergekit-yaml &>/dev/null; then
  # MERGEKIT_CMD="mergekit-yaml"
  MERGEKIT_CMD="python -m mergekit.scripts.run_yaml"

else
  MERGEKIT_CMD="python -m mergekit.scripts.run_yaml"
fi

TOTAL=0; SKIPPED=0; FAILED=0; SUCCESS=0
FAILED_LIST=()

# for config in "$CONFIG_DIR"/linear-10-checkpoint_*.yaml; do
#   ckpt=$(basename "$config" .yaml | sed 's/linear-10-//')
#   out_dir="$OUTPUT_BASE/$ckpt"
#   log_file="$LOG_DIR/${ckpt}.log"
#   TOTAL=$((TOTAL + 1))

#   if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir" 2>/dev/null)" ]; then
#     echo "[SKIP] $ckpt — output already exists"
#     SKIPPED=$((SKIPPED + 1))
#     continue
#   fi

#   mkdir -p "$out_dir"
#   echo "[RUN]  $ckpt → $out_dir"

#   if $MERGEKIT_CMD "$config" "$out_dir" \
#       --lazy-unpickle \
#       --allow-crimes \
#       --cuda \
#       > "$log_file" 2>&1; then
#     echo "[OK]   $ckpt"
#     SUCCESS=$((SUCCESS + 1))
#   else
#     echo "[FAIL] $ckpt — see $log_file"
#     FAILED=$((FAILED + 1))
#     FAILED_LIST+=("$ckpt")
#     rm -rf "$out_dir"
#   fi
# done

# echo ""
# echo "════════════════════════════════════"
# echo "  Total:    $TOTAL"
# echo "  Success:  $SUCCESS"
# echo "  Skipped:  $SKIPPED"
# echo "  Failed:   $FAILED"
# if [ ${#FAILED_LIST[@]} -gt 0 ]; then
#   echo "  Failed checkpoints:"
#   for f in "${FAILED_LIST[@]}"; do
#     echo "    - $f"
#   done
# fi
# echo "════════════════════════════════════"



# for n in 2 4 6 8; do
#   config="$CONFIG_DIR/linear-${n}-checkpoint_0047684.yaml"
#   # Get the base name (e.g., checkpoint_0047684) for output/log folders, match others' naming scheme
#   ckpt="checkpoint_0047684-linear-${n}"
#   out_dir="$OUTPUT_BASE/$ckpt"
#   log_file="$LOG_DIR/${ckpt}.log"
#   TOTAL=$((TOTAL + 1))

#   if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir" 2>/dev/null)" ]; then
#     echo "[SKIP] $ckpt — output already exists"
#     SKIPPED=$((SKIPPED + 1))
#     continue
#   fi

#   mkdir -p "$out_dir"
#   echo "[RUN]  $ckpt → $out_dir"

#   if $MERGEKIT_CMD "$config" "$out_dir" \
#       --lazy-unpickle \
#       --allow-crimes \
#       --cuda \
#       > "$log_file" 2>&1; then
#     echo "[OK]   $ckpt"
#     SUCCESS=$((SUCCESS + 1))
#   else
#     echo "[FAIL] $ckpt — see $log_file"
#     FAILED=$((FAILED + 1))
#     FAILED_LIST+=("$ckpt")
#     rm -rf "$out_dir"
#   fi
# done

# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/task-10-checkpoint_0047684.yaml /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/task-10-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes --cuda 
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/dareties-10-checkpoint_0047684.yaml /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/dareties-10-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes  
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/ties-10-checkpoint_0047684.yaml /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/ties-10-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes 
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/task-aya.yaml /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/task-aya-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes --cuda 
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-aya.yaml /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/linear-aya-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes --cuda 
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-2-checkpoint_0047684.yaml models/merged-2-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes --cuda 
#/home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-8-checkpoint_0047684.yaml models/merged-8-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
#/home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-en1fr9-checkpoint_0047684.yaml models/merged-en1fr9-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
#/home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-en9fr1-checkpoint_0047684.yaml models/merged-en9fr1-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
#/home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-en5fr5-checkpoint_0047684.yaml models/merged-en5fr5-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
#/home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-en7fr3-checkpoint_0047684.yaml models/merged-en7fr3-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
#/home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-en3fr7-checkpoint_0047684.yaml models/merged-en3fr7-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-3-checkpoint_0047684.yaml models/merged-3-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-5-checkpoint_0047684.yaml models/merged-5-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-7-checkpoint_0047684.yaml models/merged-7-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes
# /home/saycock/miniconda3/envs/gvllm/bin/python -m mergekit.scripts.run_yaml merging/linear-9-checkpoint_0047684.yaml models/merged-9-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes

# langs=("eng" "nld" "spa" "fra" "rus" "ita" "tur" "ara" "deu" "zhos")
# num_langs=${#langs[@]}
# for ((i=0; i<$num_langs-1; i++)); do
#   for ((j=i+1; j<$num_langs; j++)); do
#     lang1=${langs[$i]}
#     lang2=${langs[$j]}
#     echo "Running: merging/linear-${lang1}${lang2}-checkpoint_0047684.yaml"
#     $MERGEKIT_CMD merging/linear-${lang1}${lang2}-checkpoint_0047684.yaml /fnwi_fs/ivi/irlab/personal/saycock/multimerge/models/merged-${lang1}${lang2}-checkpoints/checkpoint_0047684 --lazy-unpickle --allow-crimes --cuda
#   done
# done