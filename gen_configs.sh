#!/bin/bash
set -e

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR="merging"
mkdir -p "$OUTPUT_DIR"

MODELS=(
  HPLT/hplt2c_ara_checkpoints
  HPLT/hplt2c_deu_checkpoints
  HPLT/hplt2c_eng_checkpoints
  HPLT/hplt2c_fra_checkpoints
  HPLT/hplt2c_ita_checkpoints
  HPLT/hplt2c_nld_checkpoints
  HPLT/hplt2c_rus_checkpoints
  HPLT/hplt2c_spa_checkpoints
  HPLT/hplt2c_tur_checkpoints
  HPLT/hplt2c_zhos_checkpoints
)

CHECKPOINTS=(
  checkpoint_0001000 checkpoint_0002000 checkpoint_0003000 checkpoint_0004000
  checkpoint_0005000 checkpoint_0006000 checkpoint_0007000 checkpoint_0008000
  checkpoint_0009000 checkpoint_0010000 checkpoint_0011000 checkpoint_0012000
  checkpoint_0013000 checkpoint_0014000 checkpoint_0015000 checkpoint_0016000
  checkpoint_0017000 checkpoint_0018000 checkpoint_0019000 checkpoint_0020000
  checkpoint_0021000 checkpoint_0022000 checkpoint_0023000 checkpoint_0024000
  checkpoint_0025000 checkpoint_0026000 checkpoint_0027000 checkpoint_0028000
  checkpoint_0029000 checkpoint_0030000 checkpoint_0031000 checkpoint_0032000
  checkpoint_0033000 checkpoint_0034000 checkpoint_0035000 checkpoint_0036000
  checkpoint_0037000 checkpoint_0038000 checkpoint_0039000 checkpoint_0040000
  checkpoint_0041000 checkpoint_0042000 checkpoint_0043000 checkpoint_0044000
  checkpoint_0045000 checkpoint_0046000 checkpoint_0047000 checkpoint_0047684
)
# ─────────────────────────────────────────────────────────────────────────────

for ckpt in "${CHECKPOINTS[@]}"; do
  YAML_FILE="$OUTPUT_DIR/linear-10-${ckpt}.yaml"

  {
    echo "models:"
    for model in "${MODELS[@]}"; do
      echo "  - model: ${model}@${ckpt}"
      echo "    parameters:"
      echo "      weight: 0.1"
    done
    echo "merge_method: linear"
    echo "dtype: float16"
  } > "$YAML_FILE"

  echo "Written: $YAML_FILE"
done

echo ""
echo "Done. $(ls $OUTPUT_DIR/linear-10-checkpoint_*.yaml | wc -l) config files generated in $OUTPUT_DIR/"



# Create final checkpoint config for the following merges:

FINAL_CKPT="checkpoint_0047684"

# 8-language merge (remove zhos, tur)
MODELS_8=(
  HPLT/hplt2c_ara_checkpoints
  HPLT/hplt2c_deu_checkpoints
  HPLT/hplt2c_eng_checkpoints
  HPLT/hplt2c_fra_checkpoints
  HPLT/hplt2c_ita_checkpoints
  HPLT/hplt2c_nld_checkpoints
  HPLT/hplt2c_rus_checkpoints
  HPLT/hplt2c_spa_checkpoints
)
YAML_FILE="$OUTPUT_DIR/linear-8-${FINAL_CKPT}.yaml"
{
  echo "models:"
  for model in "${MODELS_8[@]}"; do
    echo "  - model: ${model}@${FINAL_CKPT}"
    echo "    parameters:"
    echo "      weight: 0.125"
  done
  echo "merge_method: linear"
  echo "dtype: float16"
} > "$YAML_FILE"
echo "Written: $YAML_FILE"

# 6-language merge (remove spa, rus)
MODELS_6=(
  HPLT/hplt2c_ara_checkpoints
  HPLT/hplt2c_deu_checkpoints
  HPLT/hplt2c_eng_checkpoints
  HPLT/hplt2c_fra_checkpoints
  HPLT/hplt2c_ita_checkpoints
  HPLT/hplt2c_nld_checkpoints
)
YAML_FILE="$OUTPUT_DIR/linear-6-${FINAL_CKPT}.yaml"
{
  echo "models:"
  for model in "${MODELS_6[@]}"; do
    echo "  - model: ${model}@${FINAL_CKPT}"
    echo "    parameters:"
    echo "      weight: 0.16666666666666666"
  done
  echo "merge_method: linear"
  echo "dtype: float16"
} > "$YAML_FILE"
echo "Written: $YAML_FILE"

# 4-language merge (remove ita, nld)
MODELS_4=(
  HPLT/hplt2c_ara_checkpoints
  HPLT/hplt2c_deu_checkpoints
  HPLT/hplt2c_eng_checkpoints
  HPLT/hplt2c_fra_checkpoints
)
YAML_FILE="$OUTPUT_DIR/linear-4-${FINAL_CKPT}.yaml"
{
  echo "models:"
  for model in "${MODELS_4[@]}"; do
    echo "  - model: ${model}@${FINAL_CKPT}"
    echo "    parameters:"
    echo "      weight: 0.25"
  done
  echo "merge_method: linear"
  echo "dtype: float16"
} > "$YAML_FILE"
echo "Written: $YAML_FILE"

# 2-language merge (remove deu, ara) -> eng, fra
MODELS_2=(
  HPLT/hplt2c_eng_checkpoints
  HPLT/hplt2c_fra_checkpoints
)
YAML_FILE="$OUTPUT_DIR/linear-2-${FINAL_CKPT}.yaml"
{
  echo "models:"
  for model in "${MODELS_2[@]}"; do
    echo "  - model: ${model}@${FINAL_CKPT}"
    echo "    parameters:"
    echo "      weight: 0.5"
  done
  echo "merge_method: linear"
  echo "dtype: float16"
} > "$YAML_FILE"
echo "Written: $YAML_FILE"

