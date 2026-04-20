TEST_LANGS=("ara" "deu" "eng" "fra" "hin" "ita" "nld" "por" "rus" "spa" "tur" "zhos")

for lang in "${TEST_LANGS[@]}"; do
    echo "Running inference for $lang"
    # echo "bash infer.sh -m HPLT/hplt2c_${lang}_checkpoints -t all"
    sbatch infer.sh -m HPLT/hplt2c_${lang}_checkpoints -t all -l all
done