#!/bin/bash
# Run PONDEROSA-IMPROVED on every dataset in datasets.tsv as a Slurm array.
#
#   N=$(grep -cv '^#' hive/datasets.tsv)
#   sbatch --array=1-$N hive/run_datasets.sbatch
#
# Per dataset: input validation -> inference -> duplicate QC -> ranked family trees
# -> KING validation (if a .kin0 is given). Outputs: outputs/<name>/<name>*.
#
# Edit the #SBATCH lines for your allocation (partition/account are not hardcoded).
#SBATCH --job-name=ponderosa
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
##SBATCH --partition=high
##SBATCH --account=YOUR_ACCOUNT
set -euo pipefail

RUN_DIR=${RUN_DIR:-$SLURM_SUBMIT_DIR}
VENV=${VENV:-$RUN_DIR/venv}
DATASETS=${DATASETS:-$RUN_DIR/hive/datasets.tsv}
cd "$RUN_DIR"
mkdir -p logs outputs
export PYTHONPYCACHEPREFIX=/tmp/${USER:-$(id -un)}_pycache
source "$VENV/bin/activate"

line=$(grep -v '^#' "$DATASETS" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID:-1}p")
IFS=$'\t' read -r NAME IBD CALLER FAM MAP TRAINING KING AGES MIN_TOTAL <<< "$line"
opt() { [[ -n "${2:-}" && "$2" != "-" ]] && echo "$1 $2" || true; }

OUT=outputs/$NAME/$NAME
mkdir -p "outputs/$NAME"
echo "[$(date)] dataset=$NAME caller=$CALLER"

# --ibd-caller is always passed explicitly (CLI and YAML now agree, but this keeps runs unambiguous)
COMMON="--ibd $IBD --fam $FAM --ibd-caller $CALLER $(opt --map "$MAP") $(opt --ages "$AGES")"

# 1. Check every input before running
python -m ponderosa validate $COMMON $(opt --training "$TRAINING") $(opt --king "$KING")

# 2. Inference (pretrained classifier if given, otherwise train on the FAM pedigree)
python -m ponderosa $COMMON $(opt --training "$TRAINING") \
    --min-segment-length 3 --min-total-ibd "${MIN_TOTAL:-50}" \
    --output "$OUT" -v

# 3. Duplicate / MZ-like QC
python -m ponderosa qc-duplicates --pairs "${OUT}_pairs.txt" --output "$OUT"

# 4. Ranked family trees
python -m ponderosa pedigree --pairs "${OUT}_pairs.txt" --fam "$FAM" $(opt --ages "$AGES") \
    --output "$OUT" -v

# 5. External validation against KING
if [[ -n "${KING:-}" && "$KING" != "-" ]]; then
    python -m ponderosa king-validate --pairs "${OUT}_pairs.txt" --king "$KING" --output "$OUT" --close-only
fi
echo "[$(date)] done: outputs/$NAME/"
