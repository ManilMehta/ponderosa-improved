#!/bin/bash
# Simulation-based training for a cohort with no usable pedigree (V3 section 5).
#
#   sbatch hive/train_simulation.sbatch hive/simulation_template.yaml
#
# Produces <output_path>.classif.pkl and <output_path>.training_metadata.json.
# Put the classifier in the dataset's "training" column of datasets.tsv.
#SBATCH --job-name=ponderosa_sim
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --partition=high
##SBATCH --account=YOUR_ACCOUNT
set -euo pipefail

SIM_YAML=${1:?usage: sbatch train_simulation.sbatch SIMULATION_YAML}
RUN_DIR=${RUN_DIR:-$SLURM_SUBMIT_DIR}
VENV=${VENV:-$RUN_DIR/venv}
cd "$RUN_DIR"
mkdir -p logs outputs
export PYTHONPYCACHEPREFIX=/tmp/${USER:-$(id -un)}_pycache
module load java 2>/dev/null || true          # hap-IBD
source "$VENV/bin/activate"

python -m ponderosa simulate --config "$SIM_YAML" -v
