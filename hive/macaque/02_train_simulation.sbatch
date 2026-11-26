#!/bin/bash
# Macaque step 2: train PONDEROSA on Ped-sim families built from the 138 animals
# (unrelated founders chosen with KING; macaque map; no human interference model;
# IBD called by hap-IBD on the same thinned markers as the real data).
#   sbatch ponderosa-improved/hive/macaque/02_train_simulation.sbatch
#SBATCH --job-name=mac_simtrain
#SBATCH --account=bmhenngrp
#SBATCH --partition=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mac_simtrain_%j.out
#SBATCH --error=logs/mac_simtrain_%j.err
set -euo pipefail
PEDSIM_DIR=${PEDSIM_DIR:?set PEDSIM_DIR=/path/to/ped-sim directory}
export HAPIBD_JAR=${HAPIBD_JAR:?set HAPIBD_JAR=/path/to/hap-ibd.jar}
W=$PWD/macaque_work
export MACAQUE_PLINK_MAP=$W/rheMac10.plink.map
module load java 2>/dev/null || module load openjdk 2>/dev/null || true
source venv/bin/activate
cat > $W/sim_macaque.yaml <<YAML
pedsim:
  pedsim_path: "$PEDSIM_DIR"
  vcf_file: "$W/macaque140.thin.vcf.gz"
  simmap_file: "$W/rheMac10.simmap"
  interference_file: none
  random_seed: 20261126
king_file: "$PWD/MACAQUE_140/macaque140_king.kin0"
training:
  n_pairs_per_relationship: ${N_REPLICATES:-60}
  max_kinship: 0.05
ibd_caller: "$PWD/ponderosa-improved/hive/macaque/hap-ibd.sh"
output_path: "$W/macaque_sim_training"
cleanup_temp: false
YAML
python -m ponderosa simulate --config $W/sim_macaque.yaml -v
ln -sf $W/macaque_sim_training.classif.pkl MACAQUE_140/training.classif.pkl
echo "[$(date)] done. Next: ponderosa run MACAQUE_140"
