#!/bin/bash
# IBD-calling contract for `ponderosa simulate` (the script name gives the format: hap-ibd).
# Called as: bash hap-ibd.sh WORKSPACE_DIR
# Must read  WORKSPACE_DIR/simulation.vcf(.gz) and write WORKSPACE_DIR/simulation.ibd.gz
# plus WORKSPACE_DIR/genetic.map (PLINK format) for the bp -> cM conversion.
set -euo pipefail

dirpath=${1:?workspace directory}
prefix=${dirpath}/simulation
if [[ -f "${prefix}.vcf.gz" ]]; then vcffile="${prefix}.vcf.gz"
elif [[ -f "${prefix}.vcf" ]]; then vcffile="${prefix}.vcf"
else echo "ERROR: no simulated VCF at ${prefix}.vcf(.gz)"; exit 1; fi

# ===== edit for your setup =====
HAPIBD_JAR=${HAPIBD_JAR:-$HOME/tools/hap-ibd.jar}
# Directory with plink.chr{1..22}.GRCh37.map (the same maps used for the real data)
MAP_DIR=${MAP_DIR:-$HOME/maps/plink.GRCh37.map}
MEM_GB=${MEM_GB:-12}
N_AUTOSOMES=${N_AUTOSOMES:-22}   # 20 for rhesus macaque
THREADS=${SLURM_CPUS_PER_TASK:-4}
# ===============================

[[ -f "$HAPIBD_JAR" ]] || { echo "ERROR: hap-ibd jar not found: $HAPIBD_JAR"; exit 1; }
[[ -d "$MAP_DIR" ]] || { echo "ERROR: map directory not found: $MAP_DIR"; exit 1; }

: > "${dirpath}/genetic.map"
for i in $(seq 1 $N_AUTOSOMES); do cat "${MAP_DIR}/plink.chr${i}.GRCh37.map" >> "${dirpath}/genetic.map"; done

java -Xmx${MEM_GB}g -jar "$HAPIBD_JAR" gt="$vcffile" map="${dirpath}/genetic.map" \
    nthreads="$THREADS" out="$prefix"
