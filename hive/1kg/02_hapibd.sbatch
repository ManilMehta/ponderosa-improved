#!/bin/bash
# Step 2: hap-IBD on each filtered chromosome with the GRCh38 map.
#   sbatch --array=1-22 ponderosa-improved/hive/1kg/02_hapibd.sbatch
#SBATCH --job-name=1kg_hapibd
#SBATCH --account=bmhenngrp
#SBATCH --partition=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hapibd_%A_%a.out
#SBATCH --error=logs/hapibd_%A_%a.err
set -euo pipefail
module load java 2>/dev/null || module load openjdk 2>/dev/null || true
command -v java >/dev/null || { echo "ERROR: java not found"; exit 1; }
C=${SLURM_ARRAY_TASK_ID}
VCF=vcf/chr${C}.snps.maf05.vcf.gz
MAP=ref/maps/plink.chr${C}.GRCh38.map
[[ -f $VCF && -f $MAP ]] || { echo "ERROR: missing $VCF or $MAP"; exit 1; }
# hap-IBD needs the map's chromosome names to match the VCF's (panel uses "chr1")
VCHR=$(zcat "$VCF" | grep -v '^#' | head -1 | cut -f1)
awk -v c="$VCHR" 'BEGIN{OFS="\t"}{$1=c; print}' "$MAP" > ibd/map.chr${C}.tmp
java -Xmx20g -jar ref/hap-ibd.jar gt="$VCF" map=ibd/map.chr${C}.tmp nthreads=${SLURM_CPUS_PER_TASK} out=ibd/1kg.chr${C}
rm -f ibd/map.chr${C}.tmp
echo "[$(date)] chr$C: $(zcat ibd/1kg.chr${C}.ibd.gz | wc -l) IBD segments"
