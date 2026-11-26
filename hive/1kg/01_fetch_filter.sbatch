#!/bin/bash
# Step 1: stream each chromosome of the phased panel and keep biallelic SNVs with
# MAF >= 0.05 (plenty for IBD detection, ~10x smaller). One array task per chromosome.
#   sbatch --array=1-22 ponderosa-improved/hive/1kg/01_fetch_filter.sbatch
#SBATCH --job-name=1kg_fetch
#SBATCH --account=bmhenngrp
#SBATCH --partition=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fetch_%A_%a.out
#SBATCH --error=logs/fetch_%A_%a.err
set -euo pipefail
module load bcftools 2>/dev/null || true
command -v bcftools >/dev/null || { echo "ERROR: bcftools not found (module load bcftools)"; exit 1; }
C=${SLURM_ARRAY_TASK_ID}
URL=$(grep -E "[._]chr${C}[._]" ref/vcf_urls.txt | head -1)
[[ -n "$URL" ]] || { echo "ERROR: no URL for chr$C"; exit 1; }
OUT=vcf/chr${C}.snps.maf05.vcf.gz
echo "[$(date)] chr$C from $URL"
curl -fsSL --retry 5 "$URL" | bcftools view -m2 -M2 -v snps -q 0.05:minor -Oz -o "$OUT.tmp" -
mv "$OUT.tmp" "$OUT" && bcftools index -t "$OUT"
echo "[$(date)] chr$C: $(bcftools index -n $OUT) SNVs kept"
