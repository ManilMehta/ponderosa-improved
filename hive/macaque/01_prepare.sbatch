#!/bin/bash
# Macaque step 1: build PONDEROSA inputs for Macaque_140.
#   - COPIES the source files into PONDEROSA_MACAQUE_INPUT/ (the original project
#     folders are only read, never written to or modified)
#   - recombination map (RecombMap_rh10, recombmix) -> PLINK map + Ped-sim map
#   - thinned marker set (common SNPs, >= 5 kb apart) used for ALL IBD calling
#   - hap-IBD on the real animals
#   - dataset folder MACAQUE_140/ for `ponderosa run`
# Run from ponderosa_v3_run:   sbatch ponderosa-improved/hive/macaque/01_prepare.sbatch
#SBATCH --job-name=mac_prepare
#SBATCH --account=bmhenngrp
#SBATCH --partition=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mac_prepare_%j.out
#SBATCH --error=logs/mac_prepare_%j.err
set -euo pipefail
M=${MACAQUE_DIR:-/quobyte/bmhenngrp/from-lssc0/projects/macaques/Macaque_140}
SRC_G=$M/GLIMPSE1/GLIMPSE1_Final/GLOBAL_ANCESTRY_ROH_KING
SRC_MAP=$M/RecombMap_rh10
IN=PONDEROSA_MACAQUE_INPUT
HAPIBD_JAR=${HAPIBD_JAR:?set HAPIBD_JAR=/path/to/hap-ibd.jar}
THIN_BP=${THIN_BP:-5000}
W=macaque_work
mkdir -p $W/maps logs $IN/GLIMPSE1_Final $IN/RecombMap_rh10

echo "[$(date)] 0/4 copying inputs to $IN (originals untouched; existing copies are not overwritten)"
copy_once() { local dest=$2/$(basename "$1"); [[ -e "$dest" ]] || cp -p "$1" "$dest"; }   # never overwrite
for f in $SRC_G/all_filtered.vcf.gz $SRC_G/all_filtered.vcf.gz.tbi $SRC_G/all_filtered.fam \
         $SRC_G/KING/all_filtered_IDs.kin0 $SRC_G/KING/all_filtered_IDs.log; do copy_once "$f" $IN/GLIMPSE1_Final; done
for f in $SRC_MAP/*_cM_recombmix.map; do copy_once "$f" $IN/RecombMap_rh10; done
chmod a-w $IN/GLIMPSE1_Final/* $IN/RecombMap_rh10/*       # protect the copies too
( cd $IN && find . -type f | sort | xargs md5sum ) > $IN/MD5SUMS.txt
G=$IN/GLIMPSE1_Final
MAPDIR=$IN/RecombMap_rh10
module load bcftools 2>/dev/null || true
module load java 2>/dev/null || module load openjdk 2>/dev/null || true
command -v bcftools >/dev/null || { echo "ERROR: bcftools not found"; exit 1; }
command -v java >/dev/null || { echo "ERROR: java not found"; exit 1; }
source venv/bin/activate

echo "[$(date)] 1/4 maps"
python - "$MAPDIR" "$W" <<'PY'
import sys, pandas as pd
from pathlib import Path
mapdir, w = Path(sys.argv[1]), Path(sys.argv[2])
plink, simmap = [], []
for c in range(1, 21):
    f = mapdir / f"{c}_cM_recombmix.map"
    m = pd.read_csv(f, sep=r"\s+")
    m.columns = ["chr", "bp", "rate", "cm"]
    m = m.sort_values("bp")
    m["chr"] = c
    m[["chr"]].assign(id=".", cm=m.cm, bp=m.bp).to_csv(w / "maps" / f"rheMac10.chr{c}.map", sep="\t", header=False, index=False)
    plink.append(m[["chr"]].assign(id=".", cm=m.cm, bp=m.bp))
    # Ped-sim only assigns sexes (needed for paternal vs maternal relatives) with a
    # sex-specific map: give the sex-averaged map as identical male and female columns
    simmap.append(m[["chr", "bp"]].assign(male_cm=m.cm, female_cm=m.cm))
pd.concat(plink).to_csv(w / "rheMac10.plink.map", sep="\t", header=False, index=False)
sm = pd.concat(simmap)
sm.to_csv(w / "rheMac10.simmap", sep="\t", header=False, index=False)
tot = sum(g.male_cm.max() - g.male_cm.min() for _, g in sm.groupby("chr"))
print(f"maps: 20 chromosomes, {tot:.1f} cM total")
PY

echo "[$(date)] 2/4 thinned marker set (biallelic SNPs, MAF >= 0.05, >= ${THIN_BP} bp apart)"
bcftools view -m2 -M2 -v snps -q 0.05:minor -r $(seq -s, 1 20) $G/all_filtered.vcf.gz -Ou \
 | bcftools view -Ov | awk -v d=$THIN_BP 'BEGIN{OFS="\t"} /^#/{print; next} {if($1!=c || $2-p>=d){print; c=$1; p=$2}}' \
 | bcftools view -Oz -o $W/macaque140.thin.vcf.gz
bcftools index -t $W/macaque140.thin.vcf.gz
echo "markers kept: $(bcftools index -n $W/macaque140.thin.vcf.gz)"

echo "[$(date)] 3/4 hap-IBD on the real animals"
java -Xmx28g -jar "$HAPIBD_JAR" gt=$W/macaque140.thin.vcf.gz map=$W/rheMac10.plink.map \
     nthreads=${SLURM_CPUS_PER_TASK:-8} out=$W/macaque140 > $W/hapibd_real.stdout
echo "IBD segments: $(zcat $W/macaque140.ibd.gz | wc -l)"

echo "[$(date)] 4/4 dataset folder"
D=MACAQUE_140
mkdir -p $D
ln -sf $PWD/$W/macaque140.ibd.gz $D/
for c in $(seq 1 20); do ln -sf $PWD/$W/maps/rheMac10.chr$c.map $D/; done
ln -sf $PWD/$G/all_filtered.fam $D/macaque140.fam
ln -sf $PWD/$G/all_filtered_IDs.kin0 $D/macaque140_king.kin0
printf "name: MACAQUE_140\nspecies: macaque\nibd_caller: hap-ibd\nmin_total_ibd: 30\n" > $D/ponderosa.yaml
echo "[$(date)] done. Next: sbatch ponderosa-improved/hive/macaque/02_train_simulation.sbatch"
