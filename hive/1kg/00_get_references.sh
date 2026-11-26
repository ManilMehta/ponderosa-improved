#!/bin/bash
# Step 0 (login node, small downloads only): pedigree, GRCh38 genetic maps, hap-IBD,
# and the list of phased-panel VCF URLs. Run from the 1000 Genomes work folder:
#   bash ponderosa-improved/hive/1kg/00_get_references.sh
set -euo pipefail
BASE=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage
PANEL=$BASE/working/20220422_3202_phased_SNV_INDEL_SV
mkdir -p ref vcf ibd logs

echo "== phased panel VCFs"
curl -fsSL "$PANEL/" | grep -oE 'href="[^"]*chr[0-9]+[^"]*\.vcf\.gz"' | sed 's/href="//; s/"$//' | sort -u > ref/vcf_names.txt
awk -v p="$PANEL/" '{print p $0}' ref/vcf_names.txt > ref/vcf_urls.txt
wc -l < ref/vcf_urls.txt; head -3 ref/vcf_urls.txt

echo "== pedigree"
PED=$(curl -fsSL "$BASE/" | grep -oE 'href="[^"]*pedigree[^"]*"' | sed 's/href="//; s/"$//' | head -1)
[[ -n "$PED" ]] || { echo "ERROR: no pedigree file listed at $BASE/"; exit 1; }
curl -fsSL -o ref/$(basename "$PED") "$BASE/$PED"; ls -lh ref/$(basename "$PED"); head -3 ref/$(basename "$PED")

echo "== GRCh38 genetic maps (Beagle, PLINK format)"
if [[ ! -f ref/maps/plink.chr1.GRCh38.map ]]; then
  curl -fsSL -o ref/plink.GRCh38.map.zip https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip
  mkdir -p ref/maps && unzip -q -o ref/plink.GRCh38.map.zip -d ref/maps
fi
ls ref/maps | head -3; head -2 ref/maps/plink.chr1.GRCh38.map

echo "== hap-IBD"
[[ -f ref/hap-ibd.jar ]] || curl -fsSL -o ref/hap-ibd.jar https://faculty.washington.edu/browning/hap-ibd.jar
ls -lh ref/hap-ibd.jar
echo "Step 0 done."
