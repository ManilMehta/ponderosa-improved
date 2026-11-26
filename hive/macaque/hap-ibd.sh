#!/bin/bash
# IBD-calling contract for `ponderosa simulate` (macaque): reads WORKSPACE/simulation.vcf(.gz),
# writes WORKSPACE/simulation.ibd.gz and WORKSPACE/genetic.map (same map as the real data).
set -euo pipefail
dirpath=${1:?workspace}
prefix=$dirpath/simulation
vcf=$( [[ -f $prefix.vcf.gz ]] && echo $prefix.vcf.gz || echo $prefix.vcf )
MAP=${MACAQUE_PLINK_MAP:?set MACAQUE_PLINK_MAP}
cp "$MAP" "$dirpath/genetic.map"
java -Xmx${MEM_GB:-28}g -jar "${HAPIBD_JAR:?set HAPIBD_JAR}" gt="$vcf" map="$dirpath/genetic.map" \
     nthreads=${SLURM_CPUS_PER_TASK:-8} out="$prefix" > "$dirpath/hapibd.stdout" 2>&1
