# SacMEGA PONDEROSA Progress Summary

## Overall Goal

We are building a usable real-data pipeline for running PONDEROSA on the SacMEGA SNP-array dataset from the UC Davis HIVE server. The objective is to turn genotype-level data into the required PONDEROSA inputs, train the model appropriately, and then run relationship inference on the real SacMEGA data.

## Completed Work

### 1. Understood PONDEROSA's Required Inputs

We reviewed the PONDEROSA codebase and confirmed that the tool needs:

- IBD segment file
- FAM / pedigree file
- Genetic map, depending on IBD caller
- Optional ages file
- Optional truth / evaluation file
- Optional pretrained classifier file

We also confirmed that the current implementation depends heavily on known relationship labels from the FAM file for training.

### 2. Ran a Synthetic End-to-End Demo

Using `tests/generate_files.py`, we generated synthetic input files and successfully ran PONDEROSA locally.

During that demo, we found and documented several issues:

- The default `min_total_ibd = 50` filtered out many 4th-degree simulated relatives.
- PONDEROSA failed with a blank error when training classes had too few examples.
- QDA crashed on collinear synthetic data.
- We patched `ponderosa/classifiers.py` to use `QuadraticDiscriminantAnalysis(reg_param=1e-2)`.
- The demo achieved strong degree-level accuracy but weak specific-relationship accuracy, especially for 2nd-degree subtypes.

A full editable report was created at:

`demo_run/PONDEROSA_demo_report.md`

### 3. Set Up a Clean SacMEGA Working Folder on HIVE

We created and used a dedicated working directory:

`/quobyte/bmhenngrp/from-lssc0/data/snp-array/SAC_MEGA/ponderosa_sacmega_input_data`

This separated our work from previous labmate-generated files, which we decided not to trust.

### 4. Identified the Trusted Starting Dataset

We inspected the SacMEGA directory and decided to start from the original lab-standard PLINK genotype files:

`SAC_MEGA_zCalled_20180306_allele_rsID_nodup_IID_pTB.bed/.bim/.fam`

We avoided relying on later derived files such as previous VCFs, phasing output, hap-ibd output, and old PONDEROSA results.

### 5. Checked for Real Pedigree / Metadata

We inspected the original FAM file and found:

- 839 samples
- 0 individuals with parent IDs
- 839 distinct family IDs
- No usable pedigree structure

We also searched sample sheets and manifests:

- `nBowker_MEGA_SampleSheet.csv`
- `nBowker_SampleManifest.xlsx`

Those files only contained genotyping logistics / DNA QC information, not family relationships, ages, or usable subject metadata.

Conclusion: there is no recoverable real pedigree or age metadata in the available files.

### 6. QC'd the SacMEGA Genotype Data

We created a clean autosomal PLINK dataset using PLINK:

- Kept chromosomes 1-22
- Kept only clean A/C/G/T SNPs
- Removed variants with high missingness
- Removed rare variants with MAF < 0.01

Result:

- 839 samples retained initially
- 964,714 autosomal SNPs retained
- Genotyping rate approximately 97.4%

### 7. Ran KING Relatedness Inference

We ran KING on the QC'd genotype set and found real relatedness in the cohort:

- 33 duplicate / MZ pairs
- 28 parent-offspring pairs
- 31 full-sibling pairs
- 48 second-degree pairs
- 46 third-degree pairs
- Some additional distant / background relatedness

This confirmed the cohort has real family structure.

### 8. Resolved Duplicates / Replicates

KING revealed that most duplicate / MZ pairs were from one sample replicated eight times, likely a QC/control replicate.

We removed:

- 7 extra copies from the 8-copy replicate group
- 5 additional duplicate pairs

Result:

- 12 samples removed
- 827 samples retained
- 964,714 SNPs retained
- Genotyping rate improved to approximately 97.9%

### 9. Reconstructed a Pedigree with KING

We ran `king --build` on the deduplicated data.

KING reconstructed:

- 52 family clusters
- 52 nonfounders with parent assignments
- PO and FS relationships were recovered reasonably well

However, the reconstructed pedigree had very limited specific second-degree labels:

- PO pairs: 19
- FS pairs: 31
- Half-sib pairs: 0
- Grandparent pairs: 0
- Avuncular pairs: 1

Conclusion: the reconstructed real pedigree is useful for validation, especially PO/FS, but not enough to train PONDEROSA's full relationship classifiers.

### 10. Decided on Simulation-Based Training

Because the real pedigree lacks enough labeled second-degree relationship types, we decided that PONDEROSA cannot be trained from the real SacMEGA pedigree alone.

We selected the simulation-based route:

- Use real SacMEGA haplotypes / founders
- Simulate labeled relatives with `ped-sim`
- Train PONDEROSA on simulated relationship labels
- Apply the trained model to real SacMEGA IBD pairs
- Use the KING-reconstructed PO/FS relationships as partial validation

### 11. Determined Genome Build

We inspected the MEGA array manifest and BIM positions.

The dataset is consistent with:

**GRCh37 / hg19**

Evidence included chromosome 22 starting around 16.05 Mb and no chromosome 22 SNPs below 16 Mb.

### 12. Downloaded GRCh37 Genetic Maps

We downloaded Beagle-compatible GRCh37 genetic maps into the working folder:

`ponderosa_sacmega_input_data/maps/`

These maps will support phasing and later IBD / simulation steps.

### 13. Built a Phasing-Ready VCF

We created and indexed:

`sacmega_phasing_input.vcf.gz`

Result:

- 827 samples
- 964,714 variants
- 0 duplicate-position variants
- Ready for Beagle phasing

## Current Status

We are paused right before Beagle phasing.

The next concrete task is to run Beagle 5.4 on HIVE, likely as a Slurm array job over chromosomes 1-22.

## Remaining Major Steps

1. Run Beagle phasing on chromosomes 1-22.
2. Install or configure `ped-sim`.
3. Simulate labeled relatives from real SacMEGA haplotypes.
4. Install or configure an IBD caller, likely `phasedibd`.
5. Call IBD on simulated and real phased data.
6. Fix or extend PONDEROSA's broken simulation / training code.
7. Train PONDEROSA on simulated labels.
8. Run inference on real SacMEGA pairs.
9. Validate against the available PO/FS truth from KING.
10. Document the final pipeline and findings.
