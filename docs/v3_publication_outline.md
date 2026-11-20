# PONDEROSA V3 Publication Outline

## Working Title

PONDEROSA V3: QC-aware and simulation-trained relationship inference for real cohorts with sparse pedigrees

## Central Claim

PONDEROSA V3 improves the usability and interpretability of PONDEROSA on real-world genetic datasets by adding generalized input validation, duplicate/MZ-aware QC, sparse-pedigree support, external validation utilities, simulation-training metadata, and readable reporting.

## Motivation

Many real genotype datasets do not have complete pedigree labels. Some contain duplicate samples, technical replicates, monozygotic twins, or other non-independent records. In those settings, ordinary relationship inference tools can produce technically valid but biologically misleading labels unless QC and validation are built into the workflow.

SacMEGA exposed this failure mode clearly: duplicate/MZ-like pairs were forced into PONDEROSA's `FS` class because the model did not have a separate duplicate/MZ-aware QC layer.

## Methods Contributions

1. Generalized pre-run input validation.

   Detect missing files, incompatible caller/map settings, sparse pedigrees, missing truth files, and insufficient real training labels before long jobs run.

2. Duplicate/MZ-like QC.

   Detect near-identical sample pairs from IBD feature patterns and build connected duplicate-like components.

3. Sparse-pedigree/no-truth support.

   Allow inference with pretrained or simulation-trained classifiers when real labels are insufficient.

4. External validation module.

   Compare PONDEROSA predictions against KING or truth files using generic ID normalization.

5. Reporting.

   Produce human-readable reports for QC, inference, validation, and warnings.

6. Simulation-training metadata.

   Save reproducibility metadata alongside simulation-trained classifier files.

## Primary Case Study

SacMEGA:

- Real SNP-array cohort.
- Sparse or unusable real pedigree labels for training.
- Simulation-trained classifiers.
- Real inference on 23,375 IBD-sharing pairs.
- KING validation for close relatives.
- Detection and interpretation of a 16-sample duplicate-like component.

## Benchmarks To Include

Minimum:

- Synthetic known-truth benchmark.
- SacMEGA real-data case study.

Preferred:

- Additional public or collaborator dataset with known relatedness labels.

## Key Results To Report

- Input validation catches sparse-pedigree conditions before training.
- Duplicate/MZ QC detects known or suspected duplicate-like components.
- V3 prevents duplicate/MZ-like samples from being reported only as ordinary `FS`.
- KING validation outputs reproduce manual SacMEGA comparisons.
- Simulation metadata makes classifier provenance auditable.

## Expected Figures

- V3 workflow diagram.
- Duplicate/MZ component graph from SacMEGA.
- Before/after close-relative count comparison.
- PONDEROSA versus KING agreement summary.
- Synthetic benchmark accuracy or confusion summary.

## Limitations

- Duplicate/MZ thresholds may need calibration by marker density, IBD caller, and cohort type.
- KING validation is degree-level for many relationships and cannot prove all PONDEROSA subtype calls.
- Simulation-trained classifiers depend on map choice, founders, pedigree definitions, and IBD caller behavior.
- SacMEGA is a motivating case study, not by itself proof of generality.

## Publication Readiness Criteria

- V3 modules are implemented and tested.
- Documentation covers all major user workflows.
- SacMEGA case study is reproducible.
- At least one synthetic known-truth benchmark is complete.
- If possible, one additional dataset supports generalization beyond SacMEGA.

