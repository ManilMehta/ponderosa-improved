# PONDEROSA V3 Generalized Workflows

PONDEROSA V3 is intended to support real datasets beyond SacMEGA. SacMEGA is the motivating case study, but core code must avoid dataset-specific paths, identifiers, and thresholds unless they are exposed as configuration.

## Standard Inputs

Every workflow starts from standard user-provided files:

- FAM file with PLINK-style sample IDs and optional parent IDs.
- IBD segment file from a supported caller such as `hap-ibd` or `phasedibd`.
- Genetic map when the IBD caller requires coordinate conversion.
- Optional pretrained classifier file.
- Optional KING output for validation.
- Optional truth file for validation.
- Optional simulation-training configuration.

## Workflow A: Real Labels Available

Use this when the FAM/pedigree contains enough known relationships for training.

Expected behavior:

- Validate the FAM and IBD inputs.
- Summarize known relationship counts.
- Train classifiers from real labels.
- Run relationship inference.
- Write pair-level predictions and run summary.
- Optionally validate against KING or a truth file.

## Workflow B: Sparse Or Missing Pedigree

Use this when the real dataset has few or no known relationship labels.

Expected behavior:

- Validate that the dataset can be loaded.
- Warn that real labels are insufficient for training.
- Allow inference only when a pretrained classifier is supplied.
- Recommend simulation-based training when no classifier exists.
- Continue without an internal truth evaluation when no truth file is provided.

## Workflow C: Duplicate/MZ-Like Samples Present

Use this when the dataset may contain duplicated samples, technical replicates, monozygotic twins, or other near-identical sample records.

Expected behavior:

- Compute duplicate/MZ-like pair flags from generic IBD features.
- Build connected components of duplicate-like samples.
- Write duplicate-pair and duplicate-component outputs.
- Mark duplicate-like pairs separately from ordinary biological relationships.
- Optionally exclude or collapse duplicate-like samples before inference.

## Workflow D: External Validation Available

Use this when KING output or known truth labels are available.

Expected behavior:

- Normalize pair IDs across formats such as `FID/IID` and `FID_IID`.
- Compare PONDEROSA predictions to KING/truth labels.
- Write agreement, discordance, and missing-validation outputs.
- Highlight ambiguous comparisons such as PONDEROSA `FS` versus KING `Dup/MZ`.

## Workflow E: Simulation-Based Training

Use this when the dataset lacks enough real relationship labels.

Expected behavior:

- Validate ped-sim inputs, genetic maps, and IBD caller wrapper.
- Generate or validate simulated pedigree definitions.
- Run simulation and IBD calling with reproducible metadata.
- Train classifiers from simulated labels.
- Save classifier and metadata for future inference.

## Non-Goals

PONDEROSA V3 should not hardcode:

- SacMEGA paths.
- HIVE account names or partitions.
- A single ID format.
- A single genetic map location.
- A single IBD caller.
- SacMEGA-specific validation counts.

SacMEGA-specific details belong in the SacMEGA case study documentation, not in reusable modules.
