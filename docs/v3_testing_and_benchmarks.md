# PONDEROSA V3 Testing And Benchmark Plan

PONDEROSA V3 should be tested at three levels: small unit tests, workflow-level integration tests, and publication-facing benchmarks.

## Unit Tests

Required coverage:

- Input validation for FAM, IBD, map, truth, and KING files.
- Sparse-pedigree warnings when FAM parent IDs are absent.
- Duplicate/MZ-like pair detection from generic IBD features.
- Duplicate/MZ-like connected-component construction.
- KING `.kin0` parsing and ID normalization.
- Prediction-versus-KING comparison counts.
- Report writer output creation.
- Simulation-training metadata serialization.

## Integration Tests

Required workflow checks:

- No-truth inference should complete when pretrained classifiers are supplied.
- Empty sibling sets should not crash pedigree setup.
- Classifier inference should preserve row alignment when NaN-feature rows are skipped.
- Duplicate/MZ-like pairs should be detected before close-relative interpretation.
- Validation outputs should be generated from a small synthetic KING file.

## Benchmark Datasets

Minimum benchmark set:

- Synthetic known-truth dataset generated from repository test utilities.
- SacMEGA case study as a real sparse-pedigree dataset with KING validation.

Preferred additional benchmark:

- One independent public or collaborator dataset with known or partially known relationships.

## Before/After Analyses

The key V3 comparison is current PONDEROSA behavior versus V3 behavior:

- Current behavior may label duplicate/MZ-like pairs as `FS`.
- V3 should flag duplicate/MZ-like pairs separately.
- Current validation requires custom scripts.
- V3 should provide standard validation outputs.
- Current sparse-pedigree runs can fail or require manual patches.
- V3 should run cleanly with warnings and documented next steps.

## Publication-Facing Metrics

Report at minimum:

- Number of pairs analyzed.
- Number of close relatives predicted by class.
- Number of duplicate/MZ-like pairs and components detected.
- PONDEROSA versus KING/truth agreement counts.
- Runtime and memory for training and inference.
- Before/after reduction in misleading `FS` calls caused by duplicate/MZ-like samples.

