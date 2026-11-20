# PONDEROSA V3 Development

This repository is a private development copy of PONDEROSA for building generalized improvements inspired by real-world SacMEGA analysis.

The goal is to make PONDEROSA easier to use on many datasets, not only SacMEGA. SacMEGA is treated as a motivating case study and benchmark for issues that can occur in real SNP-array cohorts:

- Sparse or missing pedigree labels
- Simulation-based classifier training
- Duplicate/MZ-like samples that should not be reported as ordinary full siblings
- Optional validation against KING or truth files
- Human-readable QC, inference, validation, and reporting outputs

Planned V3 modules include:

- General input validation
- Duplicate/MZ QC and component reporting
- Sparse-pedigree/no-truth inference support
- KING/truth validation utilities
- Readable run reports
- Hardened simulation-training workflow
- Tests, examples, and user documentation

The original PONDEROSA repository should remain untouched. All V3 development should happen in this copy.
