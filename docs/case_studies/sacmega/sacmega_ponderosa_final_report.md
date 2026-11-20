# SacMEGA PONDEROSA Technical Report

## Executive Summary

We successfully built and ran a PONDEROSA workflow on the SacMEGA SNP-array dataset on the UC Davis HIVE server. The final real-data run analyzed 23,375 IBD-sharing SacMEGA pairs, generated relationship predictions, and produced interpretable close-relative output files.

The main technical obstacle was that SacMEGA did not contain enough known pedigree structure to train PONDEROSA directly from the real FAM file. To solve this, we used a simulation-based training strategy: simulate labeled relatives, train PONDEROSA classifiers on the simulated labels, then apply the trained model to real SacMEGA hap-IBD output. Several PONDEROSA code issues and HIVE environment issues had to be debugged and patched before this worked end-to-end.

The final close-relative predictions were:

- 184 full-sibling-like (`FS`) pairs
- 28 parent-offspring (`PO`) pairs
- 24 maternal half-sibling (`MHS`) pairs
- 21 avuncular (`AV`) pairs

Validation against KING supported all 28 `PO` calls, all 45 second-degree calls (`AV` and `MHS` mapping to KING `2nd`), and 31 normal `FS` calls. Another 33 PONDEROSA `FS` calls were called `Dup/MZ` by KING, and 120 additional PONDEROSA `FS` calls formed one 16-sample duplicate-like component with near whole-genome IBD2 sharing. Those 120 should not be interpreted as ordinary full siblings.

## Main HIVE Run Directory

The final working directory was:

`/quobyte/bmhenngrp/from-lssc0/data/snp-array/SAC_MEGA/sacmega_ponderosa_run_20260727`

Important subdirectories:

- `inputs/`: real SacMEGA PONDEROSA input files
- `outputs/`: trained classifiers, real inference output, validation summaries
- `logs/`: Slurm stdout/stderr logs
- `simulation/`: simulated training data and hap-IBD output
- `PONDEROSA_patched/`: private patched copy of PONDEROSA used for the HIVE runs
- `venv/`: private Python virtual environment used to avoid shared-environment dependency problems

## Final Input Files

The successful real-data run used:

- IBD file: `inputs/SAC_MEGA_phased_hapibd.ibd.gz`
- FAM file: `inputs/SAC_MEGA_autosomes_acgt.fid_iid.fam`
- Genetic map: `/quobyte/bmhenngrp/from-lssc0/data/snp-array/SAC_MEGA/genetic_hapibd.map`
- Trained classifier: `outputs/SAC_MEGA_sim_training.classif.pkl`

The real run used `hap-ibd` mode, not `phasedibd` mode. This mattered because hap-IBD output stores physical positions that PONDEROSA must convert to genetic positions using the genetic map.

## Why Simulation-Based Training Was Needed

PONDEROSA needs labeled relationship examples to train its classifiers. The available SacMEGA pedigree information was too sparse for this:

- The real FAM did not contain parent-child structure.
- The real data did not provide enough labeled second-degree subtypes.
- KING could validate some relationships, especially `PO` and `FS`, but could not provide enough labeled examples across all PONDEROSA relationship classes for training.

We therefore trained PONDEROSA on simulated relatives and then applied those trained classifiers to real SacMEGA IBD pairs.

The successful simulation-training job produced:

`outputs/SAC_MEGA_sim_training.classif.pkl`

Training completed successfully in Slurm job `19289731`. The log showed training examples for:

- `AV`
- `MGP`
- `MHS`
- `PGP`
- `PHS`

The simulated training job completed in about 2 minutes and used about 629 MB RAM.

## Major Issues and Fixes

### 1. HIVE Python Environment Did Not Have Required Packages

Early HIVE runs failed because the default Python environment did not reliably provide required PONDEROSA dependencies such as `pandas`, `polars`, `scikit-learn`, `networkx`, `pyyaml`, and related packages.

Fix:

- Created a private virtual environment inside the run folder.
- Installed PONDEROSA and its dependencies into that environment.
- Activated that environment inside each Slurm script before running `python -m ponderosa`.

What the fix did:

This made the run reproducible and independent of login-node or module-state differences. It also prevented dependency changes in shared environments from changing PONDEROSA behavior.

### 2. Python Bytecode Cache Permission Error

Some Python commands failed with permission errors while trying to create `__pycache__` files.

Fix:

- Set `PYTHONPYCACHEPREFIX=/tmp/${USER}_pycache` in the Slurm scripts.

What the fix did:

This redirected Python bytecode cache writes to a user-writable temporary directory, avoiding permission problems in shared or read-only directories.

### 3. Wrong IBD File Format Attempt

One attempted IBD input failed with an error like:

```text
could not parse '530_1446' as dtype 'i8' at column 'id1_haplotype'
```

Root cause:

The attempted file had been reformatted incorrectly for PONDEROSA's hap-IBD loader. PONDEROSA expected hap-IBD's fixed column order, and the IDs appeared where haplotype numbers were expected.

Fix:

- Switched back to the original hap-IBD output:

`hapibd_output/SAC_MEGA_phased_hapibd.ibd.gz`

What the fix did:

This restored the expected hap-IBD column layout, allowing PONDEROSA to parse IBD segments correctly.

### 4. hap-IBD Genetic Map Was Not Passed Into the Loader

PONDEROSA failed with a missing `genetic_map` error when using `hap-ibd`.

Root cause:

The hap-IBD loader needs a genetic map to convert physical positions to cM, but `load_pairs()` did not pass the map into `load_ibd_from_file()` when `ibd_caller` was `hap-ibd`.

Fix:

Patched `ponderosa/data_loading.py` so `load_pairs()` builds a `GeneticMap` and passes it into the loader:

```python
load_kwargs = {}
if files.ibd_caller == "hap-ibd":
    load_kwargs["genetic_map"] = GeneticMap.add_plink_list(files.map_file_list)
```

What the fix did:

This made hap-IBD input usable by giving the loader the map needed to convert base-pair intervals into genetic cM intervals. Without this, PONDEROSA could not correctly compute IBD sharing features from hap-IBD output.

### 5. CLI Defaults Overrode YAML Configuration

PONDEROSA initially behaved as though the input was `phasedibd` even when the YAML config specified `hap-ibd`.

Root cause:

The command-line interface defaulted `--ibd-caller` to `phasedibd`, and that default could override the YAML setting.

Fix:

- Passed key arguments explicitly in the Slurm command, including:
  - `--ibd`
  - `--fam`
  - `--ibd-caller hap-ibd`
  - `--map`
  - `--output`
  - `--min-segment-length`
  - `--min-total-ibd`
  - `--training` or `--train-only`

What the fix did:

This removed ambiguity between config-file values and CLI defaults. The run used the intended hap-IBD loading path.

### 6. No Sibling Pairs Caused a Crash

PONDEROSA crashed while trying to build sibling training data when no sibling pairs were present in the real pedigree.

Root cause:

`Siblings.from_pedigree_data()` assumed that `sibling_pairs` was non-empty. With an empty real FAM, there were no sibling pairs, and downstream NumPy indexing failed.

Fix:

Patched `ponderosa/pedigree.py`:

```python
if sibling_pairs.shape[0] == 0:
    sibling_ibd_data = np.empty((0, 2))
else:
    sibling_ibd_data = pairs.get_pair_data_from(
        sibling_pairs, "IBD1", "IBD2", output_style="flatten"
    )
```

What the fix did:

This allowed PONDEROSA to continue cleanly when the real pedigree had no sibling labels. The fix preserves the empty state instead of trying to index IBD arrays with an empty pair set.

### 7. QDA Failed Because Simulated Features Were Collinear

Simulation training failed with:

```text
The covariance matrix of class GPAV is not full rank.
Increase the value of reg_param to reduce the collinearity.
```

Root cause:

PONDEROSA uses quadratic discriminant analysis for some classifiers. Simulated training features can be highly collinear, making the class covariance matrix singular or near-singular.

Fix:

Patched `ponderosa/classifiers.py` to use:

```python
QuadraticDiscriminantAnalysis(reg_param=1e-2)
```

What the fix did:

`reg_param` shrinks each class covariance estimate toward the identity matrix. This stabilizes matrix inversion and prevents QDA from crashing when features are collinear. The model can still learn class differences, but with a more stable covariance estimate.

### 8. `ped-sim` Build Failed Because Boost Headers Were Missing

Building `ped-sim` initially failed with a missing Boost header error:

```text
fatal error: boost/math/special_functions/gamma.hpp: No such file or directory
```

Root cause:

The necessary Boost development headers were not available in the active HIVE environment.

Fix:

- Loaded HIVE modules including:
  - `gcc/13.2.0`
  - `boost/1.84.0`
- Rebuilt `ped-sim`.

What the fix did:

This made the required C++ headers available to the compiler and allowed the `ped-sim` executable to build successfully.

### 9. No-Truth Runs Could Crash After Inference

The real SacMEGA inference run did not have a PONDEROSA truth file. PONDEROSA could reach the output stage and then fail because the `evaluation` variable was only defined inside the optional truth-file block.

Root cause:

`evaluation` was referenced in the returned `PonderosaResults` object even when no truth file existed.

Fix:

Patched `ponderosa/core.py`:

```python
evaluation = None
if config.files.truth:
    ...
```

What the fix did:

This made no-truth inference runs complete normally. It also made the result object accurately represent that no evaluation was performed inside PONDEROSA itself.

### 10. Case-Sensitive FAM Path Caused a Failed Slurm Job

One real inference job failed immediately because the Slurm script pointed at:

`inputs/sacmega_autosomes_acgt.fid_iid.fam`

but the actual file was:

`inputs/SAC_MEGA_autosomes_acgt.fid_iid.fam`

Root cause:

Case-sensitive filename mismatch.

Fix:

Updated the Slurm script to use the correct uppercase filename.

What the fix did:

PONDEROSA could validate all required input files and proceed to load real SacMEGA data.

### 11. Classifier Row-Alignment Bug During Real Inference

The first real inference attempt with the trained classifier failed with:

```text
shape mismatch: value array of shape (23255,2) could not be broadcast to indexing result of shape (23375,2)
```

Root cause:

PONDEROSA removes rows containing NaN features before prediction. This is correct, but the inference code lost the mask describing which original pair rows were retained. One classifier returned probabilities for 23,255 valid rows, while the output matrix expected probabilities for all 23,375 pairs.

Fix:

Patched `ponderosa/classifiers.py` so inference keeps the valid-row mask:

```python
valid_mask = ~np.isnan(X).any(axis=1)
X = X[valid_mask]
return (X, valid_mask) if return_mask else X
```

Then `run_inference()` expands classifier probabilities back to the full pair matrix:

```python
if proba.shape[0] != pairs.n_pairs():
    full_proba = np.full((pairs.n_pairs(), proba.shape[1]), np.nan)
    full_proba[valid_mask] = proba
    proba = full_proba
```

What the fix did:

This preserved the alignment between predicted probabilities and original pair IDs. Rows with unusable haplotype features remain present in the output matrix as missing values rather than causing a shape mismatch.

## Successful Simulation Training Run

Slurm job:

`19289731`

Status:

`COMPLETED`, exit code `0:0`

Runtime and memory:

- Walltime: about 2 minutes
- Max memory: about 629 MB

Output:

`outputs/SAC_MEGA_sim_training.classif.pkl`

The training log showed the following relationship classes represented in simulated training:

- `AV`
- `MGP`
- `MHS`
- `PGP`
- `PHS`

This produced the pretrained classifier used in the final real SacMEGA inference run.

## Successful Real SacMEGA Inference Run

Slurm job:

`19292798`

Status:

`COMPLETED`, exit code `0:0`

Runtime and memory:

- Walltime: about 2 minutes
- Max memory: about 619 MB

Run summary:

- Individuals loaded: 839
- IBD-sharing pairs analyzed: 23,375
- Known relationships found in real FAM: 0
- Classifiers loaded: `degree`, `no_segments`, `hap_score`
- Files written: 3

Output files:

- `outputs/SAC_MEGA_real_inference_pairs.txt`
- `outputs/SAC_MEGA_real_inference.mhier.pkl`
- `outputs/SAC_MEGA_real_inference.classif.pkl`

The readable pair output was further filtered to close relatives:

`outputs/SAC_MEGA_real_inference_close_relatives.tsv`

That file contains 257 inferred close-relative pairs.

## Real Inference Results

PONDEROSA analyzed 23,375 real SacMEGA IBD-sharing pairs.

Prediction counts:

- `3rd`: 23,118
- `FS`: 184
- `PO`: 28
- `MHS`: 24
- `AV`: 21

Degree counts:

- `3rd`: 23,118
- `1st`: 212
- `2nd`: 45

Close-relative counts:

- `FS`: 184
- `PO`: 28
- `MHS`: 24
- `AV`: 21

The real FAM did not contain known relationships, so PONDEROSA reported `known_rel = unknown` for all 23,375 pairs. Validation therefore had to be done externally against KING.

## KING Validation

The validation file used was:

`/quobyte/bmhenngrp/from-lssc0/data/snp-array/SAC_MEGA/ponderosa_sacmega_input_data/sacmega_king.kin0`

KING relationship counts in that file:

- `UN`: 1,103
- `2nd`: 48
- `3rd`: 46
- `Dup/MZ`: 33
- `FS`: 31
- `PO`: 28
- `4th`: 22

Comparison between PONDEROSA close-relative predictions and KING:

- PONDEROSA `AV`: 21 matched KING `2nd`
- PONDEROSA `MHS`: 24 matched KING `2nd`
- PONDEROSA `PO`: 28 matched KING `PO`
- PONDEROSA `FS`: 31 matched KING `FS`
- PONDEROSA `FS`: 33 matched KING `Dup/MZ`
- PONDEROSA `FS`: 120 were missing from the KING validation file

The validation table was written to:

`outputs/SAC_MEGA_ponderosa_vs_king_close_relatives.tsv`

## Interpretation of the 120 Missing FS Calls

The 120 PONDEROSA `FS` calls missing from the KING validation file were investigated separately.

Result:

- All 120 were `FS` predictions.
- Neither sample ID in those pairs appeared in the KING `kin0` validation file.
- The 120 pairs formed exactly one connected component of 16 samples.
- `120 = 16 choose 2`, meaning every pair among those 16 samples was called `FS`.

The component IDs were:

```text
315_1007 316_1008 317_1010 318_1011 358_1094 377_1157 502_1410 52_141 598_1556 61_156 627_1605 63_159 685_1780 700_1868 780_2710 879_3408
```

This list was written to:

`outputs/SAC_MEGA_duplicate_like_component_ids.txt`

The pair-level IBD values in this component had:

- `ibd1 = 0`
- `ibd2` around 4,080 cM

That is not a normal full-sibling pattern. It is much more consistent with duplicate/MZ-like whole-genome sharing. PONDEROSA does not have a `Dup/MZ` output class, so duplicate-like pairs are forced into the closest first-degree label, often `FS`.

Therefore, the 120 missing `FS` calls should not be interpreted as ordinary siblings. They should be reported as a duplicate/MZ-like sample cluster requiring QC review.

## Clean Results Interpretation

The most reliable validated result set is:

- 28 parent-offspring pairs validated by KING as `PO`
- 31 full-sibling pairs validated by KING as `FS`
- 45 second-degree pairs validated by KING as `2nd`
  - 21 PONDEROSA `AV`
  - 24 PONDEROSA `MHS`
- 33 duplicate/MZ pairs detected by KING and labeled as `FS` by PONDEROSA because PONDEROSA lacks a duplicate/MZ class
- 120 additional duplicate-like pairs forming one 16-sample component, also labeled `FS` by PONDEROSA but not interpretable as ordinary full siblings

In practical terms, PONDEROSA successfully recovered close relatives from SacMEGA, but the `FS` class must be interpreted carefully because duplicate/MZ-like samples collapse into the `FS` label.

## Main Outputs Produced

Training output:

- `outputs/SAC_MEGA_sim_training.classif.pkl`

Real inference outputs:

- `outputs/SAC_MEGA_real_inference_pairs.txt`
- `outputs/SAC_MEGA_real_inference.mhier.pkl`
- `outputs/SAC_MEGA_real_inference.classif.pkl`

Derived summary outputs:

- `outputs/SAC_MEGA_real_inference_close_relatives.tsv`
- `outputs/SAC_MEGA_ponderosa_vs_king_close_relatives.tsv`
- `outputs/SAC_MEGA_ponderosa_missing_from_king.tsv`
- `outputs/SAC_MEGA_duplicate_like_component_ids.txt`

Important scripts:

- `run_ponderosa_train_sim.sh`
- `run_ponderosa_real_inference.sh`
- Slurm logs in `logs/`

## Recommendations and Remaining Work

1. Treat the 16-sample duplicate-like component as a QC issue.

   These samples show near whole-genome IBD2 sharing. They should be reviewed as possible duplicated samples, technical replicates, MZ twins, or other non-independent sample records.

2. Do not report all PONDEROSA `FS` calls as ordinary full siblings.

   Split them into:

   - KING-supported `FS`
   - KING `Dup/MZ`
   - duplicate-like component inferred from PONDEROSA IBD2

3. Use KING as the primary validation source for degree-level close relationships.

   KING directly supports the `PO` calls and the degree-level second-degree calls. PONDEROSA adds subtype predictions within second degree, but those subtype labels should be interpreted as model predictions rather than externally proven truth.

4. Consider rerunning inference after removing duplicate-like samples.

   This would produce a cleaner family-relationship output and reduce misleading `FS` calls caused by duplicate/MZ-like sharing.

5. Preserve the patched PONDEROSA copy and document patches clearly.

   Several patches were required for hap-IBD input, sparse pedigrees, simulation training, no-truth inference, and row-aligned probabilities. Those changes should be retained or upstreamed before future SacMEGA runs.

6. If a final supervisor-facing report is needed, simplify this technical report into a shorter methods/results summary.

   This file is intended to preserve the full debugging and technical trail. A shorter version can focus on goal, methods, successful outputs, validation, and biological interpretation.

