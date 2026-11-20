# PONDEROSA — End-to-End Test Run Report

**Author:** _<your name>_
**Date:** 2026-06-17
**Audience:** _<boss / advisor>_
**Repo:** `PONDEROSA` (v2 rewrite, package version reports `v0.2.0`)
**Machine:** macOS (Apple Silicon), Python 3.12, isolated virtual environment

---

## 1. Objective

Get PONDEROSA running end-to-end on a controlled dataset so we can (a) confirm the
analysis pipeline actually works, (b) understand exactly what inputs it needs, and
(c) document any bugs/rough edges that block a smooth user experience. This is the
groundwork for making the data-input pipeline more user-friendly.

Rather than use real genotype data, we used the project's built-in **synthetic data
generator** (`tests/generate_files.py`), which fabricates pedigrees with *known*
relationships and emits a complete, correctly-formatted input set. This lets us check
predictions against ground truth.

---

## 2. Summary (TL;DR)

- ✅ **PONDEROSA now runs end-to-end** and produces interpretable output.
- 🔧 Getting there required **working around 2 issues** and **fixing 1 code bug**:
  1. Training hard-fails with a **blank error message** when any relationship class is under-represented (brittle `assert`).
  2. The default IBD filter (`min_total_ibd = 50 cM`) **silently removes 4th-degree pairs**, starving that class and triggering issue #1.
  3. The classifier-selection logic **crashed on collinear features** (QDA "covariance not full rank"). *Fixed in code.*
- 📊 On the synthetic data, PONDEROSA gets **degree classification 88.3%** and **1st-degree relationships 100%**, but **cannot yet resolve specific 2nd-degree relationships** (they collapse to the intermediate `GPAV` node; half-siblings are never separated out). This is partly confounded by the small synthetic genome.

---

## 3. Environment setup

Run from the repository root.

```bash
cd ~/Desktop/henn/PONDEROSA

# Isolated environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PONDEROSA + dependencies (reads setup.py)
pip install -e .
```

**Note:** `scipy` is imported by the code but not listed in `setup.py`; it gets pulled
in transitively via scikit-learn here, but on a clean machine you may need
`pip install scipy`. *(Recommendation: add `scipy` to `setup.py`.)*

Verify:

```bash
python -c "import ponderosa, polars, sklearn, networkx, scipy; print('environment OK')"
```

> First import is slow (~10-40s) the first time in a fresh venv because many compiled
> extensions load and macOS verifies the new binaries. Subsequent imports are instant.

---

## 4. Procedure

### Step 1 — Generate a synthetic dataset with known truth

`tests/generate_files.py` builds pedigrees of known relationship types and writes a full
input set under a single prefix. We generated **50 pairs per relationship type**
(see §5 for why 15 was not enough) into a dedicated folder:

```bash
mkdir -p demo_run
python -c "from tests.generate_files import GeneratePairs; g = GeneratePairs(50); g.generate_all(); g.write_out('demo_run/demo')"
```

This produced:

| File | Purpose |
|------|---------|
| `demo_phasedibd.txt` | IBD segments (phasedibd format, cM, with header) |
| `demo_hapibd.txt` | Same segments in hap-ibd format (bp, no header) |
| `demo.fam` | PLINK pedigree → defines the *known* relationships |
| `demo.map` | PLINK genetic map (4 cols: chr, rsid, cM, bp) |
| `demo_ages.txt` | Ages per individual |
| `demo_truth.txt` | Ground-truth relationship labels (for scoring) |
| `demo_config.yaml` | Ready-made config pointing at the phasedibd file |

### Step 2 — Run PONDEROSA

```bash
python -m ponderosa --config demo_run/demo_config.yaml --min-total-ibd 20 -v
```

- `--min-total-ibd 20` overrides the config's default of 50 (see §5, Problem B).
- `-v` enables progress logging; add `--debug` for full tracebacks.

### Step 3 — Inspect outputs

```bash
cat demo_run/demo_ponderosa_output_evaluation.txt
column -t -s$'\t' demo_run/demo_ponderosa_output_pairs.txt | head -20
```

Outputs produced (prefix `demo_run/demo_ponderosa_output`):

| File | Contents |
|------|----------|
| `..._pairs.txt` | Main result: per-pair prediction, probability, IBD features, per-node probabilities |
| `..._evaluation.txt` | Accuracy vs. truth |
| `....classif.pkl` | Trained classifiers (binary) |
| `....mhier.pkl` | Probability-hierarchy object (binary) |

---

## 5. Problems encountered (and how they were resolved)

### Problem A — Training fails with a blank `Error:` message
**Symptom:** Run reached "Training classifiers…" then printed `Error:` with no text.
With `--debug`, the traceback pointed to:

```
File ".../ponderosa/classifiers.py", line 87, in _validate_training_data
    assert np.all(counts > min_training_pairs)
AssertionError
```

**Cause:** Training requires **every** relationship class to have **more than 5**
usable pairs (`MIN_TRAINING_PAIRS = 5`). The guard is a bare `assert` with no message,
so the CLI prints an empty error — unhelpful to any user.

**Resolution (workaround):** Generate more pairs and (see Problem B) lower the IBD
threshold. **Recommendation:** replace the bare `assert` with a real error that names
*which* relationship is under-represented and how many pairs it found.

### Problem B — Increasing the pair count did not help; one class stayed empty
**Symptom:** Even with 50 pairs/relationship, the same assertion failed.

**Diagnosis:** We wrote a small diagnostic (loads the data the same way PONDEROSA does
and prints usable-pair counts per class). It showed the **`4th`-degree class dropped
from 50 pairs to 2** after feature computation, while every other class was healthy:

```
[degree] 4th: before NaN-mask 50  ->  after NaN-mask 2   (must be > 5)
```

**Cause:** Synthetic 4th-degree (half-cousin / HCO) pairs share only ~40 cM of IBD, but
the config's `min_total_ibd = 50 cM` filter removes any pair sharing < 50 cM **before**
features are computed. So ~48 of 50 4th-degree pairs were silently dropped.

**Resolution:** Override the threshold: `--min-total-ibd 20`.

**Broader implication:** The default `min_total_ibd` (50 cM) is *higher* than the IBD
that genuine 3rd/4th-degree relatives typically share, so distant relatives get filtered
out before classification. With defaults, PONDEROSA effectively cannot be trained to
recognize 4th-degree relationships. Worth surfacing this in docs and/or warning the user.

### Problem C — QDA crash on collinear features *(code bug — fixed)*
**Symptom:** After Problems A/B were resolved, training advanced through the first two
classifiers, then crashed on the haplotype-score classifier:

```
ValueError: The covariance matrix of class GPAV is not full rank.
Increase the value of `reg_param` to reduce the collinearity.
```

**Cause:** The haplotype-score classifier uses **QDA**, which needs each class's
covariance matrix to be full-rank (invertible). For the `GPAV` class the two features
were perfectly collinear, so the covariance was singular. The selector
`choose_discriminant_classifier()` guards against *zero-variance* features but **not
against collinearity**, so it chose QDA and QDA failed.

**Fix (applied):** Added a small regularization term to the QDA model, which is exactly
what scikit-learn's error message recommends.

```python
# ponderosa/classifiers.py, choose_discriminant_classifier()
model = QuadraticDiscriminantAnalysis(reg_param=1e-2)
```

`reg_param` shrinks each class covariance slightly toward the identity, keeping it
full-rank even with collinear features, without meaningfully distorting the decision
boundary.

### Known latent issue (did not trigger, but should be fixed)
In `ponderosa/core.py`, the `evaluation` variable is only assigned inside the
`if config.files.truth:` block, yet it is always referenced when building the final
results object. A run **without** a truth file (or where evaluation throws and is caught)
will fail at the very end with `UnboundLocalError: evaluation`. Our config included a
truth file and evaluation succeeded, so we did not hit it — but it will bite any user who
omits `--truth`.

---

## 6. Results

### Accuracy vs. ground truth

| Level | Accuracy | Pairs evaluated |
|-------|----------|-----------------|
| Degree (1st / 2nd / 3rd / 4th) | **88.3%** (393/445) | 445 |
| Specific relationship | **43.6%** (152/349) | 349 |

### Known vs. predicted (specific relationship) — 494 pairs

```
pred_rel   2nd  3rd  4th  FS  GP  GPAV  MGP  PO
known_rel
AV           2    1    0   0   0    47    0   0
CO           0   50    0   0   0     0    0   0
FS           0    0    0  50   0     0    0   0
HCO          0    0   44   0   0     0    0   0
MGP          3    0    0   0   3    44    0   0
MHS          0    0    0   0   0    50    0   0    <- true half-sibs, all -> GPAV
PGP          0    0    0   0   2    48    0   0
PHS          2    2    0   0   1    45    0   0    <- true half-sibs, all -> GPAV
PO           0    0    0   0   0     0    0  50
unknown      0    0    0  20  10    18    2   0
```

### Interpretation

**Works well**
- **1st degree perfect:** PO→PO (50/50), FS→FS (50/50), probabilities ≈ 1.0.
- **3rd / 4th degree detection perfect at degree level:** CO→"3rd" (50/50), HCO→"4th" (44/44).
- Degree calls are confident (mean prob: 1st 0.997, 3rd 0.87, 4th 0.89).

**Main weakness — 2nd-degree relationships collapse to `GPAV`**
- Every 2nd-degree subtype (AV, MGP, MHS, PGP, PHS) is predicted `GPAV`.
- Two effects: (1) the **half-sib (`HS`) branch is never chosen** — all 100 PHS+MHS pairs
  are misrouted into `GPAV`; the haplotype-score classifier isn't separating these groups.
  (2) Even inside `GPAV`, deeper nodes don't clear the `min_probability = 0.5` threshold,
  so the prediction stops at the intermediate `GPAV` node (mean 2nd-degree prob ≈ 0.594).
- Because `GPAV` is not a specific leaf relationship, these all score as wrong at the
  specific level — which is why specific accuracy is only 43.6%.

**Bottom line:** PONDEROSA reliably distinguishes *degree* (1st/2nd/3rd/4th) and nails
1st-degree relationships, but cannot yet pin down *which* 2nd-degree relationship a pair is.

---

## 7. Caveats (important for interpreting the numbers)

These results are from a **synthetic toy dataset**; do not over-read the accuracy:

- Only **2 chromosomes / 320 cM** total genome (real human ≈ 3545 cM) → far less signal,
  especially for the subtle haplotype patterns that separate half-sibs from
  grandparent/avuncular.
- We **lowered `min_total_ibd` to 20**, which also admits noisier distant pairs.
- The `simple` segment generator assigns haplotypes crudely, which likely hurts the
  haplotype-score classifier specifically — the exact classifier underperforming above.

A fairer test would use the `complex` generator (full 22-chromosome, 3545 cM) or real data.

---

## 8. Code changes made in this session

| File | Change | Reason |
|------|--------|--------|
| `ponderosa/classifiers.py` | `QuadraticDiscriminantAnalysis(reg_param=1e-2)` in `choose_discriminant_classifier()` | Fix QDA crash on collinear features (Problem C) |

No other source files were modified. All other fixes were runtime overrides
(`--min-total-ibd 20`) or test-data parameters (`GeneratePairs(50)`).

---

## 9. Recommendations / next steps

1. **Friendlier failures (high impact, low effort).** Replace the bare training `assert`
   with a clear message naming the under-represented relationship and its pair count.
2. **Add an input preflight + ID-reconciliation step.** Validate file schemas and check
   that IDs match across IBD/FAM/ages/truth before any computation — most "it doesn't
   work" cases are silent ID mismatches or below-threshold filtering.
3. **Document the `min_total_ibd` trade-off** (and consider warning when a degree class is
   thin because of it), since the default removes most distant relatives.
4. **Fix the latent `evaluation` UnboundLocalError** in `core.py` so runs without `--truth`
   don't crash at the end.
5. **Add `scipy` to `setup.py`.**
6. **Re-run on realistic data** (`complex` generator, 22 chromosomes / 3545 cM, or real
   data) to fairly assess 2nd-degree resolution before drawing conclusions about accuracy.

---

## Appendix A — Exact commands used

```bash
# Setup
cd ~/Desktop/henn/PONDEROSA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

# Generate synthetic data (50 pairs per relationship)
mkdir -p demo_run
python -c "from tests.generate_files import GeneratePairs; g = GeneratePairs(50); g.generate_all(); g.write_out('demo_run/demo')"

# Run PONDEROSA
python -m ponderosa --config demo_run/demo_config.yaml --min-total-ibd 20 -v

# Inspect results
cat demo_run/demo_ponderosa_output_evaluation.txt
column -t -s$'\t' demo_run/demo_ponderosa_output_pairs.txt | head -20
```

## Appendix B — Diagnostic snippet (used to find the empty class in Problem B)

```python
import numpy as np
from ponderosa.config import PonderosaConfig
from ponderosa.data_loading import load_individuals, load_pairs
from ponderosa.pedigree import build_pedigree, PedigreeHierarchy
from ponderosa.classifiers import CLASSIF_DICT, prepare_classifier_data

config = PonderosaConfig.from_yaml("demo_run/demo_config.yaml")
config.validate()
hierarchy = PedigreeHierarchy.from_yaml()
ind   = load_individuals(config.files)
pairs = load_pairs(config.files, config.algorithm)
registry = build_pedigree(ind, pairs, hierarchy)

for name, cls in CLASSIF_DICT.items():
    c = cls()
    pair_arr = registry.get_relatives_from(c.nodes)
    y = pair_arr[:, 2]
    X = pairs.get_pair_data_from(pair_arr[:, :2], *c.features, output_style="flatten")
    _, yc = prepare_classifier_data(X, y)
    print(name, dict(zip(*np.unique(yc, return_counts=True))))
```
