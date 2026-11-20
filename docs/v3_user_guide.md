# PONDEROSA V3 User Guide

PONDEROSA V3 extends PONDEROSA for real datasets where pedigree labels may be sparse, duplicate/MZ-like samples may be present, and external validation may be needed.

## Choose Your Workflow

### If You Have Real Pedigree Labels

Use standard PONDEROSA training and inference.

Recommended steps:

1. Validate FAM, IBD, and map inputs.
2. Train classifiers from known relationships.
3. Run inference.
4. Generate a run report.
5. Validate against KING or truth labels if available.

### If You Have Sparse Or No Pedigree Labels

Use pretrained or simulation-trained classifiers.

Recommended steps:

1. Validate inputs and confirm that real labels are insufficient.
2. Train classifiers from simulation or use an existing compatible classifier.
3. Run inference with `--training`.
4. Treat subtype calls as model predictions unless externally validated.

### If You May Have Duplicate Or MZ-Like Samples

Run duplicate/MZ QC before interpreting close relatives.

Recommended steps:

1. Run standard pair-feature generation or inference.
2. Run duplicate/MZ-like pair detection.
3. Inspect duplicate-like connected components.
4. Decide whether to flag, exclude, or collapse samples.
5. Interpret `FS` predictions only after duplicate/MZ-like samples are accounted for.

### If You Have KING Output

Use validation utilities after inference.

Recommended steps:

1. Provide PONDEROSA pair predictions.
2. Provide KING `.kin0` output.
3. Normalize IDs using the configured separator.
4. Write prediction-versus-KING and count summaries.
5. Inspect discordant pairs, especially `FS` versus `Dup/MZ`.

## Example Python API

```python
import pandas as pd

from ponderosa.qc import detect_duplicate_like_pairs, duplicate_components
from ponderosa.validation import load_king_kin0, compare_predictions_to_king, validation_counts
from ponderosa.reporting import write_run_report

pairs = pd.read_csv("my_ponderosa_pairs.txt", sep="\t")
duplicates = detect_duplicate_like_pairs(pairs)
components = duplicate_components(duplicates)

king = load_king_kin0("my_king.kin0")
joined = compare_predictions_to_king(pairs, king)
counts = validation_counts(joined)

write_run_report(
    "my_ponderosa_v3_report.md",
    metadata={"dataset": "my_dataset"},
    duplicate_components=components,
    validation_counts=counts,
)
```

## Output Interpretation

PONDEROSA relationship labels should be interpreted with QC context.

- `PO` calls are usually easier to validate with degree-level tools such as KING.
- Second-degree subtype calls can be useful but should be treated as model predictions unless truth labels confirm them.
- `FS` calls can include duplicate/MZ-like samples if no duplicate/MZ class is modeled.
- Duplicate/MZ-like components should be reported separately from ordinary family relationships.

## SacMEGA Case Study

SacMEGA should be used as a real-data example showing why V3 features matter:

- Sparse real pedigree labels required simulation-based training.
- KING validation confirmed `PO`, `FS`, and second-degree degree-level predictions.
- A 16-sample duplicate-like component produced 120 `FS` labels in standard inference.
- V3 duplicate/MZ QC turns that failure mode into an explicit QC output.

