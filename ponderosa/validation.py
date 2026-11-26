"""Validation utilities for comparing PONDEROSA output to external calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


PairKey = Tuple[str, str]


@dataclass(frozen=True)
class KingValidationColumns:
    """Column names for KING kin0-like validation files."""

    fid1: str = "FID1"
    id1: str = "ID1"
    fid2: str = "FID2"
    id2: str = "ID2"
    relationship: str = "InfType"
    kinship: str = "Kinship"
    ibd1: str = "IBD1Seg"
    ibd2: str = "IBD2Seg"
    prop_ibd: str = "PropIBD"


@dataclass(frozen=True)
class PredictionColumns:
    """Column names for PONDEROSA pair output."""

    id1: str = "id1"
    id2: str = "id2"
    prediction: str = "pred_rel"
    degree: str = "degree"
    probability: str = "prob"


def normalized_pair_key(id1: object, id2: object) -> PairKey:
    """Return an order-independent pair key."""

    left, right = str(id1), str(id2)
    return tuple(sorted((left, right)))


def combine_fid_iid(fid: object, iid: object, sep: str = "_") -> str:
    """Convert split KING FID/IID fields into a PONDEROSA-style sample ID."""

    return f"{fid}{sep}{iid}"


KING_RENAMES = {"IID1": "ID1", "IID2": "ID2", "KINSHIP": "Kinship", "INFTYPE": "InfType", "IBS0": "IBS0",
                "HETHET": "HetHet", "NSNP": "N_SNP"}


def infer_king_relationship(kinship: float, ibs0: Optional[float] = None) -> str:
    """KING's degree boundaries (Manichaikul et al. 2010); PO vs FS from IBS0 when available."""
    if pd.isna(kinship):
        return "UN"
    if kinship > 0.354:
        return "Dup/MZ"
    if kinship > 0.177:
        if ibs0 is not None and not pd.isna(ibs0):
            return "PO" if ibs0 < 0.0012 else "FS"
        return "1st"
    if kinship > 0.0884:
        return "2nd"
    if kinship > 0.0442:
        return "3rd"
    return "UN"


def read_king_table(path: Path) -> pd.DataFrame:
    """
    Read KING output (.kin0 / .kin / .seg) or a plink2 --make-king-table file and return
    standard columns FID1 ID1 FID2 ID2 Kinship InfType (+ any others). plink2 tables
    (``#FID1 IID1 FID2 IID2 ... KINSHIP``) get an InfType derived from KING's kinship
    boundaries; files without FID columns use the IID as FID.
    """
    df = pd.read_csv(path, sep=r"\s+", dtype=str)
    df.columns = [c.lstrip("#") for c in df.columns]
    df = df.rename(columns={c: KING_RENAMES[c.upper()] for c in df.columns
                            if c.upper() in KING_RENAMES and c not in KING_RENAMES.values()})
    if "ID1" not in df.columns and "IID1" in df.columns:
        df = df.rename(columns={"IID1": "ID1", "IID2": "ID2"})
    for n in ("1", "2"):
        if f"FID{n}" not in df.columns and f"ID{n}" in df.columns:
            df[f"FID{n}"] = df[f"ID{n}"]
    if "Kinship" not in df.columns and "PropIBD" in df.columns:
        df["Kinship"] = (pd.to_numeric(df["PropIBD"], errors="coerce") / 2).astype(str)
    if "InfType" not in df.columns and "Kinship" in df.columns:
        kin = pd.to_numeric(df["Kinship"], errors="coerce")
        ibs0 = pd.to_numeric(df["IBS0"], errors="coerce") if "IBS0" in df.columns else [None] * len(df)
        df["InfType"] = [infer_king_relationship(k, i) for k, i in zip(kin, ibs0)]
    return df


def load_king_kin0(
    king_file: Path,
    columns: KingValidationColumns = KingValidationColumns(),
    fid_iid_sep: str = "_",
    iid_only: bool = False,
) -> pd.DataFrame:
    """Load KING kin0 output and add normalized pair IDs.

    By default sample IDs are FID and IID joined with ``fid_iid_sep`` (PONDEROSA's form
    when the FAM uses FID_IID sample IDs); ``iid_only=True`` matches on IID alone.
    """

    df = read_king_table(king_file)
    required = [columns.fid1, columns.id1, columns.fid2, columns.id2, columns.relationship]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"KING file missing required columns: {', '.join(missing)}")

    df = df.copy()
    if iid_only:
        df["king_id1"] = df[columns.id1].astype(str)
        df["king_id2"] = df[columns.id2].astype(str)
    else:
        df["king_id1"] = [combine_fid_iid(fid, iid, fid_iid_sep) for fid, iid in zip(df[columns.fid1], df[columns.id1])]
        df["king_id2"] = [combine_fid_iid(fid, iid, fid_iid_sep) for fid, iid in zip(df[columns.fid2], df[columns.id2])]
    df["pair_key"] = [normalized_pair_key(id1, id2) for id1, id2 in zip(df["king_id1"], df["king_id2"])]
    return df


def _king_lookup(king_df: pd.DataFrame) -> Dict[PairKey, pd.Series]:
    return {row["pair_key"]: row for _, row in king_df.iterrows()}


def compare_predictions_to_king(
    predictions: pd.DataFrame,
    king: pd.DataFrame,
    prediction_columns: PredictionColumns = PredictionColumns(),
    king_columns: KingValidationColumns = KingValidationColumns(),
) -> pd.DataFrame:
    """Join PONDEROSA pair predictions to KING calls by normalized pair ID."""

    required = [
        prediction_columns.id1,
        prediction_columns.id2,
        prediction_columns.prediction,
        prediction_columns.degree,
        prediction_columns.probability,
    ]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction file missing required columns: {', '.join(missing)}")

    lookup = _king_lookup(king)
    rows = []
    for _, pred_row in predictions.iterrows():
        row = pred_row.copy()
        key = normalized_pair_key(row[prediction_columns.id1], row[prediction_columns.id2])
        king_row = lookup.get(key)

        if king_row is None:
            row["king_InfType"] = "missing"
            row["king_Kinship"] = pd.NA
            row["king_IBD1Seg"] = pd.NA
            row["king_IBD2Seg"] = pd.NA
            row["king_PropIBD"] = pd.NA
        else:
            row["king_InfType"] = king_row.get(king_columns.relationship, pd.NA)
            row["king_Kinship"] = king_row.get(king_columns.kinship, pd.NA)
            row["king_IBD1Seg"] = king_row.get(king_columns.ibd1, pd.NA)
            row["king_IBD2Seg"] = king_row.get(king_columns.ibd2, pd.NA)
            row["king_PropIBD"] = king_row.get(king_columns.prop_ibd, pd.NA)

        rows.append(row)

    return pd.DataFrame(rows)


def validation_counts(joined: pd.DataFrame, prediction_col: str = "pred_rel", king_col: str = "king_InfType") -> pd.DataFrame:
    """Return counts by PONDEROSA prediction and KING relationship."""

    return (
        joined.groupby([prediction_col, king_col], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values([prediction_col, king_col])
        .reset_index(drop=True)
    )


def missing_validation_pairs(joined: pd.DataFrame, king_col: str = "king_InfType") -> pd.DataFrame:
    """Return prediction rows that were absent from the external validation file."""

    return joined.loc[joined[king_col] == "missing"].copy()


def write_king_validation_outputs(
    predictions_file: Path,
    king_file: Path,
    output_prefix: str,
    fid_iid_sep: str = "_",
    iid_only: bool = False,
) -> Tuple[Path, Path, Path]:
    """Compare PONDEROSA predictions to KING and write reusable TSV outputs."""

    predictions = pd.read_csv(predictions_file, sep="\t", dtype=str)
    king = load_king_kin0(king_file, fid_iid_sep=fid_iid_sep, iid_only=iid_only)
    joined = compare_predictions_to_king(predictions, king)
    counts = validation_counts(joined)
    missing = missing_validation_pairs(joined)

    joined_path = Path(f"{output_prefix}_prediction_vs_king.tsv")
    counts_path = Path(f"{output_prefix}_validation_counts.tsv")
    missing_path = Path(f"{output_prefix}_missing_from_validation.tsv")

    joined.to_csv(joined_path, sep="\t", index=False)
    counts.to_csv(counts_path, sep="\t", index=False)
    missing.to_csv(missing_path, sep="\t", index=False)

    return joined_path, counts_path, missing_path

