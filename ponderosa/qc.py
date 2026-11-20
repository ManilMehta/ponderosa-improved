"""Quality-control helpers for PONDEROSA V3.

The first V3 QC layer focuses on duplicate/MZ-like samples. These pairs should
be handled separately from ordinary biological relationships because a model
without a duplicate class will often force them into ``FS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class DuplicateQCConfig:
    """Thresholds for duplicate/MZ-like pair detection."""

    genome_length: float = 3545.0
    min_ibd2_fraction: float = 0.90
    max_ibd1_fraction: float = 0.05
    min_total_ibd_fraction: float = 0.90
    id1_col: str = "id1"
    id2_col: str = "id2"
    ibd1_col: str = "ibd1"
    ibd2_col: str = "ibd2"


@dataclass(frozen=True)
class DuplicateComponent:
    """A connected component of duplicate/MZ-like samples."""

    component_id: str
    samples: Tuple[str, ...]
    pair_count: int

    @property
    def sample_count(self) -> int:
        return len(self.samples)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _numeric(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"Column {name} contains non-numeric or missing values")
    return values


def detect_duplicate_like_pairs(pair_df: pd.DataFrame, config: DuplicateQCConfig = DuplicateQCConfig()) -> pd.DataFrame:
    """Return rows that look duplicate/MZ-like from generic IBD features."""

    required = [config.id1_col, config.id2_col, config.ibd1_col, config.ibd2_col]
    _require_columns(pair_df, required)

    ibd1 = _numeric(pair_df[config.ibd1_col], config.ibd1_col)
    ibd2 = _numeric(pair_df[config.ibd2_col], config.ibd2_col)
    total_ibd = ibd1 + ibd2

    ibd2_threshold = config.min_ibd2_fraction * config.genome_length
    ibd1_threshold = config.max_ibd1_fraction * config.genome_length
    total_threshold = config.min_total_ibd_fraction * config.genome_length

    mask = (
        (ibd2 >= ibd2_threshold)
        & (ibd1 <= ibd1_threshold)
        & (total_ibd >= total_threshold)
    )

    out = pair_df.loc[mask].copy()
    if out.empty:
        return out

    out["duplicate_qc_reason"] = (
        f"ibd2>={ibd2_threshold:.3f};"
        f"ibd1<={ibd1_threshold:.3f};"
        f"total_ibd>={total_threshold:.3f}"
    )
    return out


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def duplicate_components(
    duplicate_pairs: pd.DataFrame,
    id1_col: str = "id1",
    id2_col: str = "id2",
) -> List[DuplicateComponent]:
    """Build connected components from duplicate/MZ-like pair rows."""

    if duplicate_pairs.empty:
        return []

    _require_columns(duplicate_pairs, [id1_col, id2_col])

    uf = _UnionFind()
    pair_counts: Dict[str, int] = {}

    for row in duplicate_pairs[[id1_col, id2_col]].itertuples(index=False):
        id1, id2 = str(row[0]), str(row[1])
        uf.union(id1, id2)

    grouped: Dict[str, List[str]] = {}
    for sample in list(uf.parent):
        grouped.setdefault(uf.find(sample), []).append(sample)

    for row in duplicate_pairs[[id1_col, id2_col]].itertuples(index=False):
        root = uf.find(str(row[0]))
        pair_counts[root] = pair_counts.get(root, 0) + 1

    components = []
    for idx, (root, samples) in enumerate(sorted(grouped.items(), key=lambda item: (-len(item[1]), sorted(item[1]))), 1):
        components.append(
            DuplicateComponent(
                component_id=f"duplicate_component_{idx}",
                samples=tuple(sorted(samples)),
                pair_count=pair_counts.get(root, 0),
            )
        )

    return components


def components_to_dataframe(components: Iterable[DuplicateComponent]) -> pd.DataFrame:
    """Convert duplicate components to a readable dataframe."""

    rows = [
        {
            "component_id": component.component_id,
            "sample_count": component.sample_count,
            "pair_count": component.pair_count,
            "samples": ",".join(component.samples),
        }
        for component in components
    ]
    return pd.DataFrame(rows, columns=["component_id", "sample_count", "pair_count", "samples"])


def write_duplicate_qc_outputs(
    pair_df: pd.DataFrame,
    output_prefix: str,
    config: DuplicateQCConfig = DuplicateQCConfig(),
) -> Tuple[Path, Path]:
    """Write duplicate/MZ-like pair and component outputs."""

    duplicate_pairs = detect_duplicate_like_pairs(pair_df, config)
    components = duplicate_components(duplicate_pairs, config.id1_col, config.id2_col)

    pair_path = Path(f"{output_prefix}_duplicate_like_pairs.tsv")
    component_path = Path(f"{output_prefix}_duplicate_like_components.tsv")

    duplicate_pairs.to_csv(pair_path, sep="\t", index=False)
    components_to_dataframe(components).to_csv(component_path, sep="\t", index=False)

    return pair_path, component_path

