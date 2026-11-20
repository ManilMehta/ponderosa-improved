"""General input validation helpers for PONDEROSA V3 workflows.

The functions in this module are intentionally lightweight: they inspect files
and return structured messages instead of starting expensive IBD parsing or
training work. This lets users catch common setup problems before submitting
long jobs on local machines or HPC systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .config import PonderosaConfig


MISSING_PARENT_VALUES = {"0", "-9", "", "NA", "NaN", "nan", None}
SUPPORTED_IBD_CALLERS = {"phasedibd", "hap-ibd", "ibdseq"}
KING_KIN0_COLUMNS = {"FID1", "ID1", "FID2", "ID2", "Kinship", "InfType"}


@dataclass(frozen=True)
class ValidationMessage:
    """A single validation finding."""

    level: str
    code: str
    message: str
    path: Optional[Path] = None


@dataclass
class FamSummary:
    """Basic FAM metadata relevant to PONDEROSA readiness."""

    sample_count: int = 0
    family_count: int = 0
    paternal_links: int = 0
    maternal_links: int = 0
    duplicate_iids: List[str] = field(default_factory=list)

    @property
    def parent_link_count(self) -> int:
        return self.paternal_links + self.maternal_links

    @property
    def has_pedigree_labels(self) -> bool:
        return self.parent_link_count > 0


@dataclass
class InputValidationReport:
    """Structured result returned by input validation."""

    messages: List[ValidationMessage] = field(default_factory=list)
    fam_summary: Optional[FamSummary] = None
    ibd_files: List[Path] = field(default_factory=list)
    map_files: List[Path] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationMessage]:
        return [msg for msg in self.messages if msg.level == "error"]

    @property
    def warnings(self) -> List[ValidationMessage]:
        return [msg for msg in self.messages if msg.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, code: str, message: str, path: Optional[Path] = None) -> None:
        self.messages.append(ValidationMessage(level, code, message, path))

    def summary_lines(self) -> List[str]:
        lines = []
        if self.fam_summary:
            fam = self.fam_summary
            lines.append(f"FAM samples: {fam.sample_count}")
            lines.append(f"FAM families: {fam.family_count}")
            lines.append(f"FAM parent links: {fam.parent_link_count}")
        lines.append(f"IBD files: {len(self.ibd_files)}")
        lines.append(f"Map files: {len(self.map_files)}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        return lines


def _as_existing_path(path: Optional[Path], report: InputValidationReport, code: str, label: str, required: bool) -> Optional[Path]:
    if path is None:
        if required:
            report.add("error", code, f"Missing required {label}.")
        return None

    path = Path(path)
    if not path.exists():
        level = "error" if required else "warning"
        report.add(level, code, f"{label} does not exist: {path}", path)
        return None

    return path


def _existing_paths(paths: Optional[Sequence[Path]], report: InputValidationReport, code: str, label: str) -> List[Path]:
    out: List[Path] = []
    if not paths:
        return out

    for path in paths:
        existing = _as_existing_path(Path(path), report, code, label, required=True)
        if existing:
            out.append(existing)
    return out


def summarize_fam(fam_path: Path) -> FamSummary:
    """Read a PLINK FAM file and return basic pedigree metadata."""

    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, dtype=str)
    if fam.shape[1] < 6:
        raise ValueError(f"FAM file must have at least 6 columns: {fam_path}")

    fam = fam.iloc[:, :6]
    fam.columns = ["FID", "IID", "PAT", "MAT", "SEX", "PHENO"]

    paternal_links = int((~fam["PAT"].isin(MISSING_PARENT_VALUES)).sum())
    maternal_links = int((~fam["MAT"].isin(MISSING_PARENT_VALUES)).sum())

    duplicate_counts = fam["IID"].value_counts()
    duplicate_iids = sorted(duplicate_counts[duplicate_counts > 1].index.tolist())

    return FamSummary(
        sample_count=int(fam.shape[0]),
        family_count=int(fam["FID"].nunique()),
        paternal_links=paternal_links,
        maternal_links=maternal_links,
        duplicate_iids=duplicate_iids,
    )


def validate_king_kin0(path: Path) -> List[ValidationMessage]:
    """Validate that a KING kin0-like file has the columns needed for comparison."""

    messages: List[ValidationMessage] = []
    header = pd.read_csv(path, sep=r"\s+", nrows=0)
    missing = sorted(KING_KIN0_COLUMNS - set(header.columns))
    if missing:
        messages.append(
            ValidationMessage(
                "error",
                "king-missing-columns",
                f"KING file is missing required columns: {', '.join(missing)}",
                path,
            )
        )
    return messages


def validate_truth_file(path: Path) -> List[ValidationMessage]:
    """Validate a simple truth file with at least two IDs and one relationship column."""

    messages: List[ValidationMessage] = []
    header = pd.read_csv(path, sep=r"\s+", nrows=0)
    if len(header.columns) < 3:
        messages.append(
            ValidationMessage(
                "error",
                "truth-too-few-columns",
                "Truth file must have at least three columns: ID1, ID2, relationship.",
                path,
            )
        )
    return messages


def validate_run_inputs(config: "PonderosaConfig", king_file: Optional[Path] = None) -> InputValidationReport:
    """Validate generalized PONDEROSA V3 inputs without launching analysis."""

    report = InputValidationReport()
    files = config.files

    fam_path = _as_existing_path(files.fam, report, "missing-fam", "FAM file", required=True)
    if fam_path:
        try:
            report.fam_summary = summarize_fam(fam_path)
            if not report.fam_summary.has_pedigree_labels and not files.training:
                report.add(
                    "warning",
                    "sparse-pedigree",
                    "FAM has no parent links. Training from real labels is unlikely to work without a pretrained classifier or simulation.",
                    fam_path,
                )
            if report.fam_summary.duplicate_iids:
                report.add(
                    "warning",
                    "duplicate-iids",
                    f"FAM contains duplicate IID values: {', '.join(report.fam_summary.duplicate_iids[:10])}",
                    fam_path,
                )
        except Exception as exc:
            report.add("error", "invalid-fam", str(exc), fam_path)

    if files.ibd_caller not in SUPPORTED_IBD_CALLERS:
        report.add("error", "unsupported-ibd-caller", f"Unsupported IBD caller: {files.ibd_caller}")

    ibd_files = files._ibd_file_list or ([files.ibd] if files.ibd else files.ibd_files)
    map_files = files._map_file_list or ([files.mapf] if files.mapf else files.map_files)

    report.ibd_files = _existing_paths(ibd_files, report, "missing-ibd", "IBD file")
    report.map_files = _existing_paths(map_files, report, "missing-map", "genetic map")

    if files.ibd_caller == "hap-ibd" and not report.map_files:
        report.add("error", "hap-ibd-needs-map", "hap-IBD input requires a genetic map for bp-to-cM conversion.")

    for optional_path, code, label in [
        (files.training, "missing-training", "pretrained classifier file"),
        (files.truth, "missing-truth", "truth file"),
        (files.ages, "missing-ages", "ages file"),
        (files.populations, "missing-populations", "populations file"),
    ]:
        _as_existing_path(optional_path, report, code, label, required=False)

    if files.truth and Path(files.truth).exists():
        report.messages.extend(validate_truth_file(Path(files.truth)))

    if king_file:
        king_path = _as_existing_path(Path(king_file), report, "missing-king", "KING validation file", required=False)
        if king_path:
            report.messages.extend(validate_king_kin0(king_path))

    return report


def format_validation_report(report: InputValidationReport) -> str:
    """Return a human-readable validation summary."""

    lines = ["PONDEROSA V3 Input Validation", "=" * 30, ""]
    lines.extend(report.summary_lines())
    if report.messages:
        lines.extend(["", "Messages:"])
        for msg in report.messages:
            path = f" [{msg.path}]" if msg.path else ""
            lines.append(f"- {msg.level.upper()} {msg.code}: {msg.message}{path}")
    return "\n".join(lines)

