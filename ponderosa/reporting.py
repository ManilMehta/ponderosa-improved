"""Readable reporting utilities for PONDEROSA V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .input_validation import InputValidationReport
from .qc import DuplicateComponent


@dataclass
class ReportSection:
    """A Markdown report section."""

    title: str
    lines: List[str] = field(default_factory=list)

    def render(self) -> str:
        body = "\n".join(self.lines).strip()
        return f"## {self.title}\n\n{body}\n" if body else f"## {self.title}\n"


@dataclass
class PonderosaRunReport:
    """A lightweight Markdown report assembled from V3 workflow summaries."""

    title: str = "PONDEROSA V3 Run Report"
    metadata: Dict[str, object] = field(default_factory=dict)
    sections: List[ReportSection] = field(default_factory=list)

    def add_section(self, title: str, lines: Iterable[str]) -> None:
        self.sections.append(ReportSection(title, list(lines)))

    def render(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.metadata:
            lines.extend(["## Run Metadata", ""])
            for key, value in self.metadata.items():
                lines.append(f"- `{key}`: {value}")
            lines.append("")

        for section in self.sections:
            lines.append(section.render())

        return "\n".join(lines).rstrip() + "\n"

    def write(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.write_text(self.render())
        return output_path


def validation_section(report: InputValidationReport) -> ReportSection:
    """Create a report section from an input validation report."""

    lines = report.summary_lines()
    if report.messages:
        lines.extend(["", "Messages:"])
        for message in report.messages:
            path = f" `{message.path}`" if message.path else ""
            lines.append(f"- **{message.level.upper()} `{message.code}`**: {message.message}{path}")
    return ReportSection("Input Validation", lines)


def duplicate_qc_section(components: Iterable[DuplicateComponent]) -> ReportSection:
    """Create a report section for duplicate/MZ-like components."""

    components = list(components)
    if not components:
        return ReportSection("Duplicate/MZ QC", ["No duplicate/MZ-like components were detected."])

    lines = [f"Detected {len(components)} duplicate/MZ-like component(s).", ""]
    for component in components:
        lines.append(
            f"- `{component.component_id}`: {component.sample_count} samples, "
            f"{component.pair_count} duplicate-like pairs"
        )
        lines.append(f"  Samples: `{', '.join(component.samples)}`")
    return ReportSection("Duplicate/MZ QC", lines)


def dataframe_counts_section(title: str, df: pd.DataFrame, max_rows: int = 25) -> ReportSection:
    """Render a compact dataframe summary without requiring Markdown table support."""

    if df.empty:
        return ReportSection(title, ["No rows to report."])

    lines = [f"Rows: {len(df)}", ""]
    for idx, row in df.head(max_rows).iterrows():
        parts = [f"{col}={row[col]}" for col in df.columns]
        lines.append(f"- {', '.join(parts)}")

    if len(df) > max_rows:
        lines.append(f"- ... {len(df) - max_rows} more row(s)")

    return ReportSection(title, lines)


def write_run_report(
    output_path: Path,
    metadata: Optional[Dict[str, object]] = None,
    input_report: Optional[InputValidationReport] = None,
    duplicate_components: Optional[Iterable[DuplicateComponent]] = None,
    validation_counts: Optional[pd.DataFrame] = None,
) -> Path:
    """Write a generalized PONDEROSA V3 Markdown report."""

    report = PonderosaRunReport(metadata=metadata or {})

    if input_report is not None:
        report.sections.append(validation_section(input_report))

    if duplicate_components is not None:
        report.sections.append(duplicate_qc_section(duplicate_components))

    if validation_counts is not None:
        report.sections.append(dataframe_counts_section("External Validation Counts", validation_counts))

    return report.write(Path(output_path))

