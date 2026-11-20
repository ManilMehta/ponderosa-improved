"""Metadata helpers for reproducible simulation-based training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SimulationTrainingMetadata:
    """Metadata describing how a simulation-trained classifier was produced."""

    output_prefix: str
    ibd_caller: str
    fam_file: Optional[str] = None
    ibd_file: Optional[str] = None
    map_file: Optional[str] = None
    pedsim_path: Optional[str] = None
    vcf_file: Optional[str] = None
    def_file: Optional[str] = None
    simmap_file: Optional[str] = None
    interference_file: Optional[str] = None
    random_seed: Optional[int] = None
    min_segment_length: Optional[float] = None
    min_total_ibd: Optional[float] = None
    relationship_counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def write_training_metadata(metadata: SimulationTrainingMetadata, output_path: Path) -> Path:
    """Write simulation-training metadata as pretty JSON."""

    output_path = Path(output_path)
    output_path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n")
    return output_path


def metadata_path_for_classifier(classifier_path: Path) -> Path:
    """Return the conventional metadata path for a classifier pickle."""

    classifier_path = Path(classifier_path)
    if classifier_path.name.endswith(".classif.pkl"):
        return classifier_path.with_name(classifier_path.name.replace(".classif.pkl", ".training_metadata.json"))
    return classifier_path.with_suffix(classifier_path.suffix + ".training_metadata.json")

