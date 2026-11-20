from pathlib import Path

import pandas as pd

from ponderosa.config import PonderosaConfig
from ponderosa.input_validation import validate_run_inputs
from ponderosa.qc import DuplicateQCConfig, detect_duplicate_like_pairs, duplicate_components
from ponderosa.reporting import write_run_report
from ponderosa.simulation.metadata import (
    SimulationTrainingMetadata,
    metadata_path_for_classifier,
    write_training_metadata,
)
from ponderosa.validation import (
    compare_predictions_to_king,
    load_king_kin0,
    validation_counts,
)


def test_input_validation_warns_for_sparse_pedigree(tmp_path: Path):
    fam = tmp_path / "samples.fam"
    ibd = tmp_path / "segments.ibd"
    fam.write_text("F1 I1 0 0 1 -9\nF2 I2 0 0 2 -9\n")
    ibd.write_text("id1 id2 start end is_full_ibd\n")

    config = PonderosaConfig.from_dict(
        {
            "files": {
                "fam": str(fam),
                "ibd": str(ibd),
                "ibd_caller": "phasedibd",
            }
        }
    )

    report = validate_run_inputs(config)

    assert report.ok
    assert report.fam_summary.sample_count == 2
    assert any(message.code == "sparse-pedigree" for message in report.warnings)


def test_duplicate_qc_detects_connected_component():
    pairs = pd.DataFrame(
        [
            {"id1": "A", "id2": "B", "ibd1": 0.0, "ibd2": 3400.0},
            {"id1": "B", "id2": "C", "ibd1": 0.0, "ibd2": 3400.0},
            {"id1": "A", "id2": "D", "ibd1": 1700.0, "ibd2": 900.0},
        ]
    )

    duplicate_pairs = detect_duplicate_like_pairs(
        pairs,
        DuplicateQCConfig(genome_length=3545.0, min_ibd2_fraction=0.90, min_total_ibd_fraction=0.90),
    )
    components = duplicate_components(duplicate_pairs)

    assert duplicate_pairs.shape[0] == 2
    assert len(components) == 1
    assert components[0].samples == ("A", "B", "C")
    assert components[0].pair_count == 2


def test_king_validation_compares_predictions(tmp_path: Path):
    king_file = tmp_path / "king.kin0"
    king_file.write_text(
        "FID1\tID1\tFID2\tID2\tKinship\tIBD1Seg\tIBD2Seg\tPropIBD\tInfType\n"
        "F1\tI1\tF2\tI2\t0.25\t0.5\t0.0\t0.25\tPO\n"
    )
    predictions = pd.DataFrame(
        [
            {"id1": "F1_I1", "id2": "F2_I2", "degree": "1st", "pred_rel": "PO", "prob": "0.99"},
            {"id1": "F3_I3", "id2": "F4_I4", "degree": "1st", "pred_rel": "FS", "prob": "0.90"},
        ]
    )

    king = load_king_kin0(king_file)
    joined = compare_predictions_to_king(predictions, king)
    counts = validation_counts(joined)

    assert list(joined["king_InfType"]) == ["PO", "missing"]
    assert set(counts["king_InfType"]) == {"PO", "missing"}


def test_reporting_and_simulation_metadata_write_files(tmp_path: Path):
    metadata = SimulationTrainingMetadata(
        output_prefix="demo",
        ibd_caller="hap-ibd",
        relationship_counts={"PO": 10, "FS": 12},
    )
    metadata_path = write_training_metadata(metadata, tmp_path / "demo.training_metadata.json")
    report_path = write_run_report(tmp_path / "run_report.md", metadata={"dataset": "synthetic"})

    assert metadata_path.exists()
    assert report_path.exists()
    assert metadata_path_for_classifier(tmp_path / "demo.classif.pkl").name == "demo.training_metadata.json"

