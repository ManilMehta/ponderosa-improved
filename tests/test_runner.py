"""Tests for the one-command runner (`ponderosa run` / `ponderosa status`)."""

import shutil
from pathlib import Path

import pytest

from ponderosa.runner import RunSetupError, cmd_run, detect_ibd_caller, discover_inputs, job_script, write_run_config

DEMO = Path(__file__).resolve().parents[1] / "demo_run"


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "COHORT"
    d.mkdir()
    shutil.copy(DEMO / "demo_hapibd.txt", d / "cohort.ibd")
    for f in ("demo.fam", "demo.map"):
        shutil.copy(DEMO / f, d / f)
    (d / "cohort.hbd").write_text("")                       # hap-IBD HBD output must be ignored
    (d / "ponderosa.yaml").write_text("min_total_ibd: 20\ngenome_length: 320\n")
    return d


def test_discovers_inputs_and_caller(folder):
    inp = discover_inputs(folder)
    assert Path(inp["ibd"]).name == "cohort.ibd" and inp["ibd_caller"] == "hap-ibd"
    assert Path(inp["fam"]).name == "demo.fam" and Path(inp["map"]).name == "demo.map"
    assert inp["training"] is None and inp["king"] is None
    assert inp["min_total_ibd"] == 20 and inp["name"] == "COHORT"
    assert detect_ibd_caller(DEMO / "demo_phasedibd.txt") == "phasedibd"


def test_ambiguous_inputs_need_yaml(folder):
    shutil.copy(folder / "cohort.ibd", folder / "cohort_fixed.ibd")
    with pytest.raises(RunSetupError, match="Several IBD file"):
        discover_inputs(folder)
    (folder / "ponderosa.yaml").write_text("ibd: cohort_fixed.ibd\n")
    assert Path(discover_inputs(folder)["ibd"]).name == "cohort_fixed.ibd"


def test_per_chromosome_files_become_lists(tmp_path):
    d = tmp_path / "C"
    d.mkdir()
    for c in (10, 2, 1):
        shutil.copy(DEMO / "demo_hapibd.txt", d / f"x.chr{c}.ibd.gz.txt")
        (d / f"x.chr{c}.ibd").write_text((DEMO / "demo_hapibd.txt").read_text())
    shutil.copy(DEMO / "demo.fam", d / "a.fam")
    shutil.copy(DEMO / "demo.map", d / "a.map")
    inp = discover_inputs(d)
    assert [Path(p).name for p in inp["ibd"]] == ["x.chr1.ibd", "x.chr2.ibd", "x.chr10.ibd"]
    cfg = write_run_config(inp, d / "outputs" / "C")
    assert "ibd_files" in cfg.read_text()


def test_job_script_has_all_steps(folder):
    inp = discover_inputs(folder)
    inp["slurm"]["account"] = "bmhenngrp"
    (folder / "kin.kin0").write_text("")
    inp["king"] = str(folder / "kin.kin0")
    cfg = write_run_config(inp, folder / "outputs" / "COHORT")
    s = job_script(inp, cfg, folder / "outputs" / "COHORT")
    for needle in ("#SBATCH --account=bmhenngrp", "#SBATCH --partition=high", "-m ponderosa --config",
                   "qc-duplicates", "pedigree --pairs", "king-validate", "--close-only"):
        assert needle in s


def test_dry_run_writes_script(folder, capsys):
    assert cmd_run([str(folder), "--dry-run"]) == 0
    assert (folder / "ponderosa_job.sbatch").exists()
    assert "Errors: 0" in capsys.readouterr().out
