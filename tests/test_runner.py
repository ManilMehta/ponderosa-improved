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


def test_macaque_preset(tmp_path):
    from ponderosa.config import PonderosaConfig
    from ponderosa.species import get_species
    d = tmp_path / "MAC"
    d.mkdir()
    # 20 per-chromosome maps named like a rhesus map set; IBD only on chr1-2
    lines = (DEMO / "demo.map").read_text().splitlines()
    for c in range(1, 21):
        src = [l for l in lines if l.split()[0] == str(c)] or [f"{c} r{c}a 0.0 1000", f"{c} r{c}b 100.0 100000000"]
        (d / f"rheMac10.chr{c}.map").write_text("\n".join(src) + "\n")
    shutil.copy(DEMO / "demo_hapibd.txt", d / "mac.ibd")
    shutil.copy(DEMO / "demo.fam", d / "mac.fam")
    (d / "ponderosa.yaml").write_text("species: macaque\nmin_total_ibd: 20\n")
    inp = discover_inputs(d)
    assert inp["species"] == "macaque" and len(inp["map"]) == 20
    assert inp["genome_length"] > 0 and inp["genome_length"] != 3545
    assert inp["priors"].endswith("priors_macaque.yaml")
    cfg = PonderosaConfig.from_yaml(write_run_config(inp, d / "outputs" / "MAC"))
    cfg.validate()
    assert cfg.files.n_autosomes == 20 and cfg.files.priors.name == "priors_macaque.yaml"
    assert get_species("macaque").n_autosomes == 20
    # a "chr1" map name expands to chr1..chr20 for macaque
    single = PonderosaConfig.from_dict({"files": {"ibd": str(d / "mac.ibd"), "fam": str(d / "mac.fam"),
                                                  "ibd_caller": "hap-ibd", "mapf": str(d / "rheMac10.chr1.map")},
                                        "algorithm": {"species": "macaque"}})
    single.validate()
    assert len(single.files._map_file_list) == 20
