"""Regression tests for fixes made while implementing PONDEROSA V3 in PONDEROSA-IMPROVED."""

from pathlib import Path

import numpy as np
import pytest

from ponderosa.classifiers import DegreeClassifier
from ponderosa.config import PonderosaConfig
from ponderosa.data_loading import HapIBDLoader
from ponderosa.map_tools import GeneticMap
from ponderosa.simulation.founders import calculate_relatedness, relatedness_from_king
from ponderosa.simulation.pedsim import PedSim
from ponderosa.simulation.simulate import ibd_format_from_script


def test_hapibd_length_recomputed_from_map_and_filtered_in_cm(tmp_path):
    # Map: 0.5 cM per Mb on chr1, so a 20 Mb segment is 10 cM (hap-IBD's own column says 99)
    mapf = tmp_path / "m.map"
    mapf.write_text("".join(f"1 rs{i} {i * 0.5:.3f} {i * 1_000_000}\n" for i in range(0, 201)))
    ibd = tmp_path / "x.ibd"
    ibd.write_text("A\t1\tB\t2\t1\t10000000\t30000000\t99.0\n"     # 10 cM
                   "A\t2\tB\t1\t1\t50000000\t52000000\t99.0\n"     # 1 cM -> dropped by 3 cM filter
                   "A\t1\tB\t1\t1\t100000000\t180000000\t99.0\n")  # 40 cM
    loader = HapIBDLoader(min_segment_length=3, min_total_ibd=45)
    seg = loader.load_filtered_segments(str(ibd), genetic_map=GeneticMap.add_plink(mapf))
    assert sorted(seg["length_cm"].round(6).to_list()) == [10.0, 40.0]
    assert set(seg["id1_haplotype"].to_list()) == {0}             # 1/2 -> 0/1
    loader_strict = HapIBDLoader(min_segment_length=3, min_total_ibd=60)
    assert loader_strict.load_filtered_segments(str(ibd), genetic_map=GeneticMap.add_plink(mapf)).height == 0


def test_hapibd_requires_map(tmp_path):
    ibd = tmp_path / "x.ibd"
    ibd.write_text("A\t1\tB\t2\t1\t10000000\t30000000\t99.0\n")
    with pytest.raises(ValueError, match="genetic map"):
        HapIBDLoader(3, 0).load_filtered_segments(str(ibd))


def test_king_relatedness_uses_kinship_scale(tmp_path):
    kin0 = tmp_path / "k.kin0"
    kin0.write_text("FID1\tID1\tFID2\tID2\tKinship\tInfType\nF1\tA\tF2\tB\t0.25\tPO\n")
    seg = tmp_path / "k.seg"
    seg.write_text("FID1\tID1\tFID2\tID2\tPropIBD\tInfType\nF1\tA\tF2\tB\t0.5\tPO\n")
    r1, r2 = relatedness_from_king(kin0), relatedness_from_king(seg)
    assert r1[("A", "B")] == pytest.approx(0.25) == r2[("A", "B")]
    assert r1[("F1_A", "F2_B")] == pytest.approx(0.25)            # FID_IID spelling
    assert calculate_relatedness(king_file=kin0, sample_ids=["A", "C"]) == {}


def test_ibd_format_from_script_name():
    assert ibd_format_from_script("hap-ibd.sh") == "hap-ibd"
    assert ibd_format_from_script("ponderosa/simulation/call_hap-ibd.sh") == "hap-ibd"
    assert ibd_format_from_script("phasedibd.sh") == "phasedibd"
    with pytest.raises(ValueError):
        ibd_format_from_script("my_script.sh")


def test_pedsim_replicates_and_phase_flags(tmp_path):
    d = tmp_path / "pedigree.def"
    d.write_text("def full3gen 1 3\n1 1\n2 2 2\n3 1\n")
    ps = PedSim("ped-sim", tmp_path / "simulation", def_file=str(d), vcf_file="in.vcf", seed=7)
    ps.set_replicates(100)
    assert Path(ps.flags["-d"]).read_text().startswith("def full3gen 100 3")
    cmd = ps._get_command()
    assert "--keep_phase" in cmd and "--miss_rate 0" in cmd and "--seed 7" in cmd and "--fam" in cmd
    assert "--keep_phase" not in ps._get_command(dry_run=True)


def test_too_few_training_pairs_gives_clear_error():
    X = np.random.rand(20, 2)
    y = np.array(["PO"] * 17 + ["FS"] * 3)
    with pytest.raises(ValueError, match="too few for"):
        DegreeClassifier()._validate_training_data(X, y)


def test_cli_does_not_override_yaml_ibd_caller(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("files:\n  fam: a.fam\n  ibd: a.ibd\n  ibd_caller: hap-ibd\n  map_files: [m1.map, m2.map]\n"
                   "output:\n  write_training: true\n")
    c = PonderosaConfig.from_cli_and_yaml({"config": cfg, "output": "x", "write_training": False})
    assert c.files.ibd_caller == "hap-ibd"
    assert [p.name for p in c.files.map_files] == ["m1.map", "m2.map"]
    assert c.output.write_training is True
    c2 = PonderosaConfig.from_cli_and_yaml({"config": cfg, "ibd_caller": "phasedibd"})
    assert c2.files.ibd_caller == "phasedibd"
