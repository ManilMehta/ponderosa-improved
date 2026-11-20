"""Tests for ranked family trees (ponderosa/dynamic_pedigree.py)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from ponderosa.dynamic_pedigree import (
    DISTANT, FATHER, MOTHER, Lock, PairEvidence, PedigreeSearchConfig, PedigreeState,
    build_families, collapse_duplicates, find_duplicates, main as pedigree_main,
    realise_label, reconstruct_pedigrees, search_family,
)
from tests.pedigree_sim import NODE_COLUMNS, ponderosa_call, simulate_cohort


def posterior(**leaves):
    """Build a full PONDEROSA node row from leaf probabilities."""
    row = {c: 0.0 for c in NODE_COLUMNS}
    row.update(leaves)
    row["HS"] = row["PHS"] + row["MHS"]
    row["GP"] = row["PGP"] + row["MGP"]
    row["GPAV"] = row["GP"] + row["AV"]
    row["1st"] = row["PO"] + row["FS"]
    row["2nd"] = row["GPAV"] + row["HS"]
    return row


def pairs_df(rows):
    out = []
    for a, b, post in rows:
        call, prob, degree = ponderosa_call(post)
        out.append({"id1": a, "id2": b, "pred_rel": call, "prob": prob, "degree": degree,
                    "ibd1": 1500.0, "ibd2": 0.0, **post})
    return pd.DataFrame(out)


@pytest.fixture
def cfg():
    return PedigreeSearchConfig()


@pytest.fixture
def nuclear():
    """Father F, mother M, children A and B (all sampled)."""
    rows = [
        ("F", "A", posterior(PO=0.95, FS=0.03, AV=0.01, **{"3rd": 0.01})),
        ("F", "B", posterior(PO=0.94, FS=0.04, AV=0.01, **{"3rd": 0.01})),
        ("A", "M", posterior(PO=0.96, FS=0.02, AV=0.01, **{"3rd": 0.01})),
        ("B", "M", posterior(PO=0.93, FS=0.05, AV=0.01, **{"3rd": 0.01})),
        ("A", "B", posterior(FS=0.90, PO=0.06, PHS=0.02, **{"3rd": 0.02})),
    ]
    return pairs_df(rows), {"F": 1, "M": 2, "A": 1, "B": 2}


# 1
def test_lock_parsing():
    lk = Lock.parse("F003:ID1:ID2:av")
    assert (lk.family_id, lk.id1, lk.id2, lk.label) == ("F003", "ID1", "ID2", "AV")
    assert Lock.parse("ID2:ID1:distant").key == ("ID1", "ID2")
    with pytest.raises(ValueError):
        Lock.parse("F003:ID1:ID2:XX")


# 2
def test_nan_leaf_takes_equal_share_of_parent(cfg):
    post = posterior(PGP=0.2, MGP=0.2, AV=0.3, PHS=0.1, MHS=0.1)
    post["PGP"] = post["MGP"] = np.nan          # GP stays 0.4 -> each leaf 0.2
    post["AV"] = post["GP"] = np.nan            # GPAV 0.7 -> AV 0.35, GP 0.35 -> PGP 0.175
    ev = PairEvidence(pairs_df([("a", "b", post)]), cfg)
    p = ev.probs(("a", "b"))
    assert p["AV"] == pytest.approx(0.35)
    assert p["PGP"] == pytest.approx(0.175) and p["MGP"] == pytest.approx(0.175)


# 3
def test_third_degree_and_none_probabilities(cfg):
    post = posterior(PO=0.1, AV=0.2, **{"3rd": 0.6})
    ev = PairEvidence(pairs_df([("a", "b", post)]), cfg)
    p = ev.probs(("a", "b"))
    assert p["CO"] == p["HAV"] == p["GGP"] == pytest.approx(0.6)
    assert p["none"] == p["HCO"] == pytest.approx(1 - 0.1 - 0.2)
    assert ev.logp(("a", "b"), "FS") == pytest.approx(math.log(cfg.prob_floor))


# 4
def test_label_options_top4_plus_distant(cfg):
    post = posterior(PO=0.0005, PHS=0.3, MHS=0.25, AV=0.2, PGP=0.15, MGP=0.0995)
    ev = PairEvidence(pairs_df([("a", "b", post)]), cfg)
    labels = [lab for lab, _ in ev.label_options(("a", "b"))]
    assert labels == ["PHS", "MHS", "AV", "PGP", DISTANT]


# 5
def test_union_find_never_merges_sampled_and_rejects_sex_conflict():
    s = PedigreeState({"A": 1, "B": 1, "C": 0})
    assert not s.copy().merge("A", "B")                       # two sampled people
    t = s.copy()
    assert t.set_parent("C", MOTHER, "A") is False            # A is male, cannot be a mother
    u = s.copy()
    assert u.set_parent("C", FATHER, "A") and u.sex["A"] == 1


# 6
def test_merge_carries_parents_recursively():
    s = PedigreeState({"A": 1, "B": 2, "C": 1, "G": 1})
    s.share_parent("A", "B", FATHER)                 # A, B share latent father L1
    s.set_parent(s.parent("A", FATHER), FATHER, "G")  # G is father of L1
    assert s.set_parent("C", FATHER, "A")
    # Now make B's father the sampled C's grandfather's son: merge L1 with a new latent that has a mother
    lat = s.new_latent(1)
    s.par[lat][MOTHER] = s.new_latent(2)
    assert s.merge(lat, s.parent("B", FATHER))
    f = s.parent("A", FATHER)
    assert s.parent(f, FATHER) == "G" and s.parent(f, MOTHER) is not None
    assert s.relationship("A", "B") == "PHS"
    assert s.relationship("G", "A") == "PGP" and s.relationship("G", "C") == "GGP"


# 7
def test_implied_relationship_types():
    people = {  # id: (father, mother, sex)
        "g1": (None, None, 1), "g2": (None, None, 2),
        "p1": ("g1", "g2", 1), "p2": ("g1", "g2", 2), "s1": (None, None, 2), "s2": (None, None, 1),
        "c1": ("p1", "s1", 1), "c2": ("s2", "p2", 2), "h1": ("g1", None, 1), "hc": ("h1", None, 2),
    }
    s = PedigreeState({k: v[2] for k, v in people.items()})
    for k, (f, m, _) in people.items():
        s.par[k] = [f, m]
    rel = s.relationship
    assert rel("p1", "c1") == "PO" and rel("p1", "p2") == "FS"
    assert rel("g1", "c1") == "PGP" and rel("g1", "c2") == "MGP"
    assert rel("p2", "c1") == "AV" and rel("c1", "c2") == "CO"
    assert rel("p1", "h1") == "PHS" and rel("h1", "c1") == "HAV"
    assert rel("hc", "c1") == "HCO" and rel("s1", "s2") == "none"


# 8
def test_cycle_and_half_sib_inequality_are_rejected():
    s = PedigreeState({"A": 1, "B": 1})
    s.set_parent("B", FATHER, "A")
    t = s.copy()
    t.set_parent("A", FATHER, "B")
    assert t.has_cycle() and not s.has_cycle()

    u = PedigreeState({"A": 0, "B": 0, "M": 2})
    (hs,) = realise_label(u, "A", "B", "PHS")
    assert hs.set_parent("A", MOTHER, "M") and hs.set_parent("B", MOTHER, "M")
    assert hs.violates_inequality()


# 9
def test_age_windows(cfg):
    s = PedigreeState({"P": 1, "C": 1, "G": 2})
    s.set_parent("C", FATHER, "P")
    assert not s.violates_ages({"P": 40, "C": 10}, cfg)
    assert s.violates_ages({"P": 20, "C": 10}, cfg)          # 10-year gap < 12
    s.set_parent("P", MOTHER, "G")
    assert s.violates_ages({"P": 40, "C": 10, "G": 140}, cfg)  # GP gap 130 > 120


# 10
def test_realisation_branch_counts():
    unknown = PedigreeState({"A": 0, "B": 0})
    assert len(realise_label(unknown, "A", "B", "PO")) == 4    # direction x slot
    assert len(realise_label(unknown, "A", "B", "FS")) == 1
    assert len(realise_label(unknown, "A", "B", "PGP")) == 4   # direction x slot
    assert len(realise_label(unknown, "A", "B", "AV")) == 4    # direction x which parent
    known = PedigreeState({"A": 2, "B": 1})
    assert len(realise_label(known, "A", "B", "PO")) == 2
    assert len(realise_label(known, "A", "B", DISTANT)) == 1


# 11
def test_families_and_size_limit(cfg):
    rows = [("a", "b", posterior(PO=0.9, **{"3rd": 0.1})), ("b", "c", posterior(PHS=0.3, AV=0.3, **{"3rd": 0.4})),
            ("d", "e", posterior(**{"3rd": 0.9})), ("x", "y", posterior(FS=0.8, **{"3rd": 0.2}))]
    ev = PairEvidence(pairs_df(rows), cfg)
    assert build_families(ev, cfg) == [["a", "b", "c"], ["x", "y"]]
    small = PedigreeSearchConfig(max_family_size=2)
    results, _, _ = reconstruct_pedigrees(pairs_df(rows), small)
    assert [r.searched for r in results] == [False, True]
    assert "max_family_size" in results[0].reason


# 12
def test_duplicates_are_collapsed(cfg):
    rows = [("A", "A2", posterior(FS=1.0)), ("A", "B", posterior(PO=0.9, **{"3rd": 0.1})),
            ("A2", "B", posterior(PO=0.8, **{"3rd": 0.2}))]
    df = pairs_df(rows)
    df.loc[0, ["ibd1", "ibd2"]] = [10.0, 3400.0]
    groups = find_duplicates(df, cfg)
    assert len(groups) == 1 and groups[0].members == ["A", "A2"]
    collapsed = collapse_duplicates(df, groups)
    assert len(collapsed) == 1 and set(collapsed.iloc[0][["id1", "id2"]]) == {"A", "B"}


# 13
def test_nuclear_family_ranked_first(cfg, nuclear):
    df, sexes = nuclear
    results, _, _ = reconstruct_pedigrees(df, cfg, sexes=sexes)
    best = results[0].trees[0]
    assert best.implied[("A", "F")] == "PO" and best.implied[("A", "M")] == "PO"
    assert best.implied[("A", "B")] == "FS" and best.implied[("F", "M")] == "none"
    assert best.state.n_latent() == 0
    assert best.relative_probability > 0.5
    assert sum(t.relative_probability for t in results[0].trees) == pytest.approx(1.0)


# 14
def test_family_context_resolves_gpav_call(cfg):
    """A pairwise call stuck at GPAV is resolved to AV by the rest of the family."""
    gpav = posterior(AV=0.30, PGP=0.15, MGP=0.14, PHS=0.2, MHS=0.2, **{"3rd": 0.01})
    rows = [("A", "U", posterior(FS=0.92, PO=0.05, **{"3rd": 0.03})),   # aunt U is A's sister
            ("A", "N", posterior(PO=0.95, FS=0.03, **{"3rd": 0.02})),   # A is N's parent
            ("N", "U", gpav)]
    df = pairs_df(rows)
    assert df.loc[2, "pred_rel"] in ("2nd", "GPAV", "HS")
    sexes = {"A": 2, "U": 2, "N": 1}
    results, _, _ = reconstruct_pedigrees(df, cfg, sexes=sexes, ages={"A": 40, "U": 38, "N": 12})
    assert results[0].trees[0].implied[("N", "U")] == "AV"


# 15
def test_lock_restricts_and_filters(cfg, nuclear):
    df, sexes = nuclear
    lock = Lock.parse("F001:A:B:PHS")
    results, _, _ = reconstruct_pedigrees(df, cfg, sexes=sexes, locks=[lock])
    assert results[0].trees, "a tree implying the locked label must exist"
    assert all(t.implied[("A", "B")] == "PHS" for t in results[0].trees)


# 16
def test_cli_outputs_and_viewer(tmp_path):
    df, sexes, _ = simulate_cohort(4, seed=3)
    pairs = tmp_path / "sim_pairs.txt"
    df.to_csv(pairs, sep="\t", index=False)
    fam = tmp_path / "sim.fam"
    fam.write_text("".join(f"F {i} 0 0 {s} -9\n" for i, s in sexes.items()))
    prefix = tmp_path / "out"
    assert pedigree_main(["--pairs", str(pairs), "--fam", str(fam), "--output", str(prefix),
                          "--beam-width", "32"]) == 0
    data = json.loads((tmp_path / "out_pedigrees.json").read_text())
    assert data["families"] and data["families"][0]["trees"][0]["rank"] == 1
    summary = pd.read_csv(tmp_path / "out_pedigree_summary.tsv", sep="\t")
    assert {"family_id", "rank", "relative_probability"} <= set(summary.columns)
    html = (tmp_path / "out_pedigrees.html").read_text()
    assert "/*__PEDIGREE_DATA__*/null" not in html and '"family_id": "F001"' in html
