"""
Simulated families with PONDEROSA-like pair posteriors, for testing and benchmarking
ranked family trees (ponderosa/dynamic_pedigree.py).

Families have three generations with spouses, occasional half-siblings and about 35%
unsampled members. For every related sampled pair a posterior is generated that
mimics PONDEROSA: the degree is usually right, PO vs FS is easy, and HS vs GPAV,
AV vs GP and paternal vs maternal are only weakly separated. Sexes are known; ages
are not supplied.
"""

from __future__ import annotations

import itertools as it
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ponderosa.dynamic_pedigree import PedigreeState, REL_DEGREE, pair_key

NODE_COLUMNS = ["1st", "FS", "PO", "2nd", "GPAV", "HS", "AV", "GP", "PHS", "MHS",
                "3rd", "HAV", "GGP", "CO", "PGP", "MGP", "4th", "HCO"]


@dataclass
class TrueFamily:
    family_index: int
    people: Dict[str, Tuple[Optional[str], Optional[str], int]]  # id -> (father, mother, sex)
    sampled: List[str]
    truth: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def state(self) -> PedigreeState:
        s = PedigreeState({p: sex for p, (_, _, sex) in self.people.items()})
        for p, (f, m, _) in self.people.items():
            s.par[p] = [f, m]
        return s


@dataclass
class NoiseModel:
    """How well PONDEROSA separates relationships (probabilities of the true branch)."""

    p_degree_right: float = 0.92       # degree usually right
    degree_conf: Tuple[float, float] = (0.70, 0.995)
    po_fs: Tuple[float, float] = (0.90, 0.995)          # easy
    hs_gpav: Tuple[float, float] = (0.45, 0.80)         # weak
    av_gp: Tuple[float, float] = (0.45, 0.75)           # weak
    pat_mat: Tuple[float, float] = (0.40, 0.70)         # weak
    p_observe_3rd: float = 0.97
    p_observe_4th: float = 0.70

    @classmethod
    def noisy(cls) -> "NoiseModel":
        return cls(p_degree_right=0.80, degree_conf=(0.55, 0.95))


def _split(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def simulate_family(idx: int, rng: np.random.Generator, p_unsampled: float = 0.35) -> TrueFamily:
    people: Dict[str, Tuple[Optional[str], Optional[str], int]] = {}
    counter = it.count(1)

    def add(f, m, sex=None):
        pid = f"f{idx}_{next(counter)}"
        people[pid] = (f, m, int(sex if sex else rng.integers(1, 3)))
        return pid

    gf, gm = add(None, None, 1), add(None, None, 2)
    gen2 = [add(gf, gm) for _ in range(int(rng.integers(2, 5)))]
    # occasional half-sibling through a grandparent's second partner
    if rng.random() < 0.25:
        shared = gf if rng.random() < 0.5 else gm
        other = add(None, None, 2 if shared == gf else 1)
        f, m = (shared, other) if shared == gf else (other, shared)
        gen2.append(add(f, m))
    for p in list(gen2):
        if rng.random() < 0.75:
            spouse = add(None, None, 3 - people[p][2])
            f, m = (p, spouse) if people[p][2] == 1 else (spouse, p)
            for _ in range(int(rng.integers(1, 4))):
                add(f, m)
            if rng.random() < 0.12:  # half-siblings in generation 3
                spouse2 = add(None, None, 3 - people[p][2])
                f2, m2 = (p, spouse2) if people[p][2] == 1 else (spouse2, p)
                add(f2, m2)

    ids = list(people)
    sampled = [p for p in ids if rng.random() >= p_unsampled]
    fam = TrueFamily(idx, people, sorted(sampled))
    st = fam.state()
    for a, b in it.combinations(sorted(sampled), 2):
        fam.truth[(a, b)] = st.relationship(a, b)
    return fam


def _posterior(rel: str, rng: np.random.Generator, noise: NoiseModel) -> Dict[str, float]:
    true_degree = REL_DEGREE[rel]
    degrees = ["1st", "2nd", "3rd", "4th"]
    tdi = degrees.index(true_degree) if true_degree in degrees else 3
    probs = np.zeros(4)
    if rng.random() < noise.p_degree_right:
        c = _split(rng, noise.degree_conf)
        probs[tdi] = c
        rest = [i for i in range(4) if i != tdi]
        w = rng.dirichlet(np.ones(3)) * (1 - c)
        for i, wi in zip(rest, w):
            probs[i] = wi
    else:
        wrong = int(np.clip(tdi + rng.choice([-1, 1]), 0, 3))
        if wrong == tdi:
            wrong = tdi + 1 if tdi < 3 else tdi - 1
        c = _split(rng, (0.5, 0.85))
        probs[wrong] = c
        probs[tdi] = (1 - c) * rng.uniform(0.5, 0.95)
        others = [i for i in range(4) if i not in (tdi, wrong)]
        left = 1 - probs.sum()
        for i, wi in zip(others, rng.dirichlet(np.ones(len(others))) * left):
            probs[i] = wi
    p1, p2, p3, p4 = probs
    out = {"1st": p1, "2nd": p2, "3rd": p3, "4th": p4}

    # 1st degree: PO vs FS (easy)
    q = _split(rng, noise.po_fs)
    po_share = q if rel == "PO" else (1 - q) if rel == "FS" else rng.uniform(0.3, 0.7)
    out["PO"], out["FS"] = p1 * po_share, p1 * (1 - po_share)

    # 2nd degree: HS vs GPAV (weak), AV vs GP (weak), paternal vs maternal (weak)
    q = _split(rng, noise.hs_gpav)
    hs_share = q if rel in ("PHS", "MHS") else (1 - q) if rel in ("AV", "PGP", "MGP") else rng.uniform(0.3, 0.7)
    out["HS"], out["GPAV"] = p2 * hs_share, p2 * (1 - hs_share)
    q = _split(rng, noise.av_gp)
    av_share = q if rel == "AV" else (1 - q) if rel in ("PGP", "MGP") else rng.uniform(0.3, 0.7)
    out["AV"], out["GP"] = out["GPAV"] * av_share, out["GPAV"] * (1 - av_share)
    q = _split(rng, noise.pat_mat)
    pat = q if rel in ("PHS", "PGP") else (1 - q) if rel in ("MHS", "MGP") else rng.uniform(0.3, 0.7)
    out["PHS"], out["MHS"] = out["HS"] * pat, out["HS"] * (1 - pat)
    q = _split(rng, noise.pat_mat)
    pat = q if rel == "PGP" else (1 - q) if rel == "MGP" else rng.uniform(0.3, 0.7)
    out["PGP"], out["MGP"] = out["GP"] * pat, out["GP"] * (1 - pat)
    out["HAV"] = out["GGP"] = out["CO"] = p3 / 3
    out["HCO"] = p4
    return out


def ponderosa_call(post: Dict[str, float], min_p: float = 0.5) -> Tuple[str, float, str]:
    """PONDEROSA's descent rule: move to the most probable child while P >= min_p."""
    tree = {"relatives": ["1st", "2nd", "3rd", "4th"], "1st": ["FS", "PO"], "2nd": ["GPAV", "HS"],
            "GPAV": ["AV", "GP"], "GP": ["PGP", "MGP"], "HS": ["PHS", "MHS"],
            "3rd": ["HAV", "GGP", "CO"], "4th": ["HCO"]}
    node, prob, degree = "relatives", 1.0, None
    while node in tree:
        child = max(tree[node], key=lambda c: post.get(c, 0))
        if post.get(child, 0) < min_p:
            break
        node, prob = child, post[child]
        if node in ("1st", "2nd", "3rd", "4th"):
            degree = node
    return node, prob, degree or node


def simulate_cohort(n_families: int, seed: int = 1, noise: Optional[NoiseModel] = None,
                    p_unsampled: float = 0.35) -> Tuple[pd.DataFrame, Dict[str, int], List[TrueFamily]]:
    rng = np.random.default_rng(seed)
    noise = noise or NoiseModel()
    fams, rows, sexes = [], [], {}
    for idx in range(1, n_families + 1):
        fam = simulate_family(idx, rng, p_unsampled)
        fams.append(fam)
        for p in fam.sampled:
            sexes[p] = fam.people[p][2]
        for (a, b), rel in fam.truth.items():
            deg = REL_DEGREE[rel]
            if deg == "none":
                continue
            if deg == "3rd" and rng.random() > noise.p_observe_3rd:
                continue
            if deg == "4th" and rng.random() > noise.p_observe_4th:
                continue
            post = _posterior(rel, rng, noise)
            call, prob, degree = ponderosa_call(post)
            row = {"id1": a, "id2": b, "degree": degree, "pred_rel": call, "prob": prob,
                   "known_rel": rel, "ibd1": 1000.0, "ibd2": 0.0}
            row.update({c: post.get(c, np.nan) for c in NODE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows), sexes, fams
