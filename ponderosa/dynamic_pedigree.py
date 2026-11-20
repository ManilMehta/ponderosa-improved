"""
Joint pedigree reconstruction ("ranked family trees") for PONDEROSA V3.

PONDEROSA scores every pair independently. Inside a family those calls constrain
one another (if A is the parent of both B and C then B and C are siblings), and a
pairwise call can stall at an internal hierarchy node such as GPAV. This module
reconstructs whole pedigrees from the pair output (``{prefix}_pairs.txt``) and
ranks them.

A *hypothesis* is a concrete pedigree over a family's sampled individuals plus any
unsampled (latent) parents needed to connect them. Each hypothesis T implies a
relationship r_T(a, b) for every pair of sampled members and is scored by a
composite log-likelihood::

    log L(T) =   sum_{observed pairs}   log max(P_PONDEROSA(r_T(a,b)), prob_floor)
               + sum_{unobserved pairs} log absent_penalty   if r_T(a,b) is 1st/2nd degree
               + sum_{couples (f,m)}    log pi(r_T(f,m)),    pi(1st)=1e-3, pi(2nd)=0.05, else 1

The search is a beam search over pairs ordered by their most confident label.
Partial pedigrees are scored by the likelihood of the relationships they actually
imply (not by the labels applied so far).

Usage::

    python -m ponderosa pedigree --pairs out_pairs.txt --fam cohort.fam \\
        --ages ages.txt --output out [--lock F003:ID1:ID2:AV]

Outputs ``{output}_pedigrees.json``, ``{output}_pedigree_summary.tsv`` and the
self-contained viewer ``{output}_pedigrees.html``.
"""

from __future__ import annotations

import argparse
import itertools as it
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).parent
DEFAULT_PRIORS = _PACKAGE_DIR / "config" / "priors.yaml"
DEFAULT_CODES = _PACKAGE_DIR / "config" / "pedigree_codes.yaml"
VIEWER_TEMPLATE = _PACKAGE_DIR / "templates" / "pedigree_viewer.html"

# Relationship labels that impose pedigree structure.
STRUCTURAL_LABELS = ("PO", "FS", "PHS", "MHS", "PGP", "MGP", "AV")
DISTANT = "distant"

# Every relationship a pedigree can imply, with its degree.
REL_DEGREE = {
    "PO": "1st", "FS": "1st",
    "PHS": "2nd", "MHS": "2nd", "PGP": "2nd", "MGP": "2nd", "AV": "2nd",
    "GGP": "3rd", "HAV": "3rd", "CO": "3rd", "GAV": "3rd",
    "HCO": "4th",
    "none": "none",
}

CLOSE_RELS = frozenset(r for r, d in REL_DEGREE.items() if d in ("1st", "2nd"))

FATHER, MOTHER = 0, 1
SLOT_SEX = {FATHER: 1, MOTHER: 2}
SLOT_NAME = {FATHER: "F", MOTHER: "M"}


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

@dataclass
class PedigreeSearchConfig:
    """Parameters of the ranked-tree search (defaults from the V3 appendix)."""

    min_close_prob: float = 0.5        # P(1st)+P(2nd) to join a family
    max_family_size: int = 40          # larger families are listed, not searched
    beam_width: int = 128
    max_labels_per_pair: int = 4
    min_label_prob: float = 1e-3
    prob_floor: float = 1e-6
    absent_pair_penalty: float = 1e-4
    mating_prior_1st: float = 1e-3
    mating_prior_2nd: float = 0.05
    export_n: int = 10
    top_k: int = 3
    genome_length: float = 3545.0
    dup_min_ibd2_frac: float = 0.90
    dup_max_ibd1_frac: float = 0.05
    dup_min_total_frac: float = 0.90
    po_age_window: Tuple[float, float] = (12.0, 70.0)
    gp_age_window: Tuple[float, float] = (25.0, 120.0)

    @classmethod
    def with_age_windows_from(cls, priors_yaml: Optional[Path] = None, **kwargs) -> "PedigreeSearchConfig":
        """Read the PO and GP age windows from config/priors.yaml."""
        cfg = cls(**kwargs)
        path = Path(priors_yaml) if priors_yaml else DEFAULT_PRIORS
        try:
            with open(path) as f:
                priors = yaml.safe_load(f).get("age_priors", {})
            if "PO" in priors:
                cfg.po_age_window = (float(priors["PO"].get("min_gap", -np.inf)),
                                     float(priors["PO"].get("max_gap", np.inf)))
            if "GP" in priors:
                cfg.gp_age_window = (float(priors["GP"].get("min_gap", -np.inf)),
                                     float(priors["GP"].get("max_gap", np.inf)))
        except FileNotFoundError:
            logger.warning(f"Priors file not found ({path}); using default age windows")
        return cfg


@dataclass(frozen=True)
class Lock:
    """Restrict one pair to one label (``F003:ID1:ID2:AV``)."""

    family_id: Optional[str]
    id1: str
    id2: str
    label: str

    @classmethod
    def parse(cls, text: str) -> "Lock":
        parts = text.split(":")
        if len(parts) == 4:
            fam, a, b, label = parts
        elif len(parts) == 3:
            fam, (a, b, label) = None, parts
        else:
            raise ValueError(f"Lock must look like FAMILY:ID1:ID2:LABEL or ID1:ID2:LABEL, got '{text}'")
        label = DISTANT if label.lower() in {"distant", "none", "unrelated"} else label.upper()
        if label not in STRUCTURAL_LABELS and label != DISTANT:
            raise ValueError(f"Lock label must be one of {', '.join(STRUCTURAL_LABELS)} or distant, got '{label}'")
        return cls(fam or None, a, b, label)

    @property
    def key(self) -> Tuple[str, str]:
        return pair_key(self.id1, self.id2)


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def lock_satisfied(label: str, implied: str) -> bool:
    if label == DISTANT:
        return REL_DEGREE[implied] not in {"1st", "2nd"}
    return implied == label


# ----------------------------------------------------------------------------
# Pair evidence: P_PONDEROSA(r)
# ----------------------------------------------------------------------------

def _load_hierarchy_parents(codes_yaml: Path = DEFAULT_CODES) -> Dict[str, str]:
    """child -> parent node for the PONDEROSA relationship hierarchy."""
    with open(codes_yaml) as f:
        codes = yaml.safe_load(f)
    parents = {}
    for name, info in codes.items():
        if not isinstance(info, dict):
            continue
        parent = info.get("parent", info.get("degree"))
        if parent and parent != name:
            parents[name] = parent
    for d in ("1st", "2nd", "3rd", "4th"):
        parents.pop(d, None)
    return parents


class PairEvidence:
    """
    Per-pair PONDEROSA posteriors and the log-probability of any implied relationship.

    * Leaves are read from the node columns of ``{prefix}_pairs.txt``; if a leaf is
      NaN it takes an equal share of its parent node's posterior, recursively.
    * 3rd-degree relationships (GGP, HAV, CO, great-avuncular) use P(3rd).
    * "none" and HCO use 1 - P(1st) - P(2nd).
    """

    def __init__(self, df: pd.DataFrame, config: PedigreeSearchConfig,
                 hierarchy_parents: Optional[Dict[str, str]] = None):
        self.config = config
        self.parents = hierarchy_parents or _load_hierarchy_parents()
        self.children: Dict[str, List[str]] = {}
        for child, parent in self.parents.items():
            self.children.setdefault(parent, []).append(child)

        self.df = df.reset_index(drop=True)
        self.index: Dict[Tuple[str, str], int] = {}
        for i, (a, b) in enumerate(zip(self.df["id1"].astype(str), self.df["id2"].astype(str))):
            self.index[pair_key(a, b)] = i

        self._prob_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._logp_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    def __contains__(self, key: Tuple[str, str]) -> bool:
        return key in self.index

    def row(self, key: Tuple[str, str]) -> pd.Series:
        return self.df.iloc[self.index[key]]

    def _node(self, row: pd.Series, node: str, depth: int = 0) -> float:
        val = row.get(node, np.nan)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = np.nan
        if not np.isnan(val):
            return val
        parent = self.parents.get(node)
        if parent is None or depth > 10:
            return np.nan
        n_sib = max(1, len(self.children.get(parent, [])))
        p_val = self._node(row, parent, depth + 1)
        return p_val / n_sib if not np.isnan(p_val) else np.nan

    def probs(self, key: Tuple[str, str]) -> Dict[str, float]:
        """P_PONDEROSA(r) for every relationship a pedigree can imply."""
        if key in self._prob_cache:
            return self._prob_cache[key]
        row = self.row(key)
        z = lambda v: 0.0 if np.isnan(v) else float(v)
        p1 = z(self._node(row, "1st"))
        p2 = z(self._node(row, "2nd"))
        p3 = z(self._node(row, "3rd"))
        p_none = min(1.0, max(0.0, 1.0 - p1 - p2))
        out = {lab: z(self._node(row, lab)) for lab in STRUCTURAL_LABELS}
        for lab in ("GGP", "HAV", "CO", "GAV"):
            out[lab] = p3
        out["HCO"] = p_none
        out["none"] = p_none
        out[DISTANT] = p_none
        out["_close"] = p1 + p2
        self._prob_cache[key] = out
        return out

    def logp(self, key: Tuple[str, str], rel: str) -> float:
        cache = self._logp_cache.get(key)
        if cache is None:
            probs = self.probs(key)
            floor = self.config.prob_floor
            cache = {r: math.log(max(probs[r], floor)) for r in REL_DEGREE}
            self._logp_cache[key] = cache
        return cache[rel]

    def label_options(self, key: Tuple[str, str]) -> List[Tuple[str, float]]:
        """Top labels with P >= min_label_prob (at most max_labels_per_pair), plus distant."""
        probs = self.probs(key)
        ranked = sorted(((lab, probs[lab]) for lab in STRUCTURAL_LABELS), key=lambda x: (-x[1], x[0]))
        opts = [(lab, p) for lab, p in ranked if p >= self.config.min_label_prob][: self.config.max_labels_per_pair]
        opts.append((DISTANT, probs[DISTANT]))
        return opts

    def confidence(self, key: Tuple[str, str]) -> float:
        """Probability of the pair's most confident label."""
        return max(p for _, p in self.label_options(key))

    def ponderosa_call(self, key: Tuple[str, str]) -> Tuple[Optional[str], Optional[float]]:
        row = self.row(key)
        call = row.get("pred_rel")
        prob = row.get("prob")
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            prob = None
        return (None if pd.isna(call) else str(call)), prob


# ----------------------------------------------------------------------------
# Pedigree hypotheses
# ----------------------------------------------------------------------------

class PedigreeState:
    """
    A concrete pedigree: nodes with a sex (1, 2 or 0 = unknown) and two parent slots.

    Latent (unsampled) nodes are created on demand. A union-find structure merges
    nodes that constraints force to be the same person: two sampled individuals can
    never merge, a merge carries both nodes' parents (recursively unifying them) and
    a sex conflict is rejected.
    """

    __slots__ = ("sex", "par", "sampled", "uf", "ineq", "next_latent", "placed", "log_l", "signature",
                 "_rel_cache")

    def __init__(self, sampled: Dict[str, int]):
        self.sex: Dict[str, int] = dict(sampled)
        self.par: Dict[str, List[Optional[str]]] = {n: [None, None] for n in sampled}
        self.sampled: frozenset = frozenset(sampled)
        self.uf: Dict[str, str] = {}
        # inequalities: tuples of refs; a ref is (node, slot) or (node, None)
        self.ineq: List[Tuple[Tuple[str, Optional[int]], Tuple[str, Optional[int]]]] = []
        self.next_latent = 1
        self.placed: frozenset = frozenset()
        self.log_l: float = 0.0
        self.signature = None
        self._rel_cache = None

    # -- bookkeeping ---------------------------------------------------------
    def copy(self) -> "PedigreeState":
        new = PedigreeState.__new__(PedigreeState)
        new.sex = dict(self.sex)
        new.par = {k: list(v) for k, v in self.par.items()}
        new.sampled = self.sampled
        new.uf = dict(self.uf)
        new.ineq = list(self.ineq)
        new.next_latent = self.next_latent
        new.placed = self.placed
        new.log_l = self.log_l
        new.signature = None
        new._rel_cache = None
        return new

    def find(self, x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        root = x
        while root in self.uf:
            root = self.uf[root]
        while x in self.uf and self.uf[x] != root:  # path compression
            self.uf[x], x = root, self.uf[x]
        return root

    def parent(self, x: str, slot: int) -> Optional[str]:
        return self.find(self.par[self.find(x)][slot])

    def parents(self, x: str) -> Tuple[Optional[str], Optional[str]]:
        p = self.par[self.find(x)]
        return self.find(p[0]), self.find(p[1])

    def roots(self) -> List[str]:
        return [n for n in self.par if n not in self.uf]

    def latent_roots(self) -> List[str]:
        return [n for n in self.roots() if n not in self.sampled]

    def n_latent(self) -> int:
        return len(self.latent_roots())

    def new_latent(self, sex: int) -> str:
        name = f"L{self.next_latent}"
        self.next_latent += 1
        self.sex[name] = sex
        self.par[name] = [None, None]
        return name

    # -- structural operations (return False on contradiction) ----------------
    def merge(self, x: str, y: str) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return True
        if rx in self.sampled and ry in self.sampled:
            return False
        sx, sy = self.sex[rx], self.sex[ry]
        if sx and sy and sx != sy:
            return False
        keep, drop = (rx, ry) if (rx in self.sampled or (ry not in self.sampled and rx < ry)) else (ry, rx)
        self.uf[drop] = keep
        self.sex[keep] = sx or sy
        for slot in (FATHER, MOTHER):
            pk, pd_ = self.par[keep][slot], self.par[drop][slot]
            if pk is None:
                self.par[keep][slot] = pd_
            elif pd_ is not None:
                if not self.merge(pk, pd_):
                    return False
        return True

    def set_parent(self, child: str, slot: int, parent: str) -> bool:
        c, p = self.find(child), self.find(parent)
        if c == p:
            return False
        need = SLOT_SEX[slot]
        if self.sex[p] == 0:
            self.sex[p] = need
        elif self.sex[p] != need:
            return False
        cur = self.find(self.par[c][slot])
        if cur is None:
            self.par[c][slot] = p
            return True
        return self.merge(cur, p)

    def ensure_parent(self, child: str, slot: int) -> str:
        c = self.find(child)
        cur = self.find(self.par[c][slot])
        if cur is None:
            cur = self.new_latent(SLOT_SEX[slot])
            self.par[c][slot] = cur
        return cur

    def share_parent(self, a: str, b: str, slot: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        pa, pb = self.find(self.par[ra][slot]), self.find(self.par[rb][slot])
        if pa is None and pb is None:
            latent = self.new_latent(SLOT_SEX[slot])
            self.par[ra][slot] = latent
            self.par[rb][slot] = latent
            return True
        if pa is None:
            return self.set_parent(ra, slot, pb)
        if pb is None:
            return self.set_parent(rb, slot, pa)
        return self.merge(pa, pb)

    # -- validity ------------------------------------------------------------
    def _resolve_ref(self, ref: Tuple[str, Optional[int]]) -> Optional[str]:
        node, slot = ref
        return self.find(node) if slot is None else self.parent(node, slot)

    def violates_inequality(self) -> bool:
        for r1, r2 in self.ineq:
            x, y = self._resolve_ref(r1), self._resolve_ref(r2)
            if x is not None and x == y:
                return True
        return False

    def has_cycle(self) -> bool:
        """Iterative depth-first search for a directed cycle in child -> parent edges."""
        WHITE, GREY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.roots()}
        for start in color:
            if color[start] != WHITE:
                continue
            stack = [(start, iter(self.parents(start)))]
            color[start] = GREY
            while stack:
                node, children = stack[-1]
                nxt = next(children, "__done__")
                if nxt == "__done__":
                    color[node] = BLACK
                    stack.pop()
                    continue
                if nxt is None:
                    continue
                if color.get(nxt, WHITE) == GREY:
                    return True
                if color.get(nxt, WHITE) == WHITE:
                    color[nxt] = GREY
                    stack.append((nxt, iter(self.parents(nxt))))
        return False

    def violates_ages(self, ages: Dict[str, float], config: PedigreeSearchConfig) -> bool:
        if not ages:
            return False
        lo_po, hi_po = config.po_age_window
        lo_gp, hi_gp = config.gp_age_window
        for c in self.sampled:
            ac = ages.get(c)
            if ac is None:
                continue
            for p in self.parents(c):
                if p is None:
                    continue
                ap = ages.get(p) if p in self.sampled else None
                if ap is not None and not (lo_po <= ap - ac <= hi_po):
                    return True
                for g in self.parents(p):
                    if g is None or g not in self.sampled:
                        continue
                    ag = ages.get(g)
                    if ag is not None and not (lo_gp <= ag - ac <= hi_gp):
                        return True
        return False

    def is_valid(self, ages: Dict[str, float], config: PedigreeSearchConfig) -> bool:
        return not (self.has_cycle() or self.violates_inequality() or self.violates_ages(ages, config))

    # -- implied relationships ------------------------------------------------
    def _snapshot(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        if self._rel_cache is None:
            self._rel_cache = {n: self.parents(n) for n in self.roots()}
        return self._rel_cache

    def relationship(self, a: str, b: str) -> str:
        """Implied relationship between two nodes (closest single relationship)."""
        return self.directed_relationship(a, b)[0]

    def directed_relationship(self, a: str, b: str) -> Tuple[str, int]:
        """
        Implied relationship and its direction: 1 if ``a`` is the elder/ancestral
        member (parent, grandparent, aunt/uncle, ...), -1 if ``b`` is, 0 if symmetric.
        """
        P = self._snapshot()
        a, b = self.find(a), self.find(b)
        if a == b:
            return "none", 0
        fa, ma = P[a]
        fb, mb = P[b]

        # 1st degree
        if b == fa or b == ma:
            return "PO", -1
        if a == fb or a == mb:
            return "PO", 1
        share_f = fa is not None and fa == fb
        share_m = ma is not None and ma == mb
        if share_f and share_m:
            return "FS", 0
        # 2nd degree
        if share_f:
            return "PHS", 0
        if share_m:
            return "MHS", 0
        for x, y, d in ((a, b, 1), (b, a, -1)):
            fy, my = P[y]
            if fy is not None and x in P[fy]:
                return "PGP", d
            if my is not None and x in P[my]:
                return "MGP", d

        def full_sibs(u, v):
            if u is None or v is None or u == v:
                return False
            fu, mu = P[u]
            fv, mv = P[v]
            return fu is not None and mu is not None and fu == fv and mu == mv

        def half_sibs(u, v):
            if u is None or v is None or u == v:
                return False
            fu, mu = P[u]
            fv, mv = P[v]
            sf = fu is not None and fu == fv
            sm = mu is not None and mu == mv
            return sf != sm

        for x, y, d in ((a, b, 1), (b, a, -1)):
            if any(full_sibs(x, p) for p in P[y]):
                return "AV", d
        # 3rd degree
        for x, y, d in ((a, b, 1), (b, a, -1)):
            for p in P[y]:
                if p is None:
                    continue
                for g in P[p]:
                    if g is not None and x in P[g]:
                        return "GGP", d
        for x, y, d in ((a, b, 1), (b, a, -1)):
            if any(half_sibs(x, p) for p in P[y]):
                return "HAV", d
        if any(full_sibs(p, q) for p in (fa, ma) for q in (fb, mb)):
            return "CO", 0
        for x, y, d in ((a, b, 1), (b, a, -1)):
            for p in P[y]:
                if p is not None and any(full_sibs(x, g) for g in P[p]):
                    return "GAV", d
        # 4th degree
        if any(half_sibs(p, q) for p in (fa, ma) for q in (fb, mb)):
            return "HCO", 0
        return "none", 0

    def ancestry_key(self, x: str, depth: int, memo: dict):
        """Nested (node, father-key, mother-key) tuple down to ``depth`` generations.

        Every implied relationship modelled here (up to great-grandparents, great-
        avuncular and half-first-cousins) is a function of the two nodes' depth-3
        ancestry keys, so relationships can be cached across pedigree states.
        """
        if x is None:
            return None
        if depth == 0:
            return x
        k = (x, depth)
        hit = memo.get(k)
        if hit is not None:
            return hit
        f, m = self._snapshot()[x]
        val = (x, self.ancestry_key(f, depth - 1, memo), self.ancestry_key(m, depth - 1, memo))
        memo[k] = val
        return val

    def couples(self) -> Set[Tuple[str, str]]:
        out = set()
        for n, (f, m) in self._snapshot().items():
            if f is not None and m is not None:
                out.add((f, m))
        return out


# ----------------------------------------------------------------------------
# Applying a label to a pedigree: every structural realisation
# ----------------------------------------------------------------------------

def _sex_slots(state: PedigreeState, node: str) -> List[int]:
    s = state.sex[state.find(node)]
    return [FATHER] if s == 1 else [MOTHER] if s == 2 else [FATHER, MOTHER]


def realise_label(state: PedigreeState, a: str, b: str, label: str) -> List[PedigreeState]:
    """Return every pedigree obtained by applying ``label`` to (a, b) in ``state``."""
    out: List[PedigreeState] = []

    def attempt(fn):
        s = state.copy()
        if fn(s) is not False:
            out.append(s)

    if label == DISTANT:
        out.append(state.copy())

    elif label == "PO":
        for parent, child in ((a, b), (b, a)):
            for slot in _sex_slots(state, parent):
                attempt(lambda s, p=parent, c=child, sl=slot: s.set_parent(c, sl, p))

    elif label == "FS":
        attempt(lambda s: s.share_parent(a, b, FATHER) and s.share_parent(a, b, MOTHER))

    elif label in ("PHS", "MHS"):
        shared = FATHER if label == "PHS" else MOTHER
        other = MOTHER if shared == FATHER else FATHER

        def hs(s):
            if not s.share_parent(a, b, shared):
                return False
            s.ineq.append(((a, other), (b, other)))
            return True
        attempt(hs)

    elif label in ("PGP", "MGP"):
        side = FATHER if label == "PGP" else MOTHER
        for g, c in ((a, b), (b, a)):
            for slot in _sex_slots(state, g):
                def gp(s, g=g, c=c, slot=slot):
                    mid = s.ensure_parent(c, side)
                    return s.set_parent(mid, slot, g)
                attempt(gp)

    elif label == "AV":
        for u, n in ((a, b), (b, a)):
            for side in (FATHER, MOTHER):
                def av(s, u=u, n=n, side=side):
                    p = s.ensure_parent(n, side)
                    if s.find(p) == s.find(u):
                        return False
                    s.ineq.append(((u, None), (n, side)))
                    return s.share_parent(u, p, FATHER) and s.share_parent(u, p, MOTHER)
                attempt(av)
    else:
        raise ValueError(f"Unknown label: {label}")

    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

class FamilyScorer:
    """Composite log-likelihood of a pedigree, restricted to a set of sampled individuals."""

    def __init__(self, members: Sequence[str], evidence: PairEvidence, config: PedigreeSearchConfig):
        self.members = list(members)
        self.evidence = evidence
        self.config = config
        self.log_absent = math.log(config.absent_pair_penalty)
        self.log_pi = {"1st": math.log(config.mating_prior_1st),
                       "2nd": math.log(config.mating_prior_2nd)}
        self.pair_logp = {}
        for a, b in it.combinations(sorted(members), 2):
            if (a, b) in evidence.index:
                evidence.logp((a, b), "none")
                self.pair_logp[(a, b)] = evidence._logp_cache[(a, b)]
        self._intern: Dict[tuple, int] = {}
        self._rel_cache: Dict[Tuple[int, int], Tuple[str, int]] = {}

    def _key_id(self, state: PedigreeState, x: str, memo: dict) -> int:
        anc = state.ancestry_key(state.find(x), 3, memo)
        k = self._intern.get(anc)
        if k is None:
            k = self._intern[anc] = len(self._intern)
        return k

    def _rel_ids(self, state: PedigreeState, a: str, b: str, ka: int, kb: int) -> Tuple[str, int]:
        hit = self._rel_cache.get((ka, kb))
        if hit is None:
            hit = self._rel_cache[(ka, kb)] = state.directed_relationship(a, b)
        return hit

    def score(self, state: PedigreeState, individuals: Iterable[str]) -> Tuple[float, Tuple]:
        inds = sorted(individuals)
        memo: dict = {}
        kid = [self._key_id(state, x, memo) for x in inds]
        total = 0.0
        rels = []
        pair_logp = self.pair_logp
        log_absent = self.log_absent
        n = len(inds)
        for i in range(n):
            a, ka = inds[i], kid[i]
            for j in range(i + 1, n):
                rd = self._rel_ids(state, a, inds[j], ka, kid[j])
                rels.append(rd)
                lp = pair_logp.get((a, inds[j]))
                if lp is not None:
                    total += lp[rd[0]]
                elif rd[0] in CLOSE_RELS:
                    total += log_absent
        for f, m in state.couples():
            r = self._rel_ids(state, f, m, self._key_id(state, f, memo), self._key_id(state, m, memo))[0]
            total += self.log_pi.get(REL_DEGREE[r], 0.0)
        # Directed relationships in the signature: "A parent of B" and "B parent of A" are
        # different partial pedigrees that later pairs can extend differently.
        signature = (tuple(rels), tuple(state.sex[state.find(i)] for i in inds), state.n_latent())
        return total, signature

    def implied(self, state: PedigreeState) -> Dict[Tuple[str, str], str]:
        return {(a, b): state.relationship(a, b) for a, b in it.combinations(sorted(self.members), 2)}


# ----------------------------------------------------------------------------
# Families
# ----------------------------------------------------------------------------

@dataclass
class DuplicateGroup:
    representative: str
    members: List[str]


def find_duplicates(df: pd.DataFrame, config: PedigreeSearchConfig) -> List[DuplicateGroup]:
    """Duplicate / MZ-like samples: IBD2 >= 90%, IBD1 <= 5%, IBD1+IBD2 >= 90% of the genome."""
    if "ibd1" not in df.columns or "ibd2" not in df.columns:
        return []
    G = config.genome_length
    ibd1 = pd.to_numeric(df["ibd1"], errors="coerce").fillna(0)
    ibd2 = pd.to_numeric(df["ibd2"], errors="coerce").fillna(0)
    mask = (ibd2 >= config.dup_min_ibd2_frac * G) & (ibd1 <= config.dup_max_ibd1_frac * G) & \
           ((ibd1 + ibd2) >= config.dup_min_total_frac * G)
    uf: Dict[str, str] = {}

    def find(x):
        while uf.get(x, x) != x:
            x = uf[x]
        return x

    for a, b in zip(df.loc[mask, "id1"].astype(str), df.loc[mask, "id2"].astype(str)):
        uf.setdefault(a, a)
        uf.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[max(ra, rb)] = min(ra, rb)
    groups: Dict[str, List[str]] = {}
    for x in uf:
        groups.setdefault(find(x), []).append(x)
    return [DuplicateGroup(rep, sorted(m)) for rep, m in sorted(groups.items()) if len(m) > 1]


def collapse_duplicates(df: pd.DataFrame, groups: List[DuplicateGroup]) -> pd.DataFrame:
    """Map duplicates onto one node; keep the most-sharing row for each collapsed pair."""
    if not groups:
        return df
    rep = {m: g.representative for g in groups for m in g.members}
    out = df.copy()
    out["id1"] = out["id1"].astype(str).map(lambda x: rep.get(x, x))
    out["id2"] = out["id2"].astype(str).map(lambda x: rep.get(x, x))
    out = out[out["id1"] != out["id2"]].copy()
    tot = pd.to_numeric(out.get("ibd1", 0), errors="coerce").fillna(0) + \
          pd.to_numeric(out.get("ibd2", 0), errors="coerce").fillna(0)
    out["_tot"] = tot
    out["_key"] = [pair_key(a, b) for a, b in zip(out["id1"], out["id2"])]
    out = out.sort_values("_tot", ascending=False).drop_duplicates("_key").drop(columns=["_tot", "_key"])
    return out.reset_index(drop=True)


def build_families(evidence: PairEvidence, config: PedigreeSearchConfig) -> List[List[str]]:
    """Connected components of the graph with an edge where P(1st)+P(2nd) >= min_close_prob."""
    uf: Dict[str, str] = {}

    def find(x):
        while uf.get(x, x) != x:
            uf[x] = uf.get(uf[x], uf[x])
            x = uf[x]
        return x

    for key in evidence.index:
        if evidence.probs(key)["_close"] >= config.min_close_prob:
            a, b = key
            uf.setdefault(a, a)
            uf.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                uf[max(ra, rb)] = min(ra, rb)
    comps: Dict[str, List[str]] = {}
    for x in uf:
        comps.setdefault(find(x), []).append(x)
    fams = [sorted(c) for c in comps.values() if len(c) > 1]
    fams.sort(key=lambda c: (-len(c), c[0]))
    return fams


# ----------------------------------------------------------------------------
# Beam search
# ----------------------------------------------------------------------------

@dataclass
class RankedTree:
    state: PedigreeState
    log_likelihood: float
    implied: Dict[Tuple[str, str], str]
    relative_probability: float = 0.0


def search_family(members: Sequence[str], evidence: PairEvidence, config: PedigreeSearchConfig,
                  sexes: Optional[Dict[str, int]] = None, ages: Optional[Dict[str, float]] = None,
                  locks: Sequence[Lock] = ()) -> List[RankedTree]:
    """Beam search over consistent pedigrees for one family; returns ranked trees."""
    sexes = sexes or {}
    ages = {k: v for k, v in (ages or {}).items() if k in set(members)}
    scorer = FamilyScorer(members, evidence, config)
    member_set = set(members)

    lock_map = {lk.key: lk.label for lk in locks}
    pair_keys = [k for k in (pair_key(a, b) for a, b in it.combinations(sorted(members), 2)) if k in evidence]
    for k in lock_map:
        if k not in evidence and k[0] in member_set and k[1] in member_set:
            pair_keys.append(k)

    def options(k):
        if k in lock_map:
            lab = lock_map[k]
            p = evidence.probs(k)[lab] if k in evidence else 1.0
            return [(lab, p)]
        return evidence.label_options(k)

    # Locked pairs first, then pairs ordered by their most confident label.
    pair_keys.sort(key=lambda k: (k not in lock_map, -(max(p for _, p in options(k))), k))

    root = PedigreeState({m: int(sexes.get(m, 0) or 0) for m in members})
    beam = [root]

    for a, b in pair_keys:
        expanded: Dict[Tuple, PedigreeState] = {}
        for state in beam:
            placed = state.placed | {a, b}
            for label, _p in options((a, b)):
                if label == DISTANT and placed == state.placed:
                    # no new structure and no newly placed individuals: same state, same score
                    prev = expanded.get(state.signature)
                    if prev is None or state.log_l > prev.log_l:
                        expanded[state.signature] = state
                    continue
                for child in realise_label(state, a, b, label):
                    if not child.is_valid(ages, config):
                        continue
                    child.placed = placed
                    child.log_l, child.signature = scorer.score(child, placed)
                    prev = expanded.get(child.signature)
                    if prev is None or child.log_l > prev.log_l:
                        expanded[child.signature] = child
        if not expanded:
            logger.warning(f"No consistent pedigree after pair {a}-{b}; stopping search for this family")
            break
        beam = sorted(expanded.values(), key=lambda s: -s.log_l)[: config.beam_width]

    # Finished states: full log L over all members, merge identical implied relationships.
    finished: Dict[Tuple, RankedTree] = {}
    for state in beam:
        state.placed = frozenset(members)
        log_l, _ = scorer.score(state, members)
        implied = scorer.implied(state)
        if any(not lock_satisfied(lab, implied.get(k, "none")) for k, lab in lock_map.items()
               if k[0] in member_set and k[1] in member_set):
            continue
        sig = tuple(implied[k] for k in sorted(implied))
        prev = finished.get(sig)
        cand = RankedTree(state, log_l, implied)
        if prev is None or (state.n_latent(), -log_l) < (prev.state.n_latent(), -prev.log_likelihood):
            finished[sig] = cand

    ranked = sorted(finished.values(), key=lambda t: (-t.log_likelihood, t.state.n_latent()))[: config.export_n]
    if ranked:
        best = ranked[0].log_likelihood
        w = np.array([math.exp(t.log_likelihood - best) for t in ranked])
        w = w / w.sum()
        for t, p in zip(ranked, w):
            t.relative_probability = float(p)
    return ranked


# ----------------------------------------------------------------------------
# Whole-cohort driver and outputs
# ----------------------------------------------------------------------------

@dataclass
class FamilyResult:
    family_id: str
    members: List[str]
    searched: bool
    reason: Optional[str] = None
    n_observed_pairs: int = 0
    trees: List[RankedTree] = field(default_factory=list)
    seconds: float = 0.0


def load_pairs_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"id1": str, "id2": str})
    missing = {"id1", "id2", "1st", "2nd"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing PONDEROSA node columns: {', '.join(sorted(missing))}")
    return df


def load_sexes(fam: Optional[Path]) -> Dict[str, int]:
    if not fam:
        return {}
    fam_df = pd.read_csv(fam, sep=r"\s+", header=None, dtype=str)
    out = {}
    for iid, sex in zip(fam_df[1], fam_df[4]):
        try:
            s = int(sex)
        except (TypeError, ValueError):
            s = 0
        out[str(iid)] = s if s in (1, 2) else 0
    return out


def load_ages(ages: Optional[Path]) -> Dict[str, float]:
    if not ages:
        return {}
    df = pd.read_csv(ages, sep=r"\s+", header=None, dtype=str)
    out = {}
    for iid, age in zip(df[0], df[1]):
        try:
            val = float(age)
        except (TypeError, ValueError):
            continue
        if not np.isnan(val):
            out[str(iid)] = val
    return out


def reconstruct_pedigrees(pairs_df: pd.DataFrame, config: Optional[PedigreeSearchConfig] = None,
                          sexes: Optional[Dict[str, int]] = None, ages: Optional[Dict[str, float]] = None,
                          locks: Sequence[Lock] = (), families: Optional[Sequence[str]] = None
                          ) -> Tuple[List[FamilyResult], List[DuplicateGroup], PairEvidence]:
    """Build families from PONDEROSA pair output and rank pedigrees for each."""
    config = config or PedigreeSearchConfig.with_age_windows_from()
    dups = find_duplicates(pairs_df, config)
    if dups:
        logger.info(f"Collapsing {sum(len(g.members) - 1 for g in dups)} duplicate/MZ-like samples "
                    f"in {len(dups)} groups")
    df = collapse_duplicates(pairs_df, dups)
    evidence = PairEvidence(df, config)
    fams = build_families(evidence, config)
    logger.info(f"{len(fams)} families (P(1st)+P(2nd) >= {config.min_close_prob})")

    results = []
    for i, members in enumerate(fams, start=1):
        fid = f"F{i:03d}"
        n_obs = sum(1 for k in it.combinations(members, 2) if pair_key(*k) in evidence)
        res = FamilyResult(fid, members, searched=False, n_observed_pairs=n_obs)
        if families and fid not in families:
            res.reason = "not selected"
        elif len(members) > config.max_family_size:
            res.reason = f"family larger than max_family_size={config.max_family_size}"
        else:
            fam_locks = [lk for lk in locks if lk.family_id in (None, fid)
                         and lk.id1 in members and lk.id2 in members]
            t0 = time.time()
            res.trees = search_family(members, evidence, config, sexes, ages, fam_locks)
            res.seconds = time.time() - t0
            res.searched = True
            if not res.trees:
                res.reason = "no consistent pedigree found"
        results.append(res)
    return results, dups, evidence


def _tree_to_dict(tree: RankedTree, rank: int, members: Sequence[str], evidence: PairEvidence,
                  ages: Dict[str, float], dup_members: Dict[str, List[str]]) -> dict:
    s = tree.state
    nodes = []
    for n in sorted(s.roots(), key=lambda x: (x not in s.sampled, x)):
        f, m = s.parents(n)
        nodes.append({
            "id": n, "sampled": n in s.sampled, "sex": s.sex[n], "father": f, "mother": m,
            "age": ages.get(n), "duplicates": [d for d in dup_members.get(n, []) if d != n],
        })
    pairs = []
    for (a, b), rel in sorted(tree.implied.items()):
        observed = (a, b) in evidence
        call, prob = evidence.ponderosa_call((a, b)) if observed else (None, None)
        if not observed and rel == "none":
            continue
        pairs.append({
            "id1": a, "id2": b, "implied": rel, "degree": REL_DEGREE[rel], "observed": observed,
            "ponderosa_call": call, "ponderosa_prob": prob,
            "prob_implied": evidence.probs((a, b))[rel] if observed else None,
        })
    return {
        "rank": rank, "log_likelihood": tree.log_likelihood,
        "relative_probability": tree.relative_probability,
        "n_latent": s.n_latent(), "nodes": nodes, "pairs": pairs,
    }


def results_to_json(results: List[FamilyResult], dups: List[DuplicateGroup], evidence: PairEvidence,
                    config: PedigreeSearchConfig, ages: Dict[str, float], inputs: dict) -> dict:
    dup_members = {g.representative: g.members for g in dups}
    fam_out = []
    for r in results:
        fam_out.append({
            "family_id": r.family_id, "members": r.members, "n_members": len(r.members),
            "searched": r.searched, "reason": r.reason, "n_observed_pairs": r.n_observed_pairs,
            "seconds": round(r.seconds, 3),
            "trees": [_tree_to_dict(t, i + 1, r.members, evidence, ages, dup_members)
                      for i, t in enumerate(r.trees)],
        })
    cfg = asdict(config)
    return {
        "format": "ponderosa-pedigrees/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {k: (str(v) if v is not None else None) for k, v in inputs.items()},
        "parameters": cfg,
        "duplicates": [asdict(g) for g in dups],
        "families": fam_out,
    }


def write_summary_tsv(results: List[FamilyResult], evidence: PairEvidence, config: PedigreeSearchConfig,
                      path: Path) -> Path:
    rows = []
    for r in results:
        if not r.trees:
            rows.append({"family_id": r.family_id, "n_members": len(r.members), "searched": r.searched,
                         "rank": None, "relative_probability": None, "log_likelihood": None, "n_latent": None,
                         "n_close_pairs": None, "n_changed_from_ponderosa": None,
                         "members": ",".join(r.members), "close_relationships": "", "note": r.reason or ""})
            continue
        for i, t in enumerate(r.trees[: config.top_k]):
            close = [(k, v) for k, v in sorted(t.implied.items()) if REL_DEGREE[v] in ("1st", "2nd")]
            changed = 0
            for k, v in close:
                if k in evidence:
                    call, _ = evidence.ponderosa_call(k)
                    if call != v:
                        changed += 1
            rows.append({
                "family_id": r.family_id, "n_members": len(r.members), "searched": True, "rank": i + 1,
                "relative_probability": round(t.relative_probability, 6),
                "log_likelihood": round(t.log_likelihood, 4), "n_latent": t.state.n_latent(),
                "n_close_pairs": len(close), "n_changed_from_ponderosa": changed,
                "members": ",".join(r.members),
                "close_relationships": ";".join(f"{a}-{b}:{v}" for (a, b), v in close),
                "note": "",
            })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def write_viewer(payload: dict, path: Path, template: Path = VIEWER_TEMPLATE) -> Path:
    html = template.read_text()
    data = json.dumps(payload, default=str).replace("</", "<\\/")
    path.write_text(html.replace("/*__PEDIGREE_DATA__*/null", data))
    return path


def write_outputs(results, dups, evidence, config, ages, output_prefix: str, inputs: dict,
                  viewer: bool = True) -> List[Path]:
    payload = results_to_json(results, dups, evidence, config, ages, inputs)
    json_path = Path(f"{output_prefix}_pedigrees.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=1, default=str))
    written = [json_path, write_summary_tsv(results, evidence, config, Path(f"{output_prefix}_pedigree_summary.tsv"))]
    if viewer:
        written.append(write_viewer(payload, Path(f"{output_prefix}_pedigrees.html")))
    return written


# ----------------------------------------------------------------------------
# CLI: python -m ponderosa pedigree
# ----------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    d = PedigreeSearchConfig()
    p = argparse.ArgumentParser(prog="ponderosa pedigree",
                                description="Rank whole family trees from PONDEROSA pair output.")
    p.add_argument("--pairs", required=True, type=Path, help="PONDEROSA {prefix}_pairs.txt")
    p.add_argument("--fam", type=Path, help="PLINK FAM with sexes (column 5)")
    p.add_argument("--ages", type=Path, help="Ages file: IID AGE")
    p.add_argument("--priors", type=Path, default=None, help="priors.yaml with PO/GP age windows")
    p.add_argument("--output", "-o", required=True, help="Output prefix")
    p.add_argument("--lock", action="append", default=[], metavar="FAM:ID1:ID2:LABEL",
                   help="Restrict a pair to one label, e.g. F003:ID1:ID2:AV (repeatable)")
    p.add_argument("--families", default=None, help="Comma-separated family IDs to search (default all)")
    p.add_argument("--beam-width", type=int, default=d.beam_width)
    p.add_argument("--export-n", type=int, default=d.export_n)
    p.add_argument("--top-k", type=int, default=d.top_k)
    p.add_argument("--max-labels", type=int, default=d.max_labels_per_pair)
    p.add_argument("--min-label-prob", type=float, default=d.min_label_prob)
    p.add_argument("--min-close-prob", type=float, default=d.min_close_prob)
    p.add_argument("--max-family-size", type=int, default=d.max_family_size)
    p.add_argument("--prob-floor", type=float, default=d.prob_floor)
    p.add_argument("--absent-penalty", type=float, default=d.absent_pair_penalty)
    p.add_argument("--mating-prior-1st", type=float, default=d.mating_prior_1st,
                   help="Prior for a couple being 1st-degree relatives (1 disables)")
    p.add_argument("--mating-prior-2nd", type=float, default=d.mating_prior_2nd,
                   help="Prior for a couple being 2nd-degree relatives (1 disables)")
    p.add_argument("--genome-length", type=float, default=d.genome_length)
    p.add_argument("--no-viewer", action="store_true", help="Do not write the HTML viewer")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    config = PedigreeSearchConfig.with_age_windows_from(
        args.priors,
        beam_width=args.beam_width, export_n=args.export_n, top_k=args.top_k,
        max_labels_per_pair=args.max_labels, min_label_prob=args.min_label_prob,
        min_close_prob=args.min_close_prob, max_family_size=args.max_family_size,
        prob_floor=args.prob_floor, absent_pair_penalty=args.absent_penalty,
        mating_prior_1st=args.mating_prior_1st, mating_prior_2nd=args.mating_prior_2nd,
        genome_length=args.genome_length,
    )
    locks = [Lock.parse(x) for x in args.lock]
    fams = args.families.split(",") if args.families else None
    pairs_df = load_pairs_table(args.pairs)
    sexes, ages = load_sexes(args.fam), load_ages(args.ages)

    t0 = time.time()
    results, dups, evidence = reconstruct_pedigrees(pairs_df, config, sexes, ages, locks, fams)
    written = write_outputs(results, dups, evidence, config, ages, args.output,
                            {"pairs": args.pairs, "fam": args.fam, "ages": args.ages,
                             "locks": ",".join(args.lock) or None},
                            viewer=not args.no_viewer)
    n_searched = sum(r.searched for r in results)
    print(f"{len(results)} families ({n_searched} searched) reconstructed in {time.time() - t0:.1f} s")
    for r in results[:10]:
        if r.trees:
            top = ", ".join(f"{t.relative_probability:.0%}" for t in r.trees[: config.top_k])
            print(f"  {r.family_id}: {len(r.members)} members, top trees {top}")
        else:
            print(f"  {r.family_id}: {len(r.members)} members, {r.reason}")
    for w in written:
        print(f"Written: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
