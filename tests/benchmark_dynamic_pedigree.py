"""
Benchmark for ranked family trees (V3 section 6.6).

Simulated setting (tests/pedigree_sim.py):
    python tests/benchmark_dynamic_pedigree.py --seed 1 --families 120
    python tests/benchmark_dynamic_pedigree.py --seed 7 --families 90           # held out
    python tests/benchmark_dynamic_pedigree.py --seed 7 --families 90 --noisy   # noisier degree calls

Synthetic PONDEROSA demo (isolated pairs, with ages):
    python tests/benchmark_dynamic_pedigree.py --demo demo_run

Reports, for families that were searched:
  * true tree ranked 1st / in the top 3 (implied relationships equal the truth on
    every pair of sampled members);
  * accuracy on true 1st/2nd-degree pairs of PONDEROSA's own call vs the best tree.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ponderosa.dynamic_pedigree import (  # noqa: E402
    PedigreeSearchConfig, REL_DEGREE, load_ages, load_pairs_table, load_sexes, pair_key, reconstruct_pedigrees,
)
from tests.pedigree_sim import NoiseModel, simulate_cohort  # noqa: E402


def run_simulated(seed: int, n_families: int, noisy: bool, beam_width: int) -> dict:
    noise = NoiseModel.noisy() if noisy else NoiseModel()
    df, sexes, fams = simulate_cohort(n_families, seed=seed, noise=noise)
    truth = {}
    for fam in fams:
        truth.update(fam.truth)
    config = PedigreeSearchConfig(beam_width=beam_width)
    t0 = time.time()
    results, _, evidence = reconstruct_pedigrees(df, config, sexes=sexes)
    elapsed = time.time() - t0

    n_fam = top1 = top3 = 0
    close_total = call_right = tree_right = 0
    times = []
    for r in results:
        if not r.searched or not r.trees:
            continue
        n_fam += 1
        times.append(r.seconds)
        fam_truth = {pair_key(a, b): truth.get(pair_key(a, b), "none")
                     for i, a in enumerate(r.members) for b in r.members[i + 1:]}
        ranks = [all(t.implied[k] == v for k, v in fam_truth.items()) for t in r.trees]
        top1 += bool(ranks and ranks[0])
        top3 += any(ranks[:3])
        best = r.trees[0].implied
        for k, v in fam_truth.items():
            if REL_DEGREE[v] in ("1st", "2nd"):
                close_total += 1
                tree_right += best[k] == v
                if k in evidence:
                    call, _ = evidence.ponderosa_call(k)
                    call_right += call == v
    return {
        "setting": f"seed {seed}" + (" noisy" if noisy else ""),
        "families": n_fam,
        "true_tree_rank1": top1 / max(n_fam, 1),
        "true_tree_top3": top3 / max(n_fam, 1),
        "close_pairs": close_total,
        "ponderosa_call_correct": call_right / max(close_total, 1),
        "best_tree_correct": tree_right / max(close_total, 1),
        "median_family_seconds": float(pd.Series(times).median()) if times else 0.0,
        "max_family_seconds": max(times) if times else 0.0,
        "total_seconds": elapsed,
    }


def run_demo(demo_dir: Path) -> dict:
    pairs = load_pairs_table(next(demo_dir.glob("*_pairs.txt")))
    truth_df = pd.read_csv(demo_dir / "demo_truth.txt", sep=r"\s+", dtype=str)
    truth = {pair_key(a, b): t for a, b, t in truth_df.iloc[:, :3].itertuples(index=False)}
    second = {k: v for k, v in truth.items() if v in ("PHS", "MHS", "PGP", "MGP", "AV")}
    config = PedigreeSearchConfig.with_age_windows_from()
    results, _, evidence = reconstruct_pedigrees(pairs, config, load_sexes(demo_dir / "demo.fam"),
                                                 load_ages(demo_dir / "demo_ages.txt"))
    trees = {pair_key(*r.members[:2]): r.trees for r in results if len(r.members) == 2 and r.trees}
    n = internal = best = top3 = 0
    for k, v in second.items():
        if k not in evidence:
            continue
        call, _ = evidence.ponderosa_call(k)
        n += 1
        internal += call in ("GPAV", "HS", "GP", "2nd", "1st", "relatives")
        ts = trees.get(k, [])
        best += bool(ts) and ts[0].implied[k] == v
        top3 += any(t.implied[k] == v for t in ts[:3])
    return {"setting": "PONDEROSA demo, 2nd-degree pairs", "pairs": n,
            "call_at_internal_node": internal / max(n, 1),
            "best_tree_correct": best / max(n, 1), "top3_contains": top3 / max(n, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--families", type=int, default=120)
    p.add_argument("--noisy", action="store_true")
    p.add_argument("--beam-width", type=int, default=128)
    p.add_argument("--demo", type=Path, default=None)
    args = p.parse_args()
    res = run_demo(args.demo) if args.demo else run_simulated(args.seed, args.families, args.noisy, args.beam_width)
    for k, v in res.items():
        print(f"{k:28s} {v:.3f}" if isinstance(v, float) else f"{k:28s} {v}")


if __name__ == "__main__":
    main()
