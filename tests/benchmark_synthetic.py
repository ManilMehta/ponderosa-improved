"""
Synthetic benchmark (V3 section 9): degree vs exact-type accuracy and the GPAV stopping rule.

    python tests/benchmark_synthetic.py --pairs-per-rel 50 --min-total-ibd 20 --out /tmp/synth

Generates GeneratePairs(50) (50 pairs per relationship on a 2-chromosome, 320 cM toy
genome), runs PONDEROSA with min_total_ibd lowered to 20 cM so that half-cousin pairs
(~40 cM) are kept, and reports:

  * degree accuracy: called degree matches the true degree;
  * exact-type accuracy: the reported node equals the true leaf;
  * for 2nd-degree pairs, how often the call stops at an internal node (GPAV / HS / GP),
    with an example posterior showing why the descent rule (min_p = 0.5) stops there.

The 320 cM toy genome gives few segments per pair, so haplotype scores are noisy: these
numbers show the stopping behaviour, not expected accuracy on a full 3,545 cM genome.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ponderosa.config import PonderosaConfig  # noqa: E402
from ponderosa.core import run_ponderosa  # noqa: E402
from tests.generate_files import GeneratePairs  # noqa: E402

DEGREE = {"PO": "1st", "FS": "1st", "PHS": "2nd", "MHS": "2nd", "AV": "2nd", "PGP": "2nd", "MGP": "2nd",
          "HS": "2nd", "GP": "2nd", "GPAV": "2nd", "CO": "3rd", "HAV": "3rd", "GGP": "3rd", "HCO": "4th",
          "HSCO": "4th"}
# Exact type is scored over the specific 1st/2nd-degree leaves (PO, FS and the five
# 2nd-degree subtypes); degree accuracy is scored over every evaluated pair.
LEAVES = {"PO", "FS", "PHS", "MHS", "AV", "PGP", "MGP"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-per-rel", type=int, default=50)
    p.add_argument("--min-total-ibd", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("synthetic_benchmark"))
    args = p.parse_args()

    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = str(args.out / "synth")
    gen = GeneratePairs(args.pairs_per_rel)
    gen.generate_all()
    gen.write_out(prefix)

    config = PonderosaConfig.from_yaml(f"{prefix}_config.yaml")
    config.algorithm.min_total_ibd = args.min_total_ibd
    config.output.verbose = 0
    config.validate()
    run_ponderosa(config)

    pred = pd.read_csv(f"{config.output.output}_pairs.txt", sep="\t", dtype={"id1": str, "id2": str})
    truth = pd.read_csv(f"{prefix}_truth.txt", sep="\t", dtype=str)
    tmap = {tuple(sorted((a, b))): t for a, b, t in truth.itertuples(index=False)}
    pred["truth"] = [tmap.get(tuple(sorted((a, b)))) for a, b in zip(pred.id1, pred.id2)]
    pred = pred[pred.truth.notna()].copy()
    pred["true_degree"] = pred.truth.map(DEGREE)

    deg_ok = (pred.degree == pred.true_degree)
    leaf = pred[pred.truth.isin(LEAVES)]
    exact_ok = (leaf.pred_rel == leaf.truth)
    print(f"Degree accuracy      {deg_ok.mean():.0%} = {deg_ok.sum()} / {len(pred)} pairs")
    print(f"Exact-type accuracy  {exact_ok.mean():.0%} = {exact_ok.sum()} / {len(leaf)} pairs")
    print("\nCalls by true relationship:")
    print(pd.crosstab(pred.truth, pred.pred_rel).to_string())

    second = pred[pred.true_degree == "2nd"]
    internal = second.pred_rel.isin(["GPAV", "HS", "GP", "2nd"])
    print(f"\n2nd-degree pairs reported at an internal node: {internal.mean():.0%} ({internal.sum()}/{len(second)})")
    av = second[(second.truth == "AV") & (second.pred_rel == "GPAV")]
    if len(av):
        r = av.iloc[0]
        print(f"Example true AV pair {r.id1}-{r.id2}: P(2nd)={r['2nd']:.2f}, P(GPAV)={r['GPAV']:.2f}, "
              f"P(AV)={r['AV']:.2f}, P(GP)={r['GP']:.2f} -> both children of GPAV < 0.5, so the call stops at GPAV")


if __name__ == "__main__":
    main()
