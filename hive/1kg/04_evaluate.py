"""
Step 4: compare PONDEROSA to the documented 1000 Genomes pedigree.

    python ponderosa-improved/hive/1kg/04_evaluate.py 1KG_3202

Truth = every relationship the pedigree implies between two sequenced samples (PO, FS,
half-sibs, grandparent, avuncular, ...). Writes 1KG_3202/outputs/1KG_3202_vs_pedigree.tsv
and prints: recall of documented relatives, accuracy of PONDEROSA's calls on them, close
relatives PONDEROSA reports that the pedigree does not document, and family-tree accuracy.
"""
import itertools as it, json, sys
from pathlib import Path
import pandas as pd
from ponderosa.dynamic_pedigree import PedigreeState, REL_DEGREE

d = Path(sys.argv[1] if len(sys.argv) > 1 else "1KG_3202")
name = d.name
fam = pd.read_csv(d / f"{name}.fam", sep=r"\s+", header=None, dtype=str,
                  names=["fid", "iid", "father", "mother", "sex", "pheno"])
st = PedigreeState({r.iid: int(r.sex) if r.sex in ("1", "2") else 0 for r in fam.itertuples()})
for r in fam.itertuples():
    st.par[r.iid] = [None if r.father == "0" else r.father, None if r.mother == "0" else r.mother]
truth = {}
for _, grp in fam.groupby("fid"):
    for a, b in it.combinations(sorted(grp.iid), 2):
        r = st.relationship(a, b)
        if r != "none":
            truth[(a, b)] = r

p = pd.read_csv(d / "outputs" / f"{name}_pairs.txt", sep="\t", dtype=str)
p["key"] = [tuple(sorted(x)) for x in zip(p.id1, p.id2)]
calls = dict(zip(p.key, zip(p.degree, p.pred_rel)))

rows = []
for k, t in truth.items():
    deg, call = calls.get(k, ("not found", "not found"))
    rows.append({"id1": k[0], "id2": k[1], "pedigree": t, "pedigree_degree": REL_DEGREE[t],
                 "ponderosa_degree": deg, "ponderosa_call": call})
ev = pd.DataFrame(rows)
ev.to_csv(d / "outputs" / f"{name}_vs_pedigree.tsv", sep="\t", index=False)

print(f"Documented relationships between sequenced samples: {len(ev)}")
print(ev.pedigree.value_counts().to_string(), "\n")
close = ev[ev.pedigree_degree.isin(["1st", "2nd"])]
found = close.ponderosa_call != "not found"
print(f"1st/2nd-degree documented pairs found by PONDEROSA: {found.sum()}/{len(close)} ({found.mean():.1%})")
print(f"  correct degree:        {(close.ponderosa_degree == close.pedigree_degree).mean():.1%}")
print(f"  correct relationship:  {(close.ponderosa_call == close.pedigree).mean():.1%}")
print("\nPedigree (rows) vs PONDEROSA call (columns):")
print(pd.crosstab(ev.pedigree, ev.ponderosa_call).to_string(), "\n")

# close relatives PONDEROSA calls that the pedigree does not document
cc = p[p.degree.isin(["1st", "2nd"])]
extra = cc[~cc.key.isin(truth.keys())]
print(f"PONDEROSA 1st/2nd-degree calls not in the pedigree: {len(extra)} "
      f"(cryptic relatives, sample issues, or errors)")
print(extra.pred_rel.value_counts().to_string())
extra.drop(columns="key").to_csv(d / "outputs" / f"{name}_undocumented_close_calls.tsv", sep="\t", index=False)

# family trees: does the top-ranked tree reproduce every documented relationship?
pj = d / "outputs" / f"{name}_pedigrees.json"
if pj.exists():
    fams = json.loads(pj.read_text())["families"]
    n = ok1 = ok3 = 0
    for f in fams:
        if not f["trees"]:
            continue
        mem = set(f["members"])
        t = {k: v for k, v in truth.items() if k[0] in mem and k[1] in mem}
        if not t:
            continue
        n += 1
        imp = [{(x["id1"], x["id2"]): x["implied"] for x in tr["pairs"]} for tr in f["trees"]]
        hit = [all(i.get(k, "none") == v for k, v in t.items()) for i in imp]
        ok1 += hit[0]; ok3 += any(hit[:3])
    print(f"\nFamily trees with documented relatives: {n}; documented relationships all reproduced by "
          f"the top tree: {ok1} ({ok1 / max(n, 1):.1%}), within the top 3: {ok3} ({ok3 / max(n, 1):.1%})")
