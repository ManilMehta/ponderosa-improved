"""
Step 3: build the PONDEROSA dataset folder for 1000 Genomes.

    python ponderosa-improved/hive/1kg/03_make_dataset.py --classifier PATH.classif.pkl

Creates 1KG_3202/ with links to the per-chromosome hap-IBD files and GRCh38 maps, a FAM
built from the official pedigree (parents, sex), the classifier, and ponderosa.yaml.
Then: ponderosa run 1KG_3202
"""
import argparse, gzip, os, re
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--classifier", required=True, help="Pretrained classifier (*.classif.pkl)")
ap.add_argument("--out", default="1KG_3202")
ap.add_argument("--min-total-ibd", type=float, default=50)
a = ap.parse_args()

here = Path.cwd()
out = here / a.out
out.mkdir(exist_ok=True)

# Pedigree -> FAM (FID IID father mother sex pheno)
ped_path = next(Path("ref").glob("*pedigree*"))
ped = pd.read_csv(ped_path, sep=r"\s+", dtype=str)
cols = {c.lower(): c for c in ped.columns}
pick = lambda *names: next(cols[n] for n in names if n in cols)
sid = pick("sampleid", "sample", "iid", "individual_id", "id")
fa = pick("fatherid", "father", "paternal_id", "pat")
mo = pick("motherid", "mother", "maternal_id", "mat")
sx = pick("sex", "gender")
ped = ped.rename(columns={sid: "iid", fa: "father", mo: "mother", sx: "sex"})
for c in ("father", "mother"):
    ped[c] = ped[c].fillna("0").replace({"NA": "0", ".": "0", "-": "0"})
sexmap = {"1": "1", "2": "2", "male": "1", "female": "2", "m": "1", "f": "2"}
ped["sex"] = ped["sex"].astype(str).str.lower().map(sexmap).fillna("0")

# Samples actually in the IBD/VCF data
vcf = sorted(Path("vcf").glob("chr*.snps.maf05.vcf.gz"))[0]
with gzip.open(vcf, "rt") as f:
    samples = next(l for l in f if l.startswith("#CHROM")).rstrip("\n").split("\t")[9:]
in_data = set(samples)
ped = ped[ped.iid.isin(in_data)]
# parents not sequenced -> 0 (PONDEROSA only analyses FAM samples)
for c in ("father", "mother"):
    ped.loc[~ped[c].isin(in_data), c] = "0"

# FID: connected pedigree component (helps reading the output)
import networkx as nx
g = nx.Graph(); g.add_nodes_from(ped.iid)
g.add_edges_from((r.iid, p) for r in ped.itertuples() for p in (r.father, r.mother) if p != "0")
fid = {s: f"PED{i:04d}" for i, comp in enumerate(sorted(nx.connected_components(g), key=lambda c: (-len(c), min(c))), 1) for s in comp}
fam = pd.DataFrame({"fid": ped.iid.map(fid), "iid": ped.iid, "father": ped.father, "mother": ped.mother,
                    "sex": ped.sex, "pheno": "-9"})
missing = in_data - set(fam.iid)
if missing:
    fam = pd.concat([fam, pd.DataFrame({"fid": sorted(missing), "iid": sorted(missing), "father": "0",
                                        "mother": "0", "sex": "0", "pheno": "-9"})])
fam.to_csv(out / "1KG_3202.fam", sep=" ", header=False, index=False)
n_po = (fam.father != "0").sum() + (fam.mother != "0").sum()
print(f"FAM: {len(fam)} samples, {n_po} sequenced parent-child links, {sum(len(c) > 1 for c in nx.connected_components(g))} multi-sample pedigrees")

def link(src, dst):
    dst = out / dst
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(Path(src).resolve(), dst)

for f in sorted(Path("ibd").glob("1kg.chr*.ibd.gz")):
    link(f, f.name)
for f in sorted(Path("ref/maps").glob("plink.chr*.GRCh38.map")):
    if re.search(r"chr(\d+)\.", f.name) and int(re.search(r"chr(\d+)\.", f.name).group(1)) <= 22:
        link(f, f.name)
link(a.classifier, "training.classif.pkl")
(out / "ponderosa.yaml").write_text(
    f"name: 1KG_3202\nibd_caller: hap-ibd\nmin_total_ibd: {a.min_total_ibd}\n"
    "slurm: {mem: 64G, cpus: 4, time: \"06:00:00\"}\n")
print(f"IBD files: {len(list(out.glob('1kg.chr*.ibd.gz')))}, maps: {len(list(out.glob('plink.chr*.map')))}")
print(f"Ready: ponderosa run {a.out}")
