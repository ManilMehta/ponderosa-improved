# PONDEROSA V3 in PONDEROSA-IMPROVED: implementation record

This repository is `williamscole/PONDEROSA` (commit `e3ef402`) with every change in
*"PONDEROSA V3: computational description of the changes"* (Mehta & Henn, UC Davis)
implemented. Where the V3 repository (`ManilMehta/PONDEROSA-V3`, commit `f049cbd`)
already contained a change, that code was ported and checked against the document;
everything the document lists as uncommitted, stubbed or not yet wired in was written
here. Fixes for problems found along the way are listed separately.

## Section-by-section

| § | Poster item | Implementation | Status here |
|---|---|---|---|
| 1 | Baseline (segment filters, features, classifiers, hierarchy descent) | unchanged v2 code | `min_segment_length` 3 cM, `min_total_ibd` 50 cM, `max_gap` 1 cM, `min_p` 0.5 |
| 2 | Inputs: checks every input before running | `ponderosa/input_validation.py` (`validate_run_inputs`, `summarize_fam`, `validate_king_kin0`, `validate_truth_file`, `format_validation_report`) | Ported. **Now also runs automatically** at the start of every analysis (`core.run_ponderosa`, errors abort, warnings logged), via `python -m ponderosa validate`, and via `--validate-only`. |
| 3 | Clean IBD: converts hap-IBD positions correctly | `data_loading.load_pairs` passes `GeneticMap.add_plink_list(map_file_list)`; `HapIBDLoader`; `GeneticMap.interp` (`np.interp` per chromosome) | Ported, then corrected to match the document exactly: endpoints are converted **first**, length is recomputed as cM(end) − cM(start), and the 3 cM / 50 cM filters are applied in cM (V3 still filtered on hap-IBD's own length column). Haplotypes 1/2 → 0/1. Map knots are sorted by bp; endpoints outside the map are counted and a clamping warning is logged (the document's caveat). |
| 4 | Pair features: handles pairs with missing features | `RelationshipClassifier.data_from_pairs(return_mask=True)`, NaN re-expansion in `run_inference` | Ported. Training uses the same mask (`prepare_classifier_data`). |
| 5 | Classifiers: trains on simulated families | `ponderosa/simulation/` (`simulate.py`, `founders.py`, `pedsim.py`, `config.py`, `metadata.py`), `python -m ponderosa simulate` | **Completed.** `calculate_relatedness()` (a stub in V3) now uses KING (`relatedness_from_king`, kinship scale: `Kinship` column, or `PropIBD/2` for `.seg`), restricted to the VCF founder pool. Greedy independent-set founder sampling (100 attempts, neighbours removed) unchanged. Replicates = `n_pairs_per_relationship`, each with a fresh founder draw. Provenance JSON (`x.classif.pkl` → `x.training_metadata.json`) written automatically. Driver bugs fixed (see below). A real default `config/default.def` was written (it was empty). Verified end to end with Ped-sim + hap-IBD on a synthetic founder cohort. |
| 6 | Final call: ranks whole family trees | `ponderosa/dynamic_pedigree.py`, `ponderosa/templates/pedigree_viewer.html`, `python -m ponderosa pedigree` (or `--rank-trees`) | **New** (the document's `dynamic-pedigree.patch` was not in any repository). Families, composite likelihood, union-find pedigree states, structural realisations, beam search, locks, duplicate collapse and the three outputs as specified. 16 tests in `tests/test_dynamic_pedigree.py`; `tests/pedigree_sim.py`; `tests/benchmark_dynamic_pedigree.py`. |
| 7 | Fixes for real data | hap-IBD map (§3); empty pedigree (`Siblings.from_pedigree_data`); `QDA(reg_param=0.01)` + zero-variance jitter; `evaluation = None`; row mismatch (§4) | Ported. The HIVE environment issue "CLI default `--ibd-caller phasedibd` overrides the YAML" is now **fixed in code** (CLI default is unset; YAML wins; fallback `phasedibd`). |
| 8 | External validation against KING | `ponderosa/validation.py` (`load_king_kin0`, `normalized_pair_key`, `compare_predictions_to_king`, `validation_counts`, `missing_validation_pairs`, `write_king_validation_outputs`), `python -m ponderosa king-validate` | Ported; CLI added (`--close-only` filters to 1st/2nd-degree calls; `--iid-only` for cohorts whose IDs are plain IIDs). Duplicate/MZ QC: `ponderosa/qc.py`, `python -m ponderosa qc-duplicates`. |
| 9 | Synthetic benchmark | `tests/benchmark_synthetic.py` | New script reproducing the setup (GeneratePairs(50), 320 cM, `min_total_ibd` 20 cM). |

## §6 details as implemented

* **Families (6.1):** edges where P(1st)+P(2nd) ≥ `min_close_prob` (0.5); connected
  components; all observed pairs inside a family are evidence; families larger than
  `max_family_size` (40) are listed but not searched. Family IDs `F001…` are ordered by
  size, then first member.
* **Likelihood (6.2):** `log L = Σ_obs log max(P(r), 1e-6) + Σ_unobs[r 1st/2nd] log 1e-4 +
  Σ_couples log π(r)`, π(1st)=1e-3, π(2nd)=0.05. Leaves read from the node columns of
  `{prefix}_pairs.txt`; a NaN leaf takes an equal share of its parent's posterior,
  recursively (hierarchy read from `config/pedigree_codes.yaml`). 3rd-degree
  relationships use P(3rd); "none" and HCO use 1 − P(1st) − P(2nd). Set both mating
  priors to 1 (`--mating-prior-1st 1 --mating-prior-2nd 1`) for strongly
  consanguineous populations.
* **Representation (6.3):** nodes with sex 1/2/0 and F/M slots; latent parents created
  on demand; union-find merges (two sampled people never merge; merges unify parents
  recursively; sex conflicts rejected). Realisations: PO (direction × slot if sex
  unknown), FS (share F and M), PHS/MHS (share one slot + inequality on the other),
  PGP/MGP (direction × slot), AV (direction × which parent; aunt/uncle ≠ that parent),
  distant (none). Rejected: cycles (iterative DFS), half-sib inequality violations, and
  age windows from `config/priors.yaml` (PO [12, 70], GP [25, 120]) when ages are given.
  Implied relationships: PO, FS, PHS, MHS, PGP, MGP, AV, 3rd degree (GGP, HAV, CO,
  great-avuncular), HCO, none; double relationships report the closest single one.
* **Search (6.4):** pairs ordered by their most confident label; options = top 4 labels
  with P ≥ 1e-3 plus distant; beam 128; partial states scored by the implied likelihood
  over individuals placed so far; signature dedup; full log L at the end; states with
  identical implied relationships merged (fewer latent parents kept); top `export_n`
  (10) exported with relative probability `exp(logL_k − max)/Σ exp(logL_j − max)`.
* **Locks, duplicates, outputs (6.5):** `--lock F003:ID1:ID2:AV` (repeatable; `ID1:ID2:AV`
  applies to whichever family contains the pair); duplicate/MZ-like samples (IBD2 ≥ 90%,
  IBD1 ≤ 5%, IBD1+IBD2 ≥ 90% of the genome length) collapsed to one node before the search;
  outputs `_pedigrees.json`, `_pedigree_summary.tsv` (top-k trees per family) and the
  self-contained `_pedigrees.html` viewer.

Two implementation decisions the document leaves open:

1. **Directed signatures.** The beam's duplicate signature uses *directed* implied
   relationships (who is the parent/grandparent/aunt). With undirected relationships,
   "A parent of B" and "B parent of A" collapsed into one state mid-search, and without
   ages the dropped direction was sometimes the one later pairs needed: the true tree
   then out-scored the best answer. With directed signatures this fell to 1 family in
   32 in the diagnostic run, matching the document's "1 of 107". The final merge of
   finished trees remains undirected, as specified.
2. **Speed.** Relationships are cached on each node's three-generation ancestry, so a pair
   is recomputed only when its ancestry changes. Median family time in the benchmark is
   about 1 s; the largest simulated families (13–15 members) take up to ~13 s here.

## Verification

Full test suite: 85 passed, 11–12 failed (upstream at `e3ef402` in the same environment:
54 passed, 16 failed). Every remaining failure also fails upstream: outdated expectations
in `test_classifiers`/`test_ibd_tools`/`test_pedigree`; `TestSiblings[no_training]`, an
unseeded random test that fails about 1 run in 6; the end-to-end test's hard 95%
specific-accuracy threshold (91.5% here; upstream crashes before reaching it); and the
simulation tests, which need Ped-sim data files (`tests/data/ped-sim`, a scratch path on
another cluster) that are not in the repository. New tests: `test_dynamic_pedigree.py`
(16), `test_v3_fixes.py` (7) and `test_v3_generalization.py` (4) all pass.

Benchmarks (this implementation; simulators are this repository's, so tree-level numbers
are not expected to equal the poster's):

| Setting | Families | True tree 1st | True tree top 3 | 1st/2nd-degree pairs: PONDEROSA call → best tree |
|---|---|---|---|---|
| Seed 1 | 105 | 68% | 74% | 48% → 94.5% |
| Seed 7 (held out) | 76 | 54% | 65% | 46% → 93% |
| Noisier degree calls (seed 7) | 76 | 20% | 33% | 41% → 65% |
| *Document, seed 1* | *107* | *72%* | *84%* | *72% → 93%* |

The "noisier" setting here (degree right 80% of the time, wrong-degree majority
otherwise) is harsher than the document's, whose call accuracy barely moves (72% → 71%).

PONDEROSA demo, 2nd-degree pairs (250): the call stops at an internal node for 98.8%;
the best tree names the correct relationship for 26.8% and the top three contain it for
70.4% (document: all 247; 27%; 71%).

Synthetic benchmark (§9): degree accuracy 87% (432/494); exact-type accuracy 28% (99/350);
96% of 2nd-degree pairs stop at an internal node; example true AV pair P(2nd)=0.99,
P(GPAV)=0.60, P(AV)=0.27, P(GP)=0.33 (document: 88%, 44%, and 0.99/0.59/0.21/0.38). The
toy generator is random; in this draw every 2nd-degree subtype stops at GPAV, which is
the behaviour the section describes.

Simulation training, end to end: Ped-sim 1.4.7 + hap-IBD (built from source) on a
synthetic 100-founder, 4-chromosome cohort with a KING file containing two related
founder pairs. 60 simulated families, 4–5 founders each, none containing a related
founder pair; trained classes AV, MGP, MHS, PGP, PHS; metadata JSON written. The
classifier was then applied to a FAM with no parent links and no truth file (the
SacMEGA situation) and inference, duplicate QC and ranked trees completed.

## Other fixes made while implementing

* hap-IBD loader used polars' removed `dtypes=` argument and fails on polars ≥ 1.0 → `schema_overrides`.
* Ped-sim wrapper did not request phased output; hap-IBD needs phased, non-missing genotypes → `--keep_phase --miss_rate 0` (and `--fam`, `--seed`).
* Simulation driver: `run_ponderosa(..., train_only=True)` (invalid signature), `SimulationConfig.validate()` referenced a nonexistent `relationship_types`, `to_dict()` referenced a nonexistent `temp_dir`, `PedSim` received a `str` where a `Path` is required, the simmap was symlinked as `input.simmap` but passed as `input.map`, and `call_hap-ibd.sh` mapped to the format `call_hap-ibd`.
* `SampleRelatives` built `np.array(set(...))` (a 0-d array) when no sample list was given; founder-graph edges are now restricted to the founder pool.
* Training with a class of ≤ 5 pairs raised a bare `AssertionError` (blank error message) → a `ValueError` naming the classes and counts and suggesting remedies.
* `FilesConfig` built but never raised its "no files" error (→ `UnboundLocalError`); maps are optional except for hap-IBD.
* CLI/YAML merge dropped `ibd_files`, `map_files`, `rel_tree` and `priors` from the YAML, never applied `--max-gap`, and let unset CLI flags override YAML values.
* `setup.py`: `polars>=0.20.31`, `pyarrow`, `scipy`, and package data (config, templates).

## Limits carried over from the document

* Relative probabilities treat pairs as independent evidence and are best read as a ranking.
* Relationships are modelled up to first cousins; double relationships are reported as the closest single one.
* `ibdseq` passes the caller check but has no loader.
* `np.interp` clamps positions outside the map; the map must match the data's genome build.
