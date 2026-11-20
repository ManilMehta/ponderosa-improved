# Running PONDEROSA-IMPROVED on HIVE

Scripts for the UC Davis HIVE Slurm cluster, based on the setup that worked for SacMEGA
(private venv, writable bytecode cache, `gcc/13.2.0` + `boost/1.84.0` for Ped-sim,
`--ibd-caller` passed explicitly). Nothing here hardcodes an account or partition: edit
the `##SBATCH` lines.

## 1. One-time setup

```bash
cd /path/to/your/run_dir
git clone <this repo> PONDEROSA-IMPROVED
bash PONDEROSA-IMPROVED/hive/setup_env.sh "$PWD"      # venv/ and ped-sim/ in run_dir
cp -r PONDEROSA-IMPROVED/hive .                        # local copies to edit
```

## 2. List the datasets

Edit `hive/datasets.tsv`, one tab-separated line per dataset:
`name  ibd  ibd_caller  fam  map  training  king  ages  min_total_ibd` (use `-` for empty).

* hap-IBD output needs a PLINK map in the same genome build as the data.
* `training`: a classifier from step 4, or `-` to train on the FAM's pedigree.
  A FAM with no parent links cannot train classifiers; `ponderosa validate` warns about this.

## 3. Check inputs (fast, no IBD processing)

```bash
source venv/bin/activate
python -m ponderosa validate --ibd X.ibd.gz --fam X.fam --ibd-caller hap-ibd --map X.map --king X.kin0
```

## 4. (Cohorts without a usable pedigree) simulation-based training

Needs a phased cohort VCF (all autosomes), KING kinship for those samples, the hap-IBD
jar, the per-chromosome PLINK maps and Ped-sim's sex-specific map.

```bash
cp hive/simulation_template.yaml sim_COHORT.yaml         # edit paths
export HAPIBD_JAR=/path/hap-ibd.jar MAP_DIR=/path/plink.GRCh37.map
sbatch hive/train_simulation.sbatch sim_COHORT.yaml
```

Output: `outputs/COHORT_sim_training.classif.pkl` plus `.training_metadata.json`.

## 5. Run every dataset

```bash
N=$(grep -v '^#' hive/datasets.tsv | grep -cv '^\s*$')
sbatch --array=1-$N hive/run_datasets.sbatch
```

Each task writes `outputs/<name>/<name>_*`: pair calls (`_pairs.txt`), duplicate/MZ QC,
ranked family trees (`_pedigrees.json`, `_pedigree_summary.tsv`, `_pedigrees.html`) and,
when a KING file is given, `_prediction_vs_king.tsv` and `_validation_counts.tsv`.
