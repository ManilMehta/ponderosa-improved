"""
Script for managing/organizing the simulation
"""

from pathlib import Path
import logging
import random
import subprocess
import gzip

import numpy as np
import pandas as pd

from ..config import PonderosaConfig
from ..core import run_ponderosa
from .config import SimulationConfig
from .setup import simulation_workspace
from .founders import pedsim_dryrun, calculate_relatedness, create_founders_file
from .pedsim import PedSim
from .metadata import SimulationTrainingMetadata, write_training_metadata, metadata_path_for_classifier

logger = logging.getLogger(__name__)


def simulate(config: SimulationConfig) -> Path:
    """
    Simulation-based training workflow (V3, section 5).

    1. Founder pool: phased cohort VCF.
    2. Pedigree templates: ped-sim dry run gives the founder slots of each family.
    3. Unrelated founders: randomized greedy independent set on the KING kinship graph.
    4. Simulate: n_pairs_per_relationship replicates per family template.
    5. Call IBD: user script (e.g. hap-ibd.sh) on the simulated VCF.
    6. Train only: PONDEROSA training on the simulated FAM + IBD (3 cM / 50 cM).

    Returns the path to ``{output_path}.classif.pkl``. A provenance record is written
    next to it as ``{output_path}.training_metadata.json``.
    """

    config.validate()

    seed = config.pedsim.random_seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    with simulation_workspace(config) as temp_dir:

        vcf_ext = ".vcf.gz" if str(config.pedsim.vcf_file).endswith('.gz') else ".vcf"

        pedsim = PedSim(
            vcf_file=str(temp_dir / f"input{vcf_ext}"),
            def_file=str(temp_dir / "pedigree.def"),
            intf_file=str(temp_dir / "interference.tsv"),
            map_file=str(temp_dir / "input.simmap"),
            output=temp_dir / "simulation",
            executable_path=str(config.pedsim.pedsim_executable),
            seed=seed,
        )

        # 2. Dry run to get family structures (founder slots)
        logger.info("Running ped-sim dry run to obtain founder slots...")
        dry_run_families = pedsim_dryrun(pedsim)
        logger.info(f"{len(dry_run_families)} pedigree templates; "
                    f"{sum(len(f) for f in dry_run_families)} founder slots per replicate")

        # 3. Founder-pool samples and KING kinship among them
        vcf_samples = _get_vcf_samples(vcf_file=Path(pedsim.get_input("vcf")))
        relatedness_dict = calculate_relatedness(king_file=config.king_file, sample_ids=vcf_samples)
        n_edges = sum(1 for k in relatedness_dict.values() if k > config.training.max_kinship)
        logger.info(f"Founder pool: {len(vcf_samples)} samples; "
                    f"{n_edges} related pairs above max_kinship={config.training.max_kinship}")

        # 4. Create founders mapping (slot -> sample)
        founders_df = create_founders_file(
            vcf_samples=vcf_samples,
            dry_run_families=dry_run_families,
            relatedness_dict=relatedness_dict,
            max_k=config.training.max_kinship,
            n_sim=config.training.n_pairs_per_relationship
        )

        founders_file = temp_dir / "founders.txt"
        founders_df.to_csv(founders_file, sep="\t", index=False, header=False)

        # 5. Simulate: n_sim replicates of every family template
        pedsim.update_flag("--set_founders", str(founders_file))
        pedsim.set_replicates(config.training.n_pairs_per_relationship)
        logger.info("Running ped-sim...")
        pedsim.execute()

        # 6. Run IBD caller on simulated output
        logger.info(f"Calling IBD on simulated genomes with {config.ibd_caller}...")
        _run_ibd_caller(config.ibd_caller, temp_dir)

        # 7. Train only
        ponderosa_config = _create_ponderosa_config(temp_dir, config)
        ponderosa_config.algorithm.train_only = True
        ponderosa_config.output.write_training = True
        ponderosa_config.validate()

        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)

        run_ponderosa(ponderosa_config)

        classifier_path = Path(f"{config.output_path}.classif.pkl")

        # Provenance metadata
        metadata = SimulationTrainingMetadata(
            output_prefix=str(config.output_path),
            ibd_caller=ponderosa_config.files.ibd_caller,
            fam_file=str(ponderosa_config.files.fam),
            ibd_file=str(ponderosa_config.files.ibd),
            map_file=str(ponderosa_config.files.mapf) if ponderosa_config.files.mapf else None,
            pedsim_path=str(config.pedsim.pedsim_path),
            vcf_file=str(config.pedsim.vcf_file),
            def_file=str(config.pedsim.def_file) if config.pedsim.def_file else None,
            simmap_file=str(config.pedsim.simmap_file) if config.pedsim.simmap_file else None,
            interference_file=str(config.pedsim.interference_file) if config.pedsim.interference_file else None,
            random_seed=seed,
            min_segment_length=ponderosa_config.algorithm.min_segment_length,
            min_total_ibd=ponderosa_config.algorithm.min_total_ibd,
            relationship_counts=_relationship_counts(temp_dir / "simulation-everyone.fam",
                                                     ponderosa_config),
            notes=[f"king_file={config.king_file}",
                   f"max_kinship={config.training.max_kinship}",
                   f"n_pairs_per_relationship={config.training.n_pairs_per_relationship}",
                   f"ibd_script={config.ibd_caller}"],
        )
        meta_path = write_training_metadata(metadata, metadata_path_for_classifier(classifier_path))
        logger.info(f"Classifier: {classifier_path}")
        logger.info(f"Training metadata: {meta_path}")

        return classifier_path


def _relationship_counts(fam_file: Path, ponderosa_config: PonderosaConfig) -> dict:
    """Count labelled training pairs per relationship in the simulated pedigree."""
    try:
        from ..data_loading import load_individuals, load_pairs
        from ..pedigree import build_pedigree, PedigreeHierarchy
        hierarchy = PedigreeHierarchy.from_yaml()
        individuals = load_individuals(ponderosa_config.files)
        pairs = load_pairs(ponderosa_config.files, ponderosa_config.algorithm)
        registry = build_pedigree(individuals, pairs, hierarchy)
        return {rel: len(p) for rel, p in registry.nodes.items() if p}
    except Exception as exc:  # metadata must never break a finished training run
        logger.warning(f"Could not compute relationship counts for metadata: {exc}")
        return {}


def _get_vcf_samples(vcf_file: Path) -> list[str]:
    """Extract sample IDs from VCF header."""
    import gzip
    
    # Determine how to open the file
    if str(vcf_file).endswith('.gz'):
        open_func = lambda f: gzip.open(f, 'rt', encoding='utf-8')
    else:
        open_func = lambda f: open(f, 'r', encoding='utf-8')
    
    with open_func(vcf_file) as f:
        for line in f:
            if line.startswith('#CHROM'):
                # Header line found - extract sample IDs (columns 9 onward)
                columns = line.strip().split('\t')
                return columns[9:]
    
    # If we get here, no header line was found
    raise ValueError(f"No #CHROM header line found in VCF file: {vcf_file}")


def _run_ibd_caller(ibd_caller: str, temp_dir: Path):
    """
    Run the user's IBD calling script on simulated output.
    
    The script receives the temp directory as its only argument.
    It is expected to find the simulated VCF at {temp_dir}/simulation.vcf(.gz)
    and write IBD output with the prefix {temp_dir}/simulation.
    
    Args:
        ibd_caller: Path to the user's bash script
        temp_dir: Simulation workspace directory
    """
    script_path = Path(ibd_caller).resolve()
    
    if not script_path.exists():
        raise FileNotFoundError(f"IBD caller script not found: {script_path}")
    
    result = subprocess.run(
        ["bash", str(script_path), str(temp_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"IBD caller script failed (exit code {result.returncode}):\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )


def _create_ponderosa_config(temp_dir: Path, sim_config: SimulationConfig) -> PonderosaConfig:
    """
    Create PonderosaConfig from simulation output.
    
    Expects the following files in temp_dir after steps 7-8:
        - simulation-everyone.fam   (from ped-sim)
        - simulation.ibd.gz         (from IBD caller)
        - genetic.map               (from IBD caller script)
    """
    
    # Locate the fam file from ped-sim output
    fam_file = temp_dir / "simulation-everyone.fam"
    if not fam_file.exists():
        raise FileNotFoundError(f"Ped-sim fam file not found: {fam_file}")
    
    # Locate IBD output
    ibd_file = temp_dir / "simulation.ibd.gz"
    if not ibd_file.exists():
        raise FileNotFoundError(
            f"IBD output not found: {ibd_file}\n"
            f"Check that the IBD caller script wrote output with prefix: {temp_dir / 'simulation'}"
        )
    
    # Derive the IBD format from the script name (e.g. "hap-ibd.sh" or
    # "call_hap-ibd.sh" -> "hap-ibd")
    ibd_caller = ibd_format_from_script(sim_config.ibd_caller)
    
    config_dict = {
        "files": {
            "ibd": str(ibd_file),
            "fam": str(fam_file),
            "ibd_caller": ibd_caller,
        },
        "algorithm": {
            "min_segment_length": 3.0,
            "min_total_ibd": 50.0,
        },
        "output": {
            "output": sim_config.output_path,
            "write_training": True,
        }
    }
    
    # Add map file if the IBD caller script produced one
    map_file = temp_dir / "genetic.map"
    if map_file.exists():
        config_dict["files"]["mapf"] = str(map_file)
    
    return PonderosaConfig.from_dict(config_dict)


def ibd_format_from_script(script: str) -> str:
    """Map an IBD-calling script name to a PONDEROSA ``ibd_caller`` value."""
    stem = Path(script).stem.lower()
    for prefix in ("call_", "call-", "run_", "run-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    aliases = {"hapibd": "hap-ibd", "hap_ibd": "hap-ibd", "phased_ibd": "phasedibd", "phased-ibd": "phasedibd"}
    stem = aliases.get(stem, stem)
    if stem not in {"hap-ibd", "phasedibd"}:
        raise ValueError(
            f"Cannot determine the IBD format from script name '{script}'. "
            "Name the script after its output format, e.g. hap-ibd.sh or phasedibd.sh."
        )
    return stem
