"""
Species presets: autosome count, genome length and age priors.

    human    22 autosomes, 3,545 cM, config/priors.yaml (years)
    macaque  20 autosomes (rhesus, Mmul_10), genome length measured from the
             genetic map, config/priors_macaque.yaml (years)

Select with ``species: macaque`` in a run folder's ponderosa.yaml, or
``--species macaque`` on the command line.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

_CONFIG = Path(__file__).parent / "config"


@dataclass(frozen=True)
class Species:
    name: str
    n_autosomes: int
    genome_length: Optional[float]   # None: measure from the genetic map
    priors: Path


SPECIES = {
    "human": Species("human", 22, 3545.0, _CONFIG / "priors.yaml"),
    "macaque": Species("macaque", 20, None, _CONFIG / "priors_macaque.yaml"),
}
SPECIES["rhesus"] = SPECIES["macaque"]


def get_species(name: Optional[str]) -> Species:
    key = (name or "human").lower()
    if key not in SPECIES:
        raise ValueError(f"Unknown species '{name}'. Choose one of: {', '.join(sorted(set(SPECIES)))}")
    return SPECIES[key]


def map_genome_length(map_files: Union[str, Path, Sequence[Union[str, Path]]]) -> float:
    """Total autosomal length (cM) of one or more PLINK maps: sum over chromosomes of last - first cM."""
    from .map_tools import GeneticMap
    files: List[Path] = [Path(map_files)] if isinstance(map_files, (str, Path)) else [Path(f) for f in map_files]
    gm = GeneticMap.add_plink_list(files) if len(files) > 1 else GeneticMap.add_plink(files[0])
    return float(gm.map_df._calculate_genome_length())
