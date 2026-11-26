"""
Script for choosing the founders of the simulation
"""

import random
import pandas as pd
import polars as pl
import numpy as np
import networkx as nx
import itertools as it
import re
from typing import Union, List, Tuple, Dict
from pathlib import Path
from ponderosa.data_loading import load_ibd_from_file, GeneticMap, FamFile
from ponderosa.simulation.pedsim import PedSim


########    Methods for computing kinship      ########
########    from KING files, IBD files, etc.   ########

def relatedness_from_ibd(
    ibd_file: List[str],
    ibd_caller: str,
    genetic_map_file_list: List[str],
) -> Dict[Tuple[str, str], float]:
    """
    Calculate and return relatedness coefficient among all pairs that share IBD.

    Output:
        dict with keys = (id1, id2), values = relatedness coefficient (0–1)
    """

    gm = GeneticMap.add_plink_list(genetic_map_file_list)
    total_genome_cm = gm.get_genome_length()
    print(f"Total genome length: {total_genome_cm:.2f} cm")

    ibd_df_related = load_ibd_from_file(
        file_paths = ibd_file,
        ibd_caller = ibd_caller,
        min_segment_length = 5,
        min_total_ibd = 5,
        **({"genetic_map": gm} if ibd_caller == "hap-ibd" else {}),
    )

    related_summary = (
        ibd_df_related
        .group_by(["id1", "id2"])
        .agg(pl.col("length_cm").sum().alias("total_ibd_cm"))
        .with_columns(
            (pl.col("total_ibd_cm") / total_genome_cm).alias("relatedness_coefficient")
        )
    )

    # Factor of 2 is to get it in the same scale as KING
    relatedness_dict = {
        (row["id1"], row["id2"]): row["relatedness_coefficient"] / 2
        for row in related_summary.to_dicts()
    }

    return relatedness_dict

def _king_sample_ids(fid, iid, fid_iid_sep: str = "_") -> List[str]:
    """Return the sample-ID spellings a KING row can match (IID and FID_IID)."""
    iid = str(iid)
    ids = [iid]
    if fid is not None and not pd.isna(fid):
        ids.append(f"{fid}{fid_iid_sep}{iid}")
    return ids


def relatedness_from_king(king_file: Union[str, Path], fid_iid_sep: str = "_") -> Dict[Tuple[str, str], float]:
    """
    Read KING output (.kin0, .kin or .seg) and return pairwise kinship.

    V3: founder independence is defined on the KING *kinship* scale (max_kinship = 0.05
    by default). If the file has a ``Kinship`` column (.kin0/.kin) it is used directly;
    KING ``--ibdseg`` output (.seg) only has ``PropIBD``, which is converted to kinship as
    PropIBD / 2.

    Each pair is stored under both its IID and its FID_IID spelling so that it can be
    matched to VCF sample IDs written either way.
    """

    from ..validation import read_king_table
    king_df = read_king_table(king_file)

    if "ID1" not in king_df.columns or "ID2" not in king_df.columns:
        raise ValueError(f"KING file must contain ID1 and ID2 columns: {king_file}")

    if "Kinship" in king_df.columns:
        kinship = pd.to_numeric(king_df["Kinship"], errors="coerce")
    elif "PropIBD" in king_df.columns:
        kinship = pd.to_numeric(king_df["PropIBD"], errors="coerce") / 2
    else:
        raise ValueError(f"KING file must contain a Kinship or PropIBD column: {king_file}")

    fid1, fid2 = king_df["FID1"], king_df["FID2"]

    out: Dict[Tuple[str, str], float] = {}
    for f1, i1, f2, i2, k in zip(fid1, king_df["ID1"], fid2, king_df["ID2"], kinship):
        if pd.isna(k):
            continue
        for a, b in zip(_king_sample_ids(f1, i1, fid_iid_sep), _king_sample_ids(f2, i2, fid_iid_sep)):
            out[(a, b)] = float(k)
            out[(b, a)] = float(k)

    return out


def calculate_relatedness(
    king_file: Union[str, Path, None] = None,
    ibd_file: Union[List[str], None] = None,
    ibd_caller: Union[str, None] = None,
    genetic_map_file_list: Union[List[str], None] = None,
    sample_ids: Union[List[str], None] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Pairwise kinship used to define the founder graph G (V3, section 5).

    Uses KING output when ``king_file`` is given (the method used for SacMEGA);
    otherwise falls back to kinship estimated from IBD segments. If ``sample_ids`` is
    given, pairs are restricted to those samples.
    """

    if king_file is not None:
        relatedness = relatedness_from_king(king_file)
    elif ibd_file is not None:
        if ibd_caller is None:
            raise ValueError("ibd_caller is required when computing relatedness from IBD files")
        relatedness = relatedness_from_ibd(ibd_file, ibd_caller, genetic_map_file_list or [])
    else:
        raise ValueError("Provide either king_file or ibd_file to calculate relatedness")

    if sample_ids is not None:
        keep = set(map(str, sample_ids))
        relatedness = {pair: k for pair, k in relatedness.items() if pair[0] in keep and pair[1] in keep}

    return relatedness

class Relatedness:
    pass

######## Methods for sampling   ########
######## unrelated individuals  ########

####### For simple unrelated sampling
def get_simple_graph(nodes: List[str], relatedness_dict: dict, max_k: float) -> nx.Graph:
    # Initialize the graph
    G = nx.Graph()
    G.add_nodes_from(nodes) # Add the nodes

    # Now susbset to only edges between sufficiently related individuals
    node_set = set(nodes)
    edge_list = [pair for pair, k in relatedness_dict.items()
                 if k > max_k and pair[0] in node_set and pair[1] in node_set]
    G.add_edges_from(edge_list) # Add the edges between "related"

    return G

def simple_family_sample(G: nx.Graph, n_members: int):
    for _ in range(100): # Give it 100 attempts
        g = G.copy()
        nodes = g.nodes()
        out_fam = [] # Stores the IDs of the founders
        for _ in range(n_members):
            new_node = np.random.choice(sorted(nodes))
            out_fam.append(str(new_node))
            nodes = nodes - {new_node} - set(g.neighbors(new_node))
            if len(nodes) == 0:
                break
            g = g.subgraph(nodes)

        if len(out_fam) == n_members:
            return out_fam
        
    return set()


class SampleRelatives:

    def __init__(self, relatedness_dict: dict, sample_list: Union[list, np.ndarray] = None):

        self.kinship = relatedness_dict

        # Should supply a sample list! May miss out on unrelated indviduals
        if sample_list is not None and len(sample_list) > 0:
            self.samples = np.array(sample_list)
        else:
            samples = sorted(set(it.chain(*relatedness_dict.keys())))
            self.samples = np.array(samples)

        self.graph = nx.Graph()

    def set_mode(self, mode: str, **kwargs):

        self.mode = mode

        if mode == "simple":
            assert "max_k" in kwargs

            self.max_k = kwargs["max_k"]

            self.graph = get_simple_graph(self.samples,
                                          self.kinship,
                                          self.max_k)

    def get_unrelated_founders(self, n_members: int):

        if self.mode == "simple":
            fam = simple_family_sample(self.graph, n_members)

        # Ensure that a founder family was found
        if len(fam) != n_members:
            raise RuntimeError(
                f"Could not find {n_members} mutually unrelated founders (kinship <= {self.max_k}) "
                f"in 100 attempts. Lower the family size or raise max_kinship."
            )
        
        return fam




#should return nested families from the dryrun [[“A”,”B”,”C”],[“D”,”E”,”F”]]
def pedsim_dryrun(
    pedsim: PedSim
) -> List[List[str]]:
    """   
    Ped-sim dry run to obtain fam file needed for simulations
    """

    pedsim.dry_run()
    fam_file = pedsim.get_file("fam")
    fam = FamFile(fam_file)
    return fam.get_founders()


def update_fam_idx(sim_id, idx):
    return re.sub(r'(?<!\d)1(?!\d)', str(idx), sim_id, count=1)

def create_founders_file(
    vcf_samples: List[str],
    dry_run_families: List[List[str]],
    relatedness_dict: Dict[Tuple[str, str], float],
    n_sim: int,
    mode: str = "simple",
    max_k: float = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate a founders mapping file by assigning VCF samples to simulated pedigrees.

    Args:
        vcf_samples: list of sample IDs from the VCF
        dry_run_families: list of families (each family is a list of IDs, e.g. [["A","B","C"], ["D","E"]])
        relatedness_dict: dict {(id1, id2): relatedness coefficient}
        r: relatedness threshold for founders
        n_sim: number of simulated pedigrees per family

    Returns:
        pd.DataFrame with columns: dry_run_id, vcf_id
    """

    sampling = SampleRelatives(relatedness_dict, vcf_samples)

    # TODO implement other modes and ar
    if mode == "simple":
        assert max_k is not None
        sampling.set_mode("simple", max_k=max_k)

    assignments = []

    for fam in dry_run_families:
        for i in range(n_sim):
            founders = sampling.get_unrelated_founders(len(fam))
            for sim_id, sample_id in zip(fam, founders):
                assignments.append([sim_id, sample_id, i+1])

    founder_df = pd.DataFrame(assignments, columns=["sim_id", "sample_id", "fam_idx"])
    
    founder_df["sim_id"] = founder_df.apply(lambda x: update_fam_idx(x.sim_id, x.fam_idx), axis=1)

    return founder_df.drop("fam_idx", axis=1)
