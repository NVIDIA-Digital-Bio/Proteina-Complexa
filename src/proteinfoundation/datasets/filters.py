import gzip
import os
import pickle
import tempfile
from multiprocessing import Pool

import mdtraj as md
import numpy as np
from loguru import logger
from tqdm import tqdm

from proteinfoundation.datasets.transforms import Data


def structure_statistics_multiprocessing(file_paths, max_workers: int = 16, chunksize: int = 32):
    with Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(structure_statistics, file_paths, chunksize=chunksize),
                total=len(file_paths),
                desc="Computing Structure Statistics",
                unit="file",
            )
        )
    pdb_struct_stats = {}
    for struct_stats in results:
        pdb_struct_stats.update(struct_stats)
    return pdb_struct_stats


def structure_statistics(fname):
    pdb_struct_stats = {}
    pdb_id = os.path.basename(fname).split(".")[0]

    # Load structure statistics if computed
    pkl_file = os.path.join(os.path.dirname(fname), pdb_id + "_struct_stats.pkl")
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as fin:
            struct_stats = pickle.load(fin)
    else:
        suffix = os.path.splitext(fname)
        if suffix[1] == ".gz":
            # Step 1: Decompress the .gz file
            with gzip.open(fname, "rt") as file:
                content = file.read()

            # Step 2: Write the decompressed content to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=os.path.splitext(suffix[0])[1])
            temp_file.write(content.encode("utf-8"))
            temp_file_path = temp_file.name
        else:
            temp_file_path = fname

        try:
            traj = md.load(temp_file_path)
            pdb_ss = md.compute_dssp(traj, simplified=True)
            a = np.sum(pdb_ss == "C")
            b = np.sum(pdb_ss == "H")
            c = np.sum(pdb_ss == "E")
            d = np.sum(pdb_ss == "NA")
            tot = a + b + c
            pdb_coil_percent = a / tot
            pdb_helix_percent = b / tot
            pdb_strand_percent = c / tot
            pdb_rog = md.compute_rg(traj)[0]
        except Exception:
            pdb_helix_percent = 0.0
            pdb_strand_percent = 0.0
            pdb_coil_percent = 0.0
            pdb_rog = 0.0
        struct_stats = {
            "helix_percent": pdb_helix_percent,
            "strand_percent": pdb_strand_percent,
            "coil_percent": pdb_coil_percent,
            "rog": pdb_rog,
        }

        with open(pkl_file, "wb") as fout:
            pickle.dump(struct_stats, fout)

    pdb_struct_stats[pdb_id] = struct_stats
    return pdb_struct_stats


def ChainBreakFilter(graph: Data, max_chain_breaks: int = 0) -> bool:
    """Filter out proteins with a certain number of chain breaks.

    Args:
        graph (Data): _description_
        max_chain_breaks (int, optional): _description_. Defaults to 1.

    Returns:
        bool: _description_
    """
    if graph.chain_breaks:
        return graph.chain_breaks <= max_chain_breaks
    else:
        logger.error(
            "graph.chain_breaks not present; to use ChainBreakFilter as a filter, \
                     call ChainBreakCountingTransform as a transform before."
        )


def MinContactFilter(graph: Data, min_contact_ratio: int = 0.5) -> bool:
    """Filter out proteins with too few long-range contacts.
    For this filter to work,

    Args:
        graph (Data): _description_
        min_contact_ratio (int, optional): _description_. Defaults to 0.5.

    Returns:
        bool: _description_
    """
    if graph.contacts:
        num_residues = len(graph.coords)
        return graph.contacts >= min_contact_ratio * num_residues
    else:
        logger.error(
            "graph.contacts not present; to use MinContactFilter as a filter, \
                     call ContactsCountingTransform as a transform before."
        )


def CathCodeFilter(graph: Data) -> bool:
    """Filter out proteins without CATH code.
    For this filter to work, the graph must have a cath_code attribute (set when setting up PDB manager via label attribute).
    """

    return graph.cath_code
