from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import load_structure
from loguru import logger
from scipy.spatial import cKDTree

########################################################
# Functions for ipSAE calculation
# Following the original ipSAE paper: https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1.full.pdf
# As well as some hyperparameter selections from: https://www.biorxiv.org/content/10.1101/2025.08.14.670059v1.full.pdf
########################################################


def _is_ligand_chain(struct: AtomArray, chain_id: str) -> bool:
    """Check if a chain is a ligand (any atoms are HETATM)."""
    chain_atoms = struct[struct.chain_id == chain_id]
    if len(chain_atoms) == 0:
        return False
    return bool(np.any(chain_atoms.hetero))


def _get_chain_representative_atoms(struct: AtomArray, chain_id: str, protein_atom_mode: str = "ca") -> AtomArray:
    """Get representative atoms for a chain for interaction checking.

    Ligands (hetero=True): always all atoms.
    Proteins: CA atoms (default) or all atoms.

    Args:
        struct: Full AtomArray
        chain_id: Chain to extract
        protein_atom_mode: "ca" (default) or "all_atom" for protein chains.
            Ligand chains always use all atoms regardless of this setting.

    Returns:
        Filtered AtomArray for the chain
    """
    chain_atoms = struct[struct.chain_id == chain_id]
    if _is_ligand_chain(struct, chain_id):
        return chain_atoms
    if protein_atom_mode == "all_atom":
        return chain_atoms
    elif protein_atom_mode == "ca":
        return chain_atoms[chain_atoms.atom_name == "CA"]
    else:
        raise ValueError(f"Invalid protein atom mode: {protein_atom_mode}")


def _get_chain_token_count(struct: AtomArray, chain_id: str) -> int:
    """Get the number of tokens for a chain (matches PAE matrix dimension).

    Proteins: number of residues (= number of CA atoms).
    Ligands (hetero=True): number of atoms (each ligand atom is a token in AF3/RF3/Boltz2).
    """
    chain_atoms = struct[struct.chain_id == chain_id]
    if _is_ligand_chain(struct, chain_id):
        return len(chain_atoms)
    return len(chain_atoms[chain_atoms.atom_name == "CA"])


def two_chains_interacting(
    struct: AtomArray,
    target_chain: str = "A",
    binder_chain: str = "B",
    cutoff: float = 8.0,
    protein_atom_mode: str = "ca",
) -> bool:
    """Check if the target chain is interacting with the binder chain.

    Auto-detects ligand chains (hetero=True) and uses all atoms for them.
    For protein chains, uses CA atoms by default or all atoms.

    Args:
        struct: AtomArray containing the structure
        target_chain: Chain ID of the target
        binder_chain: Chain ID of the binder
        cutoff: Distance cutoff in Angstroms for interface definition
        protein_atom_mode: "ca" (default) or "all_atom" for protein chains.

    Returns:
        True if at least one atom of the target chain is within cutoff
        distance of the binder chain, False otherwise
    """
    target = _get_chain_representative_atoms(struct, target_chain, protein_atom_mode)
    binder = _get_chain_representative_atoms(struct, binder_chain, protein_atom_mode)

    if len(target) == 0 or len(binder) == 0:
        return False

    binder_tree = cKDTree(binder.coord)
    target_tree = cKDTree(target.coord)

    pairs = target_tree.query_ball_tree(binder_tree, cutoff)
    binder_interface_atoms = set(sum(pairs, []))

    return len(binder_interface_atoms) > 0


def get_chain_mask(sequence_lengths, chain_idx):
    """
    Args:
        sequence_lengths (list): list of sequence lengths
        chain_idx (int): chain index
    Returns:
        mask (torch.Tensor): (sum(sequence_lengths),)
    """
    masks = []
    for i, length in enumerate(sequence_lengths):
        if i == chain_idx:
            masks.append(torch.ones(length))
        else:
            masks.append(torch.zeros(length))
    return torch.cat(masks)


## Following 2 functions modified from the colabdesign implementation
def calc_d0(L):
    """
    Eq.2 in the ipSAE paper: https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1.full.pdf
    Args:
        L (torch.Tensor): sum of PAE matrix masks (the two chains and passed cutoff) along axis 1
    Returns:
        d0 (torch.Tensor): d0 value
    """
    L = torch.clip(L, min=27)
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return torch.clip(d0, min=1.0)


def ipSAE(
    pae_matrix: torch.Tensor,
    maskA: torch.Tensor,
    maskB: torch.Tensor,
    cutoff: float = 10.0,
) -> dict[str, float]:  # {'max': ipase_max, 'min': ipase_min}
    """
    Compute the ipSAE between two chains given the pae matrix and the masks.
    Args:
        pae_matrix (torch.Tensor): (L, L)
        maskA (torch.Tensor): (L) mask for chain A
        maskB (torch.Tensor): (L) mask for chain B
        cutoff (float): cutoff for pae
    Returns:
        Dict[str, float]: the minimum, maximum, and average ipSAE between the two chains.
    """
    # Move tensors to the same device
    maskA = maskA.to(pae_matrix.device)
    maskB = maskB.to(pae_matrix.device)

    pae_matrix.shape[0]

    # B -> A
    pae_mask_2d = maskB[:, None] * maskA[None, :] * (pae_matrix < cutoff)  # (L, L)
    d0 = calc_d0(pae_mask_2d.sum(axis=1))  # (L)
    tm_term = 1.0 / (1 + torch.square(pae_matrix) / torch.square(d0)[:, None])  # (L, L)
    mean_tm_term = (pae_mask_2d * tm_term).sum(axis=1) / (pae_mask_2d.sum(axis=1) + 1e-8)
    ipsae_ba = mean_tm_term.max()  # don't need to take index based on mask_1b, as other part will be zero

    # A -> B
    pae_mask_2d = maskA[:, None] * maskB[None, :] * (pae_matrix < cutoff)  # (L, L)
    d0 = calc_d0(pae_mask_2d.sum(axis=1))
    tm_term = 1.0 / (1 + torch.square(pae_matrix) / torch.square(d0)[:, None])  # (L, L)
    mean_tm_term = (pae_mask_2d * tm_term).sum(axis=1) / (pae_mask_2d.sum(axis=1) + 1e-8)
    ipsae_ab = mean_tm_term.max()

    ipase_min = torch.minimum(ipsae_ab, ipsae_ba).item()
    ipase_max = torch.maximum(ipsae_ab, ipsae_ba).item()
    return {"max": ipase_max, "min": ipase_min, "avg": (ipase_max + ipase_min) / 2}


### Compute the complex ipSAE
def complex_ipSAE(
    pae_matrix: torch.Tensor,
    pdb_file_path: str | Path,
    interaction_cutoff: float = 8.0,
    pae_cutoff: float = 10.0,
    protein_atom_mode: str = "ca",
) -> dict[str, float]:  # {'max': ipase_max, 'min': ipase_min, 'avg': (ipase_max + ipase_min) / 2}
    """
    Compute the complex ipSAE given the pae matrix and the pdb file path.
    Only consider the chains that are interacting with the binder chain.
    For each min, max, and average ipSAE, compute the mean over all chain pairs.
    As instructed in page 15 of https://www.biorxiv.org/content/10.1101/2025.08.14.670059v1.full.pdf

    Auto-detects ligand chains (hetero=True) and uses all atoms for interaction
    checking and token counting. Protein chains use CA atoms by default.

    Args:
        pae_matrix (torch.Tensor): (L, L)
        pdb_file_path (Union[str, Path]): path to the pdb/cif file containing the complex
        interaction_cutoff (float): cutoff for interaction
        pae_cutoff (float): cutoff for pae
        protein_atom_mode (str): "ca" (default) or "all_atom" for protein chains
            during interaction checking. Ligand chains always use all atoms.
    Returns:
        Dict[str, float]: the minimum, maximum, and average ipSAE of the complex.
    """
    struct = load_structure(pdb_file_path)
    # Handle AtomArrayStack by taking first structure if needed
    if isinstance(struct, AtomArrayStack):
        struct = struct[0]

    ## Assume the last chain is the binder chain
    all_chains = np.unique(struct.chain_id).tolist()
    all_chains = sorted(all_chains)

    # Token count per chain: CA count for proteins, all-atom count for ligands
    seq_len_list = [_get_chain_token_count(struct, c) for c in all_chains]

    target_chains = all_chains[:-1]
    binder_chain = all_chains[-1]

    ipsae_list = []
    for i in range(len(target_chains)):
        target_chain = target_chains[i]
        interacting_with_binder = two_chains_interacting(
            struct,
            target_chain,
            binder_chain,
            interaction_cutoff,
            protein_atom_mode=protein_atom_mode,
        )
        if interacting_with_binder:
            maskA = get_chain_mask(seq_len_list, i)
            maskB = get_chain_mask(seq_len_list, len(seq_len_list) - 1)
            chain_pair_ipsae = ipSAE(pae_matrix, maskA, maskB, pae_cutoff)
            ipsae_list.append(chain_pair_ipsae)
        else:
            logger.warning(
                f"Target chain {target_chain} not interacting with binder chain {binder_chain} "
                f"(cutoff={interaction_cutoff}A), skipping ipSAE for this pair"
            )

    if not ipsae_list:
        logger.warning("No interacting target-binder chain pairs found, returning zero ipSAE")
        return {"min": 0.0, "max": 0.0, "avg": 0.0}

    complex_ipsae = {
        "min": float(np.mean([ipsae["min"] for ipsae in ipsae_list])),
        "max": float(np.mean([ipsae["max"] for ipsae in ipsae_list])),
        "avg": float(np.mean([ipsae["avg"] for ipsae in ipsae_list])),
    }
    return complex_ipsae
