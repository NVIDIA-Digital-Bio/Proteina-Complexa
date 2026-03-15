"""
Shared metric utilities for protein structure evaluation.

This module consolidates commonly used metric functions:
- rmsd_metric: RMSD computation between structures
- relax_protein: Amber-based protein relaxation
- replace_seq_in_generated_pdb: Sequence replacement in PDB files
- get_interface_residues: Interface residue detection
"""

import os
from typing import Literal

import numpy as np
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack
from biotite.structure.io import load_structure, save_structure
from jaxtyping import Bool, Float
from loguru import logger
from openfold.np.protein import from_pdb_string
from scipy.spatial import cKDTree
from torch import Tensor
from transformers import logging as hf_logging

from proteinfoundation.utils.align_utils import kabsch_align_ind
from proteinfoundation.utils.coors_utils import get_atom37_bb3_mask, get_atom37_bb3o_mask, get_atom37_ca_mask

hf_logging.set_verbosity_error()


# =============================================================================
# RMSD Computation
# =============================================================================


def rmsd_metric(
    coors_1_atom37: Float[Tensor, "n 37 3"],
    coors_2_atom37: Float[Tensor, "n 37 3"],
    mask_atom_37: Bool[Tensor, "n 37"] | None = None,
    mode: Literal["ca", "bb3o", "bb3", "all_atom"] = "ca",
    align: bool = True,
    residue_indices: list[int] | None = None,
) -> Float[Tensor, ""]:
    """
    Computes RMSD between two protein structures in the Atom37 represnetation.
    For now we only use mask to check whether we have all required atoms.

    Args:
        coors_1_atom37: First structure, shape [n, 37, 3]
        coors_2_atom37: Second structure, shape [n, 37, 3]
        mask_atom37: Binary mask of first structure, shape [n, 37]. If not provided
            defaults to all n residues present, and only allows modes "ca", "bb3o" or
            "all_atom" (see below).
        mode: Modality to use, options are
            "ca": only alpha carbon
            "bb3o": four backbone atoms (N, CA, C, O)
            "bb3": three backbone atoms (N, CA, C)
            "all_atom": atoms indicated by the atom37 mask
        align: Whether to align pointclouds before computing RMSD.
        residue_indices: Optional list of residue indices to compute RMSD over.
            If provided, only these residues will be included in the calculation.

    Returns:
        RMSD value, as a Torch (float) tensor with a single element
    """
    assert coors_1_atom37.shape == coors_2_atom37.shape
    assert coors_1_atom37.shape[-1] == 3
    assert coors_1_atom37.shape[-2] == 37
    assert coors_1_atom37.ndim == 3
    n = coors_1_atom37.shape[0]

    if mask_atom_37 is not None:
        assert mask_atom_37.shape == coors_1_atom37.shape[:-1]
    else:
        assert mode != "all_atom", "`all_atom` mode not accepted for `rmsd_metric` when mask is not provided"
        mask_atom_37 = torch.zeros((n, 37), device=coors_1_atom37.device, dtype=torch.bool)
        mask_atom_37[:, :3] = True  # [N CA C]
        mask_atom_37[:, 4] = True  # [O]

    # Which atoms to select, recall atom37 order [N, CA, C, CB, O, ...]
    if mode == "ca":
        mask_f = get_atom37_ca_mask(n=n, device=coors_1_atom37.device)
    elif mode == "bb3":
        mask_f = get_atom37_bb3_mask(n=n, device=coors_1_atom37.device)
    elif mode == "bb3o":
        mask_f = get_atom37_bb3o_mask(n=n, device=coors_1_atom37.device)
    elif mode == "all_atom":
        mask_f = torch.ones((n, 37), device=coors_1_atom37.device, dtype=torch.bool)
    else:
        raise OSError(f"Mode {mode} for RMSD not valid")
    mask_atom_37 = mask_atom_37 * mask_f  # Keeps only requested atoms

    # If residue_indices is provided, create a residue mask and apply it
    if residue_indices is not None:
        residue_mask = torch.zeros(n, device=coors_1_atom37.device, dtype=torch.bool)
        residue_mask[residue_indices] = True
        # Apply residue mask to atom mask
        mask_atom_37 = mask_atom_37 & residue_mask.unsqueeze(1)

    coors_1 = coors_1_atom37[mask_atom_37, :]  # [num of atoms, 3]
    coors_2 = coors_2_atom37[mask_atom_37, :]  # [num of atoms, 3]

    if align:
        coors_1, coors_2 = kabsch_align_ind(coors_1, coors_2, ret_both=True)

    sq_err = (coors_1 - coors_2) ** 2
    return sq_err.sum(dim=-1).mean().sqrt().item()


# =============================================================================
# Protein Relaxation
# =============================================================================


def relax_protein(
    unrelaxed_pdb_path: str,
    output_directory: str,
    output_name: str,
    model_device: str = "cuda:0",
) -> str:
    """Relaxes a protein structure using Amber molecular dynamics.

    This function takes an unrelaxed protein structure and performs energy minimization
    using the Amber force field through OpenMM. The relaxation process helps remove
    steric clashes and optimize the structure geometry.

    Args:
        unrelaxed_pdb_path: Path to the input PDB file containing the unrelaxed structure.
        output_directory: Directory where the relaxed structure will be saved.
        output_name: Base name for the output file (without extension).
        model_device: Device to use for computation. Defaults to "cuda:0".

    Returns:
        Path to the relaxed PDB file.

    Raises:
        RuntimeError: If the relaxation process fails.
    """
    from openfold.np.relax import relax

    os.makedirs(output_directory, exist_ok=True)
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        exclude_residues=[],
        max_iterations=0,
        max_outer_iterations=20,
        stiffness=10.0,
        tolerance=2.39,
    )
    unrelaxed_protein = from_pdb_string(open(unrelaxed_pdb_path).read())

    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    # the struct_str will contain either a PDB-format or a ModelCIF format string
    struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    # Save the relaxed PDB.
    suffix = "_relaxed.pdb"
    relaxed_output_path = os.path.join(output_directory, f"{output_name}{suffix}")
    with open(relaxed_output_path, "w") as fp:
        fp.write(struct_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")

    return relaxed_output_path


# =============================================================================
# Sequence Replacement
# =============================================================================


def replace_seq_in_generated_pdb(
    target_pdb_path: str,
    target_pdb_chain: list[str],
    gen_pdb_path: str,
    gen_pdb_target_chain: list[str] | str,
    output_path: str,
) -> str:
    """Replaces sequence in the generated PDB file with the target sequence.

    This function takes a target PDB structure and a generated PDB structure,
    then replaces the sequence of the specified chain in the generated structure
    with the sequence from the target structure while preserving the coordinates.

    Args:
        target_pdb_path: Path to the target PDB file containing the desired sequence.
        target_pdb_chain: Chain identifier in the target PDB file.
        gen_pdb_path: Path to the generated PDB file to be modified.
        gen_pdb_target_chain: Chain identifier in the generated PDB file to replace.
        output_path: Path where the modified PDB file will be saved.

    Returns:
        The output path where the modified structure was saved.

    Raises:
        AssertionError: If the generated PDB doesn't contain only backbone atoms
            or if sequence lengths don't match.
    """
    logger.info(
        f"Replacing sequence in generated PDB with target sequence for {gen_pdb_path} and target {target_pdb_path}"
    )
    target_pdb = load_structure(target_pdb_path)
    if isinstance(target_pdb, AtomArrayStack):
        target_pdb = target_pdb[0]
    target_pdb = target_pdb[target_pdb.atom_name == "CA"]
    target_pdb = target_pdb[np.isin(target_pdb.chain_id, target_pdb_chain)]
    target_seq = target_pdb.res_name

    gen_pdb = load_structure(gen_pdb_path)
    gen_pdb = gen_pdb[gen_pdb.atom_name == "CA"]
    assert (gen_pdb.atom_name == "CA").all(), "backbone atoms only for generated PDB"
    assert (np.isin(gen_pdb.chain_id, gen_pdb_target_chain)).sum() == len(target_seq), (
        "sequence length mismatch when replacing"
    )

    seq = gen_pdb.res_name
    seq[np.isin(gen_pdb.chain_id, gen_pdb_target_chain)] = target_seq
    gen_pdb.set_annotation("res_name", seq)

    save_structure(output_path, gen_pdb)
    return output_path


# =============================================================================
# Interface Residue Detection
# =============================================================================


def get_interface_residues(
    pdb_file_path: str,
    binder_chain: str = "B",
    cutoff: float = 8.0,
) -> list[int]:
    """Get interface residues on the binder chain that are within cutoff distance of target chain.

    This function identifies interface residues using a KD-tree based approach for efficient
    distance calculations, similar to the reference implementation.

    Args:
        pdb_file_path: Path to PDB file containing both target and binder
        binder_chain: Chain ID of the binder protein
        cutoff: Distance cutoff in Angstroms for interface definition

    Returns:
        List of 0-indexed residue indices on the binder chain that are at the interface
    """
    struct = load_structure(pdb_file_path)

    if isinstance(struct, AtomArrayStack):
        struct = struct[0]

    struct_ca = struct[struct.atom_name == "CA"]

    binder = struct_ca[struct_ca.chain_id == binder_chain]
    target = struct_ca[struct_ca.chain_id != binder_chain]

    binder_tree = cKDTree(binder.coord)
    target_tree = cKDTree(target.coord)

    pairs = target_tree.query_ball_tree(binder_tree, cutoff)

    binder_interface_atoms = np.array(sorted(list(set(sum(pairs, [])))))

    if len(binder_interface_atoms) == 0:
        logger.warning(f"No interface residues found with cutoff {cutoff}Å")
        return []

    binder_interface = np.unique(binder[binder_interface_atoms].res_id)

    offset = int(binder.res_id.min())

    interface_residues = []
    for res in binder_interface:
        residue_idx = int(res) - offset
        interface_residues.append(residue_idx)

    logger.info(f"Found {len(interface_residues)} interface residues: {interface_residues}")

    return sorted(interface_residues)


def get_interface_residues_atomistic(
    pdb_file_path: str,
    binder_chain: str = "B",
    cutoff: float = 6.0,
) -> list[int]:
    """Get interface residues using all atoms (not just CA) for distance calculation.

    Args:
        pdb_file_path: Path to PDB file containing both target and binder
        binder_chain: Chain ID of the binder protein
        cutoff: Distance cutoff in Angstroms for interface definition

    Returns:
        List of 0-indexed residue indices on the binder chain that are at the interface
    """
    struct = load_any(pdb_file_path)[0]

    binder = struct[(struct.atom_name == "CA") & (struct.chain_id == binder_chain)]
    target = struct[struct.chain_id != binder_chain]

    binder_tree = cKDTree(binder.coord)
    target_tree = cKDTree(target.coord)

    pairs = target_tree.query_ball_tree(binder_tree, cutoff)

    binder_interface_atoms = np.array(sorted(list(set(sum(pairs, [])))))

    if len(binder_interface_atoms) == 0:
        logger.warning(f"No interface residues found with cutoff {cutoff}Å")
        return []

    binder_interface = np.unique(binder[binder_interface_atoms].res_id)

    offset = int(binder.res_id.min())

    interface_residues = []
    for res in binder_interface:
        residue_idx = int(res) - offset
        interface_residues.append(residue_idx)

    logger.info(f"Found {len(interface_residues)} interface residues: {interface_residues}")

    return sorted(interface_residues)
