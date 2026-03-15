import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from biotite.structure import AtomArray
from loguru import logger
from torch import Tensor
from transformers import logging as hf_logging

from proteinfoundation.metrics.inverse_folding_models import inverse_fold
from proteinfoundation.metrics.metric_utils import (
    get_interface_residues,
    get_interface_residues_atomistic,
    replace_seq_in_generated_pdb,
    rmsd_metric,
)
from proteinfoundation.utils.align_utils import kabsch_align_ind, kabsch_align_ligand
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb, pdb_name_from_path, sort_AtomArray_by_chain_id

hf_logging.set_verbosity_error()


def run_binder_eval(
    pdb_file_path: str | Path,
    target_pdb_path: str | Path,
    folding_model_specs: dict[str, dict[str, str]],
    tmp_path: str | Path = "./tmp/metrics/",
    target_pdb_chain: list[str] = ["A"],
    sequence_types: list[Literal["mpnn", "mpnn_fixed", "self"]] = ["self"],
    interface_cutoff: float = 8.0,
    is_target_ligand: bool = False,
    inverse_folding_model: str = "protein_mpnn",
    gen_target_chain: list[str] = None,  # If none, use target_pdb_chain as gen_target_chain
    binder_chain: str = None,  # If none, use the last chain id in the refolded complex
    num_redesign_seqs: int = None,  # If none, default to 8 for protein targets, 1 for ligand targets
    fixed_residues_override: list[str] | None = None,
) -> dict[str, list[dict[str, dict]]]:
    """Evaluates protein binder designs using inverse folding models and folding models.

    This function performs a comprehensive evaluation of protein binder designs by:
    1. Generating sequences using ProteinMPNN/LigandMPNN (if "mpnn" in sequence_types)
    2. Generating sequences using ProteinMPNN/LigandMPNN with interface residues fixed (if "mpnn_fixed" in sequence_types)
    3. Using self-generated sequences from PDB file (if "self" in sequence_types)
    4. Evaluating complexes using chosen folding model with target structure as template
    5. Computing RMSD metrics between different states

    Args:
        pdb_file_path: Path to input PDB file containing the target structure
        target_pdb_path: Path to input PDB file containing the target structure
        model_info: Dictionary containing information about the folding models
            model_info should be in the format:
            {
                "model_name": {"colabdesign", "RF3", "PTX"}, # Model name must be provided
                "runner": {RF3RewardRunner, PTXRewardRunner}, # Runner must be provided for RF3 and PTX
                "filter_path": PATH/TO/FILTER/FILE, # Filter path must be provided for colabdesign
                "target_msa_paths": [PATH/TO/MSA/FILE], # Target MSA paths must be provided for PTX
            }
        tmp_path: Directory to store temporary files from ProteinMPNN and AlphaFold
            Default: "./tmp/metrics/"
        target_pdb_chain: Chain identifier for the target in the original target PDB file
        binder_chain: Chain identifier for the binder in the PDB file
        sequence_types: List of sequence types to evaluate. Options:
            - "mpnn": Use ProteinMPNN redesigned sequences
            - "mpnn_fixed": Use ProteinMPNN redesigned sequences with interface residues fixed
            - "self": Use self-generated sequences from the PDB file
        interface_cutoff: Distance cutoff in Angstroms for interface definition (for mpnn_fixed)
            Default: 8.0
        is_target_ligand: Whether the target is a ligand
            Default: False
        gen_target_chain: Chain identifier for the target in the generated PDB file
            Default: None
            if None, use target_pdb_chain as gen_target_chain
        fixed_residues_override: Optional list of residue positions to fix during
            ``mpnn_fixed`` inverse folding, in ``["B45", "B46"]`` format.
            When provided, these positions are used *instead* of computing
            interface residues.  Useful for motif binder evaluation where
            the motif residues (not interface residues) should be fixed.
            Default: None (use interface-based detection)

    Returns:
        Dictionary containing the evaluation results
            - complex_statistics: List of dicts with chosen folding model metrics for each sequence
                Format: [{"mpnn_seq_1": {"model_1": {"pLDDT": float, "pTM": float, ...}}}, ...]
                All sequences are returned with prefixes to distinguish their type
            - binder_statistics: List of dicts with chosen folding model metrics for each sequence
                Format: [{"mpnn_seq_1": {"model_1": {"pLDDT": float, "pTM": float, ...}}}, ...]
                All sequences are returned with prefixes to distinguish their type
            - rmsd_results: List of dicts with RMSD metrics for each sequence
                Format: [{"mpnn_seq_1": {"binder_scRMSD": float, "binder_bound_unbound_RMSD": float}}, ...]
                All sequences are returned with prefixes to distinguish their type
            - filter_pass: List of dicts indicating which sequences passed the filters
                Format: [{"mpnn_seq_1": bool}, ...]
                All sequences are returned with prefixes to distinguish their type
    """

    model_name = folding_model_specs["model_name"]
    assert model_name in [
        "colabdesign",
        "RF3",
        "PTX",
        "BOLTZ2",
    ], f"Folding model {model_name} not supported"
    if is_target_ligand and model_name == "colabdesign":
        raise ValueError("Colabdesign does not support ligand targets")

    # Check if sequence types are valid
    valid_types = {"mpnn", "mpnn_fixed", "self"}
    invalid_types = set(sequence_types) - valid_types
    if invalid_types:
        raise ValueError(f"Invalid sequence types: {invalid_types}. Valid types are: {valid_types}")

    name = pdb_name_from_path(pdb_file_path)
    updated_pdb_path = os.path.join(os.path.dirname(pdb_file_path), name + "_updated.pdb")
    # Determine chain IDs
    # sort target_pdb_chain alphabetically to be sure that the first chain is the starting chain
    target_pdb_chain = sorted(target_pdb_chain)
    # If gen_target_chain is not provided, use target_pdb_chain as gen_target_chain
    if gen_target_chain is None:
        gen_target_chain = target_pdb_chain
    starting_chain_id = target_pdb_chain[0]
    # If binder_chain is not provided, use the last chain id in the refolded complex
    if binder_chain is None:
        all_chain_ids = [
            chr(ord(starting_chain_id) + i) for i in range(len(target_pdb_chain) + 1)
        ]  # target chains + binder chain
        binder_chain = all_chain_ids[-1]

    if not is_target_ligand:
        ### Updated pdb file has 2 changes:
        ### 1. Replace the sequence of target chains with the original target sequence
        ### 2. Only contains C-alpha atoms
        replace_seq_in_generated_pdb(
            target_pdb_path=target_pdb_path,
            target_pdb_chain=target_pdb_chain,
            gen_pdb_path=pdb_file_path,
            gen_pdb_target_chain=gen_target_chain,
            output_path=updated_pdb_path,
        )
    else:
        updated_pdb_path = updated_pdb_path.replace("_updated.pdb", ".pdb")

    logger.info(f"Binder chain ID: {binder_chain}")
    logger.info(f"Target chain IDs: {target_pdb_chain}, Is ligand: {is_target_ligand}")

    # Prepare sequences for evaluation
    all_sequences = []
    sequence_types_list = []  # Track the type of each sequence
    sequences_dict = defaultdict(list)
    all_interface_residues = []

    ## Defining target-specific inverse folding arguments
    assert inverse_folding_model in [
        "protein_mpnn",
        "ligand_mpnn",
        "soluble_mpnn",
    ], f"Inverse folding model {inverse_folding_model} not supported"
    if is_target_ligand:  # For ligand targets, use LigandMPNN, and full atom input
        inverse_folding_model = "ligand_mpnn"  # Set to ligand_mpnn anyway for ligand targets
        mpnn_input_pdb = updated_pdb_path.replace("_updated.pdb", ".pdb")
    else:
        if inverse_folding_model == "soluble_mpnn":  # Use full atom input for SolubleMPNN
            mpnn_input_pdb = updated_pdb_path.replace("_updated.pdb", ".pdb")
        else:  # For C-alpha only input for ProteinMPNN
            mpnn_input_pdb = updated_pdb_path

    if num_redesign_seqs is None:
        num_redesign_seqs = 8 if not is_target_ligand else 1
    get_interface_residues_func = get_interface_residues_atomistic if is_target_ligand else get_interface_residues

    if "mpnn" in sequence_types:
        logger.info(f"Running inverse folding: {inverse_folding_model}")
        # Use a unique output directory for mpnn
        mpnn_tmp_path = os.path.join(tmp_path, "mpnn")
        os.makedirs(mpnn_tmp_path, exist_ok=True)

        mpnn_sequences = inverse_fold(
            model_type=inverse_folding_model,
            pdb_file_path=mpnn_input_pdb,
            out_dir_root=mpnn_tmp_path,
            all_chains=gen_target_chain + [binder_chain],
            pdb_path_chains=[binder_chain],
            fix_pos=None,
            num_seq_per_target=num_redesign_seqs,
            omit_AAs=["C"],
            sampling_temp=0.1,
            verbose=False,
        )
        sequences_dict["mpnn"].extend(mpnn_sequences)
        all_sequences.extend(mpnn_sequences)
        sequence_types_list.extend(["mpnn"] * len(mpnn_sequences))
        # Compute interface residues for mpnn (if needed, else skip)
        interface_residues_mpnn = get_interface_residues_func(updated_pdb_path, binder_chain, interface_cutoff)
        all_interface_residues.extend([interface_residues_mpnn] * len(mpnn_sequences))

    if "mpnn_fixed" in sequence_types:
        # Create a separate directory for mpnn_fixed to avoid conflicts
        mpnn_fixed_tmp_path = os.path.join(tmp_path, "mpnn_fixed")
        os.makedirs(mpnn_fixed_tmp_path, exist_ok=True)

        # Determine fixed positions: use override (e.g. motif residues) or
        # fall back to interface-based detection.
        if fixed_residues_override is not None:
            fix_pos = fixed_residues_override
            logger.info(
                f"Running inverse folding: {inverse_folding_model} with "
                f"overridden fixed residues ({len(fix_pos)} positions)"
            )
        else:
            # Default: fix interface residues (standard binder eval)
            interface_residues = get_interface_residues_func(updated_pdb_path, binder_chain, interface_cutoff)
            logger.info(
                f"Running inverse folding: {inverse_folding_model} with "
                f"{len(interface_residues)} interface residues fixed"
            )
            # Convert to fix_pos format: ["ChainID-ResidueNumber"]
            fix_pos = [f"{binder_chain}{r + 1}" for r in interface_residues]  # Convert to 1-indexed

        # Always compute interface residues for AA composition tracking,
        # even when fix_pos was overridden.
        interface_residues_for_tracking = get_interface_residues_func(updated_pdb_path, binder_chain, interface_cutoff)

        mpnn_fixed_sequences = inverse_fold(
            model_type=inverse_folding_model,
            pdb_file_path=mpnn_input_pdb,
            out_dir_root=mpnn_fixed_tmp_path,
            all_chains=gen_target_chain + [binder_chain],
            pdb_path_chains=[binder_chain],
            fix_pos=fix_pos,
            num_seq_per_target=num_redesign_seqs,
            omit_AAs=["C"],
            sampling_temp=0.1,
            verbose=False,
        )
        sequences_dict["mpnn_fixed"].extend(mpnn_fixed_sequences)
        all_sequences.extend(mpnn_fixed_sequences)
        sequence_types_list.extend(["mpnn_fixed"] * len(mpnn_fixed_sequences))
        all_interface_residues.extend([interface_residues_for_tracking] * len(mpnn_fixed_sequences))

    if "self" in sequence_types:
        logger.info("Running inverse folding: self-generated sequences")
        self_sequence = {"seq": extract_seq_from_pdb(pdb_file_path, chain_id=binder_chain)}
        sequences_dict["self"].append(self_sequence)
        all_sequences.append(self_sequence)
        sequence_types_list.append("self")
        # Compute interface residues for self
        interface_residues_self = get_interface_residues_func(updated_pdb_path, binder_chain, interface_cutoff)
        all_interface_residues.append(interface_residues_self)

    if not all_sequences:
        raise ValueError("No sequences to evaluate. Please specify at least one sequence type.")
    binder_length = len(all_sequences[0]["seq"])

    logger.info("Inverse folding finished")

    if model_name == "colabdesign":
        from proteinfoundation.utils.colabdesign_utils import get_af2_advanced_settings, run_af_eval

        colabdesign_advanced_settings = get_af2_advanced_settings()
        target_settings = {
            "starting_pdb": target_pdb_path,
            "chains": ",".join(gen_target_chain),
        }
        eval_func = run_af_eval
        eval_kwargs = {
            "trajectory_pdb": pdb_file_path,
            "target_settings": target_settings,
            "advanced_settings": colabdesign_advanced_settings,
            "binder_length": binder_length,
            "binder_chain": binder_chain,
            "sequence_type_list": sequence_types_list,  # added this to save the type in the file name
        }
    elif model_name == "RF3":
        from proteinfoundation.utils.rf3_model import run_rf3_eval

        runner = folding_model_specs["runner"]
        eval_func = run_rf3_eval
        eval_kwargs = {
            "rf3_runner": runner,
            "target_chain_ids": gen_target_chain,
            "is_target_ligand": is_target_ligand,
            "sequence_type_list": sequence_types_list,
            # "updated_pdb_path": updated_pdb_path,
            "updated_pdb_path": updated_pdb_path.replace("_updated.pdb", ".pdb"),
            "binder_chain_id": binder_chain,
        }
    else:
        raise NotImplementedError(f"Folding model {model_name} not supported")
    ### Add common arguments
    eval_kwargs.update(
        {
            "binder_sequences": all_sequences,
            "design_name": name,
            "output_path": tmp_path,
        }
    )

    ### Refold and get metrics
    kwargs_to_show = {k: v for k, v in eval_kwargs.items() if k != "binder_sequences"}
    logger.info(f"Running {model_name} for kwargs: {kwargs_to_show}")
    complex_statistics, complex_pdb_paths = eval_func(
        **eval_kwargs,
    )

    ### Load generated structure
    gen_prot = load_any(pdb_file_path)[0]
    ### Compute RMSDs
    rmsd_results = []
    for seq_num, complex_pdb_path in enumerate(complex_pdb_paths):
        seq_type = sequence_types_list[seq_num]
        label = f"{seq_type}_seq_{seq_num + 1}"
        if not is_target_ligand:
            refolded_complex = load_any(complex_pdb_path, file_type="pdb")[0]
            rmsd_result = calculate_prot_prot_binder_rmsd(
                refolded_complex=refolded_complex,
                gen_complex=gen_prot,
                label=label,
            )
        else:
            refolded_complex = load_any(complex_pdb_path, file_type="pdb")[0]
            rmsd_result = calculate_ligand_binder_rmsd(
                refolded_complex=refolded_complex,
                gen_complex=gen_prot,
                ligand_chain_id="A",
                binder_chain_id=binder_chain,
                label=label,
            )
        rmsd_results.append({label: rmsd_result})

    # Add prefixes to complex and binder statistics
    # reordered_complex_stats = {k:[] for k in set(sequence_types)}
    prefixed_complex_stats = []
    sequence_type_stats = {k: {"complex_stats": [], "rmsd_stats": [], "aa_stats": []} for k in set(sequence_types)}

    for seq_num, (complex_stat, seq_type, interface_residues, rmsd_stat) in enumerate(
        zip(
            complex_statistics,
            sequence_types_list,
            all_interface_residues,
            rmsd_results,
            strict=False,
        )
    ):
        # Add prefix to sequence names
        old_seq_name = f"seq_{seq_num + 1}"
        new_seq_name = f"{seq_type}_seq_{seq_num + 1}"

        sequence = all_sequences[seq_num]["seq"]
        # Count all residues
        all_counts = Counter(sequence)
        # Count only interface residues, careful about indexing
        if interface_residues and len(interface_residues) > 0:
            # Interface indices returned by get_interface_residues should match sequence indices: adjust if not
            interface_seq = "".join([sequence[i] for i in interface_residues])
            interface_counts = Counter(interface_seq)
        else:
            interface_counts = {}

        # Add counts to complex stats and preserve complex PDB path if present:
        new_complex_stat = {
            new_seq_name: complex_stat[old_seq_name],
            "residue_counts": dict(all_counts),
            "interface_counts": dict(interface_counts),
        }
        # Preserve complex PDB path if it exists in the original stats
        if "complex_pdb_path" in complex_stat:
            new_complex_stat["complex_pdb_path"] = complex_stat["complex_pdb_path"]
            complex_stat[old_seq_name]["complex_pdb_path"] = complex_stat["complex_pdb_path"]
        prefixed_complex_stats.append(new_complex_stat)
        sequence_type_stats[seq_type]["complex_stats"].append(complex_stat[old_seq_name])
        sequence_type_stats[seq_type]["rmsd_stats"].append(rmsd_stat[new_seq_name])
        sequence_type_stats[seq_type]["aa_stats"].append(
            {
                "residue_counts": dict(all_counts),
                "interface_counts": dict(interface_counts),
                "binder_length": binder_length,
            }
        )

    return (prefixed_complex_stats, rmsd_results, sequence_type_stats, sequences_dict)


def atomarray_to_atom37_coords(
    atomarray: AtomArray,
    chains: list[str],
) -> Tensor:
    """
    Convert an atomarray (selected chains) to atom37 coordinates.
    """
    for chain in chains:
        assert chain in atomarray.chain_id, f"Chain {chain} not found in atomarray.chain_id"
    masks = [atomarray.chain_id == chain for chain in chains]
    # create a mask that's the "or" of all boolean masks in the "masks" list
    if len(masks) == 0:
        subset_atomarray = atomarray
    else:
        mask = masks[0].copy()
        for m in masks[1:]:
            mask = mask | m
        subset_atomarray = atomarray[mask]
    atom37_coords = atom_array_to_encoding(subset_atomarray, encoding=AF2_ATOM37_ENCODING)
    atom37_coords = torch.tensor(atom37_coords["xyz"], dtype=torch.float32).nan_to_num(0.0)
    return atom37_coords


def calculate_prot_prot_binder_rmsd(
    refolded_complex: AtomArray,
    gen_complex: AtomArray,
    label: str = "",
) -> dict[str, float]:
    """
    Calculate the RMSD related metrics for protein-protein complex.

    Computes CA, backbone-3, backbone-3+O, and all-atom RMSD modes for both
    the binder and the full complex, using mode-suffixed key names that are
    consistent with ``calculate_ligand_binder_rmsd``.

    Args:
        refolded_complex: The refolded complex.
        gen_complex: The generated complex.
        label: Identifier for log messages (e.g. "mpnn_fixed_seq_1").
    Returns:
        A dictionary containing the RMSD related metrics.
        Keys include (for each of binder / complex):
            "binder_scRMSD_ca", "binder_scRMSD_bb3", "binder_scRMSD_bb3o",
            "binder_scRMSD_allatom", "complex_scRMSD_ca", etc.
        Legacy keys ("binder_scRMSD", "complex_scRMSD") are also included
        and equal the CA values for backward compatibility.
    """
    # Set the occupancy to 1.0 for all chains
    refolded_complex.set_annotation(
        "occupancy",
        np.array(
            [1.0 for chain in refolded_complex.chain_id],
            dtype=refolded_complex.coord.dtype,
        ),
    )
    gen_complex.set_annotation(
        "occupancy",
        np.array([1.0 for chain in gen_complex.chain_id], dtype=gen_complex.coord.dtype),
    )

    # We always assume the last chain is the binder
    refolded_complex_chains = sorted(list(set(refolded_complex.chain_id)))
    gen_complex_chains = sorted(list(set(gen_complex.chain_id)))
    refolded_binder_chain = refolded_complex_chains[-1]
    gen_binder_chain = gen_complex_chains[-1]

    # Get the atom coordinates of the complex and the binder in the refolded complex
    refolded_complex_coors = atomarray_to_atom37_coords(refolded_complex, refolded_complex_chains)
    refolded_binder_coors = atomarray_to_atom37_coords(refolded_complex, [refolded_binder_chain])
    # Get the atom coordinates of the generated complex and binder
    gen_complex_coors = atomarray_to_atom37_coords(gen_complex, gen_complex_chains)
    gen_binder_coors = atomarray_to_atom37_coords(gen_complex, [gen_binder_chain])

    # Compute binder RMSD in all modes
    binder_scRMSD_ca = rmsd_metric(gen_binder_coors, refolded_binder_coors, mode="ca")
    mask_binder = torch.ones(gen_binder_coors.shape[:-1], device=gen_binder_coors.device, dtype=torch.bool)
    binder_scRMSD_allatom = rmsd_metric(
        gen_binder_coors,
        refolded_binder_coors,
        mode="all_atom",
        mask_atom_37=mask_binder,
    )
    binder_scRMSD_bb3o = rmsd_metric(gen_binder_coors, refolded_binder_coors, mode="bb3o", mask_atom_37=mask_binder)
    binder_scRMSD_bb3 = rmsd_metric(gen_binder_coors, refolded_binder_coors, mode="bb3", mask_atom_37=mask_binder)

    # Compute complex RMSD (CA only — full-complex all-atom is rarely useful)
    complex_scRMSD_ca = rmsd_metric(gen_complex_coors, refolded_complex_coors, mode="ca")

    tag = f"[{label}] " if label else ""
    logger.info(
        f"{tag}binder scRMSD — CA: {binder_scRMSD_ca:.4f}, "
        f"bb3: {binder_scRMSD_bb3:.4f}, bb3o: {binder_scRMSD_bb3o:.4f}, "
        f"all-atom: {binder_scRMSD_allatom:.4f}"
    )
    logger.info(f"{tag}complex scRMSD CA: {complex_scRMSD_ca:.4f}")

    rmsd_result = {
        # Mode-suffixed keys (consistent with calculate_ligand_binder_rmsd)
        "binder_scRMSD_ca": binder_scRMSD_ca,
        "binder_scRMSD_bb3": binder_scRMSD_bb3,
        "binder_scRMSD_bb3o": binder_scRMSD_bb3o,
        "binder_scRMSD_allatom": binder_scRMSD_allatom,
        "complex_scRMSD_ca": complex_scRMSD_ca,
        # Legacy keys for backward compatibility
        "binder_scRMSD": binder_scRMSD_ca,
        "complex_scRMSD": complex_scRMSD_ca,
    }

    return rmsd_result


def calculate_ligand_binder_rmsd(
    refolded_complex: AtomArray,
    gen_complex: AtomArray,
    ligand_chain_id: str = "A",
    binder_chain_id: str = "B",
    label: str = "",
) -> dict[str, float]:
    """
    Calculate the RMSD related metrics for ligand-binder complex. Now we only support 1 ligand chain and 1 binder chain.
    Args:
        refolded_complex: The refolded complex.
        gen_complex: The generated complex.
        ligand_chain_id: The ligand chain id.
        binder_chain_id: The binder chain id.
    Returns:
        A dictionary containing the RMSD related metrics.
        Keys:
            "binder_scRMSD_ca": The RMSD between the generated binder and the refolded binder based on CA atoms.
            "binder_scRMSD_allatom": The RMSD between the generated binder and the refolded binder based on all atoms.
            "ligand_scRMSD": The RMSD between the ligand in the generated complex and the ligand in the refolded complex.
            "ligand_scRMSD_aligned_ca": Align complexes based on binder backbone atoms, then compute RMSD between the ligands in the aligned complexes.
            "ligand_scRMSD_aligned_allatom": Align complexes based on binder all-atoms, then compute RMSD between the ligands in the aligned complexes.
    """
    tag = f"[{label}] " if label else ""

    # Sort the refolded complex and gen complex by chain id
    refolded_complex = sort_AtomArray_by_chain_id(refolded_complex)
    gen_complex = sort_AtomArray_by_chain_id(gen_complex)
    # Set the occupancy to 1.0 for all chains
    refolded_complex.set_annotation(
        "occupancy",
        np.array(
            [1.0 for chain in refolded_complex.chain_id],
            dtype=refolded_complex.coord.dtype,
        ),
    )
    gen_complex.set_annotation(
        "occupancy",
        np.array([1.0 for chain in gen_complex.chain_id], dtype=gen_complex.coord.dtype),
    )

    refolded_binder_coords = atomarray_to_atom37_coords(refolded_complex, [binder_chain_id])

    gen_binder_coords = atomarray_to_atom37_coords(gen_complex, [binder_chain_id])

    # Get the protein binder RMSD (both all-atom and CA)
    binder_scRMSD_ca = rmsd_metric(gen_binder_coords, refolded_binder_coords, mode="ca")
    mask_atom_37 = torch.ones(gen_binder_coords.shape[:-1], device=gen_binder_coords.device, dtype=torch.bool)
    binder_scRMSD_allatom = rmsd_metric(
        gen_binder_coords,
        refolded_binder_coords,
        mode="all_atom",
        mask_atom_37=mask_atom_37,
    )
    binder_scRMSD_bb3o = rmsd_metric(
        gen_binder_coords,
        refolded_binder_coords,
        mode="bb3o",
        mask_atom_37=mask_atom_37,
    )
    binder_scRMSD_bb3 = rmsd_metric(
        gen_binder_coords,
        refolded_binder_coords,
        mode="bb3",
        mask_atom_37=mask_atom_37,
    )
    logger.info(
        f"{tag}binder scRMSD — CA: {binder_scRMSD_ca:.4f}, "
        f"bb3: {binder_scRMSD_bb3:.4f}, bb3o: {binder_scRMSD_bb3o:.4f}, all-atom: {binder_scRMSD_allatom:.4f}"
    )

    # Calculate ligand RMSD if applicable ligand_rmsd
    try:
        ### Firstly remove OXT atoms in the protein binder for both refolded and generated complexes
        gen_complex = gen_complex[
            ((gen_complex.chain_id == binder_chain_id) & (gen_complex.atom_name != "OXT"))
            | (gen_complex.chain_id == ligand_chain_id)
        ]
        refolded_complex = refolded_complex[
            ((refolded_complex.chain_id == binder_chain_id) & (refolded_complex.atom_name != "OXT"))
            | (refolded_complex.chain_id == ligand_chain_id)
        ]

        ### For ligand RMSD
        ligand_atoms = torch.tensor(
            refolded_complex[refolded_complex.chain_id == ligand_chain_id].coord,
            dtype=torch.float32,
        )
        gen_ligand_atoms = torch.tensor(
            gen_complex[gen_complex.chain_id == ligand_chain_id].coord,
            dtype=torch.float32,
        )
        coors_1, coors_2 = kabsch_align_ind(ligand_atoms, gen_ligand_atoms, ret_both=True)
        sq_err = (coors_1 - coors_2) ** 2
        ligand_scRMSD = sq_err.sum(dim=-1).mean().sqrt().item()
        logger.info(f"{tag}ligand scRMSD: {ligand_scRMSD:.4f}")

        ### For ligand RMSD in aligned complexes based on backbone atoms
        ## Select the ligand and binder backbone atoms from the refolded complex
        refolded_complex_bb = refolded_complex[
            ((refolded_complex.chain_id == binder_chain_id) & (refolded_complex.atom_name == "CA"))
            | (refolded_complex.chain_id == ligand_chain_id)
        ]
        gen_complex_bb = gen_complex[
            ((gen_complex.chain_id == binder_chain_id) & (gen_complex.atom_name == "CA"))
            | (gen_complex.chain_id == ligand_chain_id)
        ]
        refolded_complex_bb_coord = torch.tensor(
            refolded_complex_bb.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        gen_complex_bb_coord = torch.tensor(
            gen_complex_bb.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        binder_mask = torch.tensor(
            refolded_complex_bb.chain_id == binder_chain_id,
            dtype=torch.bool,
        )
        ## Align the generated complex to the refolded complex, based only on the binder backbone atoms
        aligned_binder_centered_gen_complex, refolded_binder_centered_complex = kabsch_align_ligand(
            gen_complex_bb_coord.unsqueeze(0),  # [b, n, 3], b=1
            refolded_complex_bb_coord.unsqueeze(0),  # [b, n, 3]
            mask=binder_mask.unsqueeze(0),  # [b, n]
        )
        aligned_binder_centered_gen_complex, refolded_binder_centered_complex = (
            aligned_binder_centered_gen_complex[0],
            refolded_binder_centered_complex[0],
        )
        ## Compute the rmsd between the ligands in the aligned complexes.
        bb_aligned_ligand = aligned_binder_centered_gen_complex[~binder_mask]
        bb_aligned_refolded_ligand = refolded_binder_centered_complex[~binder_mask]
        sq_err = (bb_aligned_ligand - bb_aligned_refolded_ligand) ** 2
        ligand_scRMSD_aligned_bb = sq_err.sum(dim=-1).mean().sqrt().item()
        logger.info(f"{tag}ligand scRMSD aligned CA backbone: {ligand_scRMSD_aligned_bb:.4f}")

        ### For ligand RMSD in aligned complexes based on all atoms
        refolded_complex_allatom_coord = torch.tensor(
            refolded_complex.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        gen_complex_allatom_coord = torch.tensor(
            gen_complex.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        binder_mask = torch.tensor(
            refolded_complex.chain_id == binder_chain_id,
            dtype=torch.bool,
        )
        ## If number of atoms are different (sequence different after redesign), use the aligned backbone rmsd
        ligand_scRMSD_aligned_allatom = ligand_scRMSD_aligned_bb
        logger.debug(
            f"{tag}Atom count: gen={gen_complex_allatom_coord.shape[0]}, "
            f"refolded={refolded_complex_allatom_coord.shape[0]}"
        )
        if gen_complex_allatom_coord.shape[0] == refolded_complex_allatom_coord.shape[0]:
            ## Align the generated complex to the refolded complex, based only on the binder all atoms
            aligned_binder_centered_gen_complex, refolded_binder_centered_complex = kabsch_align_ligand(
                gen_complex_allatom_coord.unsqueeze(0),  # [b, n, 3], b=1
                refolded_complex_allatom_coord.unsqueeze(0),  # [b, n, 3]
                mask=binder_mask.unsqueeze(0),  # [b, n]
            )
            aligned_binder_centered_gen_complex, refolded_binder_centered_complex = (
                aligned_binder_centered_gen_complex[0],
                refolded_binder_centered_complex[0],
            )
            ## Compute the rmsd between the ligands in the aligned complexes.
            allatom_aligned_ligand = aligned_binder_centered_gen_complex[~binder_mask]
            allatom_aligned_refolded_ligand = refolded_binder_centered_complex[~binder_mask]
            sq_err = (allatom_aligned_ligand - allatom_aligned_refolded_ligand) ** 2
            ligand_scRMSD_aligned_allatom = sq_err.sum(dim=-1).mean().sqrt().item()
            logger.info(f"{tag}ligand scRMSD aligned all atoms: {ligand_scRMSD_aligned_allatom:.4f}")

    except Exception as e:
        logger.error(f"{tag}Could not calculate ligand RMSD: {e}")
        ligand_scRMSD = float("inf")
        ligand_scRMSD_aligned_allatom = float("inf")
        ligand_scRMSD_aligned_bb = float("inf")

    rmsd_result = {
        "binder_scRMSD_ca": binder_scRMSD_ca,
        "binder_scRMSD_bb3": binder_scRMSD_bb3,
        "binder_scRMSD_bb3o": binder_scRMSD_bb3o,
        "binder_scRMSD_allatom": binder_scRMSD_allatom,
        "ligand_scRMSD": ligand_scRMSD,
        "ligand_scRMSD_aligned_allatom": ligand_scRMSD_aligned_allatom,
        "ligand_scRMSD_aligned_ca": ligand_scRMSD_aligned_bb,
    }

    return rmsd_result
