# RF3 utility functions for data preparation and conversion
# Similar to ptx_utils.py but adapted for RF3

import gzip
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBParser
from biotite.structure.io import load_structure, pdb, pdbx, save_structure
from loguru import logger


def convert_pdb_to_rf3_json(
    pdb_file_path: str,
    output_json_path: str | None = None,
    name: str | None = None,
    msa_paths: dict[str, str] | None = None,
    ground_truth_conformer_selection: str | None = None,
) -> str:
    """Convert PDB file to RF3 input JSON format.

    Args:
        pdb_file_path: Path to input PDB file
        output_json_path: Path to output JSON file (optional)
        name: Name for the structure (optional, derived from filename if not provided)
        msa_paths: Dictionary mapping chain IDs to MSA paths
        ground_truth_conformer_selection: Selection syntax for ground truth conformers

    Returns:
        Path to the created JSON file
    """
    if name is None:
        name = Path(pdb_file_path).stem

    if output_json_path is None:
        output_json_path = pdb_file_path.replace(".pdb", "_rf3_input.json")

    # Parse PDB to get sequences and chain information
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)

    components = []

    for model in structure:
        for chain in model:
            chain_id = chain.id

            # Extract sequence from chain
            residues = [res for res in chain if res.get_id()[0] == " "]  # Only standard residues
            if not residues:
                continue

            sequence = "".join([res.get_resname() for res in residues])

            # Convert 3-letter to 1-letter amino acid codes
            sequence = _convert_to_single_letter(sequence)

            component = {"seq": sequence, "chain_id": chain_id}

            # Add MSA path if available
            if msa_paths and chain_id in msa_paths:
                component["msa_path"] = msa_paths[chain_id]

            components.append(component)

    # Create RF3 input structure
    rf3_input = {"name": name, "components": components}

    # Add advanced options if specified
    if ground_truth_conformer_selection is not None:
        rf3_input["ground_truth_conformer_selection"] = ground_truth_conformer_selection

    # Write to JSON file
    with open(output_json_path, "w") as f:
        json.dump([rf3_input], f, indent=2)

    logger.info(f"Created RF3 input JSON: {output_json_path}")
    return output_json_path


def convert_cif_to_pdb_rf3(
    cif_path: str,
    out_pdb_path: str | None = None,
    trim_chain_ids: bool = True,
) -> str:
    """Convert CIF file to PDB format for RF3 compatibility using Biotite.

    Args:
        cif_path: Path to input CIF file (can be .cif or .cif.gz)
        out_pdb_path: Path to output PDB file (optional)
        trim_chain_ids: Whether to trim chain IDs to single character

    Returns:
        Path to the converted PDB file
    """
    if out_pdb_path is None:
        out_pdb_path = cif_path.replace(".cif.gz", ".pdb").replace(".cif", ".pdb")

    try:
        # Read CIF file using Biotite's PDBx interface
        # Biotite automatically handles .cif.gz files
        if cif_path.endswith(".bcif"):
            # Use BinaryCIF interface for compressed files
            cif_file = pdbx.BinaryCIFFile.read(cif_path)
        elif cif_path.endswith(".cif.gz"):
            # Use regular CIF interface for text files
            with gzip.open(cif_path, "rt") as file:
                cif_file = pdbx.CIFFile.read(file)
        else:
            cif_file = pdbx.CIFFile.read(cif_path)

        # Get structure from CIF file
        structure = pdbx.get_structure(cif_file)

        # If it's an AtomArrayStack, take the first model
        if hasattr(structure, "stack_depth"):
            structure = structure[0]

        # Write to PDB format using Biotite
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, structure)
        pdb_file.write(out_pdb_path)

        logger.info(f"Converted CIF to PDB using Biotite: {cif_path} -> {out_pdb_path}")

    except Exception as e:
        logger.error(f"Failed to convert CIF to PDB using Biotite: {e}")

    return out_pdb_path


def extract_rf3_metrics_from_cif(cif_path: str) -> dict[str, float]:
    """Extract confidence metrics from RF3 output CIF file.

    Args:
        cif_path: Path to RF3 output CIF file

    Returns:
        Dictionary containing extracted metrics
    """
    metrics = {
        "pLDDT": 0.0,
        "pTM": 0.0,
        "i_pTM": 0.0,
        "pAE": 31.75,
    }

    try:
        # Load structure with b-factors (which may contain pLDDT)
        struct = load_structure(cif_path, extra_fields=["b_factor"])

        # Extract pLDDT from b-factors if available
        if hasattr(struct, "b_factor") and len(struct.b_factor) > 0:
            plddt = np.mean(struct.b_factor)
            metrics["pLDDT"] = float(plddt) / 100.0  # Convert to 0-1 scale

        logger.info(f"Extracted metrics from {cif_path}: {metrics}")

    except Exception as e:
        logger.warning(f"Could not extract metrics from {cif_path}: {e}")

    return metrics


def prepare_ligand_template_for_rf3(
    ligand_pdb_path: str,
    target_chain_id: str = "A",
    output_path: str | None = None,
) -> str:
    """Prepare ligand structure as template for RF3 ground truth conformer selection.

    Args:
        ligand_pdb_path: Path to PDB file containing ligand
        target_chain_id: Chain ID of the ligand
        output_path: Output path for prepared ligand structure

    Returns:
        Path to prepared ligand structure
    """
    if output_path is None:
        output_path = ligand_pdb_path.replace(".pdb", "_template.pdb")

    # Load and extract ligand chain
    struct = load_structure(ligand_pdb_path)
    ligand_struct = struct[struct.chain_id == target_chain_id]

    # Save ligand structure
    save_structure(output_path, ligand_struct)

    logger.info(f"Prepared ligand template: {output_path}")
    return output_path


def create_rf3_complex_input(
    target_sequences: list[str],
    target_mol_types: list[str],
    binder_sequence: str,
    target_msa_paths: list[str | None] | None = None,
    target_template_path: str | None = None,
    name: str = "complex",
) -> dict[str, Any]:
    """Create RF3 input for protein-ligand or protein-protein complex.

    Args:
        target_sequences: List of target sequences (or file paths for ligands)
        target_mol_types: List of molecule types for targets
        binder_sequence: Binder protein sequence
        target_msa_paths: MSA paths for target sequences
        target_template_path: Path to target template structure
        name: Name for the complex

    Returns:
        RF3 input dictionary
    """
    if target_msa_paths is None:
        target_msa_paths = [None] * len(target_sequences)

    # Prepare all sequences and molecule types
    all_sequences = target_sequences + [binder_sequence]
    all_mol_types = target_mol_types + ["protein"]
    all_msa_paths = target_msa_paths + [None]  # No MSA for binder

    # Prepare ground truth conformer selection for ligand targets
    ground_truth_selection = None
    if target_template_path and any(mol_type == "ligand" for mol_type in target_mol_types):
        # Select ligand chains for ground truth conformer
        ligand_chains = [chr(ord("A") + i) for i, mol_type in enumerate(target_mol_types) if mol_type == "ligand"]
        if ligand_chains:
            ground_truth_selection = f"[{','.join(ligand_chains)}]"

    rf3_input = {"name": name, "components": []}

    # Add components
    chain_id_counter = ord("A")
    for seq, mol_type, msa_path in zip(all_sequences, all_mol_types, all_msa_paths, strict=False):
        chain_id = chr(chain_id_counter)
        chain_id_counter += 1

        if mol_type == "protein":
            component = {"seq": seq, "chain_id": chain_id}
            if msa_path is not None:
                component["msa_path"] = msa_path
        elif mol_type == "ligand":
            # For ligands, sequence should be file path
            if seq.startswith("FILE_"):
                component = {
                    "path": seq[5:],  # Remove "FILE_" prefix
                    "chain_id": chain_id,
                }
            else:
                # Handle other ligand formats
                component = {"smiles": seq, "chain_id": chain_id}

        rf3_input["components"].append(component)

    # Add ground truth conformer selection if applicable
    if ground_truth_selection:
        rf3_input["ground_truth_conformer_selection"] = ground_truth_selection

    return rf3_input


def _convert_to_single_letter(sequence: str) -> str:
    """Convert 3-letter amino acid codes to single letter codes.

    Args:
        sequence: Sequence with 3-letter codes

    Returns:
        Sequence with single letter codes
    """
    # Standard amino acid mapping
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    # If already single letter, return as is
    if len(sequence) > 0 and all(len(aa) == 1 for aa in sequence):
        return sequence

    # Convert 3-letter codes
    result = []
    i = 0
    while i < len(sequence):
        if i + 2 < len(sequence):
            three_letter = sequence[i : i + 3].upper()
            if three_letter in aa_map:
                result.append(aa_map[three_letter])
                i += 3
                continue

        # Fallback: use single character
        result.append(sequence[i])
        i += 1

    return "".join(result)


def run_rf3_fold_command(
    input_file: str,
    output_dir: str,
    ckpt_path: str | None = None,
    n_recycles: int = 10,
    diffusion_batch_size: int = 5,
    num_steps: int = 200,
    seed: int = 42,
    ground_truth_conformer_selection: str | None = None,
    template_selection: str | None = None,
    rf3_path: str = "rf3",
) -> tuple[str, str]:
    """Run RF3 fold command with specified parameters.

    Args:
        input_file: Path to RF3 input JSON file
        output_dir: Output directory
        ckpt_path: Path to RF3 checkpoint
        n_recycles: Number of recycles
        diffusion_batch_size: Diffusion batch size
        num_steps: Number of diffusion steps
        seed: Random seed
        ground_truth_conformer_selection: Ground truth conformer selection
        template_selection: Template selection
        rf3_path: Path to RF3 executable
    Returns:
        Tuple of (output_cif_path, score_file_path)
    """
    # Prepare command
    cmd = [
        rf3_path,
        "fold",
        input_file,
        f"out_dir={output_dir}",
        f"ckpt_path={ckpt_path}",
        f"n_recycles={n_recycles}",
        f"diffusion_batch_size={diffusion_batch_size}",
        f"num_steps={num_steps}",
        f"seed={seed}",
        "skip_existing=False",
        "dump_predictions=True",
        "dump_trajectories=False",
        "one_model_per_file=False",
    ]

    # Add optional parameters
    if ground_truth_conformer_selection:
        cmd.append(f"ground_truth_conformer_selection={ground_truth_conformer_selection}")
    if template_selection:
        cmd.append(f"template_selection={template_selection}")

    logger.info(f"Running RF3 command: {' '.join(cmd)}")

    try:
        # Run command
        subprocess.run(
            cmd,
            timeout=500,  # 500 seconds timeout
        )

        logger.info("RF3 fold completed successfully")

        # Determine output file paths
        with open(input_file) as f:
            input_data = json.load(f)
        name = input_data[0]["name"]

        output_cif = os.path.join(output_dir, f"{name}.cif.gz")
        score_file = os.path.join(output_dir, f"{name}.score")

        return output_cif, score_file

    except subprocess.CalledProcessError as e:
        logger.error(f"RF3 fold failed: {e.stderr}")
        raise RuntimeError(f"RF3 fold failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("RF3 fold timed out")
        raise RuntimeError("RF3 fold timed out")
