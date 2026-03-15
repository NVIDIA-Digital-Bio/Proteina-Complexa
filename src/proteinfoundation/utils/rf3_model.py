# RF3 model utilities for structure prediction evaluation
# Similar to ptx_model.py but adapted for RF3

import os
import shutil
import tempfile
from typing import Literal

from loguru import logger

from proteinfoundation.rewards.rf3_reward import RF3RewardRunner
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb
from proteinfoundation.utils.rf3_utils import convert_cif_to_pdb_rf3, prepare_ligand_template_for_rf3


def prepare_batch_rf3_inputs(
    updated_pdb_path: str,
    binder_sequences: list[dict[str, str]],
    sequence_type_list: list[str],
    binder_chain_id: str,
    design_name: str,
    temp_dir: str,
) -> list[str]:
    """Prepare all RF3 input files in batch.

    Args:
        updated_pdb_path: Path to the prepared PDB file (from ProteinMPNN preparation)
        binder_sequences: List of binder sequences with metadata
        sequence_type_list: List of sequence types corresponding to each sequence
        binder_chain_id: Chain ID of the binder chain to replace
        design_name: Name of the design
        temp_dir: Temporary directory to store input files

    Returns:
        List of paths to created RF3 input PDB files
    """
    input_paths = []

    logger.info(f"Preparing {len(binder_sequences)} RF3 input files in batch...")

    for seq_idx, seq_info in enumerate(binder_sequences):
        seq_type = sequence_type_list[seq_idx]
        run_name = f"complex_{design_name}_{seq_type}_seq{seq_idx}"

        complex_pdb_path = os.path.join(temp_dir, f"{run_name}_complex.pdb")

        # Create PDB file for this sequence
        _create_complex_pdb_for_rf3(
            updated_pdb_path=updated_pdb_path,
            binder_sequence=seq_info["seq"],
            binder_chain_id=binder_chain_id,
            output_path=complex_pdb_path,
            run_name=run_name,
        )

        input_paths.append(complex_pdb_path)

    logger.info(f"Created {len(input_paths)} RF3 input files")
    return input_paths


def run_rf3_eval(
    rf3_runner: RF3RewardRunner,
    target_chain_ids: list[str],  # Chain IDs to template (e.g., ["A", "B"])
    is_target_ligand: bool,  # Whether target chains are ligands
    binder_sequences: list[dict[str, str]],  # List of {"seq": sequence} dicts
    sequence_type_list: list[Literal["mpnn", "mpnn_fixed", "self"]],
    design_name: str,
    output_path: str,
    updated_pdb_path: str | None = None,
    binder_chain_id: str | None = None,
    smiles: str = None,
) -> tuple[list[dict[str, dict]], list[str]]:
    """Run binder evaluation with RF3 using batch processing.

    Args:
        rf3_runner: RF3RewardRunner instance
        target_chain_ids: Chain IDs to template (e.g., ["A", "B"])
        is_target_ligand: Whether target chains are ligands
        binder_sequences: List of binder sequences
        sequence_type_list: List of sequence types
        design_name: Name of the design
        output_path: Output directory path
        updated_pdb_path: Path to the prepared PDB file (from ProteinMPNN preparation)
        binder_chain_id: Chain ID of the binder chain to replace

    Returns:
        Tuple of (complex_statistics, complex_pdb_paths)
    """
    # Create output directory
    dump_dir = os.path.join(output_path, "rf3_outputs")
    os.makedirs(dump_dir, exist_ok=True)
    # rf3_temp_dir = os.path.join(dump_dir, "rf3_temp")
    # os.makedirs(rf3_temp_dir, exist_ok=True)

    # Reset dump dir for the runner
    rf3_runner.reset_dump_dir(dump_dir)

    logger.info("================ RF3 Target info ================")
    logger.info(f"Target chain IDs: {target_chain_ids}")
    logger.info(f"Is target ligand: {is_target_ligand}")
    logger.info("================================================")

    # Prepare all inputs in batch
    temp_dir = tempfile.mkdtemp(prefix="rf3_batch_")

    try:
        logger.info("Preparing RF3 batch inputs...")
        input_paths = prepare_batch_rf3_inputs(
            updated_pdb_path=updated_pdb_path,
            binder_sequences=binder_sequences,
            sequence_type_list=sequence_type_list,
            binder_chain_id=binder_chain_id,
            design_name=design_name,
            temp_dir=temp_dir,
        )
        # for input_path in input_paths:
        #     shutil.copy(input_path, os.path.join(rf3_temp_dir, os.path.basename(input_path)))
        # Determine templating strategy based on target type - much simpler!
        template_selection = None
        ground_truth_conformer_selection = None

        # Simple logic: just use the parameters we already have!
        if is_target_ligand:
            # For ligand targets, use ground truth conformer selection
            ground_truth_conformer_selection = f"[{','.join(target_chain_ids)}]"
            logger.info(
                f"Using ground truth conformer selection for ligand targets: {ground_truth_conformer_selection}"
            )
        else:
            # For protein targets, use template selection
            template_selection = ",".join(target_chain_ids)
            logger.info(f"Using template selection for protein targets: {template_selection}")

        # Run batch prediction
        logger.info(f"Running RF3 batch prediction on {len(input_paths)} inputs...")
        predictions = rf3_runner.predict_batch_from_files(
            input_files=input_paths,
            template_selection=template_selection,
            ground_truth_conformer_selection=ground_truth_conformer_selection,
            out_dir=dump_dir,
            smiles=smiles,
        )

        # Process results
        complex_statistics = []
        complex_pdb_paths = []

        for seq_idx, (prediction, seq_info) in enumerate(zip(predictions, binder_sequences, strict=False)):
            # Extract output path from prediction
            output_cif_path = prediction.get("output_cif_path")
            if output_cif_path and os.path.exists(output_cif_path):
                # Convert CIF to PDB for compatibility
                complex_pdb_path = convert_cif_to_pdb_rf3(output_cif_path)
            else:
                complex_pdb_path = None
            complex_pdb_paths.append(complex_pdb_path)

            # Extract metrics from prediction (already in compatible format)
            #! the model outputs scores as normal so for our evaluation we need to convert them to 0-1 scale
            # TODO: normalize accross reward/search and evaluation. For search its manual, for evaluation its automatic.
            summary_conf = prediction.get("summary_confidence", [{}])[0]
            complex_metrics = {
                "pLDDT": float(summary_conf.get("plddt", 0.0)),  # Convert to 0-1 scale
                "i_pAE": float(summary_conf.get("ipAE", 31.0)) / 31.0,
                "min_ipAE": float(summary_conf.get("min_ipAE", 31.0)) / 31.0,
                "pAE": float(summary_conf.get("pAE", 31.0)) / 31.0,
                "min_ipSAE": float(summary_conf.get("min_ipSAE", 0.0)),
                "max_ipSAE": float(summary_conf.get("max_ipSAE", 0.0)),
                "avg_ipSAE": float(summary_conf.get("avg_ipSAE", 0.0)),
            }
            # Add the complex PDB path to the complex metrics
            complex_metrics["complex_pdb_path"] = complex_pdb_path
            complex_statistics.append({f"seq_{seq_idx + 1}": complex_metrics})

            logger.info("Completed prediction")

        logger.info(f"Completed RF3 batch evaluation for {len(binder_sequences)} sequences")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # logger.info(f"Leaving temporary directory: {temp_dir}")

    return complex_statistics, complex_pdb_paths


def create_rf3_runner(
    ckpt_path: str | None = None,
    dump_dir: str | None = None,
) -> RF3RewardRunner:
    """Create RF3 reward runner with specified configuration.

    Args:
        ckpt_path: Path to checkpoint file
        dump_dir: Output directory

    Returns:
        RF3RewardRunner instance
    """
    from proteinfoundation.rewards.rf3_reward import get_default_rf3_runner

    return get_default_rf3_runner(
        ckpt_path=ckpt_path,
        dump_dir=dump_dir,
    )


def prepare_rf3_ligand_complex(
    target_pdb_path: str,
    target_chain_ids: list[str],
    binder_sequence: str,
    design_name: str,
    output_dir: str,
) -> tuple[list[tuple[str, str, str | None]], str]:
    """Prepare RF3 input for protein-ligand complex evaluation.

    Args:
        target_pdb_path: Path to target PDB containing ligand
        target_chain_ids: List of target chain IDs
        binder_sequence: Binder protein sequence
        design_name: Name of the design
        output_dir: Output directory

    Returns:
        Tuple of (target_data, template_path)
    """
    target_data = []
    template_path = None

    for chain_id in target_chain_ids:
        # For ligand targets, use file path format
        ligand_file_path = f"FILE_{target_pdb_path}"
        target_data.append((ligand_file_path, "ligand", None))

        # Prepare template for ground truth conformer
        if template_path is None:
            template_path = prepare_ligand_template_for_rf3(
                target_pdb_path,
                chain_id,
                os.path.join(output_dir, f"{design_name}_ligand_template.pdb"),
            )

    return target_data, template_path


def prepare_rf3_protein_complex(
    target_pdb_path: str,
    target_chain_ids: list[str],
    binder_sequence: str,
) -> list[tuple[str, str, str | None]]:
    """Prepare RF3 input for protein-protein complex evaluation.

    Args:
        target_pdb_path: Path to target PDB
        target_chain_ids: List of target chain IDs
        binder_sequence: Binder protein sequence

    Returns:
        List of target data tuples (sequence, mol_type, template_path)
    """
    target_data = []

    for chain_id in target_chain_ids:
        # Extract sequence from target PDB
        target_sequence = extract_seq_from_pdb(target_pdb_path, chain_id=chain_id)
        # Use target PDB path as template instead of MSA
        template_path = target_pdb_path

        target_data.append((target_sequence, "protein", template_path))

    return target_data


def _create_complex_pdb_for_rf3(
    updated_pdb_path: str,
    binder_sequence: str,
    binder_chain_id: str,
    output_path: str,
    run_name: str,
) -> str:
    """Create a proper PDB file for RF3 complex prediction by copying existing PDB and replacing binder sequence.

    Args:
        updated_pdb_path: Path to the prepared PDB file (from ProteinMPNN preparation)
        binder_sequence: New binder sequence to replace in the structure
        binder_chain_id: Chain ID of the binder chain to replace
        output_path: Output PDB file path
        run_name: Name for the run

    Returns:
        Path to created PDB file

    ZC NOTE: Previous implementation will cause bug when binder residue indexing does not start from 1.
    Take BoltzDesign samples as example, the binder residue starts from 2 (1 is the ligand).
    The PDB created by this function will replace the sequence starting from the 2nd residue, and repeat the last residue.
    Causing mismatch of sequence.
    """

    logger.info(f"Creating RF3 complex PDB from {updated_pdb_path}")
    logger.info(f"Replacing chain {binder_chain_id} with sequence: {binder_sequence}")

    # Read the original PDB file
    with open(updated_pdb_path) as f:
        original_lines = f.readlines()

    # Parse the original PDB to extract binder chain information
    binder_residues = set()
    binder_atoms = []

    for line in original_lines:
        if line.startswith("ATOM") and line[21] == binder_chain_id:
            binder_atoms.append(line)
            # Extract residue number
            res_num = int(line[22:26].strip())
            binder_residues.add(res_num)

    # Create mapping of old residues to new sequence
    binder_residues = sorted(binder_residues)
    # Raise error if length mismatch
    if len(binder_residues) != len(binder_sequence):
        raise ValueError(
            f"Length mismatch: PDB has {len(binder_residues)} binder residues, sequence has {len(binder_sequence)}"
        )

    # Create new PDB file
    with open(output_path, "w") as f:
        # Write header with run information
        f.write(f"REMARK    RF3 COMPLEX INPUT FOR {run_name}\n")
        f.write(f"REMARK    ORIGINAL PDB: {updated_pdb_path}\n")
        f.write(f"REMARK    BINDER CHAIN: {binder_chain_id}\n")
        f.write(f"REMARK    BINDER SEQUENCE: {binder_sequence}\n")

        # Process each line
        for line in original_lines:
            if line.startswith("ATOM") and line[21] == binder_chain_id:
                # This is a binder chain atom - update the residue name
                res_num = int(line[22:26].strip())

                # Find position in the new sequence (residues are 1-indexed)
                seq_idx = binder_residues.index(res_num)
                if seq_idx <= len(binder_sequence):
                    # seq_idx = res_num - 1
                    new_aa = binder_sequence[seq_idx]
                    new_res_name = _single_to_three_letter(new_aa)

                    # Replace the residue name in the line
                    new_line = line[:17] + new_res_name + line[20:]
                    f.write(new_line)
                else:
                    # Residue number beyond sequence length, skip or keep original
                    logger.warning(f"Residue {res_num} beyond sequence length {len(binder_sequence)}")
                    f.write(line)
            else:
                # Not a binder chain atom, keep as is
                f.write(line)

    logger.info(f"Created RF3 complex PDB file: {output_path}")
    return output_path


def _single_to_three_letter(aa: str) -> str:
    """Convert single letter amino acid code to three letter code."""
    aa_map = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "E": "GLU",
        "Q": "GLN",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }
    return aa_map.get(aa.upper(), "UNK")
