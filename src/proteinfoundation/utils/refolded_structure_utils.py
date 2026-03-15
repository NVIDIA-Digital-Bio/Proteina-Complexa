#!/usr/bin/env python3
"""
Utilities for handling refolded structure paths and computing metrics on them.
This module provides functions to extract paths to refolded structures from
binder evaluation results and compute force field and bioinformatics metrics
on successful samples only.
"""

import glob
import os

import pandas as pd
from loguru import logger

from proteinfoundation.result_analysis.analysis_utils import SEQUENCE_TYPES


def extract_best_refolded_structure_paths_from_df(
    df: pd.DataFrame, sequence_types: list[str] = None
) -> dict[str, dict[str, str]]:
    """
    Extract paths to the best refolded structures from binder evaluation dataframe.

    For Protenix evaluation, the dataframe contains columns with the paths to the best
    refolded structures for each sequence type:
    - For mpnn and mpnn_fixed: uses the single "best" path column (not the "_all" column)
    - For self: uses the single path column

    Args:
        df: Binder evaluation results dataframe
        sequence_types: List of sequence types to consider

    Returns:
        Dictionary mapping sample names to best structure paths:
        {
            'sample_name': {
                'mpnn': 'path_to_best_mpnn_structure',
                'mpnn_fixed': 'path_to_best_mpnn_fixed_structure',
                'self': 'path_to_self_structure'
            }
        }
    """
    if sequence_types is None:
        sequence_types = SEQUENCE_TYPES

    best_paths = {}

    for _, row in df.iterrows():
        # Extract sample name from pdb_path
        pdb_path = row["pdb_path"]
        sample_name = os.path.basename(pdb_path).replace(".pdb", "").replace("tmp_", "")

        if sample_name not in best_paths:
            best_paths[sample_name] = {}

        for seq_type in sequence_types:
            # Look for the column containing the best structure path
            # This should be something like 'mpnn_complex_pdb_path' or similar
            possible_columns = [
                f"{seq_type}_complex_pdb_path",
                f"{seq_type}_structure_path",
                f"{seq_type}_pdb_path",
                f"{seq_type}_best_path",
            ]

            structure_path = None
            for col in possible_columns:
                if col in row.index and pd.notna(row[col]) and row[col] != "":
                    structure_path = row[col]
                    break

            if structure_path and os.path.exists(structure_path):
                best_paths[sample_name][seq_type] = structure_path
            else:
                logger.debug(f"No valid structure path found for {sample_name} {seq_type}")

    return best_paths


def get_successful_best_samples_with_paths(
    df: pd.DataFrame,
    best_paths_dict: dict[str, dict[str, str]],
    sequence_types: list[str] = None,
) -> list[tuple[str, str, str]]:
    """
    Get successful best samples with their corresponding refolded structure paths.

    This function checks if the best sample for each sequence type is successful
    and returns the path to its refolded structure.

    Args:
        df: Binder evaluation results dataframe
        best_paths_dict: Dictionary of best refolded structure paths
        sequence_types: List of sequence types to consider

    Returns:
        List of tuples: (sample_name, seq_type, structure_path)
    """
    if sequence_types is None:
        sequence_types = SEQUENCE_TYPES

    successful_best_samples = []

    for _, row in df.iterrows():
        # Extract sample name from pdb_path
        pdb_path = row["pdb_path"]
        sample_name = os.path.basename(pdb_path).replace(".pdb", "").replace("tmp_", "")

        if sample_name not in best_paths_dict:
            continue

        for seq_type in sequence_types:
            try:
                # Get the best metrics for this sequence type (not "_all" columns)
                best_ipae_col = f"{seq_type}_complex_i_pAE"
                best_plddt_col = f"{seq_type}_complex_pLDDT"
                best_rmsd_col = f"{seq_type}_binder_scRMSD"

                # Check if required columns exist
                if not all(col in row.index for col in [best_ipae_col, best_plddt_col, best_rmsd_col]):
                    logger.debug(f"Missing metric columns for {sample_name} {seq_type}")
                    continue

                # Get the best values
                ipae = row[best_ipae_col]
                plddt = row[best_plddt_col]
                rmsd = row[best_rmsd_col]

                # Check if the best sample is successful using the same criteria
                if pd.notna(ipae) and pd.notna(plddt) and pd.notna(rmsd):
                    if (ipae * 31 <= 7) and (plddt >= 0.9) and (rmsd < 1.5):
                        # This best sample is successful, get its structure path
                        structure_path = best_paths_dict[sample_name].get(seq_type)
                        if structure_path and os.path.exists(structure_path):
                            successful_best_samples.append((sample_name, seq_type, structure_path))
                            logger.debug(f"Found successful best sample: {sample_name} {seq_type}")
                        else:
                            logger.debug(f"Structure path not found for successful sample: {sample_name} {seq_type}")

            except Exception as e:
                logger.warning(f"Failed to process {seq_type} for sample {sample_name}: {e}")

    logger.info(f"Found {len(successful_best_samples)} successful best samples with refolded structures")
    return successful_best_samples


def extract_refolded_paths_from_evaluation_output(
    evaluation_output_dir: str, folding_method: str, sample_names: list[str]
) -> dict[str, dict[str, list[str]]]:
    """
    Alternative method to extract refolded structure paths directly from evaluation output directories.

    This function is useful when the evaluation has already been run and you want to
    extract the structure paths for post-processing.

    Args:
        evaluation_output_dir: Directory containing evaluation outputs
        folding_method: Folding method used ('colabdesign', 'protenix', etc.)
        sample_names: List of sample names to look for

    Returns:
        Dictionary mapping sample names to structure paths
    """
    refolded_paths = {}

    for sample_name in sample_names:
        sample_dir = os.path.join(evaluation_output_dir, f"tmp_{sample_name}")
        if not os.path.exists(sample_dir):
            continue

        refolded_paths[sample_name] = {"mpnn": [], "mpnn_fixed": [], "self": []}

        if folding_method == "colabdesign":
            # Look for ColabDesign output structure files
            complex_dir = os.path.join(sample_dir, "MPNN", "Complex")
            if os.path.exists(complex_dir):
                complex_files = glob.glob(os.path.join(complex_dir, "*.pdb"))
                complex_files = sorted(complex_files)

                seq_per_type = 8
                if len(complex_files) >= seq_per_type:
                    refolded_paths[sample_name]["mpnn"] = complex_files[:seq_per_type]
                if len(complex_files) >= 2 * seq_per_type:
                    refolded_paths[sample_name]["mpnn_fixed"] = complex_files[seq_per_type : 2 * seq_per_type]
                if len(complex_files) >= 2 * seq_per_type + 1:
                    refolded_paths[sample_name]["self"] = [complex_files[2 * seq_per_type]]

    return refolded_paths
