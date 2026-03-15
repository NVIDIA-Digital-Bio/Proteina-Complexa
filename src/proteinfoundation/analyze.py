#!/usr/bin/env python3
"""
Unified analysis script for aggregating and analyzing v2 evaluation results.

This script provides a clean, modular framework for analyzing evaluation results
from both monomer and binder benchmarks. It supports:

- Aggregating results from multiple parallel evaluation jobs
- Computing success rates and pass rates for binders
- Computing diversity metrics using Foldseek
- Aggregating force field and bioinformatics metrics
- Generating summary reports

Protein Types:
    - monomer: Single chain proteins (designability, novelty, diversity)
    - binder: Binder + target complexes (success rates, interface metrics, diversity)

Result Types (for binder analysis):
    - protein_binder: Standard protein-protein binding
    - ligand_binder: Small molecule binding

Usage:
    # Analyze binder results (protein target)
    python analyze.py --config-name analyze \\
        result_type=protein_binder \\
        results_dir=./evaluation_results/my_binder_run

    # Analyze binder results (ligand target)
    python analyze.py --config-name analyze \\
        result_type=ligand_binder \\
        results_dir=./evaluation_results/my_ligand_run

    # Analyze monomer results  
    python analyze.py --config-name analyze \\
        result_type=monomer \\
        results_dir=./evaluation_results/my_monomer_run

    # Analyze with custom success thresholds (use a config file override)
    # See docs/CONFIGURATION_GUIDE.md for threshold format

    # Dryrun to see what would be computed
    python analyze.py --config-name analyze dryrun=true
"""

import glob
import json
import os
import re
import shutil
import sys
from functools import partial, reduce

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openfold.np.residue_constants import restype_3to1

# Apply atomworks patches early - before any imports that use atomworks/biotite
import proteinfoundation.patches.atomworks_patches  # noqa: F401
from proteinfoundation.evaluation.binder_eval_utils import get_binder_chain_from_complex
from proteinfoundation.result_analysis.analysis import compute_timing_metrics
from proteinfoundation.result_analysis.analysis_utils import (
    FLOAT_FORMAT_PD,
    SEP_CSV_PD,
    SEQUENCE_TYPES,
    filter_columns_for_csv,
    parse_threshold_spec,
)
from proteinfoundation.result_analysis.binder_analysis import (
    compute_filter_ligand_pass_rate,
    compute_filter_pass_rate,
    filter_by_success_thresholds,
    save_combined_success_criteria_json,
)
from proteinfoundation.result_analysis.binder_analysis_utils import (
    DEFAULT_LIGAND_BINDER_THRESHOLDS,
    DEFAULT_PROTEIN_BINDER_THRESHOLDS,
    normalize_threshold_dict,
)
from proteinfoundation.result_analysis.compute_diversity import compute_foldseek_diversity, compute_mmseqs_diversity
from proteinfoundation.result_analysis.monomer_analysis import (
    compute_codesignability_by_length,
    compute_designability_by_length,
    compute_monomer_pass_rates,
    compute_single_designability_by_length,
    filter_monomer_by_single_threshold,
)
from proteinfoundation.result_analysis.monomer_analysis_utils import (
    DEFAULT_MONOMER_ALL_ATOM_CODESIGNABILITY_THRESHOLDS,
    DEFAULT_MONOMER_CA_CODESIGNABILITY_THRESHOLDS,
    DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS,
    build_thresholds_from_detected,
    detect_monomer_folding_models,
    get_codesignability_thresholds,
)
from proteinfoundation.result_analysis.motif_analysis import (
    compute_motif_region_pass_rates,
    compute_motif_rmsd_pass_rates,
    compute_motif_seq_rec_pass_rates,
    compute_motif_success_pass_rates,
    filter_by_motif_rmsd,
    filter_by_motif_success,
    save_motif_thresholds_json,
)
from proteinfoundation.result_analysis.motif_analysis_utils import (
    DEFAULT_MOTIF_CODESIGNABILITY_THRESHOLDS,
    DEFAULT_MOTIF_DESIGNABILITY_THRESHOLDS,
    DEFAULT_MOTIF_REGION_CODESIGNABILITY_THRESHOLDS,
    DEFAULT_MOTIF_REGION_DESIGNABILITY_THRESHOLDS,
    DEFAULT_MOTIF_RMSD_THRESHOLDS,
    DEFAULT_MOTIF_SEQ_REC_THRESHOLD,
    MOTIF_SUCCESS_PRESETS,
    build_motif_region_column_name,
    normalize_motif_rmsd_thresholds,
    resolve_success_criteria,
    resolve_thresholds,
)
from proteinfoundation.result_analysis.motif_binder_analysis import (
    compute_motif_binder_pass_rate,
    compute_motif_pred_metric_pass_rates,
    compute_per_task_motif_binder_pass_rates,
    filter_by_motif_binder_success,
    save_motif_binder_success_json,
)
from proteinfoundation.result_analysis.motif_binder_analysis_utils import (
    format_success_criteria_for_logging,
    get_default_motif_binder_success,
)
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb

# =============================================================================
# Constants
# =============================================================================

VALID_RESULT_TYPES = {
    "protein_binder",
    "ligand_binder",
    "monomer",
    "monomer_motif",
    "motif_protein_binder",
    "motif_ligand_binder",
}

# Column patterns for different result types
BINDER_RESULT_PATTERN = "binder_results_{config_name}_{suffix}_{job_id}.csv"
MONOMER_RESULT_PATTERN = "monomer_results_{config_name}_{job_id}.csv"

# Columns to ignore when grouping
GROUPBY_IGNORE_COLS = [
    # Config metadata (should never be used for grouping)
    "dryrun",
    "show_progress",
    "result_type",
    # Sample-specific columns
    "seed",
    "L",
    "id_gen",
    "pdb_path",
    "job_id",
    # Motif sample-specific columns (contig_string varies per sample in indexed mode)
    "contig_string",
    "mpnn_filter_pass",
    "mpnn_fixed_filter_pass",
    "self_filter_pass",
    "mpnn_filter_pass_all",
    "mpnn_fixed_filter_pass_all",
    "self_filter_pass_all",
    "mpnn_aa_counts",
    "mpnn_fixed_aa_counts",
    "self_aa_counts",
    "mpnn_aa_interface_counts",
    "mpnn_fixed_aa_interface_counts",
    "self_aa_interface_counts",
    # Sequence columns
    "sequence",
    "binder_sequence",
    "target_sequence",
    # Motif binder sample-specific columns
    "motif_rmsd_gen",
    "motif_seq_rec_gen",
    "correct_motif_sequence_gen",
    "has_ligand_clashes_gen",
]


def save_transposed_csv(df: pd.DataFrame, original_csv_path: str) -> None:
    """
    Save a transposed version of the DataFrame for easier viewing.

    In the transposed version:
    - Rows are the original column names (metrics)
    - Columns are sample indices (sample_0, sample_1, etc.)

    This makes it easier to compare metrics across samples when viewing
    in a spreadsheet or text editor, especially with many columns.

    Args:
        df: DataFrame to transpose
        original_csv_path: Path to the original CSV (used to derive transposed path)
    """
    if df.empty:
        logger.warning("Cannot save transposed CSV: DataFrame is empty")
        return

    try:
        # Transpose the DataFrame
        df_transposed = df.T

        # Rename columns to sample indices
        df_transposed.columns = [f"sample_{i}" for i in range(len(df_transposed.columns))]

        # Add the attribute name as the first column
        df_transposed.insert(0, "attribute", df_transposed.index)
        df_transposed = df_transposed.reset_index(drop=True)

        # Create the transposed filename
        transposed_path = original_csv_path.replace(".csv", "_transposed.csv")

        # Save
        df_transposed.to_csv(transposed_path, index=False)
        logger.debug(f"Transposed results saved to {transposed_path}")

    except Exception as e:
        logger.warning(f"Failed to save transposed CSV: {e}")


# =============================================================================
# Sequence Extraction
# =============================================================================


def extract_sequences_from_pdb(
    pdb_path: str,
    result_type: str,
) -> dict[str, str | None]:
    """
    Extract sequences from a PDB file based on result type.

    For protein_binder: extracts binder_sequence and target_sequence
    For ligand_binder: extracts binder_sequence only (target is a ligand, not a protein)
    For monomers: extracts sequence

    Args:
        pdb_path: Path to the PDB file
        result_type: Type of results ("protein_binder", "ligand_binder", "monomer")

    Returns:
        Dictionary with sequence column names and values
    """
    result = {}

    protein_binder_types = {"protein_binder", "motif_protein_binder"}
    ligand_binder_types = {"ligand_binder", "motif_ligand_binder"}

    if not os.path.exists(pdb_path):
        logger.warning(f"PDB file not found: {pdb_path}")
        if result_type in protein_binder_types:
            return {"binder_sequence": None, "target_sequence": None}
        elif result_type in ligand_binder_types:
            return {"binder_sequence": None}
        else:
            return {"sequence": None}

    try:
        if result_type in protein_binder_types:
            # Get binder and target chains
            binder_chain, target_chains = get_binder_chain_from_complex(pdb_path)

            # Extract binder sequence
            binder_seq = extract_seq_from_pdb(pdb_path, chain_id=binder_chain)
            result["binder_sequence"] = binder_seq

            # Extract target sequence (concatenate all target chains)
            target_seqs = []
            for chain in target_chains:
                try:
                    seq = extract_seq_from_pdb(pdb_path, chain_id=chain)
                    target_seqs.append(seq)
                except Exception as e:
                    logger.warning(f"Failed to extract target chain {chain}: {e}")

            result["target_sequence"] = ":".join(target_seqs) if target_seqs else None

        elif result_type in ligand_binder_types:
            # For ligand binders, target is a small molecule - only extract binder sequence
            binder_chain, _ = get_binder_chain_from_complex(pdb_path)
            binder_seq = extract_seq_from_pdb(pdb_path, chain_id=binder_chain)
            result["binder_sequence"] = binder_seq

        else:  # monomer
            # For monomers, extract full sequence
            result["sequence"] = extract_seq_from_pdb(pdb_path, chain_id=None)

    except Exception as e:
        logger.warning(f"Failed to extract sequences from {pdb_path}: {e}")
        if result_type in protein_binder_types:
            result = {"binder_sequence": None, "target_sequence": None}
        elif result_type in ligand_binder_types:
            result = {"binder_sequence": None}
        else:
            result = {"sequence": None}

    return result


def add_sequence_columns(
    df: pd.DataFrame,
    result_type: str,
    pdb_path_column: str = "pdb_path",
) -> pd.DataFrame:
    """
    Add sequence columns to a DataFrame by extracting from PDB files.

    For protein_binder: adds 'binder_sequence' and 'target_sequence' columns
    For ligand_binder: adds 'binder_sequence' column only (target is a ligand)
    For monomers: adds 'sequence' column

    Args:
        df: DataFrame with pdb_path column
        result_type: Type of results ("protein_binder", "ligand_binder", "monomer")
        pdb_path_column: Name of the column containing PDB file paths

    Returns:
        DataFrame with added sequence columns
    """
    if pdb_path_column not in df.columns:
        logger.warning(f"Column '{pdb_path_column}' not found in DataFrame, skipping sequence extraction")
        return df

    logger.info(f"Extracting sequences from {len(df)} PDB files...")

    # Initialize sequence columns based on result type
    if result_type in ["protein_binder", "motif_protein_binder"]:
        df["binder_sequence"] = None
        df["target_sequence"] = None
    elif result_type in ["ligand_binder", "motif_ligand_binder"]:
        df["binder_sequence"] = None
    else:  # monomer or monomer_motif
        df["sequence"] = None

    # Extract sequences for each row
    for idx, row in df.iterrows():
        pdb_path = row[pdb_path_column]
        if pd.isna(pdb_path) or not pdb_path:
            continue

        sequences = extract_sequences_from_pdb(pdb_path, result_type)

        for col, seq in sequences.items():
            df.at[idx, col] = seq

    # Log summary
    if result_type in ["protein_binder", "motif_protein_binder"]:
        n_binder = df["binder_sequence"].notna().sum()
        n_target = df["target_sequence"].notna().sum()
        logger.info(f"Extracted {n_binder} binder sequences and {n_target} target sequences")
        seq_cols = ["binder_sequence", "target_sequence"]
    elif result_type in ["ligand_binder", "motif_ligand_binder"]:
        n_binder = df["binder_sequence"].notna().sum()
        logger.info(f"Extracted {n_binder} binder sequences (ligand targets have no protein sequence)")
        seq_cols = ["binder_sequence"]
    else:
        n_seq = df["sequence"].notna().sum()
        logger.info(f"Extracted {n_seq} sequences")
        seq_cols = ["sequence"]

    # Reorder columns to place sequence columns right after run_name
    if "run_name" in df.columns:
        cols = df.columns.tolist()
        # Remove sequence columns from their current position
        for seq_col in seq_cols:
            if seq_col in cols:
                cols.remove(seq_col)
        # Find run_name position and insert sequence columns after it
        run_name_idx = cols.index("run_name")
        for i, seq_col in enumerate(seq_cols):
            cols.insert(run_name_idx + 1 + i, seq_col)
        df = df[cols]

    return df


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_config(cfg: DictConfig, results_dir: str = None) -> None:
    """
    Validate the analysis configuration.

    Args:
        cfg: Hydra configuration dictionary
        results_dir: Optional pre-computed results directory path

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate results directory (use provided or get from cfg)
    if results_dir is None:
        results_dir = cfg.get("results_dir")

    if not results_dir:
        raise ValueError("results_dir must be specified or constructable from config")

    if not os.path.isdir(results_dir):
        raise ValueError(f"results_dir does not exist: {results_dir}")

    # Validate result_type
    result_type = cfg.get("result_type", "protein_binder")
    if result_type not in VALID_RESULT_TYPES:
        raise ValueError(f"Invalid result_type '{result_type}'. Valid options: {VALID_RESULT_TYPES}")


def print_dryrun_summary(
    cfg: DictConfig,
    result_files: list[str],
    result_type: str,
    success_thresholds: dict | None,
) -> None:
    """
    Print summary of what would be computed in dryrun mode.

    Args:
        cfg: Configuration dictionary
        result_files: List of result files found
        result_type: Type of results being analyzed
        success_thresholds: Success thresholds being used
    """
    logger.info("=" * 70)
    logger.info("DRYRUN MODE - No actual analysis will be performed")
    logger.info("=" * 70)

    logger.info("\nConfiguration Summary:")
    logger.info(f"  Results directory: {cfg.get('results_dir')}")
    logger.info(f"  Config name: {cfg.get('config_name')}")
    logger.info(f"  Result type: {result_type}")
    logger.info(f"  Number of result files found: {len(result_files)}")

    if result_files:
        logger.info("\nResult files:")
        for f in result_files[:5]:
            logger.info(f"    {os.path.basename(f)}")
        if len(result_files) > 5:
            logger.info(f"    ... and {len(result_files) - 5} more")

    cfg_aggregation = cfg.get("aggregation", {})
    logger.info("\nAggregation Settings:")
    logger.info(f"  Limit: {cfg_aggregation.get('limit', 'None (all files)')}")
    logger.info(f"  Sequence types: {cfg_aggregation.get('sequence_types', ['self', 'mpnn', 'mpnn_fixed'])}")

    if success_thresholds:
        logger.info("\nSuccess Thresholds:")
        for metric_name, spec in success_thresholds.items():
            parsed = parse_threshold_spec(spec)
            scale_str = f"*{parsed['scale']}" if parsed.get("scale", 1.0) != 1.0 else ""
            logger.info(
                f"    {parsed.get('column_prefix', 'complex')}_{metric_name}{scale_str} "
                f"{parsed.get('op', '<=')} {parsed.get('threshold')}"
            )
    else:
        logger.info(f"\nUsing default thresholds for {result_type}")

    logger.info("\nAnalyses to perform:")
    logger.info("  - Aggregate results from multiple job files")
    logger.info("  - Compute success/pass rates")
    logger.info("  - Compute diversity metrics for successful samples")
    logger.info("  - Aggregate force field and bioinformatics metrics")

    logger.info("\n" + "=" * 70)
    logger.info("End of dryrun summary")
    logger.info("=" * 70)


# =============================================================================
# Result File Discovery
# =============================================================================


def find_result_files(
    results_dir: str,
    config_name: str,
    result_type: str,
    input_mode: str = "generated",
    limit: int | None = None,
) -> list[str]:
    """
    Find all result CSV files from evaluation jobs.

    This function searches for result files matching the expected naming pattern
    for the given result type and configuration.

    Args:
        results_dir: Directory containing result files
        config_name: Configuration name used for evaluation
        result_type: Type of results ("protein_binder", "ligand_binder", "monomer")
        input_mode: Input mode used during evaluation ("generated" or "pdb_dir")
        limit: Optional limit on number of files to process

    Returns:
        List of paths to result files, sorted by job ID
    """
    # Build pattern based on result type
    # Note: input_mode is accepted for API compatibility but not used in pattern
    # since evaluate.py uses the same naming regardless of input mode
    if result_type in ["protein_binder", "ligand_binder"]:
        base_pattern = f"binder_results_{config_name}_"
    elif result_type == "monomer":
        base_pattern = f"monomer_results_{config_name}_"
    elif result_type == "monomer_motif":
        base_pattern = f"motif_results_{config_name}_"
    elif result_type in ["motif_protein_binder", "motif_ligand_binder"]:
        base_pattern = f"motif_binder_results_{config_name}_"
    else:
        raise ValueError(f"Unknown result type: {result_type}")

    # Find all matching files
    all_files = os.listdir(results_dir)
    result_files = []

    # Use regex to match files with numeric job IDs
    pattern_regex = re.compile(rf"^{re.escape(base_pattern)}(\d+)\.csv$")

    for filename in all_files:
        if pattern_regex.match(filename):
            result_files.append(os.path.join(results_dir, filename))

    # Sort by job ID
    def extract_job_id(path: str) -> int:
        match = pattern_regex.match(os.path.basename(path))
        if match:
            return int(match.group(1))
        return 0

    result_files = sorted(result_files, key=extract_job_id)

    # Apply limit if specified
    if limit is not None and limit > 0:
        result_files = result_files[:limit]

    logger.info(f"Found {len(result_files)} result files matching pattern: {base_pattern}[0-9]+.csv")

    for f in result_files[:5]:
        logger.info(f"  - {os.path.basename(f)}")
    if len(result_files) > 5:
        logger.info(f"  ... and {len(result_files) - 5} more")

    return result_files


def _find_monomer_results_for_binder(
    results_dir: str,
    config_name: str,
    input_mode: str = "generated",
) -> list[str]:
    """
    Find monomer result files when running monomer analysis on binder data.

    When evaluate runs with both compute_binder_metrics=true and
    compute_monomer_metrics=true, it creates separate files:
      - binder_results_{config_name}_{job_id}.csv (binder metrics)
      - monomer_results_{config_name}_{job_id}.csv (monomer metrics like scRMSD)

    This function finds the monomer_results files for monomer analysis.

    Args:
        results_dir: Directory containing result files
        config_name: Configuration name used for evaluation
        input_mode: Input mode used during evaluation (kept for API compatibility)

    Returns:
        List of paths to monomer result files
    """
    # Note: input_mode is accepted for API compatibility but not used in pattern
    # since evaluate.py uses the same naming regardless of input mode
    base_pattern = f"monomer_results_{config_name}_"

    try:
        all_files = os.listdir(results_dir)
    except OSError:
        return []

    pattern_regex = re.compile(rf"^{re.escape(base_pattern)}(\d+)\.csv$")

    result_files = []
    for filename in all_files:
        if pattern_regex.match(filename):
            result_files.append(os.path.join(results_dir, filename))

    # Sort by job ID
    def extract_job_id(path: str) -> int:
        match = pattern_regex.match(os.path.basename(path))
        if match:
            return int(match.group(1))
        return 0

    return sorted(result_files, key=extract_job_id)


def aggregate_results(result_files: list[str]) -> pd.DataFrame:
    """
    Aggregate results from multiple job CSV files into a single DataFrame.

    Args:
        result_files: List of paths to result CSV files

    Returns:
        Combined DataFrame with all results

    Raises:
        ValueError: If no valid files could be loaded
    """
    if not result_files:
        raise ValueError("No result files provided to aggregate")

    dfs = []
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            # Extract job ID from filename
            job_id = re.search(r"_(\d+)\.csv$", file_path)
            if job_id:
                df["job_id"] = int(job_id.group(1))
            logger.debug(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

    if not dfs:
        raise ValueError("No valid result files could be loaded")

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(dfs)} files into DataFrame with {len(combined_df)} total rows")

    return combined_df


def merge_monomer_into_binder(
    binder_df: pd.DataFrame,
    results_dir: str,
    config_name: str,
    input_mode: str = "generated",
) -> pd.DataFrame:
    """
    Merge monomer evaluation results into the binder results DataFrame.

    When the pipeline runs both binder and monomer evaluation, results are stored
    in separate CSV files. This function loads the monomer results and joins them
    into the binder DataFrame on ``pdb_path``, adding only columns that are unique
    to the monomer results (e.g., _res_scRMSD_*, _res_ss_*, etc.).

    Args:
        binder_df: Aggregated binder results DataFrame (must contain ``pdb_path``)
        results_dir: Directory containing result CSV files
        config_name: Configuration name used for file pattern matching
        input_mode: Input mode used during evaluation

    Returns:
        Enriched DataFrame with monomer metric columns merged in, or the
        original ``binder_df`` unchanged if no monomer files are found.
    """
    if "pdb_path" not in binder_df.columns:
        logger.warning("binder_df has no 'pdb_path' column, cannot merge monomer results")
        return binder_df

    monomer_files = _find_monomer_results_for_binder(results_dir, config_name, input_mode)
    if not monomer_files:
        logger.info("No monomer result files found to merge into binder results")
        return binder_df

    # Load and concatenate monomer results
    monomer_dfs = []
    for f in monomer_files:
        try:
            monomer_dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.warning(f"Failed to read monomer file {f}: {e}")

    if not monomer_dfs:
        return binder_df

    monomer_df = pd.concat(monomer_dfs, ignore_index=True)
    logger.debug(f"Loaded {len(monomer_df)} monomer results from {len(monomer_dfs)} files")

    if "pdb_path" not in monomer_df.columns:
        logger.warning("Monomer results have no 'pdb_path' column, cannot merge")
        return binder_df

    # Identify columns unique to monomer results
    binder_cols = set(binder_df.columns)
    monomer_only_cols = [c for c in monomer_df.columns if c not in binder_cols]

    if not monomer_only_cols:
        logger.info("No unique monomer columns to merge (all columns already in binder results)")
        return binder_df

    # Keep pdb_path as join key + unique monomer columns only
    merge_cols = ["pdb_path"] + monomer_only_cols
    monomer_subset = monomer_df[merge_cols].drop_duplicates(subset=["pdb_path"])

    # Left-join: keep all binder rows, add monomer columns where pdb_path matches
    merged_df = pd.merge(binder_df, monomer_subset, on="pdb_path", how="left")

    n_matched = merged_df[monomer_only_cols[0]].notna().sum()
    logger.info(
        f"Merged {len(monomer_only_cols)} monomer columns into binder results "
        f"({n_matched}/{len(merged_df)} rows matched). "
        f"New columns: {monomer_only_cols}"
    )

    return merged_df


def merge_monomer_into_motif(
    motif_df: pd.DataFrame,
    results_dir: str,
    config_name: str,
    input_mode: str = "generated",
) -> pd.DataFrame:
    """Merge monomer evaluation results into the motif results DataFrame.

    When evaluate runs with both ``compute_motif_metrics`` and
    ``compute_monomer_metrics``, results are stored in separate CSV files:
      - ``motif_results_{config_name}_{job_id}.csv``  (motif metrics)
      - ``monomer_results_{config_name}_{job_id}.csv`` (monomer metrics)

    This function loads the monomer results and joins them into the motif
    DataFrame on ``pdb_path``.  Column conflicts are handled as follows:

    - **Metadata columns** (``run_name``, ``ckpt_path``, etc.): dropped from
      the monomer side — the motif DataFrame already has them.
    - **Unique monomer columns** (e.g. ``_res_scRMSD_single_*``,
      ``_res_co_seq_rec``): added directly.
    - **Conflicting metric columns** (same name in both CSVs, e.g.
      ``_res_scRMSD_ca_esmfold``): the motif version is kept as-is and
      the monomer version is added with a ``_monomer`` suffix.

    Args:
        motif_df: Aggregated motif results DataFrame (must contain ``pdb_path``)
        results_dir: Directory containing result CSV files
        config_name: Configuration name used for file pattern matching
        input_mode: Input mode used during evaluation

    Returns:
        Enriched DataFrame with monomer metric columns merged in, or the
        original ``motif_df`` unchanged if no monomer files are found.
    """
    if "pdb_path" not in motif_df.columns:
        logger.warning("motif_df has no 'pdb_path' column, cannot merge monomer results")
        return motif_df

    monomer_files = _find_monomer_results_for_binder(results_dir, config_name, input_mode)
    if not monomer_files:
        logger.debug("No monomer result files found to merge into motif results")
        return motif_df

    # Load and concatenate monomer results
    monomer_dfs = []
    for f in monomer_files:
        try:
            monomer_dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.warning(f"Failed to read monomer file {f}: {e}")

    if not monomer_dfs:
        return motif_df

    monomer_df = pd.concat(monomer_dfs, ignore_index=True)
    logger.debug(f"  Loaded {len(monomer_df)} monomer results from {len(monomer_dfs)} files")

    if "pdb_path" not in monomer_df.columns:
        logger.warning("Monomer results have no 'pdb_path' column, cannot merge")
        return motif_df

    # Classify monomer columns into: metadata (skip), unique (add), conflict (add with suffix)
    motif_cols = set(motif_df.columns)
    metadata_cols = {
        "run_name",
        "ckpt_path",
        "ckpt_name",
        "file_limit",
        "ignore_generated_pdb_suffix",
        "id_gen",
        "pdb_path",
        "L",
    }

    unique_cols = []  # monomer-only metric columns  → add directly
    conflict_cols = []  # exist in both CSVs            → add with _monomer suffix

    for col in monomer_df.columns:
        if col in metadata_cols:
            continue
        if col in motif_cols:
            conflict_cols.append(col)
        else:
            unique_cols.append(col)

    if not unique_cols and not conflict_cols:
        logger.debug("No new monomer metric columns to merge (all columns are metadata)")
        return motif_df

    # Build the subset to merge
    # Unique cols keep their original name; conflict cols get _monomer suffix
    rename_map = {col: f"{col}_monomer" for col in conflict_cols}
    merge_cols_from_monomer = ["pdb_path"] + unique_cols + conflict_cols
    monomer_subset = monomer_df[merge_cols_from_monomer].rename(columns=rename_map).drop_duplicates(subset=["pdb_path"])

    # Left-join: keep all motif rows, add monomer columns where pdb_path matches
    merged_df = pd.merge(motif_df, monomer_subset, on="pdb_path", how="left")

    # Log summary
    added_cols = unique_cols + [f"{c}_monomer" for c in conflict_cols]
    first_added = added_cols[0] if added_cols else None
    n_matched = merged_df[first_added].notna().sum() if first_added else 0
    logger.info(
        f"  Merged monomer results ({n_matched}/{len(merged_df)} rows matched): "
        f"{len(unique_cols)} unique cols, {len(conflict_cols)} conflict cols (→ _monomer suffix)"
    )
    if unique_cols:
        logger.debug(f"  Unique monomer cols: {unique_cols}")
    if conflict_cols:
        logger.debug(f"  Conflict cols (added as *_monomer): {conflict_cols}")

    return merged_df


# =============================================================================
# Auto-Detection Helpers
# =============================================================================


def detect_sequence_types_from_columns(df: pd.DataFrame) -> list[str]:
    """
    Auto-detect which sequence types were evaluated by examining column names.

    The evaluation step creates columns like:
        - complex_irmsd_self, complex_irmsd_mpnn, complex_irmsd_mpnn_fixed
        - complex_scrmsd_self, complex_scrmsd_mpnn, complex_scrmsd_mpnn_fixed
        - refolded_self_*, refolded_mpnn_*, refolded_mpnn_fixed_*

    This function scans the DataFrame columns to detect which sequence types
    were actually computed during evaluation.

    Args:
        df: DataFrame with evaluation results

    Returns:
        List of detected sequence types (e.g., ["self", "mpnn", "mpnn_fixed"])
    """
    # Patterns that indicate a sequence type was evaluated
    detected = set()

    for col in df.columns:
        for seq_type in SEQUENCE_TYPES:
            # Check for {seq_type}_complex_* or {seq_type}_motif_* columns
            # (binder and motif binder evals use seq_type as column prefix)
            if (
                col.startswith(f"{seq_type}_complex_")
                or col.startswith(f"{seq_type}_motif_")
                or col.startswith(f"{seq_type}_binder_")
                or col.startswith(f"{seq_type}_ligand_")
            ):
                detected.add(seq_type)

    # Return in canonical order
    result = [st for st in SEQUENCE_TYPES if st in detected]

    if result:
        logger.info(f"Auto-detected sequence types from columns: {result}")
    else:
        logger.warning("Could not auto-detect sequence types from columns, using defaults")
        result = list(SEQUENCE_TYPES)

    return result


# =============================================================================
# Groupby Column Detection
# =============================================================================


def get_groupby_columns(df: pd.DataFrame) -> list[str]:
    """
    Determine which columns to use for grouping results.

    This function identifies columns that represent hyperparameters or configuration
    settings (as opposed to results or sample-specific data).

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names suitable for grouping
    """
    groupby_cols = []

    # Substring patterns to exclude from groupby
    exclude_substr = [
        "_res_",  # Result columns
        "complex_",  # Complex metrics
        "binder_",  # Binder metrics
        "refolded_",  # Refolded structure metrics
        "generated_",  # Generated structure metrics
        "_tmol",  # TMOL metrics
        "_ligand",  # Ligand metrics
    ]

    # Prefix patterns to exclude: per-sample sequence type metrics
    exclude_prefixes = [
        "self_",  # Self (original) sequence metrics
        "mpnn_",  # ProteinMPNN redesigned metrics
        "mpnn_fixed_",  # ProteinMPNN fixed-residue metrics
    ]

    for col in df.columns:
        # Skip known sample-specific columns
        if col in GROUPBY_IGNORE_COLS:
            continue

        # Skip columns matching substring exclude patterns
        skip = any(pattern in col for pattern in exclude_substr)

        # Skip columns starting with per-sample sequence type prefixes
        if not skip:
            skip = any(col.startswith(prefix) for prefix in exclude_prefixes)

        if not skip:
            groupby_cols.append(col)

    # Drop task_name from groupby if it's all null (pure monomer runs)
    if "task_name" in groupby_cols and "task_name" in df.columns and df["task_name"].isna().all():
        groupby_cols.remove("task_name")

    # Always ensure run_name is included if present
    if "run_name" not in groupby_cols and "run_name" in df.columns:
        groupby_cols.insert(0, "run_name")

    # Fallback to run_name only if no columns found
    if not groupby_cols:
        if "run_name" in df.columns:
            groupby_cols = ["run_name"]
        else:
            logger.warning("No groupby columns detected, using first non-result column")
            for col in df.columns:
                if not any(p in col for p in exclude_substr):
                    groupby_cols = [col]
                    break

    logger.info(f"  Groupby cols:  {', '.join(groupby_cols[:5])}{'...' if len(groupby_cols) > 5 else ''}")
    return groupby_cols


# =============================================================================
# Interface Metrics Aggregation
# =============================================================================


def _get_interface_metric_columns(seq_type: str) -> list[str]:
    """Get the list of interface metric columns for a sequence type."""
    ff_metrics = [
        "generated_total_interface_hbond_energy_tmol",
        "generated_n_interface_hbonds_tmol",
        "generated_total_interface_elec_energy_tmol",
        "generated_n_interface_elec_interactions_tmol",
        f"refolded_{seq_type}_total_interface_hbond_energy_tmol",
        f"refolded_{seq_type}_n_interface_hbonds_tmol",
        f"refolded_{seq_type}_total_interface_elec_energy_tmol",
        f"refolded_{seq_type}_n_interface_elec_interactions_tmol",
    ]
    bio_metrics = [
        "generated_binder_surface_hydrophobicity",
        "generated_binder_interface_sc",
        "generated_binder_interface_dSASA",
        "generated_binder_interface_fraction",
        "generated_binder_interface_hydrophobicity",
        "generated_binder_interface_nres",
        f"refolded_{seq_type}_binder_surface_hydrophobicity",
        f"refolded_{seq_type}_binder_interface_sc",
        f"refolded_{seq_type}_binder_interface_dSASA",
        f"refolded_{seq_type}_binder_interface_fraction",
        f"refolded_{seq_type}_binder_interface_hydrophobicity",
        f"refolded_{seq_type}_binder_interface_nres",
    ]
    return ff_metrics + bio_metrics


def aggregate_interface_metrics_for_successful_samples(
    successful_dfs: dict[str, pd.DataFrame],
    groupby_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Aggregate force field and bioinformatics metrics for pre-filtered successful samples.

    Aggregates per config group using groupby_cols, allowing comparison across
    different sweep configurations.

    Args:
        successful_dfs: Dictionary mapping sequence type to pre-filtered DataFrame
            of successful samples for that sequence type
        groupby_cols: Columns to group by for aggregation. If None, computes
            global means (legacy behavior).

    Returns:
        DataFrame with aggregated metrics for successful samples
    """
    all_seq_type_dfs = []

    for seq_type, df_successful in successful_dfs.items():
        try:
            if len(df_successful) == 0:
                logger.info(f"No successful samples for {seq_type}, skipping interface metrics")
                continue

            all_metrics = _get_interface_metric_columns(seq_type)
            existing_metrics = [m for m in all_metrics if m in df_successful.columns]

            if not existing_metrics:
                logger.warning(f"No interface metric columns found for {seq_type}")
                continue

            # Replace invalid values before aggregation
            df_clean = df_successful[existing_metrics].replace([-1, float("inf"), float("-inf")], pd.NA)

            if groupby_cols:
                # Filter groupby_cols to only those present in df
                valid_groupby = [c for c in groupby_cols if c in df_successful.columns]
                if not valid_groupby:
                    logger.warning(f"No valid groupby columns for {seq_type}, using global means")
                    groupby_cols_to_use = None
                else:
                    groupby_cols_to_use = valid_groupby
            else:
                groupby_cols_to_use = None

            if groupby_cols_to_use:
                # Aggregate per config group
                df_for_agg = pd.concat([df_successful[groupby_cols_to_use], df_clean], axis=1)
                df_grouped = df_for_agg.groupby(groupby_cols_to_use, dropna=False)

                # Build mean and count DataFrames
                df_means = df_grouped[existing_metrics].mean().reset_index()
                df_counts = df_grouped[existing_metrics].count().reset_index()

                # Rename: metric -> _res_{metric}_successful_{seq_type}_mean
                rename_means = {}
                rename_counts = {}
                for m in existing_metrics:
                    clean_name = m.replace(f"_{seq_type}_", "_")
                    rename_means[m] = f"_res_{clean_name}_successful_{seq_type}_mean"
                    rename_counts[m] = f"_res_{clean_name}_successful_{seq_type}_n_valid"

                df_means = df_means.rename(columns=rename_means)
                df_counts = df_counts.rename(columns=rename_counts)

                # Add n_successful_samples per group
                df_n = (
                    df_successful.groupby(groupby_cols_to_use, dropna=False)
                    .size()
                    .reset_index(name=f"_res_n_successful_{seq_type}")
                )

                # Merge means, counts, and n_successful
                result = df_means.merge(df_counts, on=groupby_cols_to_use, how="outer")
                result = result.merge(df_n, on=groupby_cols_to_use, how="outer")
            else:
                # Legacy: global means (no groupby)
                result_dict = {
                    f"_res_n_successful_{seq_type}": len(df_successful),
                }
                for metric in existing_metrics:
                    valid_values = df_clean[metric].dropna()
                    clean_name = metric.replace(f"_{seq_type}_", "_")
                    if len(valid_values) > 0:
                        result_dict[f"_res_{clean_name}_successful_{seq_type}_mean"] = valid_values.mean()
                        result_dict[f"_res_{clean_name}_successful_{seq_type}_n_valid"] = len(valid_values)
                    else:
                        result_dict[f"_res_{clean_name}_successful_{seq_type}_mean"] = None
                        result_dict[f"_res_{clean_name}_successful_{seq_type}_n_valid"] = 0
                result = pd.DataFrame([result_dict])

            all_seq_type_dfs.append(result)

        except Exception as e:
            logger.warning(f"Failed to aggregate metrics for {seq_type}: {e}")

    if all_seq_type_dfs:
        # Merge all sequence type results
        if groupby_cols:
            valid_groupby = [c for c in groupby_cols if c in all_seq_type_dfs[0].columns]
            if valid_groupby:
                merged = all_seq_type_dfs[0]
                for df_next in all_seq_type_dfs[1:]:
                    merged = pd.merge(merged, df_next, on=valid_groupby, how="outer")
                return merged
        # Fallback: concat rows (one row per seq_type)
        return pd.concat(all_seq_type_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


# =============================================================================
# Monomer Structure Metrics
# =============================================================================


def compute_secondary_structure(
    df: pd.DataFrame,
    groupby_cols: list[str],
    results_dir: str,
    metric_suffix: str = "all_samples",
) -> pd.DataFrame:
    """
    Aggregate secondary structure proportions (alpha helix, beta sheet, coil)
    for each group of samples.

    Prefers pre-computed per-sample columns (_res_ss_alpha, _res_ss_beta,
    _res_ss_coil) from the evaluate step. Falls back to computing from PDB
    files via biotite if those columns are not present.

    Args:
        df: DataFrame with pre-computed SS columns or a 'pdb_path' column
        groupby_cols: Columns to group results by
        results_dir: Directory to save result CSV
        metric_suffix: Suffix for metric column names and output filename

    Returns:
        DataFrame with columns: groupby_cols + _res_ss_alpha_{suffix},
        _res_ss_beta_{suffix}, _res_ss_coil_{suffix}
    """
    logger.debug(f"Aggregating secondary structure metrics - {metric_suffix}")

    ss_cols = ["_res_ss_alpha", "_res_ss_beta", "_res_ss_coil"]
    has_precomputed = all(col in df.columns for col in ss_cols)

    if has_precomputed:
        df_grouped = df.groupby(groupby_cols, dropna=False)[ss_cols].mean().reset_index()
        # Rename columns with metric_suffix
        df_grouped = df_grouped.rename(columns={col: f"{col}_{metric_suffix}" for col in ss_cols})
    elif "pdb_path" in df.columns:
        # Fallback: compute from PDB files directly
        from proteinfoundation.metrics.structural_metric_ss_ca_ca import compute_ss_metrics

        logger.debug("SS columns not found, falling back to computing from PDB files")
        df_grouped = df.groupby(groupby_cols, dropna=False)["pdb_path"].agg(list).reset_index()

        ss_alpha, ss_beta, ss_coil = [], [], []
        for _, row in df_grouped.iterrows():
            pdb_paths = row["pdb_path"]
            ss_results = []
            for path in pdb_paths:
                try:
                    ss_results.append(compute_ss_metrics(path))
                except Exception as e:
                    logger.warning(f"Failed to compute SS for {path}: {e}")

            if ss_results:
                ss_alpha.append(np.mean([r["biot_alpha"] for r in ss_results]))
                ss_beta.append(np.mean([r["biot_beta"] for r in ss_results]))
                ss_coil.append(np.mean([r["biot_coil"] for r in ss_results]))
            else:
                ss_alpha.append(0.0)
                ss_beta.append(0.0)
                ss_coil.append(0.0)

        df_grouped[f"_res_ss_alpha_{metric_suffix}"] = ss_alpha
        df_grouped[f"_res_ss_beta_{metric_suffix}"] = ss_beta
        df_grouped[f"_res_ss_coil_{metric_suffix}"] = ss_coil
        df_grouped = df_grouped.drop("pdb_path", axis=1)
    else:
        logger.warning("No SS columns or pdb_path found, skipping SS computation")
        return pd.DataFrame()

    csv_path = os.path.join(results_dir, f"res_ss_{metric_suffix}.csv")
    df_grouped.to_csv(csv_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
    logger.debug(f"SS saved: {csv_path}")

    return df_grouped


def _count_residues_from_pdb(pdb_path: str) -> dict[str, int]:
    """
    Count occurrences of each standard amino acid in a PDB file.

    Reads ATOM/HETATM records and counts unique residues identified by
    (chain_id, residue_number, residue_name) tuples.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary mapping 3-letter amino acid codes to counts
    """
    residue_count = dict.fromkeys(restype_3to1, 0)
    seen_residues = set()

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                chain_id = line[21].strip()
                unique_id = (chain_id, res_seq, res_name)

                if res_name in restype_3to1 and unique_id not in seen_residues:
                    seen_residues.add(unique_id)
                    residue_count[res_name] += 1

    return residue_count


def compute_residue_type_distribution(
    df: pd.DataFrame,
    groupby_cols: list[str],
    results_dir: str,
    metric_suffix: str = "all_samples",
) -> pd.DataFrame:
    """
    Compute residue type proportions across samples for each group.

    For each group, counts all amino acid residues across all PDB files
    and computes the proportion of each of the 20 standard amino acid types.

    Args:
        df: DataFrame with a 'pdb_path' column
        groupby_cols: Columns to group results by
        results_dir: Directory to save result CSV
        metric_suffix: Suffix for metric column names and output filename

    Returns:
        DataFrame with columns: groupby_cols + _res_aa_prop_{AA}_{suffix}
        for each amino acid (1-letter code)
    """
    logger.debug(f"Computing residue type distribution - {metric_suffix}")

    if "pdb_path" not in df.columns:
        logger.warning("No 'pdb_path' column found, skipping residue type distribution")
        return pd.DataFrame()

    df_grouped = df.groupby(groupby_cols, dropna=False)["pdb_path"].agg(list).reset_index()

    all_proportions = []
    for _, row in df_grouped.iterrows():
        pdb_paths = row["pdb_path"]

        # Aggregate counts across all PDBs in this group
        total_counts = dict.fromkeys(restype_3to1, 0)
        for path in pdb_paths:
            try:
                counts = _count_residues_from_pdb(path)
                for aa, count in counts.items():
                    total_counts[aa] += count
            except Exception as e:
                logger.warning(f"Failed to count residues for {path}: {e}")

        # Convert to proportions
        total = sum(total_counts.values())
        if total > 0:
            proportions = {aa: count / total for aa, count in total_counts.items()}
        else:
            proportions = dict.fromkeys(restype_3to1, 0.0)

        all_proportions.append(proportions)

    # Add proportion columns (use 1-letter codes for readability)
    for aa_3letter, aa_1letter in restype_3to1.items():
        df_grouped[f"_res_aa_prop_{aa_1letter}_{metric_suffix}"] = [p[aa_3letter] for p in all_proportions]

    # Drop pdb_path list column and save
    df_grouped = df_grouped.drop("pdb_path", axis=1)

    csv_path = os.path.join(results_dir, f"res_aa_distribution_{metric_suffix}.csv")
    df_grouped.to_csv(csv_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
    logger.debug(f"AA distribution saved: {csv_path}")

    return df_grouped


# =============================================================================
# Shared DataFrame Merge Helpers
# =============================================================================


def _safe_merge_dfs(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge two DataFrames on shared non-result columns."""
    common = list(set(left.columns) & set(right.columns) - {c for c in right.columns if c.startswith("_res_")})
    if not common:
        # Fallback: find any shared columns
        common = list(set(left.columns) & set(right.columns))
    return pd.merge(left, right, on=common, how="outer")


def _compare_series(series: pd.Series, threshold: float, op: str) -> pd.Series:
    """Compare a pandas Series against a threshold using the given operator."""
    ops = {
        "<=": series.__le__,
        "<": series.__lt__,
        ">=": series.__ge__,
        ">": series.__gt__,
        "==": series.__eq__,
    }
    return ops.get(op, series.__eq__)(threshold)


# =============================================================================
# Overall Metrics Computation
# =============================================================================


def compute_binder_overall_metrics(
    df: pd.DataFrame,
    results_dir: str,
    config_name: str,
    result_type: str,
    sequence_types: list[str] = None,
    success_thresholds: dict | None = None,
    cfg_aggregation: dict | None = None,
) -> None:
    """
    Compute overall performance metrics including pass rates, diversity, and aggregated metrics.

    This is the main analysis function that orchestrates all metric computations.

    Args:
        df: Combined DataFrame with all evaluation results
        results_dir: Directory to save results
        config_name: Config name for output file naming
        result_type: Result type ("protein_binder", "ligand_binder", "monomer")
        sequence_types: Sequence types to analyze
        success_thresholds: Custom success thresholds. If None, uses defaults.
        cfg_aggregation: Aggregation config dict (for metric toggles like compute_mmseqs_diversity)
    """
    if cfg_aggregation is None:
        cfg_aggregation = {}

    # Auto-detect sequence types from columns if not explicitly provided
    if sequence_types is None:
        sequence_types = detect_sequence_types_from_columns(df)

    # Determine thresholds
    if success_thresholds is not None:
        thresholds = normalize_threshold_dict(success_thresholds)
    elif result_type == "protein_binder":
        thresholds = DEFAULT_PROTEIN_BINDER_THRESHOLDS.copy()
    elif result_type == "ligand_binder":
        thresholds = DEFAULT_LIGAND_BINDER_THRESHOLDS.copy()
    else:
        logger.error(f"Result type {result_type} not supported for overall metrics")
        return

    # Log thresholds compactly
    thresh_parts = []
    for metric_name, spec in thresholds.items():
        parsed = parse_threshold_spec(spec)
        scale_str = f"*{parsed['scale']}" if parsed.get("scale", 1.0) != 1.0 else ""
        thresh_parts.append(
            f"{parsed.get('column_prefix', 'complex')}_{metric_name}{scale_str} "
            f"{parsed.get('op', '<=')} {parsed.get('threshold')}"
        )
    logger.info(f"  Thresholds:    {'; '.join(thresh_parts[:3])}{'...' if len(thresh_parts) > 3 else ''}")
    logger.info(f"  Seq types:     {', '.join(sequence_types)}")

    # Get groupby columns
    groupby_cols = get_groupby_columns(df)

    # Create filter function (save_json=False to use combined JSON save later)
    filter_pass_func = partial(
        filter_by_success_thresholds,
        success_thresholds=thresholds,
        path_store_results=results_dir,
        filter_name=result_type,
        save_json=False,
    )

    # Select pass rate computation function
    compute_pass_rate_func = (
        compute_filter_pass_rate if result_type == "protein_binder" else compute_filter_ligand_pass_rate
    )
    metric_suffix = "overall" if result_type == "protein_binder" else "all_samples"

    # ------------------------------------------------------------------
    # Step 1: Filter pass rates
    # ------------------------------------------------------------------
    logger.info("  [1/4] Computing filter pass rates")
    filter_pass_df = None
    try:
        filter_pass_df = compute_pass_rate_func(
            df=df,
            groupby_cols=groupby_cols,
            path_store_results=results_dir,
            metric_suffix=metric_suffix,
            success_thresholds=thresholds,
        )
    except Exception as e:
        logger.error(f"Failed to compute filter pass rates: {e}")

    # ------------------------------------------------------------------
    # Step 2: Filter successful samples per sequence type
    # ------------------------------------------------------------------
    logger.info("  [2/4] Filtering successful samples")
    successful_dfs: dict[str, pd.DataFrame] = {}
    for seq_type in sequence_types:
        try:
            df_successful = filter_pass_func(df, seq_type)
            successful_dfs[seq_type] = df_successful
        except Exception as e:
            logger.warning(f"Failed to filter by success for {seq_type}: {e}")

    success_summary = ", ".join(f"{k}({len(v)})" for k, v in successful_dfs.items())
    logger.info(f"         Successful: {success_summary}")

    # ------------------------------------------------------------------
    # Step 3: Diversity (FoldSeek + MMseqs)
    # ------------------------------------------------------------------
    diversity_results = []
    tmp_path_diversity = os.path.join(results_dir, "tmp_diversity")
    os.makedirs(tmp_path_diversity, exist_ok=True)

    if cfg_aggregation.get("compute_diversity", True):
        logger.info("  [3/4] FoldSeek + MMseqs diversity")
        try:
            diversity_df = compute_foldseek_diversity(
                df,
                groupby_cols,
                results_dir,
                tmp_path=tmp_path_diversity,
                metric_suffix="all_generated",
                min_seq_id=0.0,
                alignment_type=1,
                diversity_mode="binder",
            )
            diversity_results.append(diversity_df)
        except Exception as e:
            logger.debug(f"FoldSeek failed for all_generated: {e}")

        for seq_type, df_successful in successful_dfs.items():
            if len(df_successful) > 0:
                try:
                    diversity_df = compute_foldseek_diversity(
                        df_successful,
                        groupby_cols,
                        results_dir,
                        tmp_path=tmp_path_diversity,
                        metric_suffix=f"successful_{seq_type}",
                        min_seq_id=0.0,
                        alignment_type=1,
                        diversity_mode="binder",
                    )
                    diversity_results.append(diversity_df)
                except Exception as e:
                    logger.error(f"FoldSeek failed for successful_{seq_type}: {e}")

    if cfg_aggregation.get("compute_mmseqs_diversity", True):
        tmp_mmseqs_path = os.path.join(results_dir, "tmp_mmseqs_diversity")
        os.makedirs(tmp_mmseqs_path, exist_ok=True)
        mmseqs_min_seq_id = cfg_aggregation.get("mmseqs_min_seq_id", 0.1)
        mmseqs_coverage = cfg_aggregation.get("mmseqs_coverage", 0.7)

        try:
            mmseqs_df = compute_mmseqs_diversity(
                df,
                groupby_cols,
                results_dir,
                tmp_path=tmp_mmseqs_path,
                metric_suffix="all_generated",
                min_seq_id=mmseqs_min_seq_id,
                coverage=mmseqs_coverage,
                diversity_mode="binder",
            )
            diversity_results.append(mmseqs_df)
        except Exception as e:
            logger.error(f"MMseqs failed for all_generated: {e}")

        for seq_type, df_successful in successful_dfs.items():
            if len(df_successful) > 0:
                try:
                    mmseqs_df = compute_mmseqs_diversity(
                        df_successful,
                        groupby_cols,
                        results_dir,
                        tmp_path=tmp_mmseqs_path,
                        metric_suffix=f"successful_{seq_type}",
                        min_seq_id=mmseqs_min_seq_id,
                        coverage=mmseqs_coverage,
                        diversity_mode="binder",
                    )
                    diversity_results.append(mmseqs_df)
                except Exception as e:
                    logger.error(f"MMseqs failed for successful_{seq_type}: {e}")

    # ------------------------------------------------------------------
    # Step 4: Aggregate interface metrics + save
    # ------------------------------------------------------------------
    logger.info("  [4/4] Interface metrics + save")
    try:
        aggregated_metrics_df = aggregate_interface_metrics_for_successful_samples(
            successful_dfs,
            groupby_cols=groupby_cols,
        )
        if not aggregated_metrics_df.empty:
            aggregated_csv_path = os.path.join(results_dir, f"aggregated_interface_metrics_{config_name}.csv")
            aggregated_metrics_df.to_csv(aggregated_csv_path, index=False)
            logger.debug(f"Interface metrics saved to {aggregated_csv_path}")
    except Exception as e:
        logger.error(f"Failed to compute aggregated interface metrics: {e}")

    save_combined_success_criteria_json(
        success_thresholds=thresholds,
        path_store_results=results_dir,
        filter_name=result_type,
        sequence_types=sequence_types,
    )

    # Combine all overall performance results
    overall_results = []
    if filter_pass_df is not None:
        overall_results.append(filter_pass_df)
    overall_results.extend(diversity_results)

    if overall_results:
        overall_df = reduce(_safe_merge_dfs, overall_results)
        overall_csv_path = os.path.join(results_dir, f"overall_binder_performance_{config_name}_aggregated.csv")
        overall_df.to_csv(overall_csv_path, index=False)
        logger.debug(f"Overall performance saved to {overall_csv_path}")


# =============================================================================
# Benchmark-Specific Analysis Functions
# =============================================================================


def run_binder_analysis(
    cfg: DictConfig,
    df: pd.DataFrame,
    results_dir: str,
    config_name: str,
    result_type: str,
) -> dict[str, pd.DataFrame]:
    """
    Run comprehensive binder analysis.

    Args:
        cfg: Configuration dictionary
        df: Combined results DataFrame
        results_dir: Output directory
        config_name: Configuration name
        result_type: "protein_binder" or "ligand_binder"

    Returns:
        Dictionary of result DataFrames
    """
    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("BINDER ANALYSIS"))
    logger.info("+" + "-" * 68 + "+")
    logger.info(f"  Samples:       {len(df)}")
    logger.info(f"  Result type:   {result_type}")

    cfg_aggregation = cfg.get("aggregation", {})

    # Get analysis settings
    sequence_types = cfg_aggregation.get("sequence_types", None)
    if sequence_types is not None:
        sequence_types = list(sequence_types) if hasattr(sequence_types, "__iter__") else [sequence_types]

    success_thresholds = cfg_aggregation.get("success_thresholds", None)
    if success_thresholds is not None:
        success_thresholds = OmegaConf.to_container(success_thresholds, resolve=True)

    thresh_src = "custom" if success_thresholds is not None else "default"
    seq_src = ", ".join(sequence_types) if sequence_types else "auto-detect"
    logger.info(f"  Seq types:     {seq_src}")
    logger.info(f"  Thresholds:    {thresh_src}")
    logger.info("")

    # Compute overall metrics
    compute_binder_overall_metrics(
        df=df,
        results_dir=results_dir,
        config_name=config_name,
        result_type=result_type,
        sequence_types=sequence_types,
        success_thresholds=success_thresholds,
        cfg_aggregation=cfg_aggregation,
    )

    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("BINDER ANALYSIS COMPLETE"))
    logger.info("+" + "-" * 68 + "+")
    logger.info("")

    return {"combined": df}


# =============================================================================
# Shared Filter-then-Cluster Helpers
# =============================================================================
# Both monomer and motif analysis follow the same pattern:
#   1. Build a list of (suffix_label, filtered_df) subsets
#   2. Run FoldSeek diversity on each subset
#   3. Run MMseqs diversity on each subset
#   4. Run secondary structure / residue distribution on each subset
#
# The helpers below factor out this shared logic so both analysis modes
# can reuse it, differing only in *what* they filter by.


def build_monomer_filtered_subsets(
    df: pd.DataFrame,
    threshold_groups: list[dict],
) -> list[dict]:
    """Build (suffix, filtered_df) pairs by filtering on monomer-style thresholds.

    Each entry in *threshold_groups* is a dict:
        {
            "thresholds": {mode: {model: {threshold, op}}},
            "metric_type": "designability" | "codesignability" | ...,
            "suffix_prefix": "des" | "codes" | ...,
        }

    Returns a list of dicts ``{"suffix": str, "df": pd.DataFrame}``.
    """
    subsets = []
    for group in threshold_groups:
        thresholds = group.get("thresholds")
        metric_type = group.get("metric_type", "designability")
        prefix = group.get("suffix_prefix", metric_type)
        if not thresholds:
            continue
        for mode, models in thresholds.items():
            for model, spec in models.items():
                threshold = spec.get("threshold", 2.0) if isinstance(spec, dict) else spec
                op = spec.get("op", "<=") if isinstance(spec, dict) else "<="
                df_filtered = filter_monomer_by_single_threshold(
                    df=df,
                    mode=mode,
                    model=model,
                    threshold=threshold,
                    metric_type=metric_type,
                    op=op,
                )
                if len(df_filtered) > 0:
                    subsets.append(
                        {
                            "suffix": f"{prefix}_{mode}_{model}",
                            "df": df_filtered,
                        }
                    )
                    logger.debug(
                        f"Filter {metric_type} {mode}/{model} {op} {threshold}: {len(df_filtered)}/{len(df)} passed"
                    )
                else:
                    logger.debug(f"No samples passed {metric_type} {mode}/{model} {op} {threshold}")
    return subsets


def build_column_filtered_subsets(
    df: pd.DataFrame,
    column_filters: list[dict],
) -> list[dict]:
    """Build (suffix, filtered_df) pairs by filtering on arbitrary column thresholds.

    Each entry in *column_filters* is a dict:
        {
            "column": str,          # column name in df
            "threshold": float,
            "op": str,              # comparison operator
            "suffix": str,          # label for this subset
        }

    Useful for motif-region columns or any metric not covered by
    ``filter_monomer_by_single_threshold``.
    """
    subsets = []
    for filt in column_filters:
        col = filt["column"]
        if col not in df.columns:
            logger.warning(f"  Column {col} not found, skipping")
            continue
        threshold = filt["threshold"]
        op = filt.get("op", "<=")
        mask = _compare_series(pd.Series(df[col]), threshold, op)
        df_filtered = df[mask]
        suffix = filt["suffix"]
        if len(df_filtered) > 0:
            subsets.append({"suffix": suffix, "df": df_filtered})
            logger.debug(f"Filter {col} {op} {threshold}: {len(df_filtered)}/{len(df)} passed")
        else:
            logger.debug(f"No samples passed {col} {op} {threshold}")
    return subsets


def run_foldseek_on_subsets(
    df_all: pd.DataFrame,
    subsets: list[dict],
    groupby_cols: list[str],
    results_dir: str,
    tmp_path: str,
    results_accum: list,
    results_dict: dict,
) -> None:
    """Run FoldSeek structural diversity on all samples + each filtered subset.

    Args:
        df_all: Full (unfiltered) DataFrame
        subsets: List of ``{"suffix": str, "df": pd.DataFrame}`` dicts
        groupby_cols: Columns to group by
        results_dir: Output directory
        tmp_path: Temp directory for FoldSeek
        results_accum: List to append diversity DataFrames to
        results_dict: Dict to store named results
    """
    # All samples first
    try:
        div_df = compute_foldseek_diversity(
            df_all,
            groupby_cols,
            results_dir,
            tmp_path=tmp_path,
            metric_suffix="all_samples",
            min_seq_id=0.0,
            alignment_type=1,
            diversity_mode="monomer",
        )
        results_accum.append(div_df)
        results_dict["diversity_all"] = div_df
    except Exception as e:
        logger.warning(f"FoldSeek failed for all_samples: {e}")

    # Each filtered subset
    for subset in subsets:
        suffix, df_filt = subset["suffix"], subset["df"]
        try:
            div_df = compute_foldseek_diversity(
                df_filt,
                groupby_cols,
                results_dir,
                tmp_path=tmp_path,
                metric_suffix=suffix,
                min_seq_id=0.0,
                alignment_type=1,
                diversity_mode="monomer",
            )
            results_accum.append(div_df)
            results_dict[f"diversity_{suffix}"] = div_df
        except Exception as e:
            logger.debug(f"FoldSeek skipped for {suffix}: {e}")


def run_mmseqs_on_subsets(
    df_all: pd.DataFrame,
    subsets: list[dict],
    groupby_cols: list[str],
    results_dir: str,
    cfg_aggregation: dict,
    results_accum: list,
    results_dict: dict,
) -> None:
    """Run MMseqs sequence diversity on all samples + each filtered subset.

    Args:
        df_all: Full (unfiltered) DataFrame
        subsets: List of ``{"suffix": str, "df": pd.DataFrame}`` dicts
        groupby_cols: Columns to group by
        results_dir: Output directory
        cfg_aggregation: Aggregation config (for mmseqs params)
        results_accum: List to append diversity DataFrames to
        results_dict: Dict to store named results
    """
    tmp_mmseqs_path = os.path.join(results_dir, "tmp_mmseqs_diversity")
    os.makedirs(tmp_mmseqs_path, exist_ok=True)

    mmseqs_min_seq_id = cfg_aggregation.get("mmseqs_min_seq_id", 0.1)
    mmseqs_coverage = cfg_aggregation.get("mmseqs_coverage", 0.7)

    all_dfs = [{"suffix": "all_samples", "df": df_all}] + subsets
    for entry in all_dfs:
        suffix, df_sub = entry["suffix"], entry["df"]
        if len(df_sub) == 0:
            continue
        try:
            mmseqs_df = compute_mmseqs_diversity(
                df_sub,
                groupby_cols,
                results_dir,
                tmp_path=tmp_mmseqs_path,
                metric_suffix=suffix,
                min_seq_id=mmseqs_min_seq_id,
                coverage=mmseqs_coverage,
            )
            results_accum.append(mmseqs_df)
            results_dict[f"mmseqs_diversity_{suffix}"] = mmseqs_df
        except Exception as e:
            logger.debug(f"MMseqs skipped for {suffix}: {e}")


def compute_metric_on_subsets(
    df_all: pd.DataFrame,
    subsets: list[dict],
    compute_fn,
    metric_name: str,
    results_accum: list,
    results_dict: dict,
) -> None:
    """Compute a metric (SS, AA distribution, etc.) on all + filtered subsets.

    Args:
        df_all: Full (unfiltered) DataFrame
        subsets: List of ``{"suffix": str, "df": pd.DataFrame}`` dicts
        compute_fn: Callable(df, metric_suffix) -> pd.DataFrame
        metric_name: Short name for this metric (used in log messages and result keys)
        results_accum: List to append result DataFrames to
        results_dict: Dict to store named results
    """
    all_entries = [{"suffix": "all_samples", "df": df_all}] + subsets
    for entry in all_entries:
        suffix, df_sub = entry["suffix"], entry["df"]
        if len(df_sub) == 0:
            continue
        try:
            result_df = compute_fn(df_sub, suffix)
            if result_df is not None and not result_df.empty:
                results_accum.append(result_df)
                results_dict[f"{metric_name}_{suffix}"] = result_df
        except Exception as e:
            logger.debug(f"Skipped {metric_name} for {suffix}: {e}")


def merge_and_save_results(
    all_pass_rate_dfs: list[pd.DataFrame],
    all_diversity_dfs: list[pd.DataFrame],
    groupby_cols: list[str],
    results_dir: str,
    config_name: str,
    prefix: str,
    results_dict: dict,
) -> None:
    """Merge pass-rate and diversity DataFrames and save CSVs.

    Shared by both ``run_monomer_analysis`` and ``run_motif_analysis``.

    Args:
        all_pass_rate_dfs: Pass-rate DataFrames to merge
        all_diversity_dfs: Diversity / metric DataFrames to merge
        groupby_cols: Columns to merge on
        results_dir: Output directory
        config_name: Config name for output filenames
        prefix: File prefix ("monomer" or "motif")
        results_dict: Dict to store named results
    """
    if all_pass_rate_dfs:
        merged = reduce(
            lambda l, r: pd.merge(l, r, on=groupby_cols, how="outer"),
            all_pass_rate_dfs,
        )
        merged_f = filter_columns_for_csv(merged, log_dropped=False)
        path = os.path.join(results_dir, f"overall_{prefix}_pass_rates_{config_name}.csv")
        merged_f.to_csv(path, index=False)
        logger.debug(f"Overall pass rates saved to {path}")
        results_dict["overall_pass_rates"] = merged_f

    if all_diversity_dfs:
        merged_div = reduce(_safe_merge_dfs, all_diversity_dfs)
        merged_div = merged_div.drop(
            columns=[c for c in merged_div.columns if c.startswith("_res") and not c.startswith("_res_diversity")],
            errors="ignore",
        )
        path = os.path.join(results_dir, f"overall_{prefix}_diversity_{config_name}.csv")
        merged_div.to_csv(path, index=False)
        logger.debug(f"Overall diversity saved to {path}")
        results_dict["overall_diversity"] = merged_div


def run_monomer_analysis(
    cfg: DictConfig,
    df: pd.DataFrame,
    results_dir: str,
    config_name: str,
) -> dict[str, pd.DataFrame]:
    """
    Run comprehensive monomer analysis with threshold-based filtering.

    This function supports:
    - Computing pass rates for designability and codesignability metrics
    - Filtering samples by user-defined RMSD thresholds (individual or combined)
    - Computing diversity metrics on filtered subsets

    Args:
        cfg: Configuration dictionary
        df: Combined results DataFrame
        results_dir: Output directory
        config_name: Configuration name

    Configuration Keys (under aggregation):
        - compute_diversity: Whether to compute diversity metrics (default: True)
        - designability_thresholds: Custom thresholds for designability filtering
        - codesignability_thresholds: Custom thresholds for codesignability filtering
        - require_all_thresholds: If True, require all thresholds to pass (default: True)

    Returns:
        Dictionary of result DataFrames
    """
    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("MONOMER ANALYSIS"))
    logger.info("+" + "-" * 68 + "+")
    logger.info(f"  Samples:       {len(df)}")

    cfg_aggregation = cfg.get("aggregation", {})

    # Get groupby columns
    groupby_cols = get_groupby_columns(df)

    results = {}
    all_pass_rate_dfs = []
    all_diversity_dfs = []

    # Compute sample counts
    if "pdb_path" in df.columns:
        df_counts = (
            df.groupby(groupby_cols, dropna=False)["pdb_path"]
            .count()
            .reset_index()
            .rename(columns={"pdb_path": "n_samples"})
        )
        counts_path = os.path.join(results_dir, f"sample_counts_{config_name}.csv")
        df_counts.to_csv(counts_path, index=False)
        results["counts"] = df_counts

    # --- Threshold configuration ---
    designability_thresholds = cfg_aggregation.get("designability_thresholds", None)
    ca_codesignability_thresholds = cfg_aggregation.get("ca_codesignability_thresholds", None)
    allatom_codesignability_thresholds = cfg_aggregation.get("allatom_codesignability_thresholds", None)
    require_all = cfg_aggregation.get("require_all_thresholds", False)

    def _get_thresholds(cfg_value, metric_type: str, mode_filter: str = None, default: dict = None):
        if cfg_value is not None:
            return OmegaConf.to_container(cfg_value, resolve=True), "config"
        detected = detect_monomer_folding_models(df.columns, metric_type)
        if mode_filter:
            detected = {k: v for k, v in detected.items() if k == mode_filter}
        if detected:
            return build_thresholds_from_detected(detected), "auto-detected"
        return default.copy(), "default"

    designability_thresholds, des_src = _get_thresholds(
        designability_thresholds,
        "designability",
        None,
        DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS,
    )
    ca_codesignability_thresholds, ca_codes_src = _get_thresholds(
        ca_codesignability_thresholds,
        "codesignability",
        "ca",
        DEFAULT_MONOMER_CA_CODESIGNABILITY_THRESHOLDS,
    )
    allatom_codesignability_thresholds, aa_codes_src = _get_thresholds(
        allatom_codesignability_thresholds,
        "codesignability",
        "all_atom",
        DEFAULT_MONOMER_ALL_ATOM_CODESIGNABILITY_THRESHOLDS,
    )
    codesignability_thresholds = get_codesignability_thresholds(
        ca_thresholds=ca_codesignability_thresholds,
        allatom_thresholds=allatom_codesignability_thresholds,
    )

    def _fmt_monomer_thresh(thresholds: dict) -> str:
        parts = []
        for mode, models in thresholds.items():
            for model, spec in models.items():
                if isinstance(spec, dict):
                    parts.append(f"{mode}/{model} {spec.get('op', '<=')} {spec.get('threshold', 2.0)}")
                else:
                    parts.append(f"{mode}/{model} <= {spec}")
        return ", ".join(parts)

    logger.info("  Thresholds:")
    logger.info(f"    Designability ({des_src}):          {_fmt_monomer_thresh(designability_thresholds)}")
    logger.info(f"    CA codesignability ({ca_codes_src}):    {_fmt_monomer_thresh(ca_codesignability_thresholds)}")
    logger.info(
        f"    AA codesignability ({aa_codes_src}):    {_fmt_monomer_thresh(allatom_codesignability_thresholds)}"
    )
    logger.info("")

    # ==========================================================================
    # Compute pass rates for all samples
    # ==========================================================================

    def _save_monomer_csv(df_to_save: pd.DataFrame, filename: str):
        df_filtered = filter_columns_for_csv(df_to_save, log_dropped=False)
        output_path = os.path.join(results_dir, filename)
        df_filtered.to_csv(output_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
        logger.debug(f"Saved {filename}")
        return df_filtered

    # ------------------------------------------------------------------
    # Step 1: Pass rates (all samples)
    # ------------------------------------------------------------------
    logger.info("  [1/5] Designability + codesignability pass rates")
    if designability_thresholds:
        for metric_type, label, key_prefix in [
            ("designability", "designability (best of N)", "des"),
            (
                "single_designability",
                "single-MPNN designability (1st seq)",
                "single_des",
            ),
        ]:
            pass_df = compute_monomer_pass_rates(
                df=df,
                groupby_cols=groupby_cols,
                thresholds=designability_thresholds,
                metric_type=metric_type,
                path_store_results=None,
            )
            if not pass_df.empty:
                _save_monomer_csv(pass_df, f"res_monomer_{key_prefix}_pass_rates_all_samples.csv")
                all_pass_rate_dfs.append(pass_df)
                results[f"{key_prefix}_pass_rates"] = pass_df

    if codesignability_thresholds:
        codes_pass_df = compute_monomer_pass_rates(
            df=df,
            groupby_cols=groupby_cols,
            thresholds=codesignability_thresholds,
            metric_type="codesignability",
            path_store_results=None,
        )
        if not codes_pass_df.empty:
            _save_monomer_csv(codes_pass_df, "res_monomer_codesignability_pass_rates_all_samples.csv")
            all_pass_rate_dfs.append(codes_pass_df)
            results["codes_pass_rates"] = codes_pass_df

    # ------------------------------------------------------------------
    # Step 2: Pass rates by sample length
    # ------------------------------------------------------------------
    if "L" in df.columns:
        logger.info("  [2/5] Pass rates by sample length")
        if designability_thresholds:
            for metric_type, label, key_prefix, by_length_fn in [
                (
                    "designability",
                    "designability",
                    "des",
                    compute_designability_by_length,
                ),
                (
                    "single_designability",
                    "single-MPNN designability",
                    "single_des",
                    compute_single_designability_by_length,
                ),
            ]:
                by_length_df = by_length_fn(
                    df=df,
                    thresholds=designability_thresholds,
                    length_column="L",
                    path_store_results=None,
                    additional_groupby_cols=groupby_cols,
                )
                if not by_length_df.empty:
                    _save_monomer_csv(by_length_df, f"res_monomer_{key_prefix}_by_length.csv")
                    results[f"{key_prefix}_by_length"] = by_length_df

        if codesignability_thresholds:
            codes_by_length_df = compute_codesignability_by_length(
                df=df,
                thresholds=codesignability_thresholds,
                length_column="L",
                path_store_results=None,
                additional_groupby_cols=groupby_cols,
            )
            if not codes_by_length_df.empty:
                _save_monomer_csv(codes_by_length_df, "res_monomer_codesignability_by_length.csv")
                results["codes_by_length"] = codes_by_length_df
    else:
        logger.info("  [2/5] Pass rates by length — skipped (no L column)")

    # ==========================================================================
    # Build filtered subsets for diversity + metric computation
    # ==========================================================================

    monomer_subsets = build_monomer_filtered_subsets(
        df,
        [
            {
                "thresholds": designability_thresholds,
                "metric_type": "designability",
                "suffix_prefix": "des",
            },
            {
                "thresholds": designability_thresholds,
                "metric_type": "single_designability",
                "suffix_prefix": "single_des",
            },
            {
                "thresholds": codesignability_thresholds,
                "metric_type": "codesignability",
                "suffix_prefix": "codes",
            },
        ],
    )

    # ------------------------------------------------------------------
    # Step 3: Build filtered subsets + diversity
    # ------------------------------------------------------------------
    logger.info("  [3/5] Building filtered subsets")

    tmp_path = os.path.join(results_dir, "tmp_diversity")
    os.makedirs(tmp_path, exist_ok=True)

    subset_summary = ", ".join(f"{s['suffix']}({len(s['df'])})" for s in monomer_subsets)
    logger.info(f"         {len(monomer_subsets)} subsets: {subset_summary}")

    if "pdb_path" in df.columns and cfg_aggregation.get("compute_diversity", True):
        logger.info("  [4/5] FoldSeek + MMseqs diversity")
        run_foldseek_on_subsets(
            df_all=df,
            subsets=monomer_subsets,
            groupby_cols=groupby_cols,
            results_dir=results_dir,
            tmp_path=tmp_path,
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    # Secondary structure
    ss_cols_present = all(col in df.columns for col in ["_res_ss_alpha", "_res_ss_beta", "_res_ss_coil"])
    if ss_cols_present or "pdb_path" in df.columns:
        compute_metric_on_subsets(
            df_all=df,
            subsets=monomer_subsets,
            compute_fn=lambda df_in, suffix: compute_secondary_structure(
                df=df_in,
                groupby_cols=groupby_cols,
                results_dir=results_dir,
                metric_suffix=suffix,
            ),
            metric_name="ss",
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    # Residue type distribution
    if "pdb_path" in df.columns and cfg_aggregation.get("compute_residue_type_distribution", True):
        compute_metric_on_subsets(
            df_all=df,
            subsets=monomer_subsets,
            compute_fn=lambda df_in, suffix: compute_residue_type_distribution(
                df=df_in,
                groupby_cols=groupby_cols,
                results_dir=results_dir,
                metric_suffix=suffix,
            ),
            metric_name="aa_distribution",
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    if "pdb_path" in df.columns and cfg_aggregation.get("compute_mmseqs_diversity", True):
        run_mmseqs_on_subsets(
            df_all=df,
            subsets=monomer_subsets,
            groupby_cols=groupby_cols,
            results_dir=results_dir,
            cfg_aggregation=cfg_aggregation,
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    # ------------------------------------------------------------------
    # Step 5: Novelty + save
    # ------------------------------------------------------------------
    novelty_cols = [c for c in df.columns if "_res_novelty" in c]
    if novelty_cols:
        logger.info(f"  [5/5] Novelty metrics ({len(novelty_cols)} columns)")
        for col in novelty_cols:
            df_novelty = df.groupby(groupby_cols, dropna=False)[col].mean().reset_index()
            results[f"novelty_{col}"] = df_novelty
    else:
        logger.info("  [5/5] Novelty — skipped (no novelty columns)")

    merge_and_save_results(
        all_pass_rate_dfs,
        all_diversity_dfs,
        groupby_cols,
        results_dir,
        config_name,
        prefix="monomer",
        results_dict=results,
    )

    combined_thresholds = {
        "filter_name": f"analysis_{config_name}",
        "require_all": require_all,
        "ca_designability": designability_thresholds,
        "ca_codesignability": ca_codesignability_thresholds,
        "allatom_codesignability": allatom_codesignability_thresholds,
    }
    thresholds_path = os.path.join(results_dir, f"monomer_thresholds_{config_name}.json")
    with open(thresholds_path, "w") as f:
        json.dump(combined_thresholds, f, indent=2)

    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("MONOMER ANALYSIS COMPLETE"))
    logger.info("+" + "-" * 68 + "+")
    logger.info("")

    return results


def run_motif_analysis(
    cfg: DictConfig,
    df: pd.DataFrame,
    results_dir: str,
    config_name: str,
) -> dict[str, pd.DataFrame]:
    """
    Run motif analysis: motif-specific pass rates + reuse monomer for shared metrics.

    Motif evaluation produces the *same* designability/codesignability columns as
    monomer evaluation (_res_scRMSD_*, _res_co_scRMSD_*), so we reuse
    ``compute_monomer_pass_rates`` for those directly. The motif-unique metrics
    (motif RMSD, seq recovery, motif-region RMSD) use dedicated functions.

    Args:
        cfg: Configuration dictionary
        df: Combined results DataFrame
        results_dir: Output directory
        config_name: Configuration name

    Returns:
        Dictionary of result DataFrames
    """
    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("MOTIF ANALYSIS"))
    logger.info("+" + "-" * 68 + "+")
    logger.info(f"  Samples:       {len(df)}")

    cfg_aggregation = cfg.get("aggregation", {})
    groupby_cols = get_groupby_columns(df)

    results = {}
    all_pass_rate_dfs = []
    all_diversity_dfs = []

    # --- Helper ---
    def _save_csv(df_to_save: pd.DataFrame, filename: str):
        df_f = filter_columns_for_csv(df_to_save, log_dropped=False)
        df_f.to_csv(
            os.path.join(results_dir, filename),
            sep=SEP_CSV_PD,
            index=False,
            float_format=FLOAT_FORMAT_PD,
        )
        logger.debug(f"Saved {filename}")
        return df_f

    def _cfg_or_default(key, default):
        val = cfg_aggregation.get(key, None)
        if val is not None:
            return OmegaConf.to_container(val, resolve=True)
        return default.copy()

    # --- Folding model (auto-detect from result columns, or override via config) ---
    folding_model = cfg_aggregation.get("folding_model", None)
    if folding_model is None:
        detected = detect_monomer_folding_models(list(df.columns), "codesignability")
        if detected:
            folding_model = next(iter(next(iter(detected.values()))))
        else:
            folding_model = "esmfold"
    logger.info(f"  Folding model: {folding_model}")

    # --- Thresholds ---
    motif_rmsd_thresh = _cfg_or_default("motif_rmsd_thresholds", DEFAULT_MOTIF_RMSD_THRESHOLDS)
    motif_seq_rec_thresh = _cfg_or_default("motif_seq_rec_threshold", DEFAULT_MOTIF_SEQ_REC_THRESHOLD)
    des_thresh = resolve_thresholds(
        _cfg_or_default("designability_thresholds", DEFAULT_MOTIF_DESIGNABILITY_THRESHOLDS),
        model=folding_model,
    )
    codes_thresh = resolve_thresholds(
        _cfg_or_default("codesignability_thresholds", DEFAULT_MOTIF_CODESIGNABILITY_THRESHOLDS),
        model=folding_model,
    )
    motif_des_thresh = resolve_thresholds(
        _cfg_or_default(
            "motif_region_designability_thresholds",
            DEFAULT_MOTIF_REGION_DESIGNABILITY_THRESHOLDS,
        ),
        model=folding_model,
    )
    motif_codes_thresh = resolve_thresholds(
        _cfg_or_default(
            "motif_region_codesignability_thresholds",
            DEFAULT_MOTIF_REGION_CODESIGNABILITY_THRESHOLDS,
        ),
        model=folding_model,
    )

    # Log thresholds compactly
    def _fmt_thresh(d):
        """Format threshold dict into compact 'mode/model op val' strings."""
        parts = []
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        parts.append(f"{k}/{k2} {v2.get('op', '<=')} {v2.get('threshold', '?')}")
                    else:
                        parts.append(f"{k} {v.get('op', '<=')} {v.get('threshold', '?')}")
                        break
            else:
                parts.append(f"{k}: {v}")
        return ", ".join(parts) if parts else str(d)

    logger.info("  Thresholds:")
    logger.info(f"    Motif RMSD:          {_fmt_thresh(motif_rmsd_thresh)}")
    logger.info(
        f"    Motif seq rec:       {motif_seq_rec_thresh.get('op', '>=')} {motif_seq_rec_thresh.get('threshold', '?')}"
    )
    logger.info(f"    Designability:       {_fmt_thresh(des_thresh)}")
    logger.info(f"    Codesignability:     {_fmt_thresh(codes_thresh)}")
    logger.info(f"    Motif-region des:    {_fmt_thresh(motif_des_thresh)}")
    logger.info(f"    Motif-region codes:  {_fmt_thresh(motif_codes_thresh)}")
    logger.info("")

    # ------------------------------------------------------------------
    # Step 1: Motif-specific pass rates
    # ------------------------------------------------------------------
    logger.info("  [1/8] Motif RMSD pass rates")
    motif_rmsd_df = compute_motif_rmsd_pass_rates(df, groupby_cols, motif_rmsd_thresh)
    if not motif_rmsd_df.empty:
        _save_csv(motif_rmsd_df, "res_motif_rmsd_pass_rates.csv")
        all_pass_rate_dfs.append(motif_rmsd_df)
        results["motif_rmsd_pass_rates"] = motif_rmsd_df

    logger.info("  [2/8] Motif sequence recovery pass rates")
    seq_rec_df = compute_motif_seq_rec_pass_rates(df, groupby_cols, motif_seq_rec_thresh)
    if not seq_rec_df.empty:
        _save_csv(seq_rec_df, "res_motif_seq_rec_pass_rates.csv")
        all_pass_rate_dfs.append(seq_rec_df)
        results["motif_seq_rec_pass_rates"] = seq_rec_df

    # ------------------------------------------------------------------
    # Step 2: Full-structure pass rates (reuses monomer functions)
    # ------------------------------------------------------------------
    logger.info("  [3/8] Full-structure designability + codesignability pass rates")
    for thresholds, metric_type, key_prefix in [
        (des_thresh, "designability", "des"),
        (codes_thresh, "codesignability", "codes"),
    ]:
        if thresholds:
            pass_df = compute_monomer_pass_rates(
                df=df,
                groupby_cols=groupby_cols,
                thresholds=thresholds,
                metric_type=metric_type,
            )
            if not pass_df.empty:
                _save_csv(pass_df, f"res_motif_{key_prefix}_pass_rates.csv")
                all_pass_rate_dfs.append(pass_df)
                results[f"{key_prefix}_pass_rates"] = pass_df

    # ------------------------------------------------------------------
    # Step 3: Motif-region pass rates
    # ------------------------------------------------------------------
    logger.info("  [4/8] Motif-region designability + codesignability pass rates")
    for thresholds, metric_type, key_prefix in [
        (motif_des_thresh, "motif_designability", "motif_des"),
        (motif_codes_thresh, "motif_codesignability", "motif_codes"),
    ]:
        if thresholds:
            pass_df = compute_motif_region_pass_rates(
                df=df,
                groupby_cols=groupby_cols,
                thresholds=thresholds,
                metric_type=metric_type,
            )
            if not pass_df.empty:
                _save_csv(pass_df, f"res_motif_{key_prefix}_pass_rates.csv")
                all_pass_rate_dfs.append(pass_df)
                results[f"{key_prefix}_pass_rates"] = pass_df

    # ------------------------------------------------------------------
    # Step 4: Combined success criteria (all must pass per sample)
    # ------------------------------------------------------------------
    criteria_to_run = dict(MOTIF_SUCCESS_PRESETS)
    custom_criteria = cfg_aggregation.get("motif_success_criteria", None)
    if custom_criteria is not None:
        criteria_to_run["custom_motif_success"] = OmegaConf.to_container(custom_criteria, resolve=True)
    logger.info(f"  [5/8] Combined success criteria ({len(criteria_to_run)} presets)")
    for preset_name, criteria in criteria_to_run.items():
        resolved = resolve_success_criteria(criteria, model=folding_model)
        rules = "  AND  ".join(f"{c['column']} {c['op']} {c['threshold']}" for c in resolved)
        logger.info(f"         {preset_name}:")
        logger.info(f"           {rules}")

    for preset_name, criteria in criteria_to_run.items():
        pass_df = compute_motif_success_pass_rates(
            df,
            groupby_cols,
            criteria,
            model=folding_model,
            suffix=preset_name,
        )
        if not pass_df.empty:
            _save_csv(pass_df, f"res_{preset_name}_pass_rates.csv")
            all_pass_rate_dfs.append(pass_df)
            results[f"{preset_name}_pass_rates"] = pass_df

    # ------------------------------------------------------------------
    # Step 5: Build filtered subsets for diversity
    # ------------------------------------------------------------------
    logger.info("  [6/8] Building filtered subsets")

    # 5a. Monomer-style threshold subsets (full-structure des + codes)
    motif_subsets = build_monomer_filtered_subsets(
        df,
        [
            {
                "thresholds": des_thresh,
                "metric_type": "designability",
                "suffix_prefix": "des",
            },
            {
                "thresholds": codes_thresh,
                "metric_type": "codesignability",
                "suffix_prefix": "codes",
            },
        ],
    )

    # 5b. Motif-region threshold subsets
    motif_region_filters = []
    for thresholds, metric_type, prefix in [
        (motif_des_thresh, "motif_designability", "motif_des"),
        (motif_codes_thresh, "motif_codesignability", "motif_codes"),
    ]:
        if not thresholds:
            continue
        for mode, models in thresholds.items():
            for model, spec in models.items():
                threshold = spec.get("threshold", 2.0) if isinstance(spec, dict) else spec
                op = spec.get("op", "<=") if isinstance(spec, dict) else "<="
                motif_region_filters.append(
                    {
                        "column": build_motif_region_column_name(metric_type, mode, model),
                        "threshold": threshold,
                        "op": op,
                        "suffix": f"{prefix}_{mode}_{model}",
                    }
                )
    motif_subsets += build_column_filtered_subsets(df, motif_region_filters)

    # 5c. Direct motif RMSD subsets
    for mode, spec in normalize_motif_rmsd_thresholds(motif_rmsd_thresh).items():
        col = f"_res_motif_rmsd_{mode}"
        if col not in df.columns:
            continue
        df_filt = filter_by_motif_rmsd(df, {mode: spec}, require_all=True)
        if len(df_filt) > 0:
            motif_subsets.append({"suffix": f"motif_rmsd_{mode}", "df": df_filt})

    # 5d. Combined success criteria subsets
    for preset_name, criteria in criteria_to_run.items():
        df_success = filter_by_motif_success(df, criteria, model=folding_model)
        if len(df_success) > 0:
            motif_subsets.append({"suffix": f"success_{preset_name}", "df": df_success})

    subset_summary = ", ".join(f"{s['suffix']}({len(s['df'])})" for s in motif_subsets)
    logger.info(f"         {len(motif_subsets)} subsets: {subset_summary}")

    # ------------------------------------------------------------------
    # Step 6: Structural + sequence diversity
    # ------------------------------------------------------------------
    tmp_path = os.path.join(results_dir, "tmp_diversity")
    os.makedirs(tmp_path, exist_ok=True)

    if "pdb_path" in df.columns and cfg_aggregation.get("compute_diversity", True):
        logger.info("  [7/8] FoldSeek structural diversity")
        run_foldseek_on_subsets(
            df_all=df,
            subsets=motif_subsets,
            groupby_cols=groupby_cols,
            results_dir=results_dir,
            tmp_path=tmp_path,
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    if "pdb_path" in df.columns and cfg_aggregation.get("compute_mmseqs_diversity", True):
        logger.info("  [7/8] MMseqs sequence diversity")
        run_mmseqs_on_subsets(
            df_all=df,
            subsets=motif_subsets,
            groupby_cols=groupby_cols,
            results_dir=results_dir,
            cfg_aggregation=cfg_aggregation,
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    # ------------------------------------------------------------------
    # Step 7: Secondary structure
    # ------------------------------------------------------------------
    ss_ok = all(c in df.columns for c in ["_res_ss_alpha", "_res_ss_beta", "_res_ss_coil"])
    if ss_ok or "pdb_path" in df.columns:
        logger.info("  [8/8] Secondary structure aggregation")
        compute_metric_on_subsets(
            df_all=df,
            subsets=motif_subsets,
            compute_fn=lambda df_in, suffix: compute_secondary_structure(
                df=df_in,
                groupby_cols=groupby_cols,
                results_dir=results_dir,
                metric_suffix=suffix,
            ),
            metric_name="ss",
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    # ------------------------------------------------------------------
    # Merge, save, and report
    # ------------------------------------------------------------------
    merge_and_save_results(
        all_pass_rate_dfs,
        all_diversity_dfs,
        groupby_cols,
        results_dir,
        config_name,
        prefix="motif",
        results_dict=results,
    )

    save_motif_thresholds_json(
        motif_rmsd_thresholds=motif_rmsd_thresh,
        designability_thresholds=des_thresh,
        codesignability_thresholds=codes_thresh,
        motif_region_designability_thresholds=motif_des_thresh,
        motif_region_codesignability_thresholds=motif_codes_thresh,
        motif_seq_rec_threshold=motif_seq_rec_thresh,
        success_criteria=criteria_to_run,
        folding_model=folding_model,
        path_store_results=results_dir,
        filter_name=f"analysis_{config_name}",
    )

    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("MOTIF ANALYSIS COMPLETE"))
    logger.info("+" + "-" * 68 + "+")
    logger.info("")
    return results


# =============================================================================
# Motif Binder Analysis
# =============================================================================


def run_motif_binder_analysis(
    cfg: DictConfig,
    df: pd.DataFrame,
    results_dir: str,
    config_name: str,
    result_type: str,
) -> dict[str, pd.DataFrame]:
    """
    Run motif binder analysis: joint binder+motif success criteria.

    Combines binder success criteria (ipAE, pLDDT, scRMSD) with motif-specific
    criteria (motif RMSD, sequence recovery, optional clashes) evaluated jointly
    per redesign.  A sample is successful when at least one redesign passes ALL
    criteria simultaneously.

    Steps:
      1. Compute combined motif+binder pass rates (grouped by config cols)
      2. Filter successful samples per sequence type
      3. Compute individual motif metric pass rates (RMSD pred, seq rec, clashes)
      4. Per-task pass rates (grouped by task_name)
      5. Diversity on filtered subsets (FoldSeek + MMseqs)
      6. Merge and save

    Args:
        cfg: Hydra configuration dict.
        df: Combined results DataFrame.
        results_dir: Output directory.
        config_name: Config name for file naming.
        result_type: ``"motif_protein_binder"`` or ``"motif_ligand_binder"``.

    Returns:
        Dictionary of result DataFrames.
    """
    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("MOTIF BINDER ANALYSIS"))
    logger.info("+" + "-" * 68 + "+")
    logger.info(f"  Samples:       {len(df)}")
    logger.info(f"  Result type:   {result_type}")

    cfg_aggregation = cfg.get("aggregation", {})
    is_ligand = result_type == "motif_ligand_binder"

    # --- Resolve success criteria from YAML or defaults ---
    motif_binder_success = cfg_aggregation.get("motif_binder_success_thresholds", None)
    if motif_binder_success is not None:
        motif_binder_success = OmegaConf.to_container(motif_binder_success, resolve=True)
        thresh_src = "custom"
    else:
        motif_binder_success = get_default_motif_binder_success(result_type)
        thresh_src = "default"

    # --- Auto-detect sequence types ---
    sequence_types = cfg_aggregation.get("sequence_types", None)
    if sequence_types is not None:
        sequence_types = list(sequence_types) if hasattr(sequence_types, "__iter__") else [sequence_types]
    else:
        sequence_types = detect_sequence_types_from_columns(df)

    seq_src = ", ".join(sequence_types) if sequence_types else "auto-detect"
    logger.info(f"  Seq types:     {seq_src}")
    logger.info(f"  Thresholds:    {thresh_src}")
    logger.info("  Criteria:")
    logger.info(format_success_criteria_for_logging(motif_binder_success))
    logger.info("")

    groupby_cols = get_groupby_columns(df)

    results = {}
    all_pass_rate_dfs = []
    all_diversity_dfs = []

    # ------------------------------------------------------------------
    # Step 1: Combined motif+binder pass rates
    # ------------------------------------------------------------------
    logger.info("  [1/6] Computing combined motif+binder pass rates")
    filter_pass_df = None
    try:
        filter_pass_df = compute_motif_binder_pass_rate(
            df=df,
            groupby_cols=groupby_cols,
            path_store_results=results_dir,
            metric_suffix="overall",
            success_thresholds=motif_binder_success,
            result_type=result_type,
        )
        if filter_pass_df is not None and not filter_pass_df.empty:
            all_pass_rate_dfs.append(filter_pass_df)
            results["motif_binder_pass_rates"] = filter_pass_df
    except Exception as e:
        logger.error(f"Failed to compute motif binder pass rates: {e}")

    # ------------------------------------------------------------------
    # Step 2: Filter successful samples per sequence type
    # ------------------------------------------------------------------
    logger.info("  [2/6] Filtering successful samples")
    successful_dfs: dict[str, pd.DataFrame] = {}
    for seq_type in sequence_types:
        try:
            df_successful = filter_by_motif_binder_success(
                df,
                seq_type,
                success_thresholds=motif_binder_success,
                result_type=result_type,
                path_store_results=results_dir,
                filter_name=f"motif_binder_{result_type}",
                save_json=False,
            )
            successful_dfs[seq_type] = df_successful
        except Exception as e:
            logger.warning(f"Failed to filter motif binder success for {seq_type}: {e}")

    success_summary = ", ".join(f"{k}({len(v)})" for k, v in successful_dfs.items())
    logger.info(f"         Successful: {success_summary}")

    # ------------------------------------------------------------------
    # Step 3: Individual motif metric pass rates
    # ------------------------------------------------------------------
    logger.info("  [3/6] Individual motif metric pass rates")
    try:
        motif_pred_df = compute_motif_pred_metric_pass_rates(
            df=df,
            groupby_cols=groupby_cols,
            path_store_results=results_dir,
            metric_suffix="overall",
            is_ligand=is_ligand,
            success_thresholds=motif_binder_success,
            result_type=result_type,
        )
        if not motif_pred_df.empty:
            all_pass_rate_dfs.append(motif_pred_df)
            results["motif_pred_metrics"] = motif_pred_df
    except Exception as e:
        logger.error(f"Failed to compute motif pred metric pass rates: {e}")

    # ------------------------------------------------------------------
    # Step 4: Per-task pass rates
    # ------------------------------------------------------------------
    if "task_name" in df.columns:
        logger.info("  [4/6] Per-task pass rates")
        try:
            per_task_df = compute_per_task_motif_binder_pass_rates(
                df=df,
                groupby_cols=groupby_cols,
                success_thresholds=motif_binder_success,
                result_type=result_type,
                path_store_results=results_dir,
            )
            if not per_task_df.empty:
                all_pass_rate_dfs.append(per_task_df)
                results["per_task_pass_rates"] = per_task_df
        except Exception as e:
            logger.error(f"Failed to compute per-task pass rates: {e}")
    else:
        logger.info("  [4/6] Per-task pass rates -- skipped (no task_name column)")

    # ------------------------------------------------------------------
    # Step 5: Diversity on filtered subsets
    # ------------------------------------------------------------------
    motif_binder_subsets = []

    # Build subsets: all samples + successful per seq_type
    motif_binder_subsets.append({"suffix": "all_generated", "df": df})
    for seq_type, df_successful in successful_dfs.items():
        if len(df_successful) > 0:
            motif_binder_subsets.append({"suffix": f"successful_{seq_type}", "df": df_successful})

    subset_summary = ", ".join(f"{s['suffix']}({len(s['df'])})" for s in motif_binder_subsets)
    logger.info(f"  [5/6] Diversity subsets: {subset_summary}")

    tmp_path = os.path.join(results_dir, "tmp_diversity")
    os.makedirs(tmp_path, exist_ok=True)

    if cfg_aggregation.get("compute_diversity", True):
        run_foldseek_on_subsets(
            df_all=df,
            subsets=motif_binder_subsets,
            groupby_cols=groupby_cols,
            results_dir=results_dir,
            tmp_path=tmp_path,
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    if cfg_aggregation.get("compute_mmseqs_diversity", True):
        run_mmseqs_on_subsets(
            df_all=df,
            subsets=motif_binder_subsets,
            groupby_cols=groupby_cols,
            results_dir=results_dir,
            cfg_aggregation=cfg_aggregation,
            results_accum=all_diversity_dfs,
            results_dict=results,
        )

    # ------------------------------------------------------------------
    # Step 6: Merge + save
    # ------------------------------------------------------------------
    logger.info("  [6/6] Merge and save")
    merge_and_save_results(
        all_pass_rate_dfs,
        all_diversity_dfs,
        groupby_cols,
        results_dir,
        config_name,
        prefix="motif_binder",
        results_dict=results,
    )

    # Save success criteria JSON for reproducibility
    save_motif_binder_success_json(
        success_thresholds=motif_binder_success,
        path_store_results=results_dir,
        filter_name=f"motif_binder_{result_type}",
        sequence_types=sequence_types,
    )

    logger.info("")
    logger.info("+" + "-" * 68 + "+")
    logger.info("|{:^68s}|".format("MOTIF BINDER ANALYSIS COMPLETE"))
    logger.info("+" + "-" * 68 + "+")
    logger.info("")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def organize_results(results_dir: str) -> None:
    """
    Organize analysis output files into subdirectories for cleaner results.

    Moves files matching known prefixes into labeled subdirectories:
        res_div_*       -> diversity/
        res_ss_*        -> secondary_structure/
        res_monomer_*   -> monomer_metrics/
        res_aa_*        -> amino_acid_distribution/
        res_filter_*    -> filter_results/
        clusters_*      -> clusters/

    Temporary directories (tmp_diversity, tmp_mmseqs_diversity) are removed.

    Args:
        results_dir: Path to the results directory to organize.
    """
    # Define prefix -> subdirectory mapping
    prefix_to_subdir = {
        "res_div_": "diversity",
        "res_ss_": "secondary_structure",
        "res_monomer_": "monomer_metrics",
        "res_motif_binder_": "motif_binder_metrics",
        "res_filter_motif_binder_": "motif_binder_metrics",
        "res_motif_": "motif_metrics",
        "res_motif_success": "motif_metrics",
        "res_refolded_motif_success": "motif_metrics",
        "res_custom_motif_success": "motif_metrics",
        "res_aa_": "amino_acid_distribution",
        "res_filter_": "filter_results",
    }

    moved_counts = {}

    for prefix, subdir in prefix_to_subdir.items():
        # Find matching CSV files
        pattern = os.path.join(results_dir, f"{prefix}*.csv")
        matches = glob.glob(pattern)
        if not matches:
            continue

        dest_dir = os.path.join(results_dir, subdir)
        os.makedirs(dest_dir, exist_ok=True)

        for filepath in matches:
            filename = os.path.basename(filepath)
            dest = os.path.join(dest_dir, filename)
            # shutil.move overwrites files natively on POSIX
            shutil.move(filepath, dest)

        moved_counts[subdir] = len(matches)

    # Move cluster directories
    cluster_pattern = os.path.join(results_dir, "clusters_*")
    cluster_dirs = [d for d in glob.glob(cluster_pattern) if os.path.isdir(d)]
    if cluster_dirs:
        dest_dir = os.path.join(results_dir, "clusters")
        os.makedirs(dest_dir, exist_ok=True)
        for dirpath in cluster_dirs:
            dirname = os.path.basename(dirpath)
            dest = os.path.join(dest_dir, dirname)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(dirpath, dest)
        moved_counts["clusters"] = len(cluster_dirs)

    # Clean up temporary directories
    for tmp_name in ["tmp_diversity", "tmp_mmseqs_diversity"]:
        tmp_path = os.path.join(results_dir, tmp_name)
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)
            logger.debug(f"Removed temporary directory: {tmp_name}")

    if moved_counts:
        summary = ", ".join(f"{subdir}/ ({count} files)" for subdir, count in moved_counts.items())
        logger.info(f"Organized results into subdirectories: {summary}")
    else:
        logger.debug("No files to organize")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="analyze",
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for unified result analysis.

    This function orchestrates the analysis pipeline for evaluation results
    from both monomer and binder benchmarks.

    Args:
        cfg: Hydra configuration dictionary containing analysis settings

    Configuration Keys:
        - results_dir: Directory containing evaluation results
        - config_name: Name of the evaluation configuration
        - result_type: Type of results ("protein_binder", "ligand_binder", "monomer")
        - input_mode: Input mode used during evaluation ("generated" or "pdb_dir")
        - dryrun: If True, show summary without executing
        - aggregation: Dictionary with aggregation settings:
            - limit: Max number of result files to process
            - sequence_types: List of sequence types to analyze
            - diversity_modes: List of diversity computation modes
            - success_thresholds: Custom success thresholds
    """
    # Get config name from Hydra or config
    config_name = cfg.get(
        "base_config_name",
        (
            hydra.core.hydra_config.HydraConfig.get().job.config_name
            if hydra.core.hydra_config.HydraConfig.initialized()
            else "analyze"
        ),
    )

    # Get basic settings
    # Check results_dir first, then output_dir (for compatibility with evaluate configs)
    results_dir = cfg.get("results_dir") or cfg.get("output_dir")
    result_type = cfg.get("result_type", "protein_binder")
    input_mode = cfg.get("input_mode", "generated")
    dryrun = cfg.get("dryrun", False)
    run_name = cfg.get("run_name", None)

    # Construct results_dir from config if not explicitly provided
    # This matches how evaluate.py constructs output_dir
    if not results_dir:
        logger.info("results_dir not set, constructing from config...")
        # Try to get target task name for binders
        target_task_name = None
        #! List of generated types of proteins
        if result_type in [
            "protein_binder",
            "ligand_binder",
            "motif_protein_binder",
            "motif_ligand_binder",
        ]:
            try:
                generation_cfg = cfg.get("generation", {})
                target_task_name = generation_cfg.get("task_name", None)
            except Exception:
                pass

        # Construct path matching evaluate.py pattern
        if target_task_name:
            results_dir = f"./evaluation_results/{config_name}_{target_task_name}"
        else:
            results_dir = f"./evaluation_results/{config_name}"

        logger.info(f"Constructed results_dir: {results_dir}")

    # Append run_name to results_dir if provided (always, not just when paths are auto-constructed)
    # This allows users to override run_name via CLI and have it reflected in the results path
    if run_name and not results_dir.endswith(run_name):
        results_dir = f"{results_dir}_{run_name}"
        logger.info(f"Appended run_name to results_dir: {results_dir}")

    # Validate configuration
    try:
        validate_config(cfg, results_dir)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Get aggregation settings
    cfg_aggregation = cfg.get("aggregation", {})
    limit = cfg_aggregation.get("limit", None)

    # Get success thresholds
    success_thresholds = cfg_aggregation.get("success_thresholds", None)
    if success_thresholds is not None:
        success_thresholds = OmegaConf.to_container(success_thresholds, resolve=True)

    # --- Opening banner ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PROTEIN FOUNDATION MODELS — ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Config:        {config_name}")
    logger.info(f"  Result type:   {result_type}")
    logger.info(f"  Input mode:    {input_mode}")
    logger.info(f"  Results dir:   {results_dir}")
    if run_name:
        logger.info(f"  Run name:      {run_name}")
    logger.info("=" * 70)
    logger.info("")

    # Find result files
    result_files = find_result_files(
        results_dir=results_dir,
        config_name=config_name,
        result_type=result_type,
        input_mode=input_mode,
        limit=limit,
    )

    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        sys.exit(1)

    # Handle dryrun mode
    if dryrun:
        print_dryrun_summary(cfg, result_files, result_type, success_thresholds)
        sys.exit(0)

    # Aggregate results
    try:
        combined_df = aggregate_results(result_files)
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")
        sys.exit(1)

    # For binder result types (including motif binder), merge monomer metrics
    # into the DataFrame so that both binder and monomer analysis modes can use
    # one unified DataFrame.
    if result_type in [
        "protein_binder",
        "ligand_binder",
        "motif_protein_binder",
        "motif_ligand_binder",
    ]:
        combined_df = merge_monomer_into_binder(combined_df, results_dir, config_name, input_mode)

    # For monomer_motif result types, merge monomer metrics into the motif DataFrame.
    # Unique monomer columns (e.g. _res_scRMSD_single_*) are added directly;
    # conflicting columns get a _monomer suffix to preserve both versions.
    if result_type == "monomer_motif":
        combined_df = merge_monomer_into_motif(combined_df, results_dir, config_name, input_mode)

    # Add sequence columns by extracting from PDB files
    combined_df = add_sequence_columns(combined_df, result_type)

    # Save combined results
    combined_csv_filename = f"RAW_{result_type}_results_{config_name}_combined.csv"
    combined_csv_path = os.path.join(results_dir, combined_csv_filename)

    # Filter out non-metric columns before saving
    df_to_save = filter_columns_for_csv(combined_df, log_dropped=False)

    df_to_save.to_csv(combined_csv_path, index=False)
    logger.info(f"Combined results saved to {combined_csv_path}")

    # Save transposed version for easier viewing
    save_transposed_csv(df_to_save, combined_csv_path)

    # Aggregate per-job timing CSVs (timing_*.csv in each eval subdir)
    if cfg_aggregation.get("compute_timing", True):
        try:
            groupby_cols = get_groupby_columns(combined_df)
            compute_timing_metrics(
                df=combined_df,
                groupby_cols=groupby_cols,
                path_store_results=results_dir,
                root_path=results_dir,
            )
        except Exception as e:
            logger.warning(f"Timing aggregation skipped: {e}")

    # Get analysis modes from config (defaults based on result_type)
    analysis_modes = cfg_aggregation.get("analysis_modes", None)

    # Set default analysis modes based on result_type if not specified
    if analysis_modes is None:
        if result_type in ["protein_binder", "ligand_binder"]:
            analysis_modes = ["binder", "monomer"]
        elif result_type == "monomer":
            analysis_modes = ["monomer"]
        elif result_type == "monomer_motif":
            analysis_modes = ["motif", "monomer"]
        elif result_type in ["motif_protein_binder", "motif_ligand_binder"]:
            analysis_modes = ["motif_binder"]
        else:
            raise ValueError(f"Unknown result type: {result_type}")

    # Convert to list if needed (OmegaConf ListConfig -> list)
    if hasattr(analysis_modes, "__iter__") and not isinstance(analysis_modes, str):
        analysis_modes = list(analysis_modes)
    else:
        analysis_modes = [analysis_modes]

    logger.info(f"  Analysis modes: {', '.join(analysis_modes)}")
    logger.info(f"  Samples:        {len(combined_df)}")
    logger.info("")

    # Run each analysis mode - each mode fails safely if required files are missing
    successful_modes = []
    failed_modes = []

    for mode in analysis_modes:
        try:
            if mode == "binder":
                # Binder analysis applies to protein/ligand binders and motif binders
                # (motif binders produce all standard binder columns)
                binder_result_types = {
                    "protein_binder",
                    "ligand_binder",
                    "motif_protein_binder",
                    "motif_ligand_binder",
                }
                if result_type not in binder_result_types:
                    logger.warning(f"Skipping 'binder' analysis mode - not applicable for result_type={result_type}")
                    continue

                # Check if we have any binder-specific columns
                binder_cols = [c for c in combined_df.columns if "complex_" in c or "binder_" in c]
                if not binder_cols:
                    logger.warning(
                        "Skipping 'binder' analysis mode - no binder metrics found in results. "
                        "Ensure evaluate ran with compute_binder_metrics=true."
                    )
                    failed_modes.append(("binder", "No binder metrics columns found"))
                    continue

                # Map motif binder result types to their binder equivalents
                binder_rt = result_type
                if result_type == "motif_protein_binder":
                    binder_rt = "protein_binder"
                elif result_type == "motif_ligand_binder":
                    binder_rt = "ligand_binder"

                run_binder_analysis(
                    cfg=cfg,
                    df=combined_df,
                    results_dir=results_dir,
                    config_name=config_name,
                    result_type=binder_rt,
                )
                successful_modes.append("binder")

            elif mode == "monomer":
                # Monomer metrics are either:
                # - Native columns (for monomer result_type)
                # - Merged from monomer_results_*.csv (for binder result_type,
                #   done earlier via merge_monomer_into_binder)
                monomer_cols = [c for c in combined_df.columns if "_res_scRMSD" in c or "_res_co_scRMSD" in c]
                if not monomer_cols:
                    logger.warning(
                        "Skipping 'monomer' analysis mode - no monomer metrics found in results. "
                        "Expected columns like '_res_scRMSD_ca_*' or '_res_co_scRMSD_ca_*'. "
                        "Ensure evaluate ran with compute_monomer_metrics=true."
                    )
                    failed_modes.append(("monomer", "No monomer metrics columns found"))
                    continue

                run_monomer_analysis(
                    cfg=cfg,
                    df=combined_df,
                    results_dir=results_dir,
                    config_name=config_name,
                )
                successful_modes.append("monomer")

            elif mode == "motif":
                # Motif metrics from motif_results_*.csv or monomer_motif evaluation
                motif_cols = [c for c in combined_df.columns if "_res_motif_rmsd" in c]
                if not motif_cols:
                    logger.warning(
                        "Skipping 'motif' analysis mode - no motif metrics found in results. "
                        "Expected columns like '_res_motif_rmsd_ca'. "
                        "Ensure evaluate ran with compute_motif_metrics=true."
                    )
                    failed_modes.append(("motif", "No motif metrics columns found"))
                    continue

                run_motif_analysis(
                    cfg=cfg,
                    df=combined_df,
                    results_dir=results_dir,
                    config_name=config_name,
                )
                successful_modes.append("motif")

            elif mode == "motif_binder":
                if result_type not in ["motif_protein_binder", "motif_ligand_binder"]:
                    logger.warning(
                        f"Skipping 'motif_binder' analysis mode - not applicable for result_type={result_type}"
                    )
                    continue

                # Check for motif binder columns (motif metrics on predicted structures)
                motif_binder_cols = [c for c in combined_df.columns if "motif_rmsd_pred" in c or "motif_seq_rec" in c]
                if not motif_binder_cols:
                    logger.warning(
                        "Skipping 'motif_binder' analysis mode - no motif binder "
                        "metrics found. Expected columns like '*_motif_rmsd_pred'. "
                        "Ensure evaluate ran with motif_binder evaluation."
                    )
                    failed_modes.append(("motif_binder", "No motif binder columns found"))
                    continue

                run_motif_binder_analysis(
                    cfg=cfg,
                    df=combined_df,
                    results_dir=results_dir,
                    config_name=config_name,
                    result_type=result_type,
                )
                successful_modes.append("motif_binder")

            else:
                logger.warning(f"Unknown analysis mode: {mode}, skipping")

        except Exception as e:
            logger.error(f"Analysis mode '{mode}' failed with error: {e}")
            failed_modes.append((mode, str(e)))
            # Continue with other modes instead of crashing
            continue

    # Organize output files into subdirectories
    if successful_modes:
        organize_results(results_dir)

    # --- Closing summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("  ANALYSIS COMPLETE")
    logger.info("=" * 70)
    if successful_modes:
        logger.info(f"  Completed:     {', '.join(successful_modes)}")
    if failed_modes:
        logger.info(f"  Skipped:       {', '.join(m[0] for m in failed_modes)}")
    logger.info(f"  Results dir:   {results_dir}")
    logger.info("=" * 70)
    logger.info("")

    if not successful_modes and failed_modes:
        logger.warning(
            "All analysis modes were skipped/failed. Check that evaluate was run with the appropriate metrics enabled."
        )


if __name__ == "__main__":
    main()
