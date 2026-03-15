"""
Diversity computation functions for protein structure analysis.

This module contains functions for computing structural and sequence diversity
metrics using Foldseek and MMseqs2. It also includes helper functions for
extracting binder chains from complex structures.

Functions:
    - compute_foldseek_diversity: Compute structural diversity using Foldseek
    - compute_mmseqs_diversity: Compute sequence diversity using MMseqs2
    - extract_binder_chains_from_complexes: Extract binder chains from PDB files
    - extract_binder_chains_from_complexes_parallel: Parallel version of above
"""

import os
from typing import Literal

import pandas as pd
from biotite.structure.io import load_structure
from loguru import logger
from tqdm import tqdm

from proteinfoundation.metrics.diversity import diversity_foldseek, diversity_sequence_mmseqs

# Import shared CSV formatting constants
from proteinfoundation.result_analysis.analysis_utils import FLOAT_FORMAT_PD, SEP_CSV_PD
from proteinfoundation.utils.pdb_utils import load_pdb, write_prot_to_pdb

# =============================================================================
# Binder Chain Extraction Functions
# =============================================================================


def extract_binder_chains_from_complexes(
    pdb_paths: list[str],
    tmp_path: str,
) -> list[str]:
    """Extract binder chains from complex PDB files and save them as separate PDB files.
    This assumes that all PDB files have the same binder chain.

    Args:
        pdb_paths: List of paths to complex PDB files
        tmp_path: Directory to save extracted binder PDB files

    Returns:
        List of paths to extracted binder PDB files
    """
    os.makedirs(tmp_path, exist_ok=True)
    binder_pdb_paths = []
    binder_chain = None

    for i, pdb_path in enumerate(pdb_paths):
        try:
            if binder_chain is None:
                chains = sorted(set(load_structure(pdb_path).chain_id.tolist()))
                binder_chain = chains[-1]
            # Load only the binder chain (chain B) directly
            binder_prot = load_pdb(pdb_path, chain_id=binder_chain)

            # Save binder PDB file
            base_name = os.path.splitext(os.path.basename(pdb_path))[0]
            binder_pdb_path = os.path.join(tmp_path, f"{base_name}_binder_{i}.pdb")

            # Convert Protein object to the format expected by write_prot_to_pdb
            write_prot_to_pdb(
                prot_pos=binder_prot.atom_positions,
                file_path=binder_pdb_path,
                aatype=binder_prot.aatype,
                chain_index=binder_prot.chain_index,
                b_factors=binder_prot.b_factors,
                no_indexing=True,  # Don't add automatic indexing to avoid path confusion
            )

            # Check if file was actually created
            if not os.path.exists(binder_pdb_path):
                logger.error(f"Failed to create binder PDB file: {binder_pdb_path}")
                binder_pdb_paths.append(None)
            else:
                binder_pdb_paths.append(binder_pdb_path)

        except Exception as e:
            logger.error(f"Error extracting binder chain from {pdb_path}: {e}")
            binder_pdb_paths.append(None)

    return binder_pdb_paths


def _extract_single_binder(args):
    """Worker function for parallel binder extraction.

    Args:
        args: Tuple of (index, pdb_path, tmp_path, binder_chain)

    Returns:
        Tuple of (index, binder_pdb_path or None)
    """
    idx, pdb_path, tmp_path, binder_chain = args

    try:
        # Determine binder chain if not provided
        if binder_chain is None:
            chains = sorted(set(load_structure(pdb_path).chain_id.tolist()))
            binder_chain = chains[-1]

        # Load only the binder chain
        binder_prot = load_pdb(pdb_path, chain_id=binder_chain)

        # Save binder PDB file
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        binder_pdb_path = os.path.join(tmp_path, f"{base_name}_binder_{idx}.pdb")

        write_prot_to_pdb(
            prot_pos=binder_prot.atom_positions,
            file_path=binder_pdb_path,
            aatype=binder_prot.aatype,
            chain_index=binder_prot.chain_index,
            b_factors=binder_prot.b_factors,
            no_indexing=True,
        )

        if not os.path.exists(binder_pdb_path):
            return (idx, None)
        return (idx, binder_pdb_path)

    except Exception as e:
        logger.error(f"Error extracting binder chain from {pdb_path}: {e}")
        return (idx, None)


def extract_binder_chains_from_complexes_parallel(
    pdb_paths: list[str],
    tmp_path: str,
    num_workers: int = 16,
    binder_chain: str = None,
    show_progress: bool = False,
    per_file_binder_detection: bool = False,
) -> list[str]:
    """Extract binder chains from complex PDB files in parallel.

    Args:
        pdb_paths: List of paths to complex PDB files
        tmp_path: Directory to save extracted binder PDB files
        num_workers: Number of parallel workers (default: 16)
        binder_chain: Chain ID of the binder (default: None, auto-detect as last chain)
        show_progress: Whether to show a progress bar
        per_file_binder_detection: If True, each file independently detects its
            own binder chain (last chain alphabetically) instead of assuming
            all files share the same chain layout. Useful when PDB files come
            from heterogeneous sources with varying chain naming.

    Returns:
        List of paths to extracted binder PDB files (in same order as input)
    """
    from multiprocessing import Pool

    os.makedirs(tmp_path, exist_ok=True)

    if per_file_binder_detection:
        logger.info("Per-file binder chain detection enabled")
        binder_chain = None
    elif binder_chain is None and len(pdb_paths) > 0:
        try:
            chains = sorted(set(load_structure(pdb_paths[0]).chain_id.tolist()))
            binder_chain = chains[-1]
            logger.info(f"Auto-detected binder chain: {binder_chain}")
        except Exception as e:
            logger.warning(f"Could not auto-detect binder chain: {e}")

    # Prepare arguments for parallel processing
    args_list = [(i, pdb_path, tmp_path, binder_chain) for i, pdb_path in enumerate(pdb_paths)]

    # Process in parallel
    results = [None] * len(pdb_paths)
    with Pool(processes=num_workers) as pool:
        for idx, result_path in tqdm(
            pool.imap_unordered(_extract_single_binder, args_list),
            total=len(pdb_paths),
            desc="Extracting binder chains",
            disable=not show_progress,
        ):
            results[idx] = result_path

    return results


# =============================================================================
# Diversity Computation Functions
# =============================================================================


def compute_foldseek_diversity(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    tmp_path: str,
    metric_suffix: str,
    min_seq_id: float,
    alignment_type: Literal[1, 2],
    diversity_mode: Literal["complex", "binder", "interface", "monomer"] = "complex",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Computes structural diversity metrics using Foldseek.

    This function uses Foldseek to compute structural diversity metrics by
    clustering structures based on TM-score. It can use either structure-only
    (alignment_type=1) or structure+sequence (alignment_type=2) clustering.

    Supports four diversity modes:
    - "complex": Uses full complex structures with default parameters
    - "binder": Extracts binder chains and computes diversity on them
    - "interface": Uses interface-based clustering with specific thresholds
    - "monomer": Uses monomer-based clustering with specific thresholds

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        tmp_path: Path for temporary files
        metric_suffix: Suffix for metric names
        min_seq_id: Minimum sequence identity threshold
        alignment_type: Type of alignment (1=structure only, 2=structure+sequence)
        diversity_mode: Type of diversity computation (default: "complex")

    Returns:
        DataFrame containing diversity metrics per run
    """
    mode_names = {
        "complex": "complex",
        "binder": "binder",
        "interface": "interface",
        "monomer": "monomer",
    }
    mode_name = mode_names[diversity_mode]

    logger.info(f"Computing {mode_name} diversity with foldseek, this may take a few minutes - {metric_suffix}")

    # Group by everything except (seed, L, id_gen (i.e. id of sample))
    # and save all paths of generations as a list
    df_grouped = df.groupby(groupby_cols, dropna=False)["pdb_path"].agg(list).reset_index()

    # Compute diversity for each group
    results_div_mc = []
    for idx, row in tqdm(df_grouped.iterrows(), total=df_grouped.shape[0], disable=not show_progress):
        if diversity_mode == "binder" or diversity_mode == "monomer":
            # Extract binder chains from complex PDB files
            if diversity_mode == "binder":
                binder_pdb_paths = extract_binder_chains_from_complexes(
                    row["pdb_path"],
                    os.path.join(tmp_path, f"group_{idx}"),
                )
            elif diversity_mode == "monomer":
                binder_pdb_paths = row["pdb_path"]

            # Filter out None values (failed extractions)
            valid_pdb_paths = [p for p in binder_pdb_paths if p is not None]

            # Debug: Check if all files exist
            for path in valid_pdb_paths:
                if not os.path.exists(path):
                    logger.error(f"Binder PDB file does not exist: {path}")

            if len(valid_pdb_paths) < 2:
                if len(valid_pdb_paths) == 0:
                    logger.warning(f"Group {idx}: No valid binder structures for diversity computation, skipping...")
                    results_div_mc.append(None)
                    continue
                elif len(valid_pdb_paths) == 1:
                    logger.warning(
                        f"Group {idx}: Only one valid binder structure for diversity computation, skipping..."
                    )
                    results_div_mc.append((1.0, 1, 1))
                    continue
        else:
            valid_pdb_paths = row["pdb_path"]

        try:
            # Extract inf_config from path: .../inf_{number}_{suffix}/n_##_id_##/filename.pdb
            # The inf_config directory is at position -3 from the end
            path_parts = valid_pdb_paths[0].split(os.path.sep)
            inf_config_name = ""
            for part in path_parts:
                if part.startswith("inf_"):
                    inf_config_name = part
                    break
        except Exception as e:
            logger.warning(f"Error extracting inf config name from {valid_pdb_paths[0]}: {e}")
            inf_config_name = ""

        # Set parameters based on diversity mode
        logger.debug(f"Computing Foldseek diversity for group {idx} with {len(valid_pdb_paths)} PDB paths")
        if diversity_mode == "interface":
            # Interface mode: use specific thresholds for interface-based clustering
            result = diversity_foldseek(
                list_of_pdb_paths=valid_pdb_paths,
                tmp_path=(tmp_path if diversity_mode != "binder" else os.path.join(tmp_path, f"group_{idx}_diversity")),
                min_seq_id=min_seq_id,
                alignment_type=alignment_type,
                tm_threshold=0.0,
                multimer_tm_threshold=0.0,
                interface_lddt_threshold=0.6,
                save_cluster_file=True,
                cluster_output_dir=os.path.join(path_store_results, f"clusters_{mode_name}_{metric_suffix}"),
                inf_config_name=inf_config_name,
            )
        else:
            # Complex and binder modes: use default parameters
            result = diversity_foldseek(
                list_of_pdb_paths=valid_pdb_paths,
                tmp_path=(tmp_path if diversity_mode != "binder" else os.path.join(tmp_path, f"group_{idx}_diversity")),
                min_seq_id=min_seq_id,
                alignment_type=alignment_type,
                save_cluster_file=True,
                cluster_output_dir=os.path.join(path_store_results, f"clusters_{mode_name}_{metric_suffix}"),
                inf_config_name=inf_config_name,
            )

        # For binder mode, save the original PDB paths for analysis
        if (diversity_mode == "binder" or diversity_mode == "monomer") and result is not None:
            cluster_dir = os.path.join(path_store_results, f"clusters_{mode_name}_{metric_suffix}")
            original_paths_file = os.path.join(cluster_dir, f"original_pdb_paths_{inf_config_name}.txt")

            # Overwrite the file with original complex PDB paths
            with open(original_paths_file, "w") as f:
                for i, pdb_path in enumerate(row["pdb_path"], 1):
                    f.write(f"{i}\t{pdb_path}\n")
            logger.debug(f"Original complex PDB paths saved to {original_paths_file}")
        results_div_mc.append(result)

    # Use appropriate column name based on diversity mode
    column_name = f"_res_diversity_foldseek_{mode_name}_{metric_suffix}"
    df_grouped[column_name] = results_div_mc

    # Save results, drop col for easy viewing
    df_grouped = df_grouped.drop("pdb_path", axis=1)

    # Keep groupby columns + result columns
    cols_to_keep = list(groupby_cols) + [c for c in df_grouped.columns if c.startswith("_res_")]
    # Deduplicate while preserving order
    seen = set()
    cols_to_keep = [c for c in cols_to_keep if c in df_grouped.columns and not (c in seen or seen.add(c))]
    df_grouped = df_grouped[cols_to_keep]
    # logger.info(f"Diversity CSV columns: {list(df_grouped.columns)}")

    # Use appropriate file name based on diversity mode
    file_name = f"res_div_foldseek_{mode_name}_{metric_suffix}.csv"
    df_grouped.to_csv(
        os.path.join(path_store_results, file_name),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def compute_mmseqs_diversity(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    tmp_path: str,
    metric_suffix: str,
    min_seq_id: float,
    coverage: float,
    diversity_mode: Literal["monomer", "binder"] = "monomer",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Computes sequence diversity metrics using MMseqs2.

    This function uses MMseqs2 to compute sequence diversity metrics by
    clustering sequences based on sequence identity and coverage.

    Supports two diversity modes:
    - "monomer": Uses PDB files directly (single chain structures)
    - "binder": Extracts binder chains from complex PDB files first

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        tmp_path: Path for temporary files
        metric_suffix: Suffix for metric names
        min_seq_id: Minimum sequence identity threshold
        coverage: Minimum coverage threshold
        diversity_mode: Type of diversity computation ("monomer" or "binder")

    Returns:
        DataFrame containing diversity metrics per run
    """
    logger.info(f"Computing {diversity_mode} diversity with mmseqs, this may take a few minutes - {metric_suffix}")

    # Group by everything except (seed, L, id_gen (i.e. id of sample))
    # and save all paths of generations as a list
    df_grouped = df.groupby(groupby_cols, dropna=False)["pdb_path"].agg(list).reset_index()

    # Compute diversity for each group
    results_div_mm = []
    for idx, row in tqdm(df_grouped.iterrows(), total=df_grouped.shape[0], disable=not show_progress):
        pdb_paths = row["pdb_path"]
        logger.debug(f"Computing MMseqs diversity for group {idx} with {len(pdb_paths)} PDB paths")
        # For binder mode, extract binder chains first
        if diversity_mode == "binder":
            binder_pdb_paths = extract_binder_chains_from_complexes(
                pdb_paths,
                os.path.join(tmp_path, f"mmseqs_binder_group_{idx}"),
            )
            pdb_paths = [p for p in binder_pdb_paths if p is not None]

            if len(pdb_paths) < 2:
                logger.warning(f"Group {idx}: fewer than 2 valid binder structures for MMseqs diversity, skipping...")
                results_div_mm.append((1.0, 1, 1) if len(pdb_paths) == 1 else None)
                continue

        result = diversity_sequence_mmseqs(
            list_of_pdb_paths=pdb_paths,
            min_seq_id=min_seq_id,
            coverage=coverage,
            tmp_path=tmp_path,
        )
        results_div_mm.append(result)

    column_name = f"_res_diversity_mmseqs_{metric_suffix}"
    df_grouped[column_name] = results_div_mm

    # Save results, drop col for easy viewing
    df_grouped = df_grouped.drop("pdb_path", axis=1)

    # Keep groupby columns + result columns
    cols_to_keep = list(groupby_cols) + [c for c in df_grouped.columns if c.startswith("_res_")]
    seen = set()
    cols_to_keep = [c for c in cols_to_keep if c in df_grouped.columns and not (c in seen or seen.add(c))]
    df_grouped = df_grouped[cols_to_keep]
    # logger.info(f"Diversity CSV columns: {list(df_grouped.columns)}")

    df_grouped.to_csv(
        os.path.join(path_store_results, f"res_div_mmseqs_{metric_suffix}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped
