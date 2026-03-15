"""
This requires a copy of all result files, and of samples produced as well.
Can be done with the script script_utils/download_results.sh

Will only process results for
"""

import argparse
import ast
import os
import re
import shutil
import sys
import warnings
from functools import reduce
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openfold.np.residue_constants import restype_1to3, restype_3to1
from tqdm import tqdm

from proteinfoundation.metrics.structural_metric_ss_ca_ca import compute_ss_metrics
from proteinfoundation.result_analysis.analysis_utils import (
    FLOAT_FORMAT_PD,
    SEP_CSV_PD,
    SEQUENCE_TYPES,
    keep_lists_separate,
)
from proteinfoundation.result_analysis.binder_analysis import (
    compute_filter_ligand_pass_rate,
    compute_filter_pass_rate,
    filter_by_binder_success,
    filter_by_ligand_binder_success,
)

# Import diversity functions from dedicated module
from proteinfoundation.result_analysis.compute_diversity import compute_foldseek_diversity, compute_mmseqs_diversity

# from proteinfoundation.utils.visual_utils import visualize_directory_all_atom


def detect_folding_models(df_columns):
    """Detect available folding models from column names."""
    models = set()
    rmsd_modes = ["ca_", "bb3o_", "all_atom_"]

    for col in df_columns:
        if col.startswith("_res_scRMSD_") and "_all" not in col:
            # Extract everything after _res_scRMSD_
            model_part = col.replace("_res_scRMSD_", "")

            # Remove RMSD mode prefixes to get the actual model name
            actual_model = model_part
            for mode_prefix in rmsd_modes:
                if model_part.startswith(mode_prefix):
                    actual_model = model_part[len(mode_prefix) :]
                    break

            if actual_model:  # Only add non-empty model names
                models.add(actual_model)

    models = list(models)
    if not models:
        logger.warning("No folding models detected")
    return models


def detect_codesignability_folding_models(df_columns, mode):
    """Detect available folding models for codesignability metrics from column names."""
    models = []
    base_pattern = f"_res_co_scRMSD_{mode}_"

    for col in df_columns:
        if col.startswith(base_pattern):
            model = col.replace(base_pattern, "")
            models.append(model)

    if not models:
        logger.warning("No co-designability folding models detected")
    return models


def detect_motif_folding_models(df_columns, mode):
    """Detect available folding models for motif metrics from column names."""
    models = []
    base_pattern = f"_res_co_motif_scRMSD_{mode}_"

    for col in df_columns:
        if col.startswith(base_pattern):
            model = col.replace(base_pattern, "")
            models.append(model)

    if not models:
        logger.warning("No co-designability motif models detected")
    return models


def merge_dfs(left: pd.DataFrame, right: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Merge two dataframes on groupby columns.

    Args:
        left: Left dataframe to merge
        right: Right dataframe to merge
        groupby_cols: Columns to merge on

    Returns:
        Merged dataframe
    """
    return pd.merge(left, right, on=groupby_cols, how="outer")


def compute_nsamples(df: pd.DataFrame, groupby_cols: list[str], path_store_results: str) -> pd.DataFrame:
    """Computes number of samples produced per run.

    This function is primarily used to verify that all jobs worked correctly by
    checking if each run produced the expected number of samples (500).

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results

    Returns:
        DataFrame containing sample counts per run

    Raises:
        Warning: If any count is different from 500
    """
    logger.info("Computing number of samples.")

    # Use pdb_path for counting since it's always available regardless of folding models
    count_col = "pdb_path"

    if count_col not in df.columns:
        logger.error(f"Column {count_col} not found. Cannot compute sample counts.")
        return pd.DataFrame()

    df_grouped = df.groupby(groupby_cols, dropna=False)[count_col].agg(list).reset_index()
    df_grouped["count"] = df_grouped[count_col].apply(lambda l: len(l))

    # Print warning if count is different from 500
    if any(df_grouped["count"] != 500):
        warnings.warn("Some count values are different from 500.")

    # Save results, drop col for easy viewing
    df_grouped = df_grouped.drop(count_col, axis=1)
    df_grouped.to_csv(
        os.path.join(path_store_results, "res_nsamples.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def compute_designability(df: pd.DataFrame, groupby_cols: list[str], path_store_results: str) -> pd.DataFrame:
    """Computes designability metrics for each set of hyperparameters.

    Designability is computed using a 2A RMSD threshold. The function computes
    both designability with 8 MPNN samples and with just 1 MPNN sample.
    Now supports multiple folding models and mode-specific columns.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results

    Returns:
        DataFrame containing designability metrics per run
    """
    logger.info("Computing designability with 2A RMSD threshold")

    # Detect available folding models
    folding_models = detect_folding_models(df.columns)
    logger.info(f"Detected folding models for designability: {folding_models}")

    # Log available scRMSD columns for debugging
    scrmsd_cols = [col for col in df.columns if col.startswith("_res_scRMSD") and not col.endswith("_all")]
    logger.info(f"Available scRMSD columns: {scrmsd_cols}")

    if not folding_models:
        logger.warning("No folding models detected, cannot compute designability")
        return pd.DataFrame()

    all_results = []
    rmsd_modes = ["ca", "bb3o", "all_atom"]

    # For each model, find all mode-specific columns
    for model in folding_models:
        for mode in rmsd_modes:
            scrmsd_col = f"_res_scRMSD_{mode}_{model}"
            scrmsd_all_col = f"_res_scRMSD_{mode}_{model}_all"
            suffix = f"_{mode}_{model}"

            # Check if this mode-specific column exists
            if scrmsd_col not in df.columns:
                continue  # Skip this mode-model combination

            logger.info(f"Computing designability for {model} ({mode})")

            if scrmsd_all_col not in df.columns:
                logger.warning(f"Column {scrmsd_all_col} not found, skipping MPNN-1 analysis for {model} ({mode})")
                # Still proceed with main designability calculation
                scrmsd_all_col = None

            # For designability with just 1 MPNN sample
            df_model = df.copy(deep=True)
            if scrmsd_all_col and scrmsd_all_col in df_model.columns:
                df_model[scrmsd_all_col] = df_model[scrmsd_all_col].apply(ast.literal_eval)
                df_model[f"_res_scRMSD_first{suffix}"] = df_model[scrmsd_all_col].apply(lambda x: x[0] if x else None)

                df_grouped = (
                    df_model.groupby(groupby_cols, dropna=False)[[scrmsd_col, f"_res_scRMSD_first{suffix}"]]
                    .agg(list)
                    .reset_index()
                )

                # MPNN 1
                df_grouped[f"des_2A_mpnn_1{suffix}"] = df_grouped[f"_res_scRMSD_first{suffix}"].apply(
                    lambda l: (
                        sum([v <= 2 for v in l if v is not None]) / len([v for v in l if v is not None])
                        if any(v is not None for v in l)
                        else 0.0
                    )
                )
                df_grouped[f"des_scRMSD_mpnn_1_mean{suffix}"] = df_grouped[f"_res_scRMSD_first{suffix}"].apply(
                    lambda l: (
                        np.mean([v for v in l if v is not None]) if any(v is not None for v in l) else float("inf")
                    )
                )

                # Clean up first column
                df_grouped = df_grouped.drop(f"_res_scRMSD_first{suffix}", axis=1)
            else:
                df_grouped = df_model.groupby(groupby_cols, dropna=False)[scrmsd_col].agg(list).reset_index()

            # MPNN 8
            df_grouped[f"des_2A{suffix}"] = df_grouped[scrmsd_col].apply(lambda l: sum([v <= 2 for v in l]) / len(l))
            df_grouped[f"des_scRMSD_mean{suffix}"] = df_grouped[scrmsd_col].apply(np.mean)

            # Clean up columns
            df_grouped = df_grouped.drop(scrmsd_col, axis=1)
            all_results.append(df_grouped)

    # Merge all results
    if len(all_results) > 1:
        result = reduce(lambda l, r: merge_dfs(l, r, groupby_cols), all_results)
    elif len(all_results) == 1:
        result = all_results[0]
    else:
        logger.warning("No designability results computed")
        return pd.DataFrame()

    result.to_csv(
        os.path.join(path_store_results, "res_designability.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return result


def compute_codesignability(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    mode: Literal["ca", "bb3", "bb3o", "all_atom"],
    incl_rmsd: bool = True,
) -> pd.DataFrame:
    """Computes codesignability metrics for each set of hyperparameters.

    Codesignability is computed using a 2A RMSD threshold. The function can compute
    metrics for different modes (ca, bb3, bb3o, all_atom) and optionally include
    RMSD statistics. Now supports multiple folding models.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        mode: Mode for codesignability computation ('ca', 'bb3', 'bb3o', or 'all_atom')
        incl_rmsd: Whether to include RMSD statistics in output

    Returns:
        DataFrame containing codesignability metrics per run
    """
    logger.info(f"Computing co-designability ({mode}) with 2A RMSD threshold")

    # Detect available folding models for this mode
    folding_models = detect_codesignability_folding_models(df.columns, mode)
    logger.info(f"Detected folding models for codesignability ({mode}): {folding_models}")

    # Log available codesignability columns for debugging
    co_scrmsd_cols = [col for col in df.columns if col.startswith(f"_res_co_scRMSD_{mode}_")]
    logger.info(f"Available co-scRMSD columns for {mode}: {co_scrmsd_cols}")

    if not folding_models:
        logger.warning(f"No folding models detected for codesignability ({mode})")
        return pd.DataFrame()

    all_results = []

    for model in folding_models:
        co_scrmsd_col = f"_res_co_scRMSD_{mode}_{model}"
        suffix = f"_{model}"

        # Validate required column exists
        if co_scrmsd_col not in df.columns:
            logger.warning(f"Column {co_scrmsd_col} not found, skipping {model}")
            continue

        logger.info(f"Computing codesignability for {model} ({mode})")

        df_grouped = df.groupby(groupby_cols, dropna=False)[co_scrmsd_col].agg(list).reset_index()
        df_grouped[f"codes_2A_{mode}{suffix}"] = df_grouped[co_scrmsd_col].apply(
            lambda l: sum([v <= 2 for v in l]) / len(l)
        )

        # Mean and std for scRMSD
        if incl_rmsd:
            df_grouped[f"codes_scRMSD_mean_{mode}{suffix}"] = df_grouped[co_scrmsd_col].apply(np.mean)
            df_grouped[f"codes_scRMSD_std_{mode}{suffix}"] = df_grouped[co_scrmsd_col].apply(np.std)

        # Clean up columns
        df_grouped = df_grouped.drop(co_scrmsd_col, axis=1)
        all_results.append(df_grouped)

    # Merge all results
    if len(all_results) > 1:
        result = reduce(lambda l, r: merge_dfs(l, r, groupby_cols), all_results)
    elif len(all_results) == 1:
        result = all_results[0]
    else:
        logger.warning(f"No codesignability results computed for {mode}")
        return pd.DataFrame()

    result.to_csv(
        os.path.join(path_store_results, f"res_codesignability_{mode}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return result


def compute_codesignability_per_len(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    mode: str,
) -> pd.DataFrame:
    """Computes codesignability metrics per length for each set of hyperparameters.

    This function computes codesignability metrics separately for each length,
    then aggregates the results into tuples for each hyperparameter configuration.
    Now supports multiple folding models.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        mode: Mode for codesignability computation ('ca', 'bb3', 'bb3o', or 'all_atom')

    Returns:
        DataFrame containing codesignability metrics per length per run
    """
    # Detect available folding models for this mode
    folding_models = detect_codesignability_folding_models(df.columns, mode)
    logger.info(f"Detected folding models for codesignability per length ({mode}): {folding_models}")

    all_results = []

    for model in folding_models:
        co_scrmsd_col = f"_res_co_scRMSD_{mode}_{model}"
        scrmsd_all_col = f"_res_scRMSD_{model}_all"
        suffix = f"_{model}"

        if co_scrmsd_col not in df.columns:
            logger.warning(f"Column {co_scrmsd_col} not found, skipping {model} for per-length analysis")
            continue

        logger.info(f"Computing codesignability per length for {model} ({mode})")

        # Create copy and keep only first element for scRMSD
        df_model = df.copy(deep=True)
        if scrmsd_all_col in df_model.columns:
            df_model[scrmsd_all_col] = df_model[scrmsd_all_col].apply(ast.literal_eval)
            df_model[f"_res_scRMSD_first{suffix}"] = df_model[scrmsd_all_col].apply(lambda x: x[0] if x else None)
        else:
            # Fallback to mode-specific scRMSD column if available
            for fallback_mode in ["ca", "bb3o", "all_atom"]:
                fallback_scrmsd_col = f"_res_scRMSD_{fallback_mode}_{model}"
                if fallback_scrmsd_col in df_model.columns:
                    df_model[f"_res_scRMSD_first{suffix}"] = df_model[fallback_scrmsd_col]
                    break
            else:
                logger.warning(f"No scRMSD data found for {model}, skipping designability calculation")
                df_model[f"_res_scRMSD_first{suffix}"] = None

        # Ensure the data is sorted by 'L' for correct tuple order
        df_model = df_model.sort_values(by="L")

        # Group including L
        agg_dict = {co_scrmsd_col: list}
        if f"_res_scRMSD_first{suffix}" in df_model.columns:
            agg_dict[f"_res_scRMSD_first{suffix}"] = list

        df_grouped = df_model.groupby(groupby_cols + ["L"], dropna=False).agg(agg_dict).reset_index()
        # Calculate metrics
        df_grouped[f"codes_2A_{mode}{suffix}"] = df_grouped[co_scrmsd_col].apply(
            lambda l: sum(v <= 2 for v in l) / len(l) if len(l) > 0 else 0
        )

        if f"_res_scRMSD_first{suffix}" in df_grouped.columns:
            df_grouped[f"des_2A_mpnn1{suffix}"] = df_grouped[f"_res_scRMSD_first{suffix}"].apply(
                lambda l: sum(v <= 2 for v in l) / len(l) if len(l) > 0 and all(v is not None for v in l) else 0
            )

        df_grouped[f"codes_scRMSD_mean_{mode}{suffix}"] = df_grouped[co_scrmsd_col].apply(np.mean)

        # Remove the list columns
        df_grouped.drop(columns=[co_scrmsd_col], inplace=True)
        if f"_res_scRMSD_first{suffix}" in df_grouped.columns:
            df_grouped.drop(columns=[f"_res_scRMSD_first{suffix}"], inplace=True)

        # Regroup without "L" to make results into tuples
        agg_dict_final = {
            f"codes_2A_{mode}{suffix}": tuple,
            f"codes_scRMSD_mean_{mode}{suffix}": tuple,
        }
        if f"des_2A_mpnn1{suffix}" in df_grouped.columns:
            agg_dict_final[f"des_2A_mpnn1{suffix}"] = tuple

        final_df = df_grouped.groupby(groupby_cols, dropna=False).agg(agg_dict_final).reset_index()

        all_results.append(final_df)

    # Merge all results
    if len(all_results) > 1:
        result = reduce(lambda l, r: merge_dfs(l, r, groupby_cols), all_results)
    elif len(all_results) == 1:
        result = all_results[0]
    else:
        logger.warning(f"No codesignability per length results computed for {mode}")
        return pd.DataFrame()

    # Save results
    output_file = os.path.join(path_store_results, f"res_codesignability_{mode}_len.csv")
    result.to_csv(output_file, sep=",", index=False, float_format="%.4f")

    return result


def compute_cosequence_recovery(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
) -> pd.DataFrame:
    """Computes co-sequence-recovery metrics for each set of hyperparameters.

    Co-sequence-recovery measures how well the model recovers the original
    sequence when co-designing structures.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results

    Returns:
        DataFrame containing co-sequence-recovery metrics per run
    """
    logger.info("Computing co-sequence-recovery")
    df_grouped = df.groupby(groupby_cols, dropna=False)["_res_co_seq_rec"].agg(list).reset_index()
    df_grouped["co_seq_rec"] = df_grouped["_res_co_seq_rec"].apply(lambda l: sum([v for v in l]) / len(l))

    # Save results, drop col for easy viewing
    df_grouped = df_grouped.drop("_res_co_seq_rec", axis=1)
    df_grouped.to_csv(
        os.path.join(path_store_results, "res_co_seq_rec.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def compute_recls_prob(df: pd.DataFrame, groupby_cols: list[str], path_store_results: str) -> pd.DataFrame:
    """Computes reclassification probability metrics for each set of hyperparameters.

    This function computes reclassification probabilities for each class category
    (common, rare, regular) based on CATH code frequencies.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results

    Returns:
        DataFrame containing reclassification probability metrics per run
    """
    logger.info("Computing re-classfication probability for each class")

    def class_category(x: pd.Series) -> str:
        """Determine class category based on CATH code frequency.

        Args:
            x: Row containing level and cath_code_frequency

        Returns:
            Category string: 'common', 'rare', or 'regular'
        """
        common_threshold = None
        rare_threshold = None
        if x["level"] == "C":
            common_threshold = 1000000
            rare_threshold = 1000000
        elif x["level"] == "A":
            common_threshold = 500000
            rare_threshold = 10000
        elif x["level"] == "T":
            common_threshold = 100000
            rare_threshold = 5000

        if x["cath_code_frequency"] >= common_threshold:
            return "common"
        if x["cath_code_frequency"] <= rare_threshold:
            return "rare"
        return "regular"

    # Save class-wise results
    df["class_category"] = df.apply(class_category, axis=1)
    df_grouped_cls = (
        df.groupby(groupby_cols + ["class_category"], dropna=False)["_res_recls_prob"].agg(list).reset_index()
    )
    df_grouped_cls["_res_cls_recls_prob_mean"] = df_grouped_cls["_res_recls_prob"].apply(lambda x: np.mean(x))
    df_grouped_cls["_res_cls_recls_prob_std"] = df_grouped_cls["_res_recls_prob"].apply(lambda x: np.std(x))
    df_grouped_cls["_res_cls_recls_prob_cnt"] = df_grouped_cls["_res_recls_prob"].apply(lambda x: len(x))
    df_grouped_cls = df_grouped_cls.drop("_res_recls_prob", axis=1)
    df_grouped_cls.to_csv(
        os.path.join(path_store_results, "res_recls_prob_per_class.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )

    # Save config-wise results
    groupby_cols_wo_class = [k for k in groupby_cols if k not in ["cath_code", "cath_code_frequency"]] + [
        "class_category"
    ]
    df_grouped = (
        df_grouped_cls.groupby(groupby_cols_wo_class, dropna=False)["_res_cls_recls_prob_mean"].agg(list).reset_index()
    )
    df_grouped["_res_recls_prob_mean"] = df_grouped["_res_cls_recls_prob_mean"].apply(lambda x: np.mean(x))
    df_grouped["_res_recls_prob_std"] = df_grouped["_res_cls_recls_prob_mean"].apply(lambda x: np.std(x))
    df_grouped["_res_recls_prob_cnt"] = df_grouped["_res_cls_recls_prob_mean"].apply(lambda x: len(x))
    df_grouped = df_grouped.drop("_res_cls_recls_prob_mean", axis=1)
    df_grouped.to_csv(
        os.path.join(path_store_results, "res_recls_prob.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def compute_ss(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
) -> pd.DataFrame:
    """Computes secondary structure content metrics for each set of hyperparameters.

    This function uses biotite to compute normalized secondary structure content
    (alpha helices, beta sheets, and coils) for each structure.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        metric_suffix: Suffix for metric names

    Returns:
        DataFrame containing secondary structure metrics per run
    """
    logger.info(f"Computing secondary structure with biotite - {metric_suffix}")

    # Group df
    df_grouped = df.groupby(groupby_cols, dropna=False)["pdb_path"].agg(list).reset_index()

    # Compute metrics
    tqdm.pandas()
    results_ss = {"biot_alpha": [], "biot_beta": [], "biot_coil": []}
    for idx, row in tqdm(df_grouped.iterrows(), total=df_grouped.shape[0]):
        list_of_pdbs = row["pdb_path"]
        results_ss_local = []
        for fname in list_of_pdbs:
            results_ss_local.append(compute_ss_metrics(fname))

        for k in results_ss:
            if len(results_ss_local) > 0:
                results_ss[k].append(np.mean([r[k] for r in results_ss_local]))
            else:
                results_ss[k].append(0.0)

    df_grouped[f"_res_ss_biot_alpha_{metric_suffix}"] = results_ss["biot_alpha"]
    df_grouped[f"_res_ss_biot_beta_{metric_suffix}"] = results_ss["biot_beta"]
    df_grouped[f"_res_ss_biot_coil_{metric_suffix}"] = results_ss["biot_coil"]

    # Save results and drop col for easy viewing
    df_grouped = df_grouped.drop("pdb_path", axis=1)
    df_grouped.to_csv(
        os.path.join(path_store_results, f"res_ss_biot_{metric_suffix}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def compute_res_ty_prop(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    plot: bool = True,
) -> pd.DataFrame:
    """Computes residue type proportions for each set of hyperparameters.

    This function computes the proportion of each amino acid type in the
    generated structures and optionally creates visualization plots.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        metric_suffix: Suffix for metric names
        plot: Whether to generate visualization plots

    Returns:
        DataFrame containing residue type proportions per run
    """

    def _count_residue_ty(fname: str) -> dict[str, int]:
        """Count occurrences of each residue type in a PDB file.

        Args:
            fname: Path to PDB file

        Returns:
            Dictionary mapping residue types to their counts
        """
        # Dictionary of amino acids with their three-letter codes
        standard_amino_acids = [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]
        # This will hold the count of each amino acid
        residue_count = dict.fromkeys(standard_amino_acids, 0)
        # Set to track unique residues
        seen_residues = set()

        # Open and read the PDB file
        with open(fname) as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    res_name = line[17:20].strip()
                    res_seq = line[22:26].strip()
                    chain_id = line[21].strip()
                    unique_residue_id = (chain_id, res_seq, res_name)

                    if res_name in standard_amino_acids and unique_residue_id not in seen_residues:
                        seen_residues.add(unique_residue_id)
                        residue_count[res_name] += 1

        return residue_count

    logger.info(f"Computing residue type counter (parsing pdb manually, faster) - {metric_suffix}")

    # Group df
    df_grouped = df.groupby(groupby_cols, dropna=False)["pdb_path"].agg(list).reset_index()

    # Compute metrics
    tqdm.pandas()
    results_rtype_prop = []
    for idx, row in tqdm(df_grouped.iterrows(), total=df_grouped.shape[0]):
        list_of_pdbs = row["pdb_path"]
        results_local = []
        for fname in list_of_pdbs:
            results_local.append(_count_residue_ty(fname))

        # Compute proportions
        residue_ty_count = {k: sum([r.get(k, 0) for r in results_local]) for k in restype_3to1}
        total = sum([residue_ty_count[k] for k in restype_3to1])
        residue_ty_propr = {k: residue_ty_count[k] / total for k in restype_3to1}

        results_rtype_prop.append(residue_ty_propr)

    for k in restype_3to1:
        df_grouped[f"_res_rtype_propr_{k}_{metric_suffix}"] = [r[k] for r in results_rtype_prop]

    # Save results and drop col for easy viewing
    df_grouped = df_grouped.drop("pdb_path", axis=1)

    if plot:
        dir_figs = os.path.join(path_store_results, f"res_type_figs_{metric_suffix}")
        if os.path.exists(dir_figs):
            shutil.rmtree(dir_figs)
        os.makedirs(dir_figs, exist_ok=False)

        plot_names = []
        for i, r in enumerate(results_rtype_prop):
            plot_name = f"props_{i}.png"
            plot_names.append(plot_name)
            sorted_keys_1 = [
                "L",
                "A",
                "E",
                "V",
                "G",
                "S",
                "I",
                "R",
                "K",
                "D",
                "T",
                "P",
                "F",
                "N",
                "Q",
                "Y",
                "M",
                "H",
                "W",
                "C",
            ]  # Same as used in Danny's plots
            sorted_keys_3 = [restype_1to3[v] for v in sorted_keys_1]
            plt.figure()
            sorted_keys = [v + f"_{restype_3to1[v]}" for v in sorted_keys_3]
            sorted_values = [r[key] for key in sorted_keys_3]
            plt.bar(sorted_keys, sorted_values)
            plt.xlabel("Residue type")
            plt.ylabel("Proportion")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig(os.path.join(dir_figs, plot_name))
            plt.close("all")

        df_grouped["_res_plot_res_prop"] = plot_names

    df_grouped.to_csv(
        os.path.join(path_store_results, f"res_type_prop_{metric_suffix}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )

    return df_grouped


def compute_novelty_tm(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    db_type: Literal["pdb", "afdb_rep_v4"],
) -> pd.DataFrame:
    """
    Computes novelty wrt some databse, returns pandas dataframe.
    """
    logger.info(f"Computing {db_type} novelty - {metric_suffix}")

    # Reduce with mean
    df_grouped = df.groupby(groupby_cols, dropna=False)[f"_res_novelty_{db_type}_tm"].mean().reset_index()
    df_grouped = df_grouped.rename(columns={f"_res_novelty_{db_type}_tm": f"_res_novelty_{db_type}_tm_{metric_suffix}"})
    df_grouped.to_csv(
        os.path.join(path_store_results, f"res_{db_type}_nov_{metric_suffix}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def get_representatative_folders(df: pd.DataFrame, groupby_cols: list[str], path_store_results: str) -> pd.DataFrame:
    """
    Gets a representative directory with samples for each set of hyperparameters (since each set of hyperparameters requires many generation and evaluation jobs),
    and returns the corresponding directories organized in a pandas dataframe.

    CAUTION: this assumes paths are of the form `./results_downloaded/<run_name>/inf_{number}_{suffix}/n_##_id_##/filename.pdb`
    and we want to extract the `inf_{number}_{suffix}` (e.g., `inf_0`, `inf_0_PDL1_myrun`)
    """
    df_grouped = df.groupby(groupby_cols, dropna=False)["pdb_path"].agg(list).reset_index()
    df_grouped["repr_dir"] = df_grouped["pdb_path"].apply(lambda paths: paths[0].split(os.path.sep)[-3])
    df_grouped = df_grouped.drop("pdb_path", axis=1)
    df_grouped.to_csv(
        os.path.join(path_store_results, "repr_dir.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def visualize_structures_top(df: pd.DataFrame, k: int = 1, limit: int = 30, save_for_publication: bool = False) -> None:
    """
    Generates PNG figures for the top-k set of hyperparameters (sorted by designability).

    Args:
        df: Pandas dataframe with results and paths
        k: top-k runs to generate figures for
        limit: max number of pdb files to visualize per set of hyperparameters
        save_for_publication: determines quality of generated images
    """
    if k == -1:
        top_k = df  # Plot all
    else:
        top_k = df.nlargest(k, "des_2A")  # Plot top_k
    for index, row in top_k.iterrows():
        name = row["repr_dir"]
        # designability = row["des_2A"]
        os.path.join(root, name)
        save_folder = os.path.join(root, "afig_" + name + "_figs")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(os.path.join(save_folder, "info.txt"), "w") as f:
            f.write(f"Info: {row}")
        # visualize_directory(
        #     pdb_folder=pdb_folder,
        #     save_folder=save_folder,
        #     save_for_publication=save_for_publication,
        #     limit=limit,
        # )
        # visualize_directory_all_atom(
        #     pdb_folder=pdb_folder,
        #     save_folder=save_folder,
        #     limit=limit,
        # )


def compute_motif_success(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    tmp_path_diversity: str,
    filter_models: list[str] = None,
    base_filter_type: str = "codesignability",
    base_threshold: float = 2.0,
    motif_threshold: float = 1.0,
    base_thresholds_by_mode: dict[str, float] = None,
    motif_thresholds_by_mode: dict[str, float] = None,
    direct_motif_thresholds_by_mode: dict[str, float] = None,
    require_all: bool = True,
    skip_unspecified_modes: bool = False,
) -> pd.DataFrame:
    """
    Computes motif success rates using the same filtering criteria as the main analysis.
    Returns pandas dataframe with results.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        metric_suffix: Suffix for metric names
        tmp_path_diversity: Path for temporary diversity files
        filter_models: List of folding model names to use for filtering
        base_filter_type: Either "designability" or "codesignability" for base filtering
        base_threshold: RMSD threshold for base filtering (fallback if base_thresholds_by_mode not provided)
        motif_threshold: RMSD threshold for motif filtering (fallback if motif_thresholds_by_mode not provided)
        base_thresholds_by_mode: Dict mapping modes to base thresholds
        motif_thresholds_by_mode: Dict mapping modes to motif thresholds
        direct_motif_thresholds_by_mode: Dict mapping modes to direct motif RMSD thresholds (before refolding)
        require_all: If True, requires all models to pass threshold; if False, requires any model to pass
        skip_unspecified_modes: If True, skip filtering for modes not specified in thresholds_by_mode dicts
    """
    logger.info(f"Computing motif success rates with advanced filtering - {metric_suffix}")

    # First apply direct motif RMSD filtering if specified
    df_working = df
    if direct_motif_thresholds_by_mode:
        df_working = filter_by_direct_motif_rmsd(
            df_working,
            thresholds_by_mode=direct_motif_thresholds_by_mode,
            require_all_modes=require_all,
            skip_unspecified_modes=skip_unspecified_modes,
        )
        logger.info(f"After direct motif RMSD filtering: {len(df_working)} out of {len(df)} samples")

    # Use the same filtering logic as the main analysis for refolded motif evaluation
    motif_filtered_dfs = []

    for motif_mode in ["ca", "bb3o", "all_atom"]:
        # Check if any motif columns exist for this mode
        base_metric_name = f"_res_co_motif_scRMSD_{motif_mode}_"
        mode_specific_cols = [col for col in df_working.columns if col.startswith(base_metric_name)]

        if mode_specific_cols:
            df_mode_filtered = filter_by_multiple_motif_models(
                df_working,
                filter_models or ["esmfold"],  # Default to esmfold if no models specified
                mode=motif_mode,
                base_filter_type=base_filter_type,
                base_threshold=base_threshold,
                motif_threshold=motif_threshold,
                base_thresholds_by_mode=base_thresholds_by_mode,
                motif_thresholds_by_mode=motif_thresholds_by_mode,
                require_all=require_all,
                skip_unspecified_modes=skip_unspecified_modes,
            )
            motif_filtered_dfs.append(df_mode_filtered)

            logic_type = "ALL" if require_all else "ANY"
            logger.info(
                f"Motif success filtering for {motif_mode} mode ({logic_type} logic): {len(df_mode_filtered)} out of {len(df_working)}"
            )

    # Take intersection of all mode filters (samples that pass in ALL modes)
    if motif_filtered_dfs:
        # Start with all indexes from the first filtered df
        valid_indexes = set(motif_filtered_dfs[0].index)

        # Intersect with indexes from all other filtered dfs
        for mode_df in motif_filtered_dfs[1:]:
            valid_indexes = valid_indexes.intersection(set(mode_df.index))

        # Get final filtered dataframe using the intersection of indexes
        df_motif = df_working.loc[list(valid_indexes)]

        logger.info(
            f"Final motif success filtering (intersection of all modes): {len(df_motif)} out of {len(df_working)}"
        )
        logger.info(f"Total filtering (including direct motif RMSD): {len(df_motif)} out of {len(df)} original samples")

        # Log thresholds used
        if direct_motif_thresholds_by_mode:
            logger.info("Direct motif RMSD thresholds used:")
            for mode, threshold in direct_motif_thresholds_by_mode.items():
                logger.info(f"  {mode}: direct_motif_rmsd≤{threshold}")

        if base_thresholds_by_mode or motif_thresholds_by_mode:
            logger.info("Mode-specific thresholds used for refolded motif success:")
            for mode in ["ca", "bb3o", "all_atom"]:
                base_thresh = (
                    base_thresholds_by_mode.get(mode, base_threshold) if base_thresholds_by_mode else base_threshold
                )
                motif_thresh = (
                    motif_thresholds_by_mode.get(mode, motif_threshold) if motif_thresholds_by_mode else motif_threshold
                )
                logger.info(f"  {mode}: {base_filter_type}≤{base_thresh}, motif≤{motif_thresh}")
        else:
            logger.info(
                f"Using uniform thresholds for refolded motif success: {base_filter_type}≤{base_threshold}, motif≤{motif_threshold}"
            )
    else:
        logger.warning("No motif data available for success calculation")
        return None

    if len(df_motif) == 0:
        logger.warning("No samples passed motif success criteria")
        return None

    df_motif["repr_dir"] = df_motif["pdb_path"].apply(lambda paths: Path(paths).parts[-3])

    # Count successes per inference config
    df_motif_grouped = df_motif.groupby("repr_dir").size().reset_index(name="motif_success")
    df_motif_grouped.to_csv(
        os.path.join(path_store_results, f"motif_success_{metric_suffix}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )

    # Log success per inference config
    for _, row in df_motif_grouped.iterrows():
        logger.info(f"Motif success for {row['repr_dir']}: {row['motif_success']}")

    # Copy successful PDB files
    for _, row in df_motif.iterrows():
        repr_dir = row["repr_dir"]
        src_folder = row["pdb_path"]
        dest_folder = os.path.join(path_store_results, f"motif_success_{metric_suffix}", repr_dir)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(src_folder, dest_folder)

    # Compute diversity metrics
    results = []
    try:
        min_seq_id = 0.0
        alignment_type = 1

        # Compute diversity for successful motif samples
        dfg = compute_foldseek_diversity(
            df_motif,
            groupby_cols,
            path_store_results,
            tmp_path=tmp_path_diversity,
            metric_suffix=f"motif_success_{metric_suffix}",
            min_seq_id=min_seq_id,
            alignment_type=alignment_type,
        )
        results.append(dfg)

    except Exception as e:
        logger.warning(f"Motif success Foldseek diversity computation failed: {e!s}")

    return pd.concat(results) if results else None


def filter_by_multiple_folding_models(
    df: pd.DataFrame,
    folding_models: list[str],
    mode: str = "ca",
    threshold: float = 2.0,
    require_all: bool = True,
) -> pd.DataFrame:
    """Filter dataframe by multiple folding models with specific mode and threshold.

    Args:
        df: Input dataframe
        folding_models: List of folding model names to filter by
        mode: RMSD mode to use (ca, bb3o, all_atom)
        threshold: RMSD threshold for filtering
        require_all: If True, requires all models to pass threshold; if False, requires any model to pass

    Returns:
        Filtered dataframe
    """
    if not folding_models:
        logger.warning("No folding models provided for filtering, returning original dataframe")
        return df

    available_cols = []

    for model in folding_models:
        # Try to find the column for this mode and model combination
        # Pattern: _res_scRMSD_{mode}_{model}
        col_name = f"_res_scRMSD_{mode}_{model}"

        if col_name in df.columns:
            available_cols.append(col_name)
        else:
            logger.warning(f"Column {col_name} not found, skipping model {model}")

    if not available_cols:
        logger.warning("No valid folding model columns found for filtering, returning original dataframe")
        return df

    logger.info(
        f"Filtering using {len(available_cols)} folding models: {[col.replace(f'_res_scRMSD_{mode}_', '') for col in available_cols]}"
    )

    # Create filter conditions
    conditions = [df[col] <= threshold for col in available_cols]

    if require_all:
        # All models must pass threshold
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition
    else:
        # Any model must pass threshold
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition

    filtered_df = df[combined_condition]

    return filtered_df


def filter_by_multiple_codesignability_models(
    df: pd.DataFrame,
    folding_models: list[str],
    mode: str = "ca",
    threshold: float = 2.0,
    thresholds_by_mode: dict[str, float] = None,
    require_all: bool = True,
) -> pd.DataFrame:
    """Filter dataframe by multiple codesignability models with specific mode and threshold.

    Now supports mode-specific thresholds.

    Args:
        df: Input dataframe
        folding_models: List of folding model names to filter by
        mode: RMSD mode to use (ca, bb3, bb3o, all_atom)
        threshold: RMSD threshold for filtering (used if thresholds_by_mode not provided)
        thresholds_by_mode: Dict mapping modes to thresholds (overrides threshold)
        require_all: If True, requires all models to pass threshold; if False, requires any model to pass

    Returns:
        Filtered dataframe
    """
    # Get mode-specific threshold or fall back to single value
    actual_threshold = thresholds_by_mode.get(mode, threshold) if thresholds_by_mode else threshold

    if not folding_models:
        logger.warning("No folding models provided for codesignability filtering, returning original dataframe")
        return df

    available_cols = []

    for model in folding_models:
        if model == "default":
            col_name = f"_res_co_scRMSD_{mode}"
        else:
            col_name = f"_res_co_scRMSD_{mode}_{model}"

        if col_name in df.columns:
            available_cols.append(col_name)
        else:
            logger.warning(f"Column {col_name} not found, skipping model {model}")

    if not available_cols:
        logger.warning("No valid codesignability model columns found for filtering, returning original dataframe")
        return df

    logger.info(
        f"Filtering using {len(available_cols)} codesignability models for {mode} mode with threshold ≤{actual_threshold}"
    )

    # Create filter conditions
    conditions = [df[col] <= actual_threshold for col in available_cols]

    if require_all:
        # All models must pass threshold
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition
    else:
        # Any model must pass threshold
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition

    filtered_df = df[combined_condition]

    return filtered_df


def filter_by_multiple_motif_models(
    df: pd.DataFrame,
    folding_models: list[str],
    mode: str = "all_atom",
    base_filter_type: str = "codesignability",
    base_threshold: float = 2.0,
    motif_threshold: float = 1.0,
    base_thresholds_by_mode: dict[str, float] = None,
    motif_thresholds_by_mode: dict[str, float] = None,
    require_all: bool = True,
    skip_unspecified_modes: bool = False,
) -> pd.DataFrame:
    """Filter dataframe by multiple motif models with specific mode and thresholds.

    Applies either designability or codesignability + motif-specific RMSD thresholds.
    Now supports mode-specific thresholds and option to skip unspecified modes.

    Args:
        df: Input dataframe
        folding_models: List of folding model names to filter by
        mode: RMSD mode to use (ca, bb3, bb3o, all_atom)
        base_filter_type: Either "designability" or "codesignability" for base filtering
        base_threshold: RMSD threshold for base filtering (used if base_thresholds_by_mode not provided)
        motif_threshold: RMSD threshold for motif-specific filtering (used if motif_thresholds_by_mode not provided)
        base_thresholds_by_mode: Dict mapping modes to base thresholds (overrides base_threshold)
        motif_thresholds_by_mode: Dict mapping modes to motif thresholds (overrides motif_threshold)
        require_all: If True, requires all models to pass threshold; if False, requires any model to pass
        skip_unspecified_modes: If True, skip filtering for modes not specified in thresholds_by_mode dicts

    Returns:
        Filtered dataframe
    """
    # Check if we should skip this mode when using mode-specific thresholds
    if skip_unspecified_modes:
        if base_thresholds_by_mode and mode not in base_thresholds_by_mode:
            logger.info(
                f"Skipping {base_filter_type} filtering for {mode} mode (not specified in mode-specific thresholds)"
            )
            # Still check motif thresholds
            if motif_thresholds_by_mode and mode not in motif_thresholds_by_mode:
                logger.info(f"Skipping motif filtering for {mode} mode (not specified in mode-specific thresholds)")
                return df  # No filtering applied for this mode
        elif motif_thresholds_by_mode and mode not in motif_thresholds_by_mode:
            logger.info(f"Skipping motif filtering for {mode} mode (not specified in mode-specific thresholds)")
            # Still apply base filtering with specified or default threshold

    # Get mode-specific thresholds or fall back to single values (if not skipping)
    if skip_unspecified_modes and base_thresholds_by_mode and mode not in base_thresholds_by_mode:
        actual_base_threshold = None  # Skip base filtering
    else:
        actual_base_threshold = (
            base_thresholds_by_mode.get(mode, base_threshold) if base_thresholds_by_mode else base_threshold
        )

    if skip_unspecified_modes and motif_thresholds_by_mode and mode not in motif_thresholds_by_mode:
        actual_motif_threshold = None  # Skip motif filtering
    else:
        actual_motif_threshold = (
            motif_thresholds_by_mode.get(mode, motif_threshold) if motif_thresholds_by_mode else motif_threshold
        )

    if not folding_models:
        logger.warning("No folding models provided for motif filtering, returning original dataframe")
        return df

    available_base_cols = []
    available_motif_cols = []

    for model in folding_models:
        if base_filter_type == "designability":
            # For designability, use mode-specific scRMSD columns
            base_col = f"_res_scRMSD_{mode}_{model}"
        else:  # codesignability
            base_col = f"_res_co_scRMSD_{mode}_{model}"
        motif_col = f"_res_co_motif_scRMSD_{mode}_{model}"

        if actual_base_threshold is not None and base_col in df.columns:
            available_base_cols.append(base_col)
        elif actual_base_threshold is not None:
            logger.warning(f"Column {base_col} not found, skipping model {model} for {base_filter_type}")

        if actual_motif_threshold is not None and motif_col in df.columns:
            available_motif_cols.append(motif_col)
        elif actual_motif_threshold is not None:
            logger.warning(f"Column {motif_col} not found, skipping model {model} for motif scoring")

    if not available_base_cols and not available_motif_cols:
        if actual_base_threshold is None and actual_motif_threshold is None:
            logger.info(f"Skipping all filtering for {mode} mode (not specified in mode-specific thresholds)")
            return df
        else:
            logger.warning("No valid motif model columns found for filtering, returning original dataframe")
            return df

    logger.info(
        f"Filtering using {len(available_base_cols)} {base_filter_type} columns and {len(available_motif_cols)} motif columns for {mode} mode"
    )
    if actual_base_threshold is not None and actual_motif_threshold is not None:
        logger.info(f"Using thresholds: {base_filter_type}≤{actual_base_threshold}, motif≤{actual_motif_threshold}")
    elif actual_base_threshold is not None:
        logger.info(f"Using threshold: {base_filter_type}≤{actual_base_threshold} (motif filtering skipped)")
    elif actual_motif_threshold is not None:
        logger.info(f"Using threshold: motif≤{actual_motif_threshold} ({base_filter_type} filtering skipped)")

    # Create filter conditions
    all_conditions = []

    # Add base conditions (designability or codesignability)
    if available_base_cols and actual_base_threshold is not None:
        base_conditions = [df[col] <= actual_base_threshold for col in available_base_cols]
        all_conditions.extend(base_conditions)

    # Add motif conditions
    if available_motif_cols and actual_motif_threshold is not None:
        motif_conditions = [df[col] <= actual_motif_threshold for col in available_motif_cols]
        all_conditions.extend(motif_conditions)

    if not all_conditions:
        logger.info(f"No filter conditions applied for {mode} mode, returning original dataframe")
        return df

    if require_all:
        # All conditions must pass
        combined_condition = all_conditions[0]
        for condition in all_conditions[1:]:
            combined_condition = combined_condition & condition
    else:
        # Any condition must pass
        combined_condition = all_conditions[0]
        for condition in all_conditions[1:]:
            combined_condition = combined_condition | condition

    filtered_df = df[combined_condition]

    return filtered_df


def filter_by_direct_motif_rmsd(
    df: pd.DataFrame,
    thresholds_by_mode: dict[str, float] = None,
    require_all_modes: bool = True,
    skip_unspecified_modes: bool = False,
) -> pd.DataFrame:
    """Filter dataframe by direct motif RMSD (before refolding) with mode-specific thresholds.

    Args:
        df: Input dataframe
        thresholds_by_mode: Dict mapping modes to RMSD thresholds for direct motif filtering
        require_all_modes: If True, requires all specified modes to pass threshold; if False, requires any mode to pass
        skip_unspecified_modes: If True, skip filtering for modes not specified in thresholds_by_mode

    Returns:
        Filtered dataframe
    """
    if not thresholds_by_mode:
        logger.info("No direct motif RMSD thresholds provided, skipping direct motif filtering")
        return df

    available_cols = []

    for mode, threshold in thresholds_by_mode.items():
        # Look for direct motif RMSD columns (before refolding)
        col_name = f"_res_motif_rmsd_{mode}"

        if col_name in df.columns:
            available_cols.append((col_name, threshold))
            logger.info(f"Found direct motif RMSD column for {mode} mode: {col_name}")
        else:
            logger.info(f"Direct motif RMSD column {col_name} not found - skipping {mode} mode filtering")

    if not available_cols:
        logger.info("No direct motif RMSD columns found for any specified modes, skipping direct motif filtering")
        return df

    logger.info(f"Filtering using {len(available_cols)} direct motif RMSD modes")
    for col_name, threshold in available_cols:
        mode = col_name.replace("_res_motif_rmsd_", "")
        logger.info(f"  {mode}: direct_motif_rmsd≤{threshold}")

    # Create filter conditions
    conditions = [df[col_name] <= threshold for col_name, threshold in available_cols]

    if require_all_modes:
        # All specified modes must pass threshold
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition
        logic_type = "ALL"
    else:
        # Any specified mode must pass threshold
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition
        logic_type = "ANY"

    filtered_df = df[combined_condition]

    logger.info(f"Direct motif RMSD filtering ({logic_type} logic): {len(filtered_df)} out of {len(df)} samples passed")

    return filtered_df


def compute_aggregated_aa_distribution(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
) -> pd.DataFrame:
    """Computes aggregated amino acid distribution for each set of hyperparameters.

    This function reads the `{seq_type}_aa_counts` and `{seq_type}_aa_interface_counts`
    columns and aggregates them elementwise for each job (groupby). Each column contains
    a list of length 20 representing counts for the 20 residue types.

    Args:
        df: Input dataframe containing results
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        metric_suffix: Suffix for metric names

    Returns:
        DataFrame containing aggregated amino acid counts per run
    """
    logger.info(f"Computing aggregated amino acid distribution - {metric_suffix}")

    # Auto-detect sequence types from dataframe columns
    sequence_types = []
    for col in df.columns:
        if col.startswith("mpnn_") and "aa_counts" in col:
            sequence_types.append("mpnn")
            break
    for col in df.columns:
        if col.startswith("mpnn_fixed_") and "aa_counts" in col:
            sequence_types.append("mpnn_fixed")
            break
    for col in df.columns:
        if col.startswith("self_") and "aa_counts" in col:
            sequence_types.append("self")
            break

    # If no sequence types found, return empty dataframe
    if not sequence_types:
        logger.warning("No sequence types detected in dataframe")
        return pd.DataFrame()

    logger.info(f"Analyzing sequence types: {sequence_types}")

    # Define columns for each sequence type
    all_columns = []
    agg_dict = {}

    for seq_type in sequence_types:
        # Columns for amino acid counts
        aa_counts_col = f"{seq_type}_aa_counts"
        aa_interface_counts_col = f"{seq_type}_aa_interface_counts"

        if aa_counts_col in df.columns:
            all_columns.append(aa_counts_col)
        if aa_interface_counts_col in df.columns:
            all_columns.append(aa_interface_counts_col)

        aa_keep = lambda series: keep_lists_separate(series, expected_len=20)
        for col in [aa_counts_col, aa_interface_counts_col]:
            if col in df.columns:
                agg_dict[col] = aa_keep

    # Filter to only include columns that exist in the dataframe
    existing_columns = [col for col in all_columns if col in df.columns]
    if not existing_columns:
        logger.warning("No amino acid count columns found in dataframe")
        return pd.DataFrame()

    df_grouped = df.groupby(groupby_cols, dropna=False)[existing_columns].agg(agg_dict).reset_index()

    # Calculate aggregated distributions for each sequence type
    for seq_type in sequence_types:
        aa_counts_col = f"{seq_type}_aa_counts"
        aa_interface_counts_col = f"{seq_type}_aa_interface_counts"

        # Process total amino acid counts
        if aa_counts_col in df_grouped.columns:
            # Sum all lists elementwise for each job
            df_grouped[f"_res_{seq_type}_aa_counts_aggregated_{metric_suffix}"] = df_grouped.apply(
                lambda row: [
                    sum(sample[i] for sample in row[aa_counts_col] if i < len(sample))
                    for i in range(20)  # 20 amino acid types
                ],
                axis=1,
            )

        # Process interface amino acid counts
        if aa_interface_counts_col in df_grouped.columns:
            # Sum all lists elementwise for each job
            df_grouped[f"_res_{seq_type}_aa_interface_counts_aggregated_{metric_suffix}"] = df_grouped.apply(
                lambda row: [
                    sum(sample[i] for sample in row[aa_interface_counts_col] if i < len(sample))
                    for i in range(20)  # 20 amino acid types
                ],
                axis=1,
            )

    # Remove the original metric columns to keep only the computed results
    columns_to_drop = [
        col
        for col in df_grouped.columns
        if any(seq_type in col for seq_type in sequence_types)
        and any(metric in col for metric in ["aa_counts", "aa_interface_counts"])
        and not any(computed in col for computed in ["aggregated"])
    ]
    df_grouped.drop(columns=columns_to_drop, inplace=True)

    df_grouped.to_csv(
        os.path.join(path_store_results, f"res_aa_distribution_{metric_suffix}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    return df_grouped


def compute_tmol_metrics(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    success_criteria: str = "self",
) -> pd.DataFrame:
    """Computes TMOL metrics for each set of hyperparameters.

    This function computes TMOL interface metrics including:
    - Number of interface hydrogen bonds
    - Total interface hydrogen bond energy
    - Total interface electrostatic energy
    - Number of interface electrostatic interactions
    - Top 20% average hydrogen bonds
    - Top 200 average hydrogen bonds (padded with zeros if needed)

    Args:
        df: Input dataframe containing results (should already be filtered by success criteria)
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        metric_suffix: Suffix for metric names
        success_criteria: Success criteria to use ("mpnn", "mpnn_fixed", or "self")

    Returns:
        DataFrame containing TMOL metrics per run
    """
    logger.info(f"Computing TMOL metrics - {metric_suffix} (success_criteria={success_criteria})")

    # Define TMOL metric columns to look for based on success criteria
    tmol_columns = []

    if success_criteria in SEQUENCE_TYPES:
        # For all success criteria: use generated_self + refolded_{success_criteria}
        # Generated structures: always use self sequence
        for metric in [
            "n_interface_hbonds_tmol",
            "total_interface_hbond_energy_tmol",
            "total_interface_elec_energy_tmol",
            "n_interface_elec_interactions_tmol",
        ]:
            tmol_columns.append(f"generated_{metric}")
            tmol_columns.append(f"refolded_{success_criteria}_{metric}")
    else:
        # Default behavior: compute for all combinations (backward compatibility)
        for struct_type in ["refolded"]:
            for seq_type in SEQUENCE_TYPES:
                tmol_columns.append(f"{struct_type}_{seq_type}_n_interface_hbonds_tmol")
                tmol_columns.append(f"{struct_type}_{seq_type}_total_interface_hbond_energy_tmol")
                tmol_columns.append(f"{struct_type}_{seq_type}_total_interface_elec_energy_tmol")
                tmol_columns.append(f"{struct_type}_{seq_type}_n_interface_elec_interactions_tmol")

    # Check which TMOL columns exist in the dataframe
    existing_tmol_cols = [col for col in tmol_columns if col in df.columns]

    if not existing_tmol_cols:
        logger.warning("No TMOL metric columns found in dataframe")
        return None

    logger.info(f"Found TMOL columns: {existing_tmol_cols}")

    # Group by hyperparameters and collect all values for each metric
    df_grouped = df.groupby(groupby_cols, dropna=False)[existing_tmol_cols].agg(list).reset_index()

    # Compute standard mean metrics
    for col in existing_tmol_cols:
        df_grouped[f"_res_{col}_{metric_suffix}"] = df_grouped[col].apply(lambda x: np.nanmean(x) if x else 0.0)

    # Remove the original list columns
    df_grouped = df_grouped.drop(columns=existing_tmol_cols)

    # Save results
    df_grouped.to_csv(
        os.path.join(path_store_results, f"res_tmol_{metric_suffix}.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )

    return df_grouped


def compute_bioinformatics_binder_interface_metrics(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    success_criteria: str = "self",
) -> pd.DataFrame:
    """Computes bioinformatics binder interface metrics for each set of hyperparameters.

    Args:
        df: Input dataframe containing results (should already be filtered by success criteria)
        groupby_cols: Columns to group results by
        path_store_results: Path to store results
        metric_suffix: Suffix for metric names
        success_criteria: Success criteria to use ("mpnn", "mpnn_fixed", or "self")
    """
    logger.info(
        f"Computing bioinformatics binder interface metrics - {metric_suffix} (success_criteria={success_criteria})"
    )

    # Define bioinformatics binder interface metric columns based on success criteria
    bioinformatics_binder_interface_cols = []

    if success_criteria in SEQUENCE_TYPES:
        # For all success criteria: use generated_self + refolded_{success_criteria}
        # Generated structures: always use self sequence
        for metric in [
            "binder_surface_hydrophobicity",
            "binder_interface_sc",
            "binder_interface_dSASA",
            "binder_interface_fraction",
            "binder_interface_hydrophobicity",
            "binder_interface_nres",
        ]:
            bioinformatics_binder_interface_cols.append(f"generated_{metric}")
            bioinformatics_binder_interface_cols.append(f"refolded_{success_criteria}_{metric}")
    else:
        # Default behavior: only self sequence type (backward compatibility)
        bioinformatics_binder_interface_cols = [
            "refolded_self_binder_surface_hydrophobicity",
            "refolded_self_binder_interface_sc",
            "refolded_self_binder_interface_dSASA",
            "refolded_self_binder_interface_fraction",
            "refolded_self_binder_interface_hydrophobicity",
            "refolded_self_binder_interface_nres",
            "generated_binder_surface_hydrophobicity",
            "generated_binder_interface_sc",
            "generated_binder_interface_dSASA",
            "generated_binder_interface_fraction",
            "generated_binder_interface_hydrophobicity",
            "generated_binder_interface_nres",
        ]

    # Check which bioinformatics binder interface metric columns exist in the dataframe
    existing_bioinformatics_binder_interface_cols = [
        col for col in bioinformatics_binder_interface_cols if col in df.columns
    ]

    if not existing_bioinformatics_binder_interface_cols:
        logger.warning("No bioinformatics binder interface metric columns found in dataframe")
        return pd.DataFrame()

    logger.info(
        f"Found bioinformatics binder interface metric columns: {existing_bioinformatics_binder_interface_cols}"
    )

    # Group by hyperparameters and aggregate bioinformatics binder interface metrics
    df_grouped = (
        df.groupby(groupby_cols, dropna=False)[existing_bioinformatics_binder_interface_cols].mean().reset_index()
    )

    # Rename columns to add metric suffix
    rename_dict = {}
    for col in existing_bioinformatics_binder_interface_cols:
        rename_dict[col] = f"_res_{col}_{metric_suffix}"

    df_grouped = df_grouped.rename(columns=rename_dict)

    # Save results
    df_grouped.to_csv(
        os.path.join(
            path_store_results,
            f"res_bioinformatics_binder_interface_{metric_suffix}.csv",
        ),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )

    return df_grouped


_TIMING_CSV_RE = re.compile(r"^timing_\d+\.csv$")


def _find_timing_csvs(root_path: str) -> list[tuple]:
    """Find per-job timing CSVs (``timing_{job_id}.csv``) under root_path.

    Supports two directory layouts:
      1. root_path directly contains timing CSVs (or a timing/ subdir)
         — used by analyze.py where results_dir IS the eval dir.
      2. root_path contains eval_* subdirs, each with timing CSVs
         — used by the legacy analysis.py script.

    Only matches ``timing_{digits}.csv`` to avoid picking up aggregated
    outputs like ``timing_summary.csv`` on re-runs.

    Returns list of (eval_config_name, csv_path) tuples.
    """

    def _collect_from_dir(search_dir: str, label: str, out: list):
        """Append (label, path) for each timing_{job_id}.csv in search_dir."""
        if not os.path.isdir(search_dir):
            return
        for fname in sorted(os.listdir(search_dir)):
            if _TIMING_CSV_RE.match(fname):
                out.append((label, os.path.join(search_dir, fname)))

    results = []

    # Layout 1: timing CSVs directly in root_path (or root_path/timing/)
    root_label = os.path.basename(root_path.rstrip("/")) or "root"
    timing_subdir = os.path.join(root_path, "timing")
    if os.path.isdir(timing_subdir):
        _collect_from_dir(timing_subdir, root_label, results)
    else:
        _collect_from_dir(root_path, root_label, results)

    # Layout 2: eval_* subdirs inside root_path
    for entry in sorted(os.listdir(root_path)):
        eval_dir = os.path.join(root_path, entry)
        if not os.path.isdir(eval_dir) or not entry.startswith("eval_"):
            continue
        sub_timing = os.path.join(eval_dir, "timing")
        if os.path.isdir(sub_timing):
            _collect_from_dir(sub_timing, entry, results)
        else:
            _collect_from_dir(eval_dir, entry, results)

    return results


def compute_timing_metrics(df, groupby_cols, path_store_results, root_path):
    """Aggregate per-job timing CSVs into a single summary per eval config."""
    logger.info("Collecting timing CSVs from eval subdirs")
    found = _find_timing_csvs(root_path)
    if not found:
        logger.warning(f"No timing CSVs found under {root_path}")
        return pd.DataFrame()
    timing_rows = []
    for eval_config, csv_path in found:
        try:
            tdf = pd.read_csv(csv_path)
            if tdf.empty:
                continue
            row = tdf.iloc[0]
            if "evaluation_time_s" in tdf.columns:
                eval_time = float(row["evaluation_time_s"])
            elif "evaluation_time" in tdf.columns:
                eval_time = float(row["evaluation_time"])
            elif "total_time" in tdf.columns:
                eval_time = float(row["total_time"])
            else:
                logger.warning(f"No time column in {csv_path}, skipping")
                continue
            timing_rows.append(
                {
                    "eval_config": eval_config,
                    "job_id": int(row.get("job_id", 0)),
                    "evaluation_time_s": eval_time,
                    "generation_time_s": float(row.get("generation_time", 0)),
                    "nsamples": int(row.get("nsamples", 0)),
                }
            )
        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")
            continue
    if not timing_rows:
        logger.warning("No valid timing data parsed")
        return pd.DataFrame()
    df_all = pd.DataFrame(timing_rows)
    logger.info(f"Read {len(df_all)} timing entries across {df_all['eval_config'].nunique()} eval configs")
    df_agg = (
        df_all.groupby("eval_config")
        .agg(
            num_jobs=("job_id", "count"),
            total_evaluation_time_s=("evaluation_time_s", "sum"),
            total_generation_time_s=("generation_time_s", "sum"),
            total_samples=("nsamples", "sum"),
            max_eval_time_s=("evaluation_time_s", "max"),
            min_eval_time_s=("evaluation_time_s", "min"),
        )
        .reset_index()
    )
    df_agg["avg_eval_time_per_sample_s"] = df_agg["total_evaluation_time_s"] / df_agg["total_samples"].replace(
        0, float("nan")
    )
    df_agg["total_evaluation_time_h"] = df_agg["total_evaluation_time_s"] / 3600
    out_path = os.path.join(path_store_results, "timing_summary.csv")
    df_agg.to_csv(out_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
    logger.info(f"Saved timing summary to {out_path}")
    df_result = df_agg.copy()
    df_result["_res_evaluation_time_seconds"] = df_result["total_evaluation_time_s"]
    df_result["_res_evaluation_time_hours"] = df_result["total_evaluation_time_h"]
    df_result["_res_total_samples"] = df_result["total_samples"]
    df_result["_res_avg_eval_time_per_sample_s"] = df_result["avg_eval_time_per_sample_s"]
    df_result.to_csv(
        os.path.join(path_store_results, "res_timing.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    logger.info(f"Timing metrics computed for {len(df_result)} eval configurations")
    return df_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--run_name",
        required=True,
        help="Name of the run/directory where results are stored, within 'results_downlaoded/'.",
    )
    parser.add_argument(
        "--visual_top_k",
        type=int,
        default=0,
        help="Number of runs for which we will generate images.",
    )
    parser.add_argument(
        "--do_ss_metric",
        action="store_true",
        help="If used will compute secondary structure metrics.",
    )
    parser.add_argument(
        "--do_foldseek_diversity_metric",
        action="store_true",
        help="If used will compute foldseek diversity metric (just structure, cluster through TM-score).",
    )
    parser.add_argument(
        "--do_foldseek_diversity_metric_seq_n_struct",
        action="store_true",
        help="If used will compute foldseek diversity metric (with sequence and structure).",
    )

    parser.add_argument(
        "--do_mmseqs_seq_diversity",
        action="store_true",
        help="If used will compute foldseek diversity metric (with sequence and structure).",
    )

    parser.add_argument(
        "--do_res_type_prop",
        action="store_true",
        help="If used will compute proportion for residue types.",
    )
    parser.add_argument(
        "--filter_folding_models",
        type=str,
        default=None,
        help="Comma-separated list of folding models to use for filtering (e.g., 'esmfold,colabfold').",
    )
    parser.add_argument(
        "--filter_mode",
        type=str,
        default="des",
        help="How to filter data, options are `des` (uses models specified in filter_folding_models), `codes`, or `motif_success`.",
    )
    parser.add_argument(
        "--filter_mode_detailed",
        type=str,
        default="all_atom",
        help="RMSD mode to use for filtering when using multiple folding models (ca, bb3, bb3o, all_atom). Default: all_atom.",
    )
    parser.add_argument(
        "--filter_require_all_models",
        action="store_true",
        help="If used, requires ALL specified folding models to pass threshold. Otherwise, ANY model passing is sufficient.",
    )
    parser.add_argument(
        "--eval_unfiltered",
        action="store_true",
        help="If used will also compute metrics for unfiltered data.",
    )
    parser.add_argument(
        "--do_motif_analysis",
        action="store_true",
        help="If used will compute motif success rates and diversity metrics.",
    )
    parser.add_argument(
        "--motif_codesignability_threshold",
        type=float,
        default=2.0,
        help="RMSD threshold for codesignability when using motif filtering. Default: 2.0. (Use --motif_codesignability_thresholds_by_mode for mode-specific thresholds)",
    )
    parser.add_argument(
        "--motif_threshold",
        type=float,
        default=1.0,
        help="RMSD threshold for motif-specific filtering. Default: 1.0. (Use --motif_thresholds_by_mode for mode-specific thresholds)",
    )
    parser.add_argument(
        "--motif_codesignability_thresholds_by_mode",
        type=str,
        default=None,
        help="Mode-specific RMSD thresholds for codesignability in format 'all_atom=2.0,ca=1.5,bb3o=1.8'. Overrides --motif_codesignability_threshold when provided. Only specified modes will be filtered.",
    )
    parser.add_argument(
        "--motif_thresholds_by_mode",
        type=str,
        default=None,
        help="Mode-specific RMSD thresholds for motif filtering in format 'all_atom=1.0,ca=0.8,bb3o=0.9'. Overrides --motif_threshold when provided. Only specified modes will be filtered.",
    )
    parser.add_argument(
        "--motif_base_filter_type",
        type=str,
        default="codesignability",
        choices=["designability", "codesignability"],
        help="Base filtering type for motif success: 'designability' or 'codesignability'. Default: codesignability.",
    )
    parser.add_argument(
        "--skip_unspecified_modes",
        action="store_true",
        help="If used, skip filtering entirely for modes not specified in mode-specific thresholds instead of using global defaults.",
    )
    parser.add_argument(
        "--direct_motif_thresholds_by_mode",
        type=str,
        default=None,
        help="Mode-specific RMSD thresholds for direct motif filtering (before refolding) in format 'all_atom=2.0,ca=1.0,bb3o=1.5'. Applied as first filtering step. Only specified modes will be filtered - unspecified modes are completely ignored (no fallback to defaults).",
    )
    parser.add_argument(
        "--require_all_direct_motif_modes",
        action="store_true",
        help="If used, requires ALL specified direct motif modes to pass threshold. Otherwise, ANY mode passing is sufficient.",
    )
    parser.add_argument(
        "--binder_success_sequence_type",
        type=str,
        default="mpnn,mpnn_fixed,self",
        help="Sequence type(s) to use for binder success filtering. Can be a single type ('mpnn', 'mpnn_fixed', 'self') or multiple types separated by commas (e.g., 'mpnn,mpnn_fixed,self'). Default: mpnn",
    )
    parser.add_argument(
        "--binder_folding_method",
        type=str,
        default="colabdesign",
        help="Binder folding method to use for binder success filtering. Can be 'colabdesign', 'ptx', or 'esmfold'. Default: colabdesign.",
    )
    parser.add_argument(
        "--success_thresholds",
        type=str,
        default=None,
        help="""Success thresholds as a JSON string. If not provided, uses defaults:
            - For protein binders (binder_success): DEFAULT_PROTEIN_BINDER_THRESHOLDS
            - For ligand binders (ligand_success): DEFAULT_LIGAND_BINDER_THRESHOLDS

            Format: '{"metric_name": {"threshold": value, "op": "<=", "scale": 1.0, "column_prefix": "complex"}}'
            Metric names are case-normalized (e.g., 'plddt' -> 'pLDDT', 'ipae' -> 'i_pAE').

            Example for protein binder with custom thresholds:
            '{"i_pAE": {"threshold": 7.0, "op": "<=", "scale": 31, "column_prefix": "complex"},
            "pLDDT": {"threshold": 0.9, "op": ">=", "column_prefix": "complex"},
            "scRMSD": {"threshold": 1.5, "op": "<", "column_prefix": "binder"},
            "i_pTM": {"threshold": 0.8, "op": ">=", "column_prefix": "complex"}}'
                    """,
    )
    parser.add_argument(
        "--do_aa_distribution",
        action="store_true",
        help="If used will compute aggregated amino acid distribution metrics.",
    )
    parser.add_argument(
        "--do_tmol_metrics",
        action="store_true",
        help="If used will compute TMOL interface metrics.",
    )
    parser.add_argument(
        "--do_bioinformatics_binder_interface_metrics",
        action="store_true",
        help="If used will compute bioinformatics binder interface metrics.",
    )
    parser.add_argument(
        "--do_timing_analysis",
        action="store_true",
        help="If used will compute timing metrics for each inference configuration.",
    )
    args = parser.parse_args()
    assert args.filter_mode in [
        "des",
        "codes",
        "motif_success",
        "binder_success",
        "ligand_success",
    ], f"Filter mode {args.filter_mode} invalid"
    assert args.filter_mode_detailed in [
        "ca",
        "bb3o",
        "all_atom",
    ], f"Filter mode detailed {args.filter_mode_detailed} invalid"

    # Validate binder success sequence types
    valid_sequence_types = SEQUENCE_TYPES
    if args.filter_mode == "binder_success":
        if "," in args.binder_success_sequence_type:
            sequence_types = [seq_type.strip() for seq_type in args.binder_success_sequence_type.split(",")]
            invalid_types = [seq_type for seq_type in sequence_types if seq_type not in valid_sequence_types]
            if invalid_types:
                raise ValueError(
                    f"Invalid sequence type(s) for binder success: {invalid_types}. Valid types are: {valid_sequence_types}"
                )
        else:
            if args.binder_success_sequence_type not in valid_sequence_types:
                raise ValueError(
                    f"Invalid sequence type for binder success: {args.binder_success_sequence_type}. Valid types are: {valid_sequence_types}"
                )

    load_dotenv()
    print(os.getenv("FOLDSEEK_EXEC"))
    print(os.getenv("MMSEQS_EXEC"))

    # Get results directory and verify that they were downloaded and exist
    root = os.path.join("./results_downloaded", args.run_name)
    if not os.path.isdir(root):
        logger.error(f"The results directory {root} does not exist.")
        sys.exit(1)

    logger.info(f"Loading results from {root}")
    logger.info(f"Filtering criteria: {args.filter_mode}")
    logger.info(f"Evaluating unfiltered data: {args.eval_unfiltered}")

    path_store_results = os.path.join(root, "a_results_processed")
    os.makedirs(path_store_results, exist_ok=True)
    logger.info(f"Will save results in {path_store_results}")

    # Parse mode-specific thresholds
    def parse_threshold_string(threshold_str):
        """Parse threshold string in format 'all_atom=2.0,ca=1.5,bb3o=1.8' into dictionary."""
        if not threshold_str:
            return None
        thresholds = {}
        for item in threshold_str.split(","):
            mode, threshold = item.split("=")
            thresholds[mode.strip()] = float(threshold.strip())
        return thresholds

    # Parse mode-specific thresholds with fallbacks
    motif_codesignability_thresholds = parse_threshold_string(args.motif_codesignability_thresholds_by_mode)
    motif_thresholds = parse_threshold_string(args.motif_thresholds_by_mode)
    direct_motif_thresholds = parse_threshold_string(args.direct_motif_thresholds_by_mode)

    logger.info(f"Mode-specific codesignability thresholds: {motif_codesignability_thresholds}")
    logger.info(f"Mode-specific motif thresholds: {motif_thresholds}")
    logger.info(f"Mode-specific direct motif thresholds: {direct_motif_thresholds}")

    # Parse flexible success thresholds from JSON string if provided
    import json

    success_thresholds = None
    if args.success_thresholds is not None:
        try:
            success_thresholds = json.loads(args.success_thresholds)
            logger.info("Using flexible success thresholds from --success_thresholds:")
            for metric_name, spec in success_thresholds.items():
                logger.info(f"  {metric_name}: {spec}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse --success_thresholds JSON: {e}")
            sys.exit(1)

    # Concatenate all dataframes
    dataframes = []
    for fname in os.listdir(root):
        if fname.endswith(".csv") and not fname.startswith("timing_"):
            fpath = os.path.join(root, fname)
            df_temp = pd.read_csv(fpath)

            # Extract inference config and job ID from filename
            # Filename format: *_results_{inf_config}_{job_id}.csv
            if "results_" in fname:
                # Split by "results_" and take the part after it
                parts_after_results = fname.split("results_")[1].replace(".csv", "")
                parts = parts_after_results.split("_")

                if len(parts) >= 2:
                    # Last part is job_id, everything before it is inf_config
                    job_id = parts[-1]  # Last part is job_id
                    inf_config = "_".join(parts[:-1])  # Everything before job_id is inf_config

                    df_temp["inf_config"] = inf_config
                    df_temp["job_id"] = job_id
                else:
                    df_temp["inf_config"] = "unknown"
                    df_temp["job_id"] = "unknown"
            else:
                df_temp["inf_config"] = "unknown"
                df_temp["job_id"] = "unknown"

            dataframes.append(df_temp)

    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"{len(df)} sample results loaded")
    columns_orig = df.columns

    # Add spaces to make ckpts names the same len
    # fn = lambda v: v + "    " if "-EMA.ckpt" not in v else v)
    # Fix paths
    df["pdb_path"] = df["pdb_path"].str.replace(r"./inference", root, regex=True)

    # Otherwise designability so low it breaks code :(
    # df = df[df["rot_sch"] == "exp"]  # Use something like this in case you want to get rid of certain runs for some parameters
    df = df.drop("run_name", axis=1)  # For nicer tables
    df = df.drop("ckpt_path", axis=1)  # For nicer tables

    # Create a copy of the original dataframe for aa_distribution computation if needed
    df_with_aa_counts = None
    if args.do_aa_distribution:
        df_with_aa_counts = df.copy()

    # Define columns to drop
    drop_keys = [
        "mpnn_aa_counts",
        "mpnn_aa_interface_counts",
        "mpnn_fixed_aa_counts",
        "mpnn_fixed_aa_interface_counts",
        "self_aa_counts",
        "self_aa_interface_counts",
    ]

    drop_key_prefixes = ["generation_reward_model_"]
    for key in df.columns:
        if key in drop_keys:
            df = df.drop(key, axis=1)  # For nicer tables
        else:
            for prefix in drop_key_prefixes:
                if key.startswith(prefix):
                    df = df.drop(key, axis=1)  # For nicer tables
                    break
    # Will use this to group by columns except [seed, L, id_gen, pdb_path]
    # Simply put, groups by all parameters that identify different inference setup
    # like checkpoint used, self conditioning, etc
    ignore = [
        "seed",
        "L",
        "id_gen",
        "pdb_path",
        "job_id",
        "mpnn_filter_pass",
        "mpnn_fixed_filter_pass",
        "self_filter_pass",
        "mpnn_filter_pass_all",
        "mpnn_fixed_filter_pass_all",
        "self_filter_pass_all",
        "L_x",
        "L_y",
    ]
    groupby_cols = [
        c
        for c in df.columns
        if c not in ignore
        and "_res_" not in c
        and "complex_" not in c
        and "binder_" not in c
        and "refolded_" not in c
        and "generated_" not in c
        and "_tmol" not in c
        and "_ligand" not in c
    ]

    # Detect available folding models
    folding_models = detect_folding_models(df.columns)
    logger.info(f"Detected folding models: {folding_models}")

    # Log available scRMSD columns for debugging
    scrmsd_cols = [col for col in df.columns if col.startswith("_res_scRMSD") and not col.endswith("_all")]
    logger.info(f"Available scRMSD columns: {scrmsd_cols}")
    # Parse filter folding models if provided
    filter_models = None
    if args.filter_folding_models:
        filter_models = [model.strip() for model in args.filter_folding_models.split(",")]
        logger.info(f"Using multiple models for filtering: {filter_models}")
        logger.info(f"Filter mode: {args.filter_mode_detailed}")

    all_dfs = []
    # Will store all results and merge at the end for a single table with all results
    df_designable = None
    df_co_designable = {}

    # Count number of samples per run
    compute_nsamples(df, groupby_cols, path_store_results)

    # Designability
    if filter_models:
        # Filter by designability using multiple models
        df_designable = filter_by_multiple_folding_models(
            df,
            filter_models,
            mode=args.filter_mode_detailed,
            threshold=2.0,
            require_all=args.filter_require_all_models,
        )
        logic_type = "ALL" if args.filter_require_all_models else "ANY"
        logger.info(f"Multi-model designable samples ({logic_type} logic): {len(df_designable)} out of {len(df)}")

        # Compute designability for all detected models
        dfg = compute_designability(df, groupby_cols, path_store_results)
        all_dfs.append(dfg)

    # Co-designability
    for codes_mode in ["ca", "bb3o", "all_atom"]:
        # Check if any codesignability columns exist for this mode
        base_metric_name = f"_res_co_scRMSD_{codes_mode}_"
        mode_specific_cols = [col for col in columns_orig if col.startswith(base_metric_name)]

        if mode_specific_cols:
            # Detect available folding models for this mode
            co_folding_models = detect_codesignability_folding_models(df.columns, codes_mode)
            logger.info(f"Detected codesignability folding models for {codes_mode}: {co_folding_models}")

            # Filter by codesignability using multiple models
            if filter_models:
                df_co_designable_tmp = filter_by_multiple_codesignability_models(
                    df,
                    filter_models,
                    mode=codes_mode,
                    threshold=2.0,
                    thresholds_by_mode=motif_codesignability_thresholds,
                    require_all=args.filter_require_all_models,
                )
                logic_type = "ALL" if args.filter_require_all_models else "ANY"
                logger.info(
                    f"Multi-model codesignable samples ({codes_mode}, {logic_type} logic): {len(df_co_designable_tmp)} out of {len(df)}"
                )
                df_co_designable[codes_mode] = df_co_designable_tmp

            # Compute codesignability metrics
            dfg = compute_codesignability(df, groupby_cols, path_store_results, codes_mode)
            all_dfs.append(dfg)

    # Co-designability with less info
    results_codes = []
    for codes_mode in ["ca", "bb3o", "all_atom"]:
        # Check if any codesignability columns exist for this mode
        base_metric_name = f"_res_co_scRMSD_{codes_mode}"
        mode_specific_cols = [col for col in columns_orig if col.startswith(base_metric_name)]

        if mode_specific_cols:
            # Compute codesignability (without RMSD details)
            dfg = compute_codesignability(
                df,
                groupby_cols,
                path_store_results,
                codes_mode,
                incl_rmsd=False,
            )
            results_codes.append(dfg)

    if len(results_codes) > 0:
        result = reduce(lambda l, r: merge_dfs(l, r, groupby_cols), results_codes)
        result.to_csv(
            os.path.join(path_store_results, "summary_codes.csv"),
            sep=SEP_CSV_PD,
            index=False,
            float_format=FLOAT_FORMAT_PD,
        )

    # Co-designability per length
    for codes_mode in ["ca", "bb3o", "all_atom"]:
        # Check if any codesignability columns exist for this mode
        base_metric_name = f"_res_co_scRMSD_{codes_mode}"
        mode_specific_cols = [col for col in columns_orig if col.startswith(base_metric_name)]

        if mode_specific_cols:
            compute_codesignability_per_len(df, groupby_cols, path_store_results, codes_mode)
            # Do not add to full table to keep it cleaner
    # Compute filter pass rate for binder tasks
    if args.filter_mode == "ligand_success":
        filter_pass_rate = compute_filter_ligand_pass_rate(
            df,
            groupby_cols,
            path_store_results,
            metric_suffix="all_samples",
            success_thresholds=success_thresholds,
        )
        all_dfs.append(filter_pass_rate)
        if df_designable is not None:
            filter_pass_rate = compute_filter_ligand_pass_rate(
                df_designable,
                groupby_cols,
                path_store_results,
                metric_suffix="des_samples",
                success_thresholds=success_thresholds,
            )
            all_dfs.append(filter_pass_rate)
    else:
        if (
            "mpnn_binder_scRMSD" in columns_orig
            or "mpnn_fixed_binder_scRMSD" in columns_orig
            or "self_binder_scRMSD" in columns_orig
        ):
            filter_pass_rate = compute_filter_pass_rate(
                df,
                groupby_cols,
                path_store_results,
                metric_suffix="all_samples",
                success_thresholds=success_thresholds,
            )
            all_dfs.append(filter_pass_rate)
            if df_designable is not None:
                filter_pass_rate = compute_filter_pass_rate(
                    df_designable,
                    groupby_cols,
                    path_store_results,
                    metric_suffix="des_samples",
                    success_thresholds=success_thresholds,
                )
                all_dfs.append(filter_pass_rate)

        if "_res_binder_scRMSD" in columns_orig:
            # Filter by designability using complex scRMSD for binder evaluation
            df["_res_scRMSD"] = df["_res_complex_scRMSD"]
            df_designable = df[df["_res_scRMSD"] <= 2]
            logger.info(f"Designable samples (binder): {len(df_designable)} out of {len(df)}")
            # Compute designability
            dfg = compute_designability(df, groupby_cols, path_store_results)
            all_dfs.append(dfg)

    # Get filtered results table
    if args.filter_mode == "des":
        df_filtered = df_designable
        if filter_models:
            logger.info(
                f"Using multi-model designability filter with {filter_models} (mode: {args.filter_mode_detailed})"
            )

    elif args.filter_mode == "codes":
        if args.filter_mode_detailed == "ca":
            df_filtered = df_co_designable.get("ca")
        elif args.filter_mode_detailed == "bb3o":
            df_filtered = df_co_designable.get("bb3o")
        elif args.filter_mode_detailed == "all_atom":
            df_filtered = df_co_designable.get("all_atom")

    elif args.filter_mode == "motif_success":
        # Motif filtering using filter_models - now supports multiple modes with different thresholds
        # First apply direct motif RMSD filtering if specified
        df_working = df
        if direct_motif_thresholds:
            df_working = filter_by_direct_motif_rmsd(
                df_working,
                thresholds_by_mode=direct_motif_thresholds,
                require_all_modes=args.require_all_direct_motif_modes,
                skip_unspecified_modes=args.skip_unspecified_modes,
            )
            logger.info(f"After direct motif RMSD filtering: {len(df_working)} out of {len(df)} samples")

        # Then apply refolded motif filtering
        motif_filtered_dfs = []

        for motif_mode in ["ca", "bb3o", "all_atom"]:
            # Check if any motif columns exist for this mode
            base_metric_name = f"_res_co_motif_scRMSD_{motif_mode}_"
            mode_specific_cols = [col for col in columns_orig if col.startswith(base_metric_name)]

            if mode_specific_cols:
                df_mode_filtered = filter_by_multiple_motif_models(
                    df_working,
                    filter_models,
                    mode=motif_mode,
                    base_filter_type=args.motif_base_filter_type,
                    base_threshold=args.motif_codesignability_threshold,
                    motif_threshold=args.motif_threshold,
                    base_thresholds_by_mode=motif_codesignability_thresholds,
                    motif_thresholds_by_mode=motif_thresholds,
                    require_all=args.filter_require_all_models,
                    skip_unspecified_modes=args.skip_unspecified_modes,
                )
                motif_filtered_dfs.append(df_mode_filtered)

                logic_type = "ALL" if args.filter_require_all_models else "ANY"
                logger.info(
                    f"Motif success filtering with {filter_models} for {motif_mode} mode ({logic_type} logic): {len(df_mode_filtered)} out of {len(df_working)}"
                )

        # Take intersection of all mode filters (samples that pass in ALL modes)
        if motif_filtered_dfs:
            # Start with all indexes from the first filtered df
            valid_indexes = set(motif_filtered_dfs[0].index)

            # Intersect with indexes from all other filtered dfs
            for mode_df in motif_filtered_dfs[1:]:
                valid_indexes = valid_indexes.intersection(set(mode_df.index))

            # Get final filtered dataframe using the intersection of indexes
            df_filtered = df_working.loc[list(valid_indexes)]

            logger.info(
                f"Final motif success filtering (intersection of all modes): {len(df_filtered)} out of {len(df_working)}"
            )
            logger.info(
                f"Total filtering (including direct motif RMSD): {len(df_filtered)} out of {len(df)} original samples"
            )

            # Log thresholds used
            if direct_motif_thresholds:
                logger.info("Direct motif RMSD thresholds used:")
                for mode, threshold in direct_motif_thresholds.items():
                    logger.info(f"  {mode}: direct_motif_rmsd≤{threshold}")

            if motif_codesignability_thresholds or motif_thresholds:
                logger.info("Mode-specific thresholds used for refolded motif filtering:")
                for mode in ["ca", "bb3o", "all_atom"]:
                    base_thresh = (
                        motif_codesignability_thresholds.get(mode, args.motif_codesignability_threshold)
                        if motif_codesignability_thresholds
                        else args.motif_codesignability_threshold
                    )
                    motif_thresh = (
                        motif_thresholds.get(mode, args.motif_threshold) if motif_thresholds else args.motif_threshold
                    )
                    logger.info(f"  {mode}: {args.motif_base_filter_type}≤{base_thresh}, motif≤{motif_thresh}")
            else:
                logger.info(
                    f"Using uniform thresholds for refolded motif filtering: {args.motif_base_filter_type}≤{args.motif_codesignability_threshold}, motif≤{args.motif_threshold}"
                )
        else:
            logger.warning("No motif data available for filtering")
            df_filtered = None

    elif args.filter_mode == "binder_success":
        # Binder success filtering based on sequence type(s)
        # Uses success_thresholds if provided, otherwise DEFAULT_PROTEIN_BINDER_THRESHOLDS
        if success_thresholds is not None:
            logger.info("Using custom success_thresholds for binder success filtering")
        else:
            logger.info("Using DEFAULT_PROTEIN_BINDER_THRESHOLDS for binder success filtering")

        if "," in args.binder_success_sequence_type:
            # Multiple sequence types specified
            sequence_types = [seq_type.strip() for seq_type in args.binder_success_sequence_type.split(",")]
            logger.info(f"Filtering by binder success using multiple sequence types: {sequence_types}")

            # Create separate filtered dataframes for each sequence type
            df_filtered_dict = {}
            for seq_type in sequence_types:
                df_filtered_seq = filter_by_binder_success(
                    df,
                    seq_type,
                    success_thresholds=success_thresholds,
                )
                if df_filtered_seq is not None and len(df_filtered_seq) > 0:
                    df_filtered_dict[seq_type] = df_filtered_seq
                    logger.info(f"Binder success filtering for {seq_type}: {len(df_filtered_seq)} samples")
                else:
                    logger.warning(f"No samples passed binder success criteria for sequence type: {seq_type}")

            # Store the dictionary of filtered dataframes
            if df_filtered_dict:
                df_filtered = df_filtered_dict  # Store as dictionary instead of concatenating
                logger.info(f"Binder success filtering: {len(df_filtered_dict)} sequence types with valid samples")
            else:
                logger.warning("No samples passed binder success criteria for any sequence type")
                df_filtered = None

        else:
            # Single sequence type
            seq_type = args.binder_success_sequence_type
            logger.info(f"Filtering by binder success using sequence type: {seq_type}")

            # Filter samples that pass binder success criteria
            df_filtered = filter_by_binder_success(
                df,
                seq_type,
                success_thresholds=success_thresholds,
            )

    elif args.filter_mode == "ligand_success":
        # Ligand binder success filtering based on sequence type(s)
        # Uses success_thresholds if provided, otherwise DEFAULT_LIGAND_BINDER_THRESHOLDS
        if success_thresholds is not None:
            logger.info("Using custom success_thresholds for ligand binder success filtering")
        else:
            logger.info("Using DEFAULT_LIGAND_BINDER_THRESHOLDS for ligand binder success filtering")

        if "," in args.binder_success_sequence_type:
            # Multiple sequence types specified
            sequence_types = [seq_type.strip() for seq_type in args.binder_success_sequence_type.split(",")]
            logger.info(f"Filtering by ligand binder success using multiple sequence types: {sequence_types}")

            # Create separate filtered dataframes for each sequence type
            df_filtered_dict = {}
            for seq_type in sequence_types:
                df_filtered_seq = filter_by_ligand_binder_success(
                    df,
                    seq_type,
                    path_store_results,
                    success_thresholds=success_thresholds,
                )
                if df_filtered_seq is not None and len(df_filtered_seq) > 0:
                    df_filtered_dict[seq_type] = df_filtered_seq
                    logger.info(f"Ligand binder success filtering for {seq_type}: {len(df_filtered_seq)} samples")
                else:
                    logger.warning(f"No samples passed ligand binder success criteria for sequence type: {seq_type}")

            # Store the dictionary of filtered dataframes
            if df_filtered_dict:
                df_filtered = df_filtered_dict  # Store as dictionary instead of concatenating
                logger.info(
                    f"Ligand binder success filtering: {len(df_filtered_dict)} sequence types with valid samples"
                )
            else:
                logger.warning("No samples passed ligand binder success criteria for any sequence type")
                df_filtered = None

        else:
            # Single sequence type
            seq_type = args.binder_success_sequence_type
            logger.info(f"Filtering by ligand binder success using sequence type: {seq_type}")

            # Filter samples that pass binder success criteria
            df_filtered = filter_by_ligand_binder_success(
                df,
                seq_type,
                path_store_results,
                success_thresholds=success_thresholds,
            )
    else:
        logger.error(f"Invalid filter mode {args.filter_mode}")
        df_filtered = None

    # Co-sequence-recovery
    if "_res_co_seq_rec" in columns_orig:
        dfg = compute_cosequence_recovery(df, groupby_cols, path_store_results)
        all_dfs.append(dfg)

    # Re-classification probability
    if "_res_recls_prob" in columns_orig:
        # Compute re-classification probability
        def class_category(x):
            common_threshold = None
            rare_threshold = None
            if x["level"] == "C":
                common_threshold = 1000000
                rare_threshold = 1000000
            elif x["level"] == "A":
                common_threshold = 500000
                rare_threshold = 10000
            elif x["level"] == "T":
                common_threshold = 100000
                rare_threshold = 5000

            if x["cath_code_frequency"] >= common_threshold:
                return "common"
            if x["cath_code_frequency"] <= rare_threshold:
                return "rare"
            return "regular"

        df["class_category"] = df.apply(class_category, axis=1)
        dfg = compute_recls_prob(df, groupby_cols, path_store_results)
        all_dfs.append(dfg)

    # Secondary structure with biotite
    if args.do_ss_metric:
        if args.eval_unfiltered:
            dfg = compute_ss(df, groupby_cols, path_store_results, metric_suffix="all_samples")
            all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_ss(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                    )
                    all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_ss(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                )
                all_dfs.append(dfg)

    # Residue type distribution
    if args.do_res_type_prop:
        if args.eval_unfiltered:
            dfg = compute_res_ty_prop(
                df=df,
                groupby_cols=groupby_cols,
                path_store_results=path_store_results,
                metric_suffix="all_samples",
            )

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_res_ty_prop(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                    )
                    # all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_res_ty_prop(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                )
                # all_dfs.append(dfg)

    # Amino acid distribution
    if args.do_aa_distribution:
        if args.eval_unfiltered:
            dfg = compute_aggregated_aa_distribution(
                df=df_with_aa_counts,
                groupby_cols=groupby_cols,
                path_store_results=path_store_results,
                metric_suffix="all_samples",
            )

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    # Apply the same filtering to the dataframe with aa_counts
                    df_filtered_with_aa = df_with_aa_counts.loc[df_filtered_seq.index]
                    dfg = compute_aggregated_aa_distribution(
                        df_filtered_with_aa,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                    )
            else:
                # Single sequence type
                # Apply the same filtering to the dataframe with aa_counts
                df_filtered_with_aa = df_with_aa_counts.loc[df_filtered.index]
                dfg = compute_aggregated_aa_distribution(
                    df_filtered_with_aa,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                )

    # TMOL metrics
    if args.do_tmol_metrics:
        if args.eval_unfiltered:
            dfg = compute_tmol_metrics(
                df=df,
                groupby_cols=groupby_cols,
                path_store_results=path_store_results,
                metric_suffix="all_samples",
            )
            if dfg is not None:
                all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_tmol_metrics(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                        success_criteria=seq_type,
                    )
                    if dfg is not None:
                        all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_tmol_metrics(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                    success_criteria=args.filter_mode,
                )
                if dfg is not None:
                    all_dfs.append(dfg)

    if args.do_bioinformatics_binder_interface_metrics:
        if args.eval_unfiltered:
            dfg = compute_bioinformatics_binder_interface_metrics(
                df=df,
                groupby_cols=groupby_cols,
                path_store_results=path_store_results,
                metric_suffix="all_samples",
            )
            all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_bioinformatics_binder_interface_metrics(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                        success_criteria=seq_type,
                    )
                    all_dfs.append(dfg)
            else:
                dfg = compute_bioinformatics_binder_interface_metrics(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                    success_criteria=args.filter_mode,
                )
                all_dfs.append(dfg)

    # Timing analysis
    if args.do_timing_analysis:
        dfg = compute_timing_metrics(
            df=df,
            groupby_cols=groupby_cols,
            path_store_results=path_store_results,
            root_path=root,
        )
        if dfg is not None and len(dfg) > 0:
            all_dfs.append(dfg)
    # Diversity with foldseek (just structure)
    if args.do_foldseek_diversity_metric:
        min_seq_id = 0.0
        alignment_type = 1
        tmp_path_diversity = os.path.join(root, "tmp_foldseek_diversity_joint")

        # if args.eval_unfiltered:
        #     dfg = compute_foldseek_diversity(
        #         df=df,
        #         groupby_cols=groupby_cols,
        #         path_store_results=path_store_results,
        #         tmp_path=tmp_path_diversity,
        #         metric_suffix="all_samples",
        #         min_seq_id=min_seq_id,
        #         alignment_type=alignment_type,
        #         diversity_mode="complex",
        #     )
        #     all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    # dfg = compute_foldseek_diversity(
                    #     df_filtered_seq,
                    #     groupby_cols,
                    #     path_store_results,
                    #     tmp_path_diversity,
                    #     metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                    #     min_seq_id=min_seq_id,
                    #     alignment_type=alignment_type,
                    #     diversity_mode="complex",
                    # )
                    # all_dfs.append(dfg)

                    # If this is binder_success mode, also compute diversity on binder chains and interfaces
                    if args.filter_mode == "binder_success":
                        # Compute diversity on binder chains for this sequence type
                        dfg_binders = compute_foldseek_diversity(
                            df_filtered_seq,
                            groupby_cols,
                            path_store_results,
                            tmp_path_diversity,
                            metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                            min_seq_id=min_seq_id,
                            alignment_type=alignment_type,
                            diversity_mode="binder",
                        )
                        all_dfs.append(dfg_binders)
                    elif args.filter_mode == "ligand_success":
                        dfg_binders = compute_foldseek_diversity(
                            df_filtered_seq,
                            groupby_cols,
                            path_store_results,
                            tmp_path_diversity,
                            metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                            min_seq_id=min_seq_id,
                            alignment_type=alignment_type,
                            diversity_mode="binder",
                        )
                        all_dfs.append(dfg_binders)
            else:
                # Single sequence type
                # dfg = compute_foldseek_diversity(
                #     df_filtered,
                #     groupby_cols,
                #     path_store_results,
                #     tmp_path_diversity,
                #     metric_suffix=f"filtered_samples_{args.filter_mode}",
                #     min_seq_id=min_seq_id,
                #     alignment_type=alignment_type,
                #     diversity_mode="complex",
                # )
                # all_dfs.append(dfg)

                # If this is binder_success mode, also compute diversity on binder chains and interfaces
                if args.filter_mode == "binder_success":
                    # Single sequence type - compute diversity on binder chains
                    dfg_binders = compute_foldseek_diversity(
                        df_filtered,
                        groupby_cols,
                        path_store_results,
                        tmp_path_diversity,
                        metric_suffix=f"filtered_samples_{args.filter_mode}",
                        min_seq_id=min_seq_id,
                        alignment_type=alignment_type,
                        diversity_mode="binder",
                    )
                    all_dfs.append(dfg_binders)
                elif args.filter_mode == "ligand_success":
                    dfg_binders = compute_foldseek_diversity(
                        df_filtered,
                        groupby_cols,
                        path_store_results,
                        tmp_path_diversity,
                        metric_suffix=f"filtered_samples_{args.filter_mode}",
                        min_seq_id=min_seq_id,
                        alignment_type=alignment_type,
                        diversity_mode="binder",
                    )
                all_dfs.append(dfg_binders)

                # Compute diversity on interfaces (additional computation for binder task)
                # dfg_interfaces = compute_foldseek_diversity(
                #     df_filtered,
                #     groupby_cols,
                #     path_store_results,
                #     tmp_path_diversity,
                #     metric_suffix=f"filtered_samples_{args.filter_mode}",
                #     min_seq_id=0.0,  # For structure-only alignment
                #     alignment_type=1,  # Structure-only alignment
                #     diversity_mode="interface",
                # )
                # all_dfs.append(dfg_interfaces)

    # Diversity with foldseek (sequence and structure)
    if args.do_foldseek_diversity_metric_seq_n_struct:
        min_seq_id = 0.1  # Could revisit this value
        alignment_type = 2
        tmp_path_diversity = os.path.join(root, "tmp_foldseek_diversity_joint")

        if args.eval_unfiltered:
            dfg = compute_foldseek_diversity(
                df=df,
                groupby_cols=groupby_cols,
                path_store_results=path_store_results,
                tmp_path=tmp_path_diversity,
                metric_suffix="joint_all_samples",
                min_seq_id=min_seq_id,
                alignment_type=alignment_type,
                diversity_mode="complex",
            )
            all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_foldseek_diversity(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        tmp_path=tmp_path_diversity,
                        metric_suffix=f"joint_filtered_samples_{args.filter_mode}_{seq_type}",
                        min_seq_id=min_seq_id,
                        alignment_type=alignment_type,
                        diversity_mode="complex",
                    )
                    all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_foldseek_diversity(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    tmp_path=tmp_path_diversity,
                    metric_suffix=f"joint_filtered_samples_{args.filter_mode}",
                    min_seq_id=min_seq_id,
                    alignment_type=alignment_type,
                    diversity_mode="complex",
                )
                all_dfs.append(dfg)

    if args.do_mmseqs_seq_diversity:
        # Could revisit these value
        min_seq_id = 0.1
        coverage = 0.7
        tmp_path_diversity = os.path.join(root, "tmp_mmseqs_diversity_seq")

        if args.eval_unfiltered:
            dfg = compute_mmseqs_diversity(
                df=df,
                groupby_cols=groupby_cols,
                path_store_results=path_store_results,
                tmp_path=tmp_path_diversity,
                metric_suffix="all_samples",
                min_seq_id=min_seq_id,
                coverage=coverage,
            )
            all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_mmseqs_diversity(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        tmp_path=tmp_path_diversity,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                        min_seq_id=min_seq_id,
                        coverage=coverage,
                    )
                    all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_mmseqs_diversity(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    tmp_path=tmp_path_diversity,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                    min_seq_id=min_seq_id,
                    coverage=coverage,
                )
                all_dfs.append(dfg)

    # Novelty against PDB
    if "_res_novelty_pdb_tm" in columns_orig:
        if args.eval_unfiltered:
            dfg = compute_novelty_tm(
                df,
                groupby_cols,
                path_store_results,
                metric_suffix="all_samples",
                db_type="pdb",
            )
            all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_novelty_tm(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                        db_type="pdb",
                    )
                    all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_novelty_tm(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                    db_type="pdb",
                )
                all_dfs.append(dfg)

    # Novelty against afdb_rep_v4
    if "_res_novelty_afdb_rep_v4_tm" in columns_orig:
        if args.eval_unfiltered:
            dfg = compute_novelty_tm(
                df,
                groupby_cols,
                path_store_results,
                metric_suffix="all_samples",
                db_type="afdb_rep_v4",
            )
            all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_novelty_tm(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                        db_type="afdb_rep_v4",
                    )
                    all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_novelty_tm(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                    db_type="afdb_rep_v4",
                )
                all_dfs.append(dfg)

    # Novelty against afdb_rep_v4 with genie filters up to length 512
    if "_res_novelty_afdb_rep_v4_geniefilters_maxlen512_tm" in columns_orig:
        if args.eval_unfiltered:
            dfg = compute_novelty_tm(
                df,
                groupby_cols,
                path_store_results,
                metric_suffix="all_samples",
                db_type="afdb_rep_v4_geniefilters_maxlen512",
            )
            all_dfs.append(dfg)

        if df_filtered is not None:
            if isinstance(df_filtered, dict):
                # Multiple sequence types - process each separately
                for seq_type, df_filtered_seq in df_filtered.items():
                    dfg = compute_novelty_tm(
                        df_filtered_seq,
                        groupby_cols,
                        path_store_results,
                        metric_suffix=f"filtered_samples_{args.filter_mode}_{seq_type}",
                        db_type="afdb_rep_v4_geniefilters_maxlen512",
                    )
                    all_dfs.append(dfg)
            else:
                # Single sequence type
                dfg = compute_novelty_tm(
                    df_filtered,
                    groupby_cols,
                    path_store_results,
                    metric_suffix=f"filtered_samples_{args.filter_mode}",
                    db_type="afdb_rep_v4_geniefilters_maxlen512",
                )
                all_dfs.append(dfg)

    # Motif analysis
    if args.do_motif_analysis:
        tmp_path_diversity = os.path.join(root, "tmp_foldseek_diversity_joint_motif")
        dfg = compute_motif_success(
            df,
            groupby_cols,
            path_store_results,
            metric_suffix="all_samples",
            tmp_path_diversity=tmp_path_diversity,
            filter_models=filter_models,
            base_filter_type=args.motif_base_filter_type,
            base_threshold=args.motif_codesignability_threshold,
            motif_threshold=args.motif_threshold,
            base_thresholds_by_mode=motif_codesignability_thresholds,
            motif_thresholds_by_mode=motif_thresholds,
            direct_motif_thresholds_by_mode=direct_motif_thresholds,
            require_all=args.filter_require_all_models,
            skip_unspecified_modes=args.skip_unspecified_modes,
        )
        if dfg is not None:
            all_dfs.append(dfg)

    # Get representative directory for each sampling config (useful to plot)
    dfg = get_representatative_folders(df, groupby_cols, path_store_results)
    all_dfs.append(dfg)

    # Save all generated results in a single table
    result = reduce(lambda l, r: merge_dfs(l, r, groupby_cols), all_dfs)
    result.to_csv(
        os.path.join(path_store_results, "res_all.csv"),
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )

    # Generate visualizations for some structures
    if args.visual_top_k != 0:
        visualize_structures_top(result, k=args.visual_top_k, limit=50)
