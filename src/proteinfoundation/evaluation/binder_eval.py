"""
Binder-specific evaluation metrics.

This module provides functions for evaluating protein binder designs:
- Refolding with structure prediction models (ColabDesign, Boltz2, RF3, Protenix)
- Interface analysis (bioinformatics metrics)
- Force field metrics (hydrogen bonds, electrostatics)
"""

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from atomworks.io.utils.io_utils import load_any
from loguru import logger
from omegaconf import DictConfig
from openfold.np.residue_constants import restypes as OF_RESTYPES

from proteinfoundation.evaluation.binder_eval_utils import (
    BIOINFORMATICS_METRIC_COLS,
    DEFAULT_INTERFACE_CUTOFF_LIGAND,
    DEFAULT_INTERFACE_CUTOFF_PROTEIN,
    DEFAULT_LIGAND_RANKING_CRITERIA,
    DEFAULT_NUM_REDESIGN_SEQS_LIGAND,
    DEFAULT_NUM_REDESIGN_SEQS_PROTEIN,
    DEFAULT_PROTEIN_RANKING_CRITERIA,
    TMOL_METRIC_COLS,
    get_binder_chain_from_complex,
    get_metric_columns,
    select_best_sample_idx,
    validate_ranking_criteria,
)
from proteinfoundation.evaluation.esm_eval import ESM_AVAILABLE, compute_esm_ppl_for_sequences
from proteinfoundation.evaluation.utils import maybe_tqdm, parse_cfg_for_table
from proteinfoundation.metrics.binder_metrics import run_binder_eval
from proteinfoundation.result_analysis.analysis_utils import SEQUENCE_TYPES
from proteinfoundation.rewards.base_reward import REWARD_KEY

# =============================================================================
# Safe Imports with Availability Flags
# =============================================================================

# TMOL (force field metrics) - may not be available in all environments
TMOL_AVAILABLE = False
try:
    from proteinfoundation.rewards.tmol_reward import TmolRewardModel

    TMOL_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    logger.warning(f"TMOL import failed: {e}. TMOL metrics will return NaN.")

# PR Alternative (bioinformatics interface scoring) - may have missing dependencies
PR_ALTERNATIVE_AVAILABLE = False
try:
    from proteinfoundation.utils.pr_alternative_utils import pr_alternative_score_interface

    PR_ALTERNATIVE_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    logger.warning(f"PR Alternative import failed: {e}. Bioinformatics metrics will return NaN.")


# =============================================================================
# Folding Model Initialization
# =============================================================================


def initialize_folding_model(
    folding_model: str,
    target_pdb_chain: list[str],
    target_task_name: str,
    is_target_ligand: bool,
) -> dict[str, Any]:
    """Initialize folding model specs for binder evaluation.

    Supported models: colabdesign, protenix_*, rf3_*, boltz2_*.

    Args:
        folding_model: Name of the folding model (e.g. ``"colabdesign"``,
            ``"protenix_v0.4.0"``, ``"rf3_latest"``, ``"boltz2_v1"``).
        target_pdb_chain: Sorted chain IDs of the target structure.
        target_task_name: Task name used to resolve MSA / template paths.
        is_target_ligand: Whether the target is a small-molecule ligand.

    Returns:
        Dictionary with ``"model_name"`` and model-specific runner / path keys.

    Raises:
        ValueError: If the model name is unsupported or incompatible with the
            target type.
    """
    target_pdb_chain = sorted(target_pdb_chain)

    if folding_model == "colabdesign":
        if is_target_ligand:
            raise ValueError("ColabDesign does not support ligand-protein complex folding")
        return {"model_name": "colabdesign"}

    elif "rf3" in folding_model:
        from proteinfoundation.rewards.rf3_reward import get_default_rf3_runner

        logger.info(f"Initializing RF3 model: {folding_model}")
        runner = get_default_rf3_runner(
            ckpt_path=os.environ.get(
                "RF3_CKPT_PATH",
                os.path.join(os.environ.get("DATA_PATH", ""), "rf3/rf3_latest.pt"),
            ),
            dump_dir=None,
            rf3_path=os.environ.get("RF3_EXEC_PATH", None),
        )
        return {"model_name": "RF3", "runner": runner}

    else:
        raise ValueError(f"Folding model '{folding_model}' not supported")


# =============================================================================
# Binder Metrics
# =============================================================================


def compute_binder_metrics(
    eval_config: DictConfig,
    sample_root_paths: list[str],
    target_pdb_path: str,
    target_pdb_chain: list[str],
    is_target_ligand: bool,
    get_fixed_residues_fn: Callable[[str, str], list[str] | None] | None = None,
) -> pd.DataFrame:
    """Compute comprehensive metrics for protein binders.

    Runs refolding evaluation and extracts metrics for each sequence type.

    Args:
        eval_config: Top-level evaluation config.
        sample_root_paths: Directories containing generated PDBs.
        target_pdb_path: Path to the target PDB.
        target_pdb_chain: Target chain IDs.
        is_target_ligand: Whether the target is a ligand.
        get_fixed_residues_fn: Optional per-sample callback for ``mpnn_fixed``
            residues.  Signature: ``(pdb_path, binder_chain) -> Optional[List[str]]``.
            When provided, called for each sample to get fixed positions
            (e.g. motif residues) instead of interface-based detection.
            Return ``None`` to fall back to interface residues for that sample.

    Returns:
        DataFrame with binder metrics, one row per sample.
    """
    logger.info("Starting binder evaluation")
    cfg_metric = eval_config.metric

    # Get dataset config - fallback to generation config only if dataset doesn't exist
    if "dataset" in eval_config:
        cfg_dataset = eval_config.dataset
    else:
        cfg_dataset = eval_config.generation.dataloader.dataset
    target_task_name = cfg_dataset.task_name

    # Initialize folding model
    folding_model = cfg_metric.get("binder_folding_method", "colabdesign")
    folding_model_specs = initialize_folding_model(folding_model, target_pdb_chain, target_task_name, is_target_ligand)

    # Evaluation parameters
    sequence_types = cfg_metric.get("sequence_types", ["self"])
    interface_cutoff = cfg_metric.get(
        "interface_cutoff",
        (DEFAULT_INTERFACE_CUTOFF_LIGAND if is_target_ligand else DEFAULT_INTERFACE_CUTOFF_PROTEIN),
    )
    num_redesign_seqs = cfg_metric.get(
        "num_redesign_seqs",
        (DEFAULT_NUM_REDESIGN_SEQS_LIGAND if is_target_ligand else DEFAULT_NUM_REDESIGN_SEQS_PROTEIN),
    )
    inverse_folding_model = cfg_metric.get("inverse_folding_model", "protein_mpnn")

    if inverse_folding_model not in ["protein_mpnn", "ligand_mpnn", "soluble_mpnn"]:
        raise ValueError(f"Inverse folding model '{inverse_folding_model}' not supported")

    # Get ranking criteria from config or use defaults
    ranking_criteria = cfg_metric.get("ranking_criteria", None)
    if ranking_criteria is None:
        ranking_criteria = DEFAULT_LIGAND_RANKING_CRITERIA if is_target_ligand else DEFAULT_PROTEIN_RANKING_CRITERIA
    else:
        ranking_criteria = validate_ranking_criteria(ranking_criteria)
    logger.info(f"Using ranking criteria: {ranking_criteria}")

    # Progress bar setting
    show_progress = eval_config.get("show_progress", False)

    # Setup columns
    columns, flat_dict = parse_cfg_for_table(eval_config)
    all_columns = columns + ["id_gen", "pdb_path", "L", "task_name"]

    results = []
    binder_chain = None
    gen_target_chain = None

    for idx, sample_root_path in enumerate(maybe_tqdm(sample_root_paths, "Binder evaluation", show_progress)):
        pdb_path = os.path.join(sample_root_path, os.path.basename(sample_root_path) + ".pdb")

        # Validate PDB file exists
        if not os.path.exists(pdb_path):
            logger.warning(f"PDB file not found: {pdb_path}, skipping")
            continue

        # Detect chains on first sample
        if binder_chain is None:
            chains = sorted(set(load_any(pdb_path)[0].chain_id.tolist()))
            binder_chain = chains[-1]
            gen_target_chain = chains[:-1]
            logger.info(f"Detected chains - binder: {binder_chain}, target: {gen_target_chain}")

        row_dict = {
            **flat_dict,
            "id_gen": idx,
            "pdb_path": pdb_path,
            "task_name": target_task_name,
        }

        if cfg_metric.get("compute_binder_metrics", True):
            # Per-sample fixed residue override (e.g. motif residues)
            fixed_residues_override = None
            if get_fixed_residues_fn is not None:
                fixed_residues_override = get_fixed_residues_fn(pdb_path, binder_chain)

            _, _, sequence_type_stats, sequences_dict = run_binder_eval(
                pdb_file_path=pdb_path,
                target_pdb_path=target_pdb_path,
                folding_model_specs=folding_model_specs,
                tmp_path=sample_root_path,
                target_pdb_chain=target_pdb_chain,
                sequence_types=sequence_types,
                inverse_folding_model=inverse_folding_model,
                gen_target_chain=gen_target_chain,
                binder_chain=binder_chain,
                interface_cutoff=interface_cutoff,
                is_target_ligand=is_target_ligand,
                num_redesign_seqs=num_redesign_seqs,
                fixed_residues_override=fixed_residues_override,
            )

            # Save raw stats
            with open(os.path.join(sample_root_path, "sequence_type_stats.json"), "w") as f:
                json.dump(sequence_type_stats, f, indent=4)

            # Extract metrics for each sequence type
            for seq_type in sequence_types:
                seq_stats = sequence_type_stats[seq_type]["complex_stats"]
                if not seq_stats:
                    logger.debug(f"No complex stats for {seq_type} at sample {idx}, skipping")
                    continue

                # Find best sample using composite ranking score
                best_idx, best_score = select_best_sample_idx(seq_stats, ranking_criteria)

                if best_idx < 0 or best_score == float("inf"):
                    # Fall back to first sample only if stats exist
                    if len(seq_stats) > 0:
                        logger.warning(f"No valid samples found for {seq_type}, defaulting to first sample")
                        best_idx = 0
                    else:
                        logger.warning(f"Empty stats for {seq_type} at sample {idx}, skipping")
                        continue
                else:
                    logger.debug(f"Best sample for {seq_type}: {best_idx} with score {best_score:.4f}")

                # Extract best sample metrics
                best_complex = seq_stats[best_idx]
                best_rmsd = sequence_type_stats[seq_type]["rmsd_stats"][best_idx]
                aa_stats = sequence_type_stats[seq_type]["aa_stats"][best_idx]

                row_dict["L"] = aa_stats["binder_length"]

                # Complex metrics (best and all)
                for metric, value in best_complex.items():
                    col = f"{seq_type}_{metric}" if metric == "complex_pdb_path" else f"{seq_type}_complex_{metric}"
                    row_dict[col] = value
                    row_dict[f"{col}_all"] = [s[metric] for s in seq_stats]
                    if idx == 0:
                        all_columns.extend([col, f"{col}_all"])

                # RMSD metrics (best and all)
                for metric, value in best_rmsd.items():
                    col = f"{seq_type}_{metric}"
                    row_dict[col] = value
                    row_dict[f"{col}_all"] = [s[metric] for s in sequence_type_stats[seq_type]["rmsd_stats"]]
                    if idx == 0:
                        all_columns.extend([col, f"{col}_all"])

                # AA composition
                res_count = [0] * len(OF_RESTYPES)
                interface_count = [0] * len(OF_RESTYPES)
                for aa, count in aa_stats["residue_counts"].items():
                    if aa in OF_RESTYPES:
                        res_count[OF_RESTYPES.index(aa)] += count
                for aa, count in aa_stats["interface_counts"].items():
                    if aa in OF_RESTYPES:
                        interface_count[OF_RESTYPES.index(aa)] += count

                row_dict[f"{seq_type}_aa_counts"] = res_count
                row_dict[f"{seq_type}_aa_interface_counts"] = interface_count
                if idx == 0:
                    all_columns.extend([f"{seq_type}_aa_counts", f"{seq_type}_aa_interface_counts"])

                # Store sequences (best and all)
                seqs = [s["seq"] for s in sequences_dict[seq_type]]
                if seqs:
                    seq_best_idx = 0 if seq_type == "self" else best_idx
                    row_dict[f"{seq_type}_sequence"] = seqs[seq_best_idx]
                    row_dict[f"{seq_type}_sequence_all"] = seqs
                    if idx == 0:
                        all_columns.extend([f"{seq_type}_sequence", f"{seq_type}_sequence_all"])

                # ESM pseudo-perplexity metrics (optional)
                if cfg_metric.get("compute_esm_metrics", False) and ESM_AVAILABLE and seqs:
                    esm_model = cfg_metric.get("esm_model", "facebook/esm2_t33_650M_UR50D")
                    esm_df = compute_esm_ppl_for_sequences(seqs, model_name=esm_model)

                    row_dict[f"{seq_type}_esm_pseudo_perplexity"] = esm_df["esm_pseudo_perplexity"].iloc[seq_best_idx]
                    row_dict[f"{seq_type}_esm_log_likelihood"] = esm_df["esm_log_likelihood"].iloc[seq_best_idx]
                    row_dict[f"{seq_type}_esm_pseudo_perplexity_all"] = esm_df["esm_pseudo_perplexity"].tolist()
                    row_dict[f"{seq_type}_esm_log_likelihood_all"] = esm_df["esm_log_likelihood"].tolist()

                    if idx == 0:
                        all_columns.extend(
                            [
                                f"{seq_type}_esm_pseudo_perplexity",
                                f"{seq_type}_esm_log_likelihood",
                                f"{seq_type}_esm_pseudo_perplexity_all",
                                f"{seq_type}_esm_log_likelihood_all",
                            ]
                        )

        results.append(row_dict)

    return pd.DataFrame(results).reindex(columns=all_columns)


# =============================================================================
# Interface Metrics - Single PDB Functions (Core Building Blocks)
# =============================================================================


def compute_bioinformatics_metrics_single(
    pdb_path: str,
    binder_chain: str,
    target_chain: str,
    sc_bin: str | None = None,
) -> dict[str, Any]:
    """
    Compute bioinformatics interface metrics for a single PDB.

    Args:
        pdb_path: Path to PDB file
        binder_chain: Chain ID of the binder
        target_chain: Chain ID(s) of the target (comma-separated if multiple)
        sc_bin: Path to shape complementarity binary

    Returns:
        Dictionary of metric names to values. Returns NaN if dependencies unavailable.
    """
    if not PR_ALTERNATIVE_AVAILABLE:
        return dict.fromkeys(BIOINFORMATICS_METRIC_COLS, np.nan)

    sc_bin = sc_bin or os.environ.get("SC_EXEC", "/usr/local/bin/sc")

    try:
        scores, _, _ = pr_alternative_score_interface(
            pdb_path,
            binder_chain=binder_chain,
            target_chain=target_chain,
            sasa_engine="auto",
            sc_bin=sc_bin,
        )
        return {
            "binder_surface_hydrophobicity": round(scores["surface_hydrophobicity"], 2),
            "binder_interface_sc": round(scores["interface_sc"], 2),
            "binder_interface_dSASA": round(scores["interface_dSASA"], 2),
            "binder_interface_fraction": round(scores["interface_fraction"], 2),
            "binder_interface_hydrophobicity": round(scores["interface_hydrophobicity"], 2),
            "binder_interface_nres": scores["interface_nres"],
        }
    except Exception as e:
        logger.error(f"Bioinformatics metrics failed for {pdb_path}: {e}")
        return dict.fromkeys(BIOINFORMATICS_METRIC_COLS, np.nan)


def compute_tmol_metrics_single(
    pdb_path: str,
    tmol_model: Any | None = None,
) -> dict[str, Any]:
    """
    Compute TMOL force field metrics for a single PDB.

    Args:
        pdb_path: Path to PDB file
        tmol_model: Initialized TmolRewardModel instance

    Returns:
        Dictionary of metric names to values. Returns NaN if TMOL unavailable.
    """
    if not TMOL_AVAILABLE or tmol_model is None:
        return dict.fromkeys(TMOL_METRIC_COLS, np.nan)

    try:
        result = tmol_model.score(pdb_path=pdb_path, requires_grad=False)
        return {
            "n_interface_hbonds_tmol": result[REWARD_KEY]["n_interface_hbonds"].item(),
            "total_interface_hbond_energy_tmol": result[REWARD_KEY]["total_interface_hbond_energy"].item(),
            "total_interface_elec_energy_tmol": result[REWARD_KEY]["total_interface_elec_energy"].item(),
            "n_interface_elec_interactions_tmol": result[REWARD_KEY]["n_interface_elec_interactions"].item(),
        }
    except Exception as e:
        logger.error(f"TMOL error for {pdb_path}: {e}")
        return dict.fromkeys(TMOL_METRIC_COLS, np.nan)


# =============================================================================
# Interface Metrics - Unified Computation
# =============================================================================


def compute_interface_metrics(
    pdb_paths: list[str],
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
    sc_bin: str | None = None,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """
    Unified function to compute interface metrics for a batch of PDB files.

    This is the main entry point for computing any combination of interface metrics.
    Each metric type can be independently enabled/disabled for full control.

    If a dependency is not available (e.g., TMOL not installed), the corresponding
    metrics will be set to NaN and a warning will be logged.

    Args:
        pdb_paths: List of PDB file paths
        compute_bioinformatics: Whether to compute bioinformatics metrics (SC, SASA, hydrophobicity)
        compute_tmol: Whether to compute TMOL force field metrics
        sc_bin: Path to shape complementarity binary (for bioinformatics)
        show_progress: Whether to show progress bar

    Returns:
        List of dictionaries containing metrics for each PDB
    """
    if not pdb_paths:
        logger.debug("compute_interface_metrics called with empty pdb_paths")
        return []

    # Log what metrics will be computed
    enabled_metrics = []
    if compute_bioinformatics:
        enabled_metrics.append("bioinformatics")
    if compute_tmol:
        enabled_metrics.append("tmol")

    logger.info(f"Computing interface metrics for {len(pdb_paths)} PDBs: {enabled_metrics}")

    # Check if any metrics are requested
    if not enabled_metrics:
        logger.warning("No metrics requested in compute_interface_metrics")
        return [{"pdb_path": p} for p in pdb_paths]

    # Log availability warnings upfront
    if compute_bioinformatics and not PR_ALTERNATIVE_AVAILABLE:
        logger.warning("Bioinformatics metrics requested but PR Alternative not available. Metrics will be NaN.")

    if compute_tmol and not TMOL_AVAILABLE:
        logger.warning("TMOL metrics requested but TMOL not available. Metrics will be NaN.")

    # Initialize TMOL model lazily if needed and available
    tmol_model = None
    if compute_tmol and TMOL_AVAILABLE:
        try:
            tmol_model = TmolRewardModel(enable_hbond=True, enable_elec=True)
        except Exception as e:
            logger.warning(f"Failed to initialize TMOL model: {e}. TMOL metrics will be NaN.")

    # Detect chains from first PDB (assume consistent across batch)
    binder_chain = None
    target_chain = None
    multi_target = False

    results = []

    for pdb_path in maybe_tqdm(pdb_paths, "Interface metrics", show_progress):
        metrics = {"pdb_path": pdb_path}

        # Detect chains on first iteration
        if binder_chain is None:
            binder_chain, target_chains, multi_target = get_binder_chain_from_complex(
                pdb_path, return_multi_target=True
            )
            target_chain = ",".join(target_chains)
            logger.debug(f"Detected chains - binder: {binder_chain}, target: {target_chain}")

        # Bioinformatics metrics (return 0 for multi-target complexes)
        if compute_bioinformatics:
            if multi_target:
                logger.info("Multi-target complex detected. Bioinformatics metrics will be 0.")
                bio_metrics = dict.fromkeys(BIOINFORMATICS_METRIC_COLS, 0)
            else:
                logger.info("Computing bioinformatics metrics...")
                bio_metrics = compute_bioinformatics_metrics_single(pdb_path, binder_chain, target_chain, sc_bin)
            metrics.update(bio_metrics)

        # TMOL metrics
        if compute_tmol:
            logger.info("Computing TMOL metrics...")
            metrics.update(compute_tmol_metrics_single(pdb_path, tmol_model))

        results.append(metrics)

    logger.info(f"Interface metrics complete: {len(results)} PDBs processed")
    return results


def compute_interface_metrics_df(
    cfg: DictConfig,
    pdb_paths: list[str],
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
    sc_bin: str | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute interface metrics and return as a DataFrame with config columns.

    This is a convenience wrapper around compute_interface_metrics that adds
    the config-derived columns needed for merging with other evaluation results.

    Args:
        cfg: Configuration (used to extract columns for DataFrame)
        pdb_paths: List of PDB file paths
        compute_bioinformatics: Whether to compute bioinformatics metrics
        compute_tmol: Whether to compute TMOL metrics
        sc_bin: Path to shape complementarity binary
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with config columns + metric columns
    """
    columns, flat_dict = parse_cfg_for_table(cfg)

    # Compute metrics
    metrics_list = compute_interface_metrics(
        pdb_paths=pdb_paths,
        compute_bioinformatics=compute_bioinformatics,
        compute_tmol=compute_tmol,
        sc_bin=sc_bin,
        show_progress=show_progress,
    )

    # Build result rows with config columns
    results = []
    for i, metrics in enumerate(metrics_list):
        row = {**flat_dict, "id_gen": i, **metrics}
        results.append(row)

    # Build column order
    metric_cols = get_metric_columns(
        compute_bioinformatics=compute_bioinformatics,
        compute_tmol=compute_tmol,
    )

    all_columns = columns + ["id_gen", "pdb_path"] + metric_cols

    return pd.DataFrame(results).reindex(columns=all_columns)


# =============================================================================
# Refolded Structure Metrics
# =============================================================================


def merge_metrics_to_df(
    df: pd.DataFrame,
    metrics_list: list[dict[str, Any]],
    sample_names: list[str],
    column_prefix: str,
    skip_cols: set | None = None,
) -> pd.DataFrame:
    """
    Merge computed metrics back into the original DataFrame.

    Args:
        df: Original DataFrame
        metrics_list: List of metric dictionaries
        sample_names: List of sample names to match against pdb_path
        column_prefix: Prefix for new column names
        skip_cols: Columns to skip when merging

    Returns:
        Updated DataFrame with merged metrics
    """
    if skip_cols is None:
        skip_cols = {"pdb_path"}

    updated_df = df.copy()

    for sample_name, metrics in zip(sample_names, metrics_list, strict=False):
        sample_mask = updated_df["pdb_path"].str.contains(sample_name, regex=False)
        if not sample_mask.any():
            continue

        for col, value in metrics.items():
            if col in skip_cols:
                continue

            col_name = f"{column_prefix}{col}"
            if col_name not in updated_df.columns:
                updated_df[col_name] = None

            if isinstance(value, list):
                for idx in updated_df[sample_mask].index:
                    updated_df.at[idx, col_name] = value
            else:
                updated_df.loc[sample_mask, col_name] = value

    return updated_df


def compute_interface_metrics_on_refolded_structures(
    df: pd.DataFrame,
    best_paths_dict: dict[str, dict[str, str]],
    cfg_metric: DictConfig,
    cfg: DictConfig,
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute force field and bioinformatics metrics on successful refolded structures.

    Args:
        df: DataFrame with evaluation results.
        best_paths_dict: Dictionary of best refolded structure paths.
        cfg_metric: Metric configuration.
        cfg: Full configuration.
        compute_bioinformatics: Whether to compute bioinformatics metrics (SC, SASA, hydrophobicity).
        compute_tmol: Whether to compute TMOL force field metrics.
        show_progress: Whether to show progress bar (default: False).

    Returns:
        DataFrame with added refolded structure metrics.
    """
    # Check if any metrics requested
    if not any([compute_bioinformatics, compute_tmol]):
        return df

    successful_samples = []
    sequence_types = cfg_metric.get("sequence_types", SEQUENCE_TYPES)
    for _, row in df.iterrows():
        # Extract sample name from pdb_path
        pdb_path = row["pdb_path"]
        sample_name = os.path.basename(pdb_path).replace(".pdb", "").replace("tmp_", "")
        if sample_name not in best_paths_dict:
            continue
        for seq_type in sequence_types:
            structure_path = best_paths_dict[sample_name].get(seq_type)
            if structure_path and os.path.exists(structure_path):
                successful_samples.append((sample_name, seq_type, structure_path))
                logger.debug(f"Found successful best sample: {sample_name} {seq_type}")
            else:
                logger.debug(f"Structure path not found for successful sample: {sample_name} {seq_type}")

    logger.info(f"Found {len(successful_samples)} successful best samples with refolded structures")
    if not successful_samples:
        logger.warning("No successful refolded structures found")
        return df

    # Log which metrics are being computed
    enabled_metrics = []
    if compute_bioinformatics:
        enabled_metrics.append("bioinformatics")
    if compute_tmol:
        enabled_metrics.append("TMOL")

    logger.info(f"Computing metrics [{', '.join(enabled_metrics)}] on {len(successful_samples)} refolded structures")

    # Group samples by sequence type for efficient processing
    samples_by_seq_type: dict[str, list[tuple[str, str]]] = {}
    for sample_name, seq_type, path in successful_samples:
        if seq_type not in samples_by_seq_type:
            samples_by_seq_type[seq_type] = []
        samples_by_seq_type[seq_type].append((sample_name, path))

    # Compute and merge metrics for each sequence type
    updated_df = df.copy()
    _, flat_dict = parse_cfg_for_table(cfg)
    skip_cols = {"pdb_path"} | set(flat_dict.keys())

    for seq_type, samples in samples_by_seq_type.items():
        sample_names = [s[0] for s in samples]
        structure_paths = [s[1] for s in samples]

        # Compute metrics using unified function with individual flags
        metrics_list = compute_interface_metrics(
            pdb_paths=structure_paths,
            compute_bioinformatics=compute_bioinformatics,
            compute_tmol=compute_tmol,
            show_progress=show_progress,
        )

        # Merge with appropriate prefix
        prefix = f"refolded_{seq_type}_"
        updated_df = merge_metrics_to_df(
            df=updated_df,
            metrics_list=metrics_list,
            sample_names=sample_names,
            column_prefix=prefix,
            skip_cols=skip_cols,
        )

    logger.info("Successfully merged refolded structure metrics")
    return updated_df
