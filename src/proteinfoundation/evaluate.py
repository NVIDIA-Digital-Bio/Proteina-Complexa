"""
Unified evaluation script supporting multiple metric types and protein types.

Protein Types:
- monomer: Single chain proteins
- binder: Binder + target complexes (binder is last chain)
- monomer_motif: Monomer scaffolding with motif constraints
- motif_binder: Binder + target with unindexed motif constraints

Metrics (controlled via compute_* flags):
- Monomer metrics: Designability, codesignability, novelty, sequence recovery
- Binder metrics: Refolding metrics, interface analysis, force field metrics
- Motif binder metrics: Binder refolding + motif RMSD, sequence recovery, clash detection

The script determines which evaluations to run based on metric flags:
- compute_monomer_metrics: Run monomer evaluation (designability, novelty, etc.)
- compute_binder_metrics: Run binder refolding/interface evaluation
- compute_motif_metrics: Run monomer motif evaluation
- compute_motif_binder_metrics: Run motif binder evaluation

Input Modes:
- generated: Model outputs with expected directory structure (job_X_*) [default]
- pdb_dir: Raw PDB files from any directory

Usage:
    # Binder evaluation on model outputs
    python evaluate.py --config-name evaluate \\
        metric.compute_binder_metrics=true
    
    # Motif binder evaluation (ligand target)
    python evaluate.py --config-name evaluate \\
        protein_type=motif_binder metric.compute_motif_binder_metrics=true
    
    # Monomer evaluation for unconditional generation
    python evaluate.py --config-name evaluate \\
        protein_type=monomer metric.compute_monomer_metrics=true
"""

import os
import random
import sys
import time
import warnings
from datetime import datetime

# Apply atomworks patches early - before any imports that use atomworks/biotite
import proteinfoundation.patches.atomworks_patches  # noqa: F401

# Suppress noisy deprecation warnings from ESMFold / transformers internals
warnings.filterwarnings("ignore", message=".*torch.get_autocast_gpu_dtype.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"transformers\.models\.esm")

import hydra
import lightning as L
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig

from proteinfoundation.evaluation.binder_eval import (  # Availability flags for optional dependencies
    PR_ALTERNATIVE_AVAILABLE,
    TMOL_AVAILABLE,
    compute_binder_metrics,
    compute_interface_metrics_df,
    compute_interface_metrics_on_refolded_structures,
)
from proteinfoundation.evaluation.binder_eval_utils import get_target_info
from proteinfoundation.evaluation.monomer_eval import compute_monomer_metrics
from proteinfoundation.evaluation.motif_binder_eval import compute_motif_binder_metrics
from proteinfoundation.evaluation.motif_binder_eval_utils import get_motif_binder_target_info
from proteinfoundation.evaluation.motif_eval import compute_motif_metrics
from proteinfoundation.evaluation.motif_eval_utils import copy_motif_csvs, get_motif_dataset_config

# Import evaluation modules
from proteinfoundation.evaluation.utils import (
    get_pdb_files_from_dir,
    parse_cfg_for_table,
    prepare_sample_paths,
    read_and_update_timing_csv,
    split_by_job_generated,
    split_pdb_files_by_job,
)

# Import shared column filtering from analysis utilities
from proteinfoundation.result_analysis.analysis_utils import SEQUENCE_TYPES, filter_columns_for_csv
from proteinfoundation.utils.refolded_structure_utils import extract_best_refolded_structure_paths_from_df

# =============================================================================
# Configuration Validation
# =============================================================================

VALID_PROTEIN_TYPES = {"monomer", "binder", "monomer_motif", "motif_binder"}
VALID_INPUT_MODES = {"generated", "pdb_dir"}


def get_enabled_evaluations(cfg_metric: DictConfig) -> tuple:
    """
    Determine which evaluations to run based on metric flags.

    Args:
        cfg_metric: The metric configuration section

    Returns:
        Tuple of (run_monomer, run_binder, run_motif, run_motif_binder) booleans
    """
    run_monomer = cfg_metric.get("compute_monomer_metrics", False)
    run_binder = cfg_metric.get("compute_binder_metrics", False)
    run_motif = cfg_metric.get("compute_motif_metrics", False)
    run_motif_binder = cfg_metric.get("compute_motif_binder_metrics", False)

    return run_monomer, run_binder, run_motif, run_motif_binder


def validate_config(
    protein_type: str,
    input_mode: str,
    run_monomer: bool,
    run_binder: bool,
    run_motif: bool = False,
    run_motif_binder: bool = False,
) -> None:
    """
    Validate configuration settings and evaluation compatibility.

    Compatibility matrix:
        monomer + binder        : OK  (monomer metrics on binder chain + binder refolding)
        monomer + motif         : OK  (monomer metrics on scaffold + motif-specific metrics)
        monomer + motif_binder  : OK  (monomer metrics on binder chain + motif binder)
        binder  + motif_binder  : OK  (both use binder infrastructure)
        binder  + motif         : INVALID (different protein types)
        motif   + motif_binder  : INVALID (monomer_motif vs motif_binder)

    Args:
        protein_type: Type of protein structures
        input_mode: How to find input structures
        run_monomer: Whether monomer evaluation is enabled
        run_binder: Whether binder evaluation is enabled
        run_motif: Whether motif (monomer_motif) evaluation is enabled
        run_motif_binder: Whether motif binder evaluation is enabled

    Raises:
        ValueError: If configuration is invalid or evaluations are incompatible
    """
    # Validate protein type
    if protein_type not in VALID_PROTEIN_TYPES:
        raise ValueError(f"Invalid protein_type '{protein_type}'. Valid options: {VALID_PROTEIN_TYPES}")

    # Validate input mode
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(f"Invalid input_mode '{input_mode}'. Valid options: {VALID_INPUT_MODES}")

    # --- Compatibility matrix ---
    if run_binder and run_motif:
        raise ValueError(
            "Incompatible evaluation modes: binder and motif cannot run together. "
            "Binder requires protein_type='binder' while motif requires "
            "protein_type='monomer_motif'. Run them as separate jobs."
        )

    if run_motif and run_motif_binder:
        raise ValueError(
            "Incompatible evaluation modes: monomer_motif and motif_binder cannot "
            "run together. monomer_motif is for scaffolding evaluation while "
            "motif_binder is for binder evaluation with motif constraints."
        )

    # Warn about protein_type mismatches
    if run_binder and protein_type not in ("binder", "motif_binder"):
        logger.warning(
            f"compute_binder_metrics=True but protein_type='{protein_type}'. "
            "Binder metrics require protein_type='binder' or 'motif_binder'."
        )

    if run_motif and protein_type != "monomer_motif":
        logger.warning(
            f"compute_motif_metrics=True but protein_type='{protein_type}'. "
            "Motif metrics are designed for protein_type='monomer_motif'."
        )

    if run_motif_binder and protein_type != "motif_binder":
        logger.warning(
            f"compute_motif_binder_metrics=True but protein_type='{protein_type}'. "
            "Motif binder metrics are designed for protein_type='motif_binder'."
        )

    # Log the validated evaluation plan
    enabled = []
    if run_monomer:
        enabled.append("monomer")
    if run_binder:
        enabled.append("binder")
    if run_motif:
        enabled.append("motif")
    if run_motif_binder:
        enabled.append("motif_binder")
    logger.info(f"Evaluation plan: {' + '.join(enabled) or 'none'} (protein_type={protein_type})")


def print_dryrun_summary(
    cfg: DictConfig,
    run_monomer: bool,
    run_binder: bool,
    protein_type: str,
    input_mode: str,
    sample_paths: list[str],
    output_dir: str,
    run_motif: bool = False,
    run_motif_binder: bool = False,
) -> None:
    """Print summary of what would be executed in dryrun mode."""
    logger.info("=" * 60)
    logger.info("DRYRUN MODE - No actual evaluation will be performed")
    logger.info("=" * 60)

    logger.info("\nConfiguration Summary:")
    logger.info(f"  Run monomer evaluation: {run_monomer}")
    logger.info(f"  Run binder evaluation: {run_binder}")
    logger.info(f"  Run motif evaluation: {run_motif}")
    logger.info(f"  Run motif binder evaluation: {run_motif_binder}")
    logger.info(f"  Protein type: {protein_type}")
    logger.info(f"  Input mode: {input_mode}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Number of samples: {len(sample_paths)}")

    if sample_paths:
        logger.info("\nFirst 5 sample paths:")
        for path in sample_paths[:5]:
            logger.info(f"    {path}")
        if len(sample_paths) > 5:
            logger.info(f"    ... and {len(sample_paths) - 5} more")

    # Dependency availability
    logger.info("\nDependency Availability:")
    logger.info(
        f"  TMOL (force field metrics): {'Available' if TMOL_AVAILABLE else 'NOT AVAILABLE - metrics will be NaN'}"
    )
    logger.info(
        f"  PR Alternative (bioinformatics): {'Available' if PR_ALTERNATIVE_AVAILABLE else 'NOT AVAILABLE - metrics will be NaN'}"
    )

    cfg_metric = cfg.get("metric", {})
    logger.info("\nMetrics to compute:")

    if run_monomer:
        logger.info("  Monomer metrics:")
        logger.info(f"    - Designability: {cfg_metric.get('compute_designability', False)}")
        logger.info(f"    - Codesignability: {cfg_metric.get('compute_codesignability', False)}")
        logger.info(f"    - Sequence recovery: {cfg_metric.get('compute_co_sequence_recovery', False)}")
        logger.info(f"    - Novelty (PDB): {cfg_metric.get('compute_novelty_pdb', False)}")
        logger.info(f"    - Novelty (AFDB): {cfg_metric.get('compute_novelty_afdb', False)}")

    if run_binder:
        logger.info("  Binder metrics:")
        logger.info(f"    - Binder refolding: {cfg_metric.get('compute_binder_metrics', False)}")
        logger.info(f"    - Folding method: {cfg_metric.get('binder_folding_method', 'colabdesign')}")
        logger.info(f"    - Sequence types: {cfg_metric.get('sequence_types', ['self'])}")

        # Pre-refolding metrics with granular flags
        pre_refolding_enabled = cfg_metric.get("compute_pre_refolding_metrics", False)
        logger.info(f"    - Pre-refolding interface metrics: {pre_refolding_enabled}")
        if pre_refolding_enabled:
            pre_cfg = cfg_metric.get("pre_refolding", {})
            bio_enabled = pre_cfg.get("bioinformatics", True)
            tmol_enabled = pre_cfg.get("tmol", True)

            bio_status = "" if PR_ALTERNATIVE_AVAILABLE else " [UNAVAILABLE]"
            tmol_status = "" if TMOL_AVAILABLE else " [UNAVAILABLE]"

            logger.info(f"        bioinformatics (SC/SASA): {bio_enabled}{bio_status}")
            logger.info(f"        tmol (force field): {tmol_enabled}{tmol_status}")

        # Refolded structure metrics with granular flags
        refolded_enabled = cfg_metric.get("compute_refolded_structure_metrics", False)
        logger.info(f"    - Refolded structure metrics: {refolded_enabled}")
        if refolded_enabled:
            refolded_cfg = cfg_metric.get("refolded", {})
            bio_enabled = refolded_cfg.get("bioinformatics", True)
            tmol_enabled = refolded_cfg.get("tmol", True)

            bio_status = "" if PR_ALTERNATIVE_AVAILABLE else " [UNAVAILABLE]"
            tmol_status = "" if TMOL_AVAILABLE else " [UNAVAILABLE]"

            logger.info(f"        bioinformatics (SC/SASA): {bio_enabled}{bio_status}")
            logger.info(f"        tmol (force field): {tmol_enabled}{tmol_status}")

    if run_motif:
        # Mirror the actual default logic from motif_eval.compute_motif_metrics:
        # When compute_motif_metrics=True, all sub-flags default to True.
        motif_on = cfg_metric.get("compute_motif_metrics", False)
        do_des = cfg_metric.get("compute_designability", motif_on)
        do_codes = cfg_metric.get("compute_codesignability", motif_on)
        logger.info("  Motif metrics:")
        logger.info(f"    - Motif RMSD: {cfg_metric.get('compute_motif_rmsd', motif_on)}")
        logger.info(f"    - Designability: {do_des}")
        logger.info(f"    - Codesignability: {do_codes}")
        logger.info(f"    - Motif designability: {cfg_metric.get('compute_motif_designability', motif_on) and do_des}")
        logger.info(
            f"    - Motif codesignability: {cfg_metric.get('compute_motif_codesignability', motif_on) and do_codes}"
        )

    if run_motif_binder:
        logger.info("  Motif binder metrics (raw scores only, success in analyze):")
        logger.info(f"    - Folding method: {cfg_metric.get('binder_folding_method', 'rf3_latest')}")
        logger.info(f"    - Inverse folding model: {cfg_metric.get('inverse_folding_model', 'ligand_mpnn')}")
        logger.info(f"    - Sequence types: {cfg_metric.get('sequence_types', ['mpnn_fixed', 'self'])}")

    logger.info("\n" + "=" * 60)
    logger.info("End of dryrun summary")
    logger.info("=" * 60)


# =============================================================================
# Main Evaluation Logic
# =============================================================================


def run_monomer_evaluation(
    cfg: DictConfig,
    sample_paths: list[str],
    output_dir: str,
    job_id: int,
    protein_type: str,
    input_mode: str,
) -> pd.DataFrame:
    """
    Run monomer evaluation (designability, novelty, etc.)

    Args:
        cfg: Configuration
        sample_paths: List of prepared sample directory paths (already copied to output_dir)
        output_dir: Output directory
        job_id: Job ID
        protein_type: "monomer" or "binder" (affects chain extraction)
        input_mode: "pdb_dir" or "generated"

    Returns:
        DataFrame with monomer metrics
    """
    cfg_metric = cfg.metric
    ncpus = cfg.get("ncpus_", 24)

    # sample_paths are now always directories in output_dir
    # Get PDB file paths from directories
    samples_file_paths = []
    for dir_path in sample_paths:
        fname = os.path.basename(dir_path) + ".pdb"
        samples_file_paths.append(os.path.join(dir_path, fname))

    logger.info(f"Running monomer evaluation on {len(samples_file_paths)} samples (protein_type={protein_type})")

    show_progress = cfg.get("show_progress", False)

    df = compute_monomer_metrics(
        cfg=cfg,
        cfg_metric=cfg_metric,
        samples_paths=samples_file_paths,
        job_id=job_id,
        ncpus=ncpus,
        root_path=output_dir,
        protein_type=protein_type,
        show_progress=show_progress,
    )

    return df


def _add_pre_refolding_metrics(
    cfg: DictConfig,
    df: pd.DataFrame,
    sample_paths: list[str],
) -> pd.DataFrame:
    """Compute interface metrics on generated (pre-refolding) structures and merge into *df*."""
    cfg_metric = cfg.metric
    show_progress = cfg.get("show_progress", False)

    if not cfg_metric.get("compute_pre_refolding_metrics", False):
        return df

    logger.info("Computing pre-refolding metrics on generated structures...")

    gen_structure_paths = []
    for sample_path in sample_paths:
        fname = os.path.basename(sample_path) + ".pdb"
        gen_structure_paths.append(os.path.join(sample_path, fname))

    fixed_col_names = [
        "run_name",
        "ckpt_path",
        "ckpt_name",
        "ncpus_",
        "seed",
        "gen_njobs",
        "eval_njobs",
        "id_gen",
        "pdb_path",
        "L",
    ]

    pre_refolding_cfg = cfg_metric.get("pre_refolding", {})
    compute_bioinformatics = pre_refolding_cfg.get("bioinformatics", True)
    compute_tmol = pre_refolding_cfg.get("tmol", True)

    enabled = []
    if compute_bioinformatics:
        enabled.append("bioinformatics")
    if compute_tmol:
        enabled.append("TMOL")
    logger.info(f"Pre-refolding metrics enabled: {enabled}")

    if not any([compute_bioinformatics, compute_tmol]):
        return df

    try:
        gen_metrics_df = compute_interface_metrics_df(
            cfg=cfg,
            pdb_paths=gen_structure_paths,
            compute_bioinformatics=compute_bioinformatics,
            compute_tmol=compute_tmol,
            show_progress=show_progress,
        )
        gen_columns = {
            col: f"generated_{col}"
            for col in gen_metrics_df.columns
            if (col not in fixed_col_names) and ("generation_" not in col)
        }
        gen_metrics_df = gen_metrics_df.rename(columns=gen_columns)

        try:
            _, flat_dict = parse_cfg_for_table(cfg)
            merge_cols = [c for c in flat_dict.keys() if c in df.columns and c in gen_metrics_df.columns]
        except Exception:
            merge_cols = []
        merge_cols = merge_cols + ["id_gen", "pdb_path"]
        merge_cols = [c for c in merge_cols if c in df.columns and c in gen_metrics_df.columns]

        df = pd.merge(df, gen_metrics_df, on=merge_cols, how="left")
        logger.info("Pre-refolding interface metrics computed successfully")
    except Exception as e:
        logger.error(f"Failed to compute pre-refolding interface metrics: {e}")

    return df


def _add_refolded_structure_metrics(
    cfg: DictConfig,
    df: pd.DataFrame,
    job_id: int,
) -> pd.DataFrame:
    """Compute interface metrics on best refolded structures and merge into *df*."""
    cfg_metric = cfg.metric
    show_progress = cfg.get("show_progress", False)

    if not cfg_metric.get("compute_refolded_structure_metrics", False):
        return df

    logger.info("Computing metrics on successful refolded structures...")

    best_paths_dict = extract_best_refolded_structure_paths_from_df(
        df,
        sequence_types=cfg_metric.get("sequence_types", SEQUENCE_TYPES),
    )

    refolded_cfg = cfg_metric.get("refolded", {})
    compute_bioinformatics = refolded_cfg.get("bioinformatics", True)
    compute_tmol = refolded_cfg.get("tmol", True)

    df = compute_interface_metrics_on_refolded_structures(
        df=df,
        best_paths_dict=best_paths_dict,
        cfg_metric=cfg_metric,
        cfg=cfg,
        compute_bioinformatics=compute_bioinformatics,
        compute_tmol=compute_tmol,
        show_progress=show_progress,
    )

    bad_columns = [c for c in df.columns if "dataset_target_dict" in c]
    if bad_columns:
        logger.warning(f"Dropping columns: {bad_columns}")
        df.drop(columns=bad_columns, inplace=True)

    return df


def run_binder_evaluation(
    cfg: DictConfig,
    sample_paths: list[str],
    output_dir: str,
    job_id: int,
    input_mode: str,
) -> pd.DataFrame:
    """
    Run binder evaluation (refolding, interface metrics, etc.)

    Args:
        cfg: Configuration
        sample_paths: List of prepared sample directory paths (already copied to output_dir)
        output_dir: Output directory
        job_id: Job ID
        input_mode: "pdb_dir" or "generated"

    Returns:
        DataFrame with binder metrics
    """
    # Get target information
    target_task_name, target_pdb_path, target_pdb_chain, is_target_ligand = get_target_info(cfg)
    logger.info(f"Running binder evaluation for target: {target_task_name}")

    df = compute_binder_metrics(
        eval_config=cfg,
        sample_root_paths=sample_paths,
        target_pdb_path=target_pdb_path,
        target_pdb_chain=target_pdb_chain,
        is_target_ligand=is_target_ligand,
    )

    df = _add_pre_refolding_metrics(cfg, df, sample_paths)
    df = _add_refolded_structure_metrics(cfg, df, job_id)

    # Note: ESM metrics are computed inside compute_binder_metrics when
    # cfg.metric.compute_esm_metrics=True. They are computed per sequence type
    # (self, mpnn, mpnn_fixed) using the sequences from inverse folding.

    return df


def run_motif_evaluation(
    cfg: DictConfig,
    sample_paths: list[str],
    output_dir: str,
    job_id: int,
    input_mode: str,
) -> pd.DataFrame:
    """
    Run motif evaluation (motif RMSD, motif-aware designability/codesignability).

    This mode is designed for monomer_motif protein type, where each sample
    is a motif scaffolding result. It combines task-driven evaluation (like binders)
    with scRMSD-based metrics (like monomers).

    Args:
        cfg: Configuration
        sample_paths: List of prepared sample directory paths
        output_dir: Output directory
        job_id: Job ID
        input_mode: "pdb_dir" or "generated"

    Returns:
        DataFrame with motif metrics
    """
    cfg_metric = cfg.metric
    ncpus = cfg.get("ncpus_", 24)

    # Get PDB file paths from directories
    samples_file_paths = []
    for dir_path in sample_paths:
        fname = os.path.basename(dir_path) + ".pdb"
        samples_file_paths.append(os.path.join(dir_path, fname))

    logger.info(f"Running motif evaluation on {len(samples_file_paths)} samples")

    show_progress = cfg.get("show_progress", False)

    df = compute_motif_metrics(
        cfg=cfg,
        cfg_metric=cfg_metric,
        samples_paths=samples_file_paths,
        job_id=job_id,
        ncpus=ncpus,
        root_path=output_dir,
        show_progress=show_progress,
    )

    return df


def run_motif_binder_evaluation(
    cfg: DictConfig,
    sample_paths: list[str],
    output_dir: str,
    job_id: int,
    input_mode: str,
) -> pd.DataFrame:
    """
    Run motif binder evaluation (binder refolding + motif metrics).

    ``compute_motif_binder_metrics`` already runs the full binder refolding
    pipeline internally (Phase 1) and layers motif overlay on top (Phase 2).
    After that, pre-refolding and post-refolding interface metrics
    (bioinformatics, TMOL) are computed via the same shared helpers
    used by ``run_binder_evaluation``, so there is no need to also enable
    ``compute_binder_metrics``.

    Args:
        cfg: Configuration
        sample_paths: List of prepared sample directory paths
        output_dir: Output directory
        job_id: Job ID
        input_mode: "pdb_dir" or "generated"

    Returns:
        DataFrame with binder + motif + interface metrics
    """
    target_task_name, target_pdb_path, target_pdb_chain, is_target_ligand = get_motif_binder_target_info(cfg)
    logger.info(f"Running motif binder evaluation for target: {target_task_name}")

    df = compute_motif_binder_metrics(
        eval_config=cfg,
        sample_root_paths=sample_paths,
        target_pdb_path=target_pdb_path,
        target_pdb_chain=target_pdb_chain,
        is_target_ligand=is_target_ligand,
    )

    df = _add_pre_refolding_metrics(cfg, df, sample_paths)
    df = _add_refolded_structure_metrics(cfg, df, job_id)

    return df


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluate",
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for unified protein structure evaluation.

    This function orchestrates the evaluation pipeline for both monomer and binder
    protein structures. Which evaluations run is determined by the metric.compute_*
    flags in the configuration.

    Supported Metrics:
        - Monomer: Designability, codesignability, novelty, sequence recovery
          (enabled by compute_designability, compute_codesignability, etc.)
        - Binder: Refolding metrics, interface analysis, force field metrics
          (enabled by compute_binder_metrics)

    Supported Protein Types:
        - monomer: Single chain proteins
        - binder: Binder + target complexes (binder is last chain by convention)

    Supported Input Modes:
        - generated: Model outputs with expected directory structure (job_X_*)
        - pdb_dir: Raw PDB files from a specified directory

    The function handles:
        1. Configuration parsing and validation
        2. Sample discovery and job distribution
        3. Running requested evaluations based on metric flags
        4. Saving results to CSV files
        5. Timing information tracking

    Args:
        cfg: Hydra configuration dictionary containing all evaluation settings

    Environment Variables:
        Required environment variables depend on which metrics are enabled.
        Use `validate_env_vars()` from evaluation.utils to check requirements.

    Configuration Keys:
        - protein_type: Type of protein structures ("monomer" or "binder")
        - input_mode: How to find input structures ("generated" or "pdb_dir")
        - sample_storage_path: Path to input samples
        - output_dir: Path for evaluation results
        - metric: Dictionary of metric-specific configuration with compute_* flags
        - dryrun: If True, print summary without executing
    """
    load_dotenv()
    torch.set_float32_matmul_precision("high")

    # Record start time
    start_time = time.time()
    logger.info(f"Starting evaluation job at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    # Get config name and job settings
    config_name = cfg.get("base_config_name", HydraConfig.get().job.config_name)
    job_id = cfg.get("job_id", 0)
    njobs = cfg.get("eval_njobs", 1)

    # Get evaluation settings
    protein_type = cfg.get("protein_type", "binder")
    input_mode = cfg.get("input_mode", "generated")

    # Get dryrun flag
    dryrun = cfg.get("dryrun", False)

    # Determine which evaluations to run based on metric flags
    cfg_metric = cfg.get("metric", {})
    run_monomer, run_binder, run_motif, run_motif_binder = get_enabled_evaluations(cfg_metric)

    # Validate configuration and compatibility
    validate_config(protein_type, input_mode, run_monomer, run_binder, run_motif, run_motif_binder)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  PROTEIN FOUNDATION MODELS — EVALUATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Config:        {config_name}")
    logger.info(f"  Job:           {job_id}/{njobs}")
    logger.info(f"  Protein type:  {protein_type}")
    logger.info(f"  Input mode:    {input_mode}")
    logger.info(f"  Monomer eval:  {run_monomer}")
    logger.info(f"  Binder eval:   {run_binder}")
    logger.info(f"  Motif eval:    {run_motif}")
    logger.info(f"  Motif binder:  {run_motif_binder}")
    logger.info("=" * 70)
    logger.info("")

    # Set seed
    seed = cfg.get("seed", 42) + job_id
    logger.info(f"Seeding everything to seed {seed}")
    L.seed_everything(seed)

    # Handle baseline_model override if provided
    baseline_model = cfg.get("baseline_model", None)
    if baseline_model is not None:
        logger.info(f"Overriding model config with baseline_model: {baseline_model}")
        cfg.run_name = baseline_model
        cfg.ckpt_path = baseline_model
        cfg.ckpt_name = f"{baseline_model}_ckpt"

    # Get sample storage path and output directory
    sample_storage_path = cfg.get("sample_storage_path", None)
    output_dir = cfg.get("output_dir", None)
    run_name = cfg.get("run_name", None)

    # Get target info for path construction if needed (for binder / motif_binder evaluation).
    # motif_binder is the primary task — its config defines the target.
    # Standard binder eval can run on motif_binder outputs using the same target.
    target_task_name = None
    if run_motif_binder:
        try:
            target_task_name, _, _, _ = get_motif_binder_target_info(cfg)
        except Exception as e:
            logger.warning(f"Could not get target info for motif_binder: {e}")
    elif run_binder:
        try:
            target_task_name, _, _, _ = get_target_info(cfg)
        except Exception as e:
            logger.warning(f"Could not get target info: {e}")

    # Construct paths if not explicitly provided
    if sample_storage_path is None or output_dir is None:
        logger.warning("sample_storage_path or output_dir not set, constructing from config")
        if target_task_name:
            sample_storage_path = sample_storage_path or f"./inference/{config_name}_{target_task_name}"
            output_dir = output_dir or f"./evaluation_results/{config_name}_{target_task_name}"
        else:
            sample_storage_path = sample_storage_path or f"./inference/{config_name}"
            output_dir = output_dir or f"./evaluation_results/{config_name}"
        if run_name:
            sample_storage_path = f"{sample_storage_path}_{run_name}"
            # output_dir = f"{output_dir}_{run_name}"
    # else:
    #     if run_name and not output_dir.endswith(run_name):
    #         output_dir = f"{output_dir}_{run_name}"
    if run_name and not output_dir.endswith(run_name):
        output_dir = f"{output_dir}_{run_name}"
        logger.info(f"Appended run_name to output_dir: {output_dir}")
    logger.info(f"Sample storage path: {sample_storage_path}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get sample paths based on input mode
    if input_mode == "pdb_dir":
        all_files = get_pdb_files_from_dir(
            sample_storage_path,
            ignore_postfix=cfg.get("ignore_generated_pdb_suffix", "_binder.pdb"),
        )
        if not all_files:
            logger.error(f"No PDB files found in {sample_storage_path}")
            sys.exit(1)
        sample_paths = split_pdb_files_by_job(all_files, job_id, njobs)
    else:  # generated mode
        sample_paths = split_by_job_generated(sample_storage_path, job_id, return_root=True)

    if not sample_paths:
        logger.warning(f"No files assigned to job {job_id}/{njobs}")
        sys.exit(0)

    # Apply file_limit if set (useful for quick debugging / test runs)
    file_limit = cfg.get("file_limit", None)
    if file_limit is not None:
        sample_paths = sorted(sample_paths)[: int(file_limit)]
        logger.info(f"file_limit={file_limit}: evaluating first {len(sample_paths)} samples")

    # Shuffle for load balancing
    random.shuffle(sample_paths)

    # Handle dryrun mode
    if dryrun:
        print_dryrun_summary(
            cfg=cfg,
            run_monomer=run_monomer,
            run_binder=run_binder,
            protein_type=protein_type,
            input_mode=input_mode,
            sample_paths=sample_paths,
            output_dir=output_dir,
            run_motif=run_motif,
            run_motif_binder=run_motif_binder,
        )
        logger.info("Exiting due to --dryrun flag")
        sys.exit(0)

    # ==========================================================================
    # Copy samples from inference to evaluation output directory
    # This ensures all evaluation artifacts are self-contained in output_dir
    # ==========================================================================
    logger.info(f"Copying {len(sample_paths)} samples from inference to evaluation directory...")
    prepared_paths = prepare_sample_paths(sample_paths, output_dir, input_mode)
    logger.info(f"Samples prepared in: {output_dir}")

    # Copy motif_info CSVs alongside the samples (only for motif evaluation)
    if run_motif:
        _, motif_task_name, _ = get_motif_dataset_config(cfg)
        copy_motif_csvs(sample_paths, output_dir, input_mode, task_name=motif_task_name)

    # Initialize results storage
    all_results = {}

    # ==========================================================================
    # Run enabled evaluations
    # ==========================================================================

    # Monomer evaluation
    if run_monomer:
        logger.info("")
        logger.info("+" + "-" * 68 + "+")
        logger.info("|{:^68s}|".format("MONOMER EVALUATION"))
        logger.info("+" + "-" * 68 + "+")

        monomer_df = run_monomer_evaluation(
            cfg=cfg,
            sample_paths=prepared_paths,
            output_dir=output_dir,
            job_id=job_id,
            protein_type=protein_type,
            input_mode=input_mode,
        )

        csv_filename = f"monomer_results_{config_name}_{job_id}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        # Filter out config/metadata columns before saving
        monomer_df_filtered = filter_columns_for_csv(monomer_df)
        monomer_df_filtered.to_csv(csv_path, index=False)
        logger.info(f"Monomer results saved to {csv_path}")
        all_results["monomer"] = monomer_df_filtered

    # Binder evaluation
    if run_binder:
        logger.info("")
        logger.info("+" + "-" * 68 + "+")
        logger.info("|{:^68s}|".format("BINDER EVALUATION"))
        logger.info("+" + "-" * 68 + "+")

        if protein_type != "binder":
            logger.warning(
                f"compute_binder_metrics=True but protein_type='{protein_type}'. Results may not be meaningful."
            )

        binder_df = run_binder_evaluation(
            cfg=cfg,
            sample_paths=prepared_paths,
            output_dir=output_dir,
            job_id=job_id,
            input_mode=input_mode,
        )

        csv_filename = f"binder_results_{config_name}_{job_id}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        # Filter out config/metadata columns before saving
        binder_df_filtered = filter_columns_for_csv(binder_df)
        binder_df_filtered.to_csv(csv_path, index=False)
        logger.info(f"Binder results saved to {csv_path}")
        all_results["binder"] = binder_df_filtered

    # Motif evaluation
    if run_motif:
        logger.info("")
        logger.info("+" + "-" * 68 + "+")
        logger.info("|{:^68s}|".format("MOTIF EVALUATION"))
        logger.info("+" + "-" * 68 + "+")

        motif_df = run_motif_evaluation(
            cfg=cfg,
            sample_paths=prepared_paths,
            output_dir=output_dir,
            job_id=job_id,
            input_mode=input_mode,
        )

        csv_filename = f"motif_results_{config_name}_{job_id}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        # Filter out config/metadata columns before saving
        motif_df_filtered = filter_columns_for_csv(motif_df)
        motif_df_filtered.to_csv(csv_path, index=False)
        # Also save a transposed version for easier debugging (TODO: remove later)
        # csv_path_T = csv_path.replace(".csv", "_transposed.csv")
        # motif_df_filtered.T.to_csv(csv_path_T)
        logger.info(f"Motif results saved to {csv_path}")
        all_results["motif"] = motif_df_filtered

    # Motif binder evaluation
    if run_motif_binder:
        logger.info("")
        logger.info("+" + "-" * 68 + "+")
        logger.info("|{:^68s}|".format("MOTIF BINDER EVALUATION"))
        logger.info("+" + "-" * 68 + "+")

        motif_binder_df = run_motif_binder_evaluation(
            cfg=cfg,
            sample_paths=prepared_paths,
            output_dir=output_dir,
            job_id=job_id,
            input_mode=input_mode,
        )

        csv_filename = f"motif_binder_results_{config_name}_{job_id}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        motif_binder_df_filtered = filter_columns_for_csv(motif_binder_df)
        motif_binder_df_filtered.to_csv(csv_path, index=False)
        logger.info(f"Motif binder results saved to {csv_path}")
        all_results["motif_binder"] = motif_binder_df_filtered

    # ==========================================================================
    # Timing & completion summary
    # ==========================================================================
    end_time = time.time()
    evaluation_time = end_time - start_time

    # Determine number of samples processed
    nsamples = 0
    for df in all_results.values():
        nsamples = max(nsamples, len(df))

    # Update timing CSV in sample_storage_path (legacy location)
    timing_csv_path = os.path.join(sample_storage_path, f"timing_{job_id}.csv")
    if os.path.exists(timing_csv_path):
        read_and_update_timing_csv(timing_csv_path, job_id, evaluation_time, nsamples)
    else:
        timing_dir = os.path.dirname(timing_csv_path)
        if timing_dir:
            os.makedirs(timing_dir, exist_ok=True)
        with open(timing_csv_path, "w") as f:
            f.write("job_id,generation_time,evaluation_time,total_time,nsamples\n")
            f.write(f"{job_id},0,{evaluation_time:.2f},{evaluation_time:.2f},{nsamples}\n")

    evals_run = []
    if run_monomer:
        evals_run.append("monomer")
    if run_binder:
        evals_run.append("binder")
    if run_motif:
        evals_run.append("motif")
    if run_motif_binder:
        evals_run.append("motif_binder")

    # Save timing CSV in output_dir alongside results
    output_timing_path = os.path.join(output_dir, f"timing_{job_id}.csv")
    with open(output_timing_path, "w") as f:
        f.write("job_id,evaluation_time_s,nsamples,evals_run\n")
        f.write(f"{job_id},{evaluation_time:.2f},{nsamples},{'+'.join(evals_run)}\n")

    # Format elapsed time nicely
    mins, secs = divmod(int(evaluation_time), 60)
    hrs, mins = divmod(mins, 60)
    elapsed_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"

    logger.info("")
    logger.info("=" * 70)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Evaluations run:  {', '.join(evals_run)}")
    logger.info(f"  Samples:          {nsamples}")
    logger.info(f"  Elapsed:          {elapsed_str} ({evaluation_time:.1f}s)")
    logger.info(f"  Output dir:       {output_dir}")
    for name, df in all_results.items():
        logger.info(f"  {name:16s}  {len(df)} rows  ->  {name}_results_{config_name}_{job_id}.csv")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
