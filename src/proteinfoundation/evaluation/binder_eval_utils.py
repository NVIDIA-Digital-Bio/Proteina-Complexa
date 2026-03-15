"""
Binder evaluation utilities and default criteria.

This module contains:
- Default ranking criteria for selecting best refolded samples
- Metric column definitions for interface analysis
- Utility functions for ranking and sample selection
- Target / MSA / template path resolution
- Chain extraction helpers
"""

import os
from typing import Any

import numpy as np
from atomworks.io.utils.io_utils import load_any
from biotite.structure import filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from loguru import logger
from omegaconf import DictConfig

# =============================================================================
# Ranking Criteria for Best Sample Selection
# =============================================================================

# Default ranking criteria for selecting best refolded sample
# Each metric has: scale (weight), direction ("minimize" or "maximize")
# Composite score = sum(metric_value * scale * direction_sign)
# Lower composite score is better (direction_sign: minimize=1, maximize=-1)

DEFAULT_PROTEIN_RANKING_CRITERIA = {
    "i_pAE": {
        "scale": 1.0,
        "direction": "minimize",  # Lower ipAE is better
    },
}

DEFAULT_LIGAND_RANKING_CRITERIA = {
    "min_ipAE": {
        "scale": 1.0,
        "direction": "minimize",  # Lower ipAE is better
    },
}

VALID_RANKING_DIRECTIONS = {"minimize", "maximize"}

# Default evaluation parameters
DEFAULT_INTERFACE_CUTOFF_PROTEIN = 8.0
DEFAULT_INTERFACE_CUTOFF_LIGAND = 6.0
DEFAULT_NUM_REDESIGN_SEQS_PROTEIN = 8
DEFAULT_NUM_REDESIGN_SEQS_LIGAND = 1


# =============================================================================
# Interface Metrics - Column Definitions
# =============================================================================

BIOINFORMATICS_METRIC_COLS = [
    "binder_surface_hydrophobicity",
    "binder_interface_sc",
    "binder_interface_dSASA",
    "binder_interface_fraction",
    "binder_interface_hydrophobicity",
    "binder_interface_nres",
]

TMOL_METRIC_COLS = [
    "n_interface_hbonds_tmol",
    "total_interface_hbond_energy_tmol",
    "total_interface_elec_energy_tmol",
    "n_interface_elec_interactions_tmol",
]

# =============================================================================
# Ranking Criteria Functions
# =============================================================================


def validate_ranking_criteria(
    ranking_criteria: dict[str, dict[str, Any]] | DictConfig | None,
) -> dict[str, dict[str, Any]]:
    """
    Validate and normalize ranking criteria configuration.

    Args:
        ranking_criteria: Dictionary or DictConfig defining ranking criteria

    Returns:
        Validated ranking criteria dictionary

    Raises:
        ValueError: If criteria format is invalid
    """
    if ranking_criteria is None:
        return {}

    # Convert DictConfig to dict if needed
    if hasattr(ranking_criteria, "items"):
        ranking_criteria = dict(ranking_criteria)

    validated = {}
    for metric_name, criteria in ranking_criteria.items():
        if not isinstance(criteria, (dict, DictConfig)):
            raise ValueError(
                f"Ranking criteria for '{metric_name}' must be a dictionary, got {type(criteria).__name__}"
            )

        # Convert DictConfig
        criteria = dict(criteria) if hasattr(criteria, "items") else criteria

        # Validate direction
        direction = criteria.get("direction", "minimize")
        if direction not in VALID_RANKING_DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{direction}' for metric '{metric_name}'. Valid options: {VALID_RANKING_DIRECTIONS}"
            )

        # Validate scale
        scale = criteria.get("scale", 1.0)
        if not isinstance(scale, (int, float)):
            raise ValueError(f"Scale for metric '{metric_name}' must be numeric, got {type(scale).__name__}")

        validated[metric_name] = {
            "scale": float(scale),
            "direction": direction,
        }

    return validated


def compute_composite_ranking_score(
    stats: dict[str, Any],
    ranking_criteria: dict[str, dict[str, Any]],
) -> float:
    """
    Compute a composite ranking score from multiple metrics.

    Args:
        stats: Dictionary of metric values for a single sample
        ranking_criteria: Dictionary defining metrics to use for ranking.
            Each entry has:
            - "scale": Weight for this metric (float)
            - "direction": "minimize" or "maximize"

    Returns:
        Composite score (lower is better)
    """
    score = 0.0
    for metric_name, criteria in ranking_criteria.items():
        if metric_name not in stats:
            logger.warning(f"Ranking metric '{metric_name}' not found in stats, skipping")
            continue

        value = stats[metric_name]
        if value is None or value == float("inf") or value == float("-inf"):
            return float("inf")  # Invalid sample

        scale = criteria.get("scale", 1.0)
        direction = criteria.get("direction", "minimize")

        # Convert to "lower is better" space
        if direction == "maximize":
            score -= value * scale  # Negate so higher values give lower scores
        else:
            score += value * scale

    return score


def select_best_sample_idx(
    stats_list: list[dict[str, Any]],
    ranking_criteria: dict[str, dict[str, Any]],
) -> tuple[int, float]:
    """
    Select the best sample from a list based on composite ranking score.

    Args:
        stats_list: List of stat dictionaries, one per sample
        ranking_criteria: Ranking criteria dictionary

    Returns:
        Tuple of (best_index, best_score)
    """
    if not stats_list:
        return -1, float("inf")

    scores = [compute_composite_ranking_score(stats, ranking_criteria) for stats in stats_list]

    best_idx = int(np.argmin(scores))
    return best_idx, scores[best_idx]


def get_metric_columns(
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
) -> list[str]:
    """
    Get list of metric columns based on which metrics are enabled.

    Args:
        compute_bioinformatics: Whether bioinformatics metrics are enabled
        compute_tmol: Whether TMOL metrics are enabled

    Returns:
        List of metric column names
    """
    cols = []
    if compute_bioinformatics:
        cols.extend(BIOINFORMATICS_METRIC_COLS)
    if compute_tmol:
        cols.extend(TMOL_METRIC_COLS)
    return cols


# =============================================================================
# Target Info Extraction
# =============================================================================


def get_target_info(cfg: DictConfig) -> tuple[str, str, list[str], bool]:
    """Extract binder target information from config.

    Reads ``target_dict_cfg`` to resolve the target PDB path, chain IDs,
    and whether the target is a ligand.

    Args:
        cfg: Hydra configuration containing ``dataset`` or ``generation``
            sections with target information.

    Returns:
        Tuple of (target_task_name, target_pdb_path, target_pdb_chain,
        is_target_ligand).

    Raises:
        ValueError: If ``target_task_name`` is missing or not found in
            ``target_dict_cfg``.
    """
    if "dataset" not in cfg:
        cfg_dataset = cfg.generation.dataloader.dataset
        target_dict_cfg = cfg.generation.target_dict_cfg
    else:
        cfg_dataset = cfg.dataset
        target_dict_cfg = cfg.dataset.target_dict_cfg

    target_task_name = cfg_dataset.get("task_name", None)

    if target_task_name is None:
        raise ValueError("target_task_name must be specified in config")
    if target_task_name not in target_dict_cfg:
        raise ValueError(
            f"target_task_name {target_task_name} not found in target_dict_cfg {list(target_dict_cfg.keys())}"
        )
    target_cfg = target_dict_cfg[target_task_name]

    # Get target PDB path
    if target_cfg.get("target_path"):
        target_pdb_path = target_cfg["target_path"]
        logger.info(f"Using target_path from config: {target_pdb_path}")
    else:
        target_pdb_path = os.path.join(
            os.environ["DATA_PATH"],
            f"target_data/{target_cfg['source']}/{target_cfg['target_filename']}.pdb",
        )

    # Determine target type: presence of "ligand" key in target config implies
    # a ligand target.  Fall back to checking the source name for backwards compat.
    if "ligand" in target_cfg or "ligand" in target_cfg.get("source", "") or target_cfg.get("is_ligand", False):
        is_target_ligand = True
        target_pdb_chain = ["A"]
    else:
        is_target_ligand = False
        target_pdb_chain = sorted(set(x[0] for x in target_cfg["target_input"].split(",")))

    return target_task_name, target_pdb_path, target_pdb_chain, is_target_ligand


# =============================================================================
# Chain Extraction Utilities
# =============================================================================


def extract_binder_chain_to_pdb(
    complex_pdb_path: str,
    output_pdb_path: str,
    binder_chain: str | None = None,
) -> str:
    """Extract the binder chain from a complex PDB file and save to a new file.

    Args:
        complex_pdb_path: Path to the complex PDB file.
        output_pdb_path: Path to save the extracted binder chain.
        binder_chain: Chain ID of the binder. If None, uses the last chain.

    Returns:
        Path to the output PDB file.
    """
    structure = load_any(complex_pdb_path)[0]
    unique_chains = sorted(set(structure.chain_id.tolist()))

    if binder_chain is None:
        binder_chain = unique_chains[-1]

    binder_mask = structure.chain_id == binder_chain
    binder_structure = structure[binder_mask]
    binder_structure = binder_structure[filter_amino_acids(binder_structure)]

    pdb_file = PDBFile()
    pdb_file.set_structure(binder_structure)
    pdb_file.write(output_pdb_path)

    logger.debug(f"Extracted binder chain {binder_chain} to {output_pdb_path}")
    return output_pdb_path


def get_binder_chain_from_complex(
    complex_pdb_path: str,
    return_multi_target: bool = False,
) -> tuple[str, list[str]] | tuple[str, list[str], bool]:
    """Determine the binder and target chains from a complex PDB file.

    By convention, the binder is the last chain alphabetically.

    Args:
        complex_pdb_path: Path to the complex PDB file.
        return_multi_target: If True, also return whether target has multiple chains.

    Returns:
        If return_multi_target=False (default):
            Tuple of (binder_chain_id, list_of_target_chain_ids)
        If return_multi_target=True:
            Tuple of (binder_chain_id, list_of_target_chain_ids, is_multi_target)
    """
    structure = load_any(complex_pdb_path)[0]
    unique_chains = sorted(set(structure.chain_id.tolist()))

    binder_chain = unique_chains[-1]
    target_chains = unique_chains[:-1]
    is_multi_target = len(target_chains) > 1

    if return_multi_target:
        return binder_chain, target_chains, is_multi_target
    return binder_chain, target_chains


# =============================================================================
# Target MSA / Template Path Resolution
# =============================================================================


def get_target_msa_path(
    target_task_name: str,
    target_pdb_chain: list[str],
    is_target_ligand: bool,
    required: bool = False,
) -> list[str]:
    """Resolve MSA directory paths for each target chain.

    Args:
        target_task_name: Task name used to construct the MSA directory name.
        target_pdb_chain: Chain IDs of the target structure.
        is_target_ligand: If True, ligand chains are skipped (no MSA).
        required: If True, raise if the MSA directory does not exist.

    Returns:
        List of MSA directory paths (one per chain), with ``None`` entries
        for missing or skipped chains.

    Raises:
        ValueError: If *required* is True and a chain's MSA is missing.
    """
    target_msa_paths = []
    for chain_id in target_pdb_chain:
        if is_target_ligand:
            continue
        target_msa_name = f"{target_task_name}_{chain_id}"
        target_msa_path = os.path.join(os.environ["DATA_PATH"], f"target_data/target_msa/{target_msa_name}")

        if not os.path.exists(target_msa_path):
            if required:
                raise ValueError(f"Target MSA is necessary, but MSA for {target_msa_name} does not exist")
            else:
                target_msa_path = None
        target_msa_paths.append(target_msa_path)
    return target_msa_paths


def get_target_template_path(target_task_name: str, is_target_ligand: bool) -> str | None:
    """Resolve the target template directory path.

    Args:
        target_task_name: Task name used to construct the template directory name.
        is_target_ligand: If True, returns None (no template for ligand targets).

    Returns:
        Path to the template directory, or None if not found or target is a ligand.
    """
    target_template_path = os.path.join(os.environ["DATA_PATH"], f"target_data/target_template/{target_task_name}")

    if not os.path.exists(target_template_path) or is_target_ligand:
        target_template_path = None
    return target_template_path
