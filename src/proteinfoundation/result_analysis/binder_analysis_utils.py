"""
Binder analysis utilities and default criteria.

This module contains:
- Default success thresholds for protein and ligand binders
- Metric column name mappings and normalization
- Threshold check helpers for binder-specific success criteria
"""

from typing import Any

import numpy as np

from proteinfoundation.result_analysis.analysis_utils import evaluate_threshold

# =============================================================================
# Metric Name Mapping
# =============================================================================

# Mapping from lowercase/alternative metric names to canonical column name suffixes
# This allows users to specify "plddt" instead of "pLDDT" etc.
METRIC_CASE_MAPPING = {
    # pLDDT variations
    "plddt": "pLDDT",
    "complex_plddt": "complex_pLDDT",
    # ipAE variations
    "ipae": "i_pAE",
    "i_pae": "i_pAE",
    "complex_ipae": "complex_i_pAE",
    "complex_i_pae": "complex_i_pAE",
    # iPTM variations
    "iptm": "i_pTM",
    "i_ptm": "i_pTM",
    "complex_iptm": "complex_i_pTM",
    "complex_i_ptm": "complex_i_pTM",
    # min_ipAE variations
    "min_ipae": "min_ipAE",
    "min_i_pae": "min_ipAE",
    "complex_min_ipae": "complex_min_ipAE",
    "complex_min_i_pae": "complex_min_ipAE",
    # avg_ipSAE variations
    "avg_ipsae": "avg_ipSAE",
    "avg_i_psae": "avg_ipSAE",
    "complex_avg_ipsae": "complex_avg_ipSAE",
    # scRMSD variations
    "scrmsd": "scRMSD",
    "binder_scrmsd": "binder_scRMSD",
    "binder_scrmsd_ca": "binder_scRMSD_ca",
    "binder_scrmsd_allatom": "binder_scRMSD_allatom",
    "ligand_scrmsd": "ligand_scRMSD",
    "ligand_scrmsd_aligned_allatom": "ligand_scRMSD_aligned_allatom",
    "ligand_scrmsd_aligned_ca": "ligand_scRMSD_aligned_ca",
    "complex_scrmsd": "complex_scRMSD",
    # pTM variations
    "ptm": "pTM",
    "binder_ptm": "binder_pTM",
}


# =============================================================================
# Default Success Thresholds
# =============================================================================

# Default threshold specification structure:
# {
#     "metric_suffix": {
#         "threshold": float,           # The threshold value
#         "op": str,                    # Comparison operator: "<=", "<", ">=", ">", "=="
#         "scale": float,               # Scale factor applied to value before comparison (default 1.0)
#         "column_prefix": str,         # "complex", "binder", "ligand" - what comes before metric name
#     }
# }

# Default protein binder success thresholds (AlphaProteo criteria)
DEFAULT_PROTEIN_BINDER_THRESHOLDS = {
    "i_pAE": {
        "threshold": 7.0,
        "op": "<=",
        "scale": 31.0,  # ipae * 31 <= 7
        "column_prefix": "complex",
    },
    "pLDDT": {
        "threshold": 0.9,
        "op": ">=",
        "scale": 1.0,
        "column_prefix": "complex",
    },
    "scRMSD_ca": {
        "threshold": 1.5,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "binder",
    },
}

# Default ligand binder success thresholds
DEFAULT_LIGAND_BINDER_THRESHOLDS = {
    "min_ipAE": {
        "threshold": 2.0,
        "op": "<",
        "scale": 31.0,  # min_ipae * 31 < 2
        "column_prefix": "complex",
    },
    "scRMSD_ca": {
        "threshold": 2.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "binder",
    },
    "scRMSD_aligned_allatom": {
        "threshold": 5.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "ligand",
    },
}


# =============================================================================
# Metric Name Utilities
# =============================================================================


def normalize_metric_name(metric_name: str) -> str:
    """Normalize a metric name to its canonical form using METRIC_CASE_MAPPING.

    Args:
        metric_name: The metric name (potentially lowercase or alternative form)

    Returns:
        The canonical metric name
    """
    # Check if it's in the mapping (case-insensitive lookup)
    lower_name = metric_name.lower()
    if lower_name in METRIC_CASE_MAPPING:
        return METRIC_CASE_MAPPING[lower_name]
    # Also check the original name in case it's already correct
    if metric_name in METRIC_CASE_MAPPING:
        return METRIC_CASE_MAPPING[metric_name]
    # Return as-is if not in mapping
    return metric_name


def normalize_threshold_dict(thresholds: dict) -> dict:
    """Normalize all metric names in a threshold dictionary.

    Args:
        thresholds: Dictionary with metric names as keys

    Returns:
        Dictionary with normalized metric names
    """
    normalized = {}
    for metric_name, spec in thresholds.items():
        normalized_name = normalize_metric_name(metric_name)
        normalized[normalized_name] = spec
    return normalized


def build_column_name(seq_type: str, column_prefix: str, metric_suffix: str) -> str:
    """Build the full column name for a metric.

    Args:
        seq_type: Sequence type ("self", "mpnn", "mpnn_fixed")
        column_prefix: Prefix like "complex", "binder", "ligand"
        metric_suffix: The metric suffix like "i_pAE", "pLDDT", "scRMSD"

    Returns:
        Full column name like "self_complex_i_pAE_all"
    """
    return f"{seq_type}_{column_prefix}_{metric_suffix}_all"


def get_thresholds_for_result_type(
    success_thresholds: dict | None,
    is_ligand_binder: bool = False,
) -> dict:
    """Get appropriate thresholds based on result type.

    Args:
        success_thresholds: User-provided thresholds (may be None)
        is_ligand_binder: Whether this is a ligand binder

    Returns:
        Threshold dictionary to use
    """
    if success_thresholds is not None:
        return success_thresholds

    if is_ligand_binder:
        return DEFAULT_LIGAND_BINDER_THRESHOLDS.copy()
    return DEFAULT_PROTEIN_BINDER_THRESHOLDS.copy()


# =============================================================================
# Threshold Check Helpers
# =============================================================================


def check_redesign_passes_all_thresholds(
    metric_values: dict[str, Any],
    parsed_thresholds: dict,
) -> bool:
    """Check if a single redesign passes all threshold criteria.

    This is the shared evaluation logic used by both filter_by_success_thresholds
    and compute_filter_pass_rate.

    Args:
        metric_values: Dictionary mapping metric names to values for one redesign
                       e.g., {"i_pAE": 0.15, "pLDDT": 0.92, "scRMSD": 1.2}
        parsed_thresholds: Dictionary of parsed threshold specs (output of parse_threshold_spec)
                          e.g., {"i_pAE": {"threshold": 7.0, "op": "<=", "scale": 31.0, "column_prefix": "complex"}}

    Returns:
        True if all criteria pass, False otherwise
    """
    for metric_name, spec in parsed_thresholds.items():
        if metric_name not in metric_values:
            return False

        value = metric_values[metric_name]

        # Handle non-float values (e.g., strings, None)
        if not isinstance(value, (int, float)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return False

        if not evaluate_threshold(value, spec["threshold"], spec["op"], spec["scale"]):
            return False

    return True


def check_sample_has_passing_redesign(
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> bool:
    """Check if ANY redesign in a sample passes ALL threshold criteria.

    Args:
        sample_metric_values: Dictionary mapping metric names to lists of values (one per redesign)
                             e.g., {"i_pAE": [0.15, 0.18], "pLDDT": [0.92, 0.88]}
        parsed_thresholds: Dictionary of parsed threshold specs

    Returns:
        True if at least one redesign passes all criteria
    """
    if not sample_metric_values:
        return False

    # Get number of redesigns from first metric
    first_metric = list(sample_metric_values.keys())[0]
    n_redesigns = len(sample_metric_values[first_metric])

    for i in range(n_redesigns):
        # Build metric values for this redesign
        redesign_values = {}
        for metric_name in parsed_thresholds:
            if metric_name in sample_metric_values:
                redesign_values[metric_name] = sample_metric_values[metric_name][i]

        if check_redesign_passes_all_thresholds(redesign_values, parsed_thresholds):
            return True

    return False


def count_passing_redesigns(
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> int:
    """Count how many redesigns in a sample pass ALL threshold criteria.

    Args:
        sample_metric_values: Dictionary mapping metric names to lists of values (one per redesign)
        parsed_thresholds: Dictionary of parsed threshold specs

    Returns:
        Number of redesigns that pass all criteria
    """
    if not sample_metric_values:
        return 0

    # Get number of redesigns from first metric
    first_metric = list(sample_metric_values.keys())[0]
    n_redesigns = len(sample_metric_values[first_metric])

    count = 0
    for i in range(n_redesigns):
        # Build metric values for this redesign
        redesign_values = {}
        for metric_name in parsed_thresholds:
            if metric_name in sample_metric_values:
                redesign_values[metric_name] = sample_metric_values[metric_name][i]

        if check_redesign_passes_all_thresholds(redesign_values, parsed_thresholds):
            count += 1

    return count
