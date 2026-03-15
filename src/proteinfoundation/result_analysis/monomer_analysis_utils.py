"""
Monomer analysis utilities and default thresholds.

This module contains:
- Default thresholds for monomer designability/codesignability filtering
- Column name building utilities for monomer metrics
- Threshold normalization and detection utilities
"""

from typing import Any

# Import shared threshold utilities
from proteinfoundation.result_analysis.analysis_utils import evaluate_threshold, parse_threshold_spec

# =============================================================================
# Constants
# =============================================================================

# Valid RMSD modes for monomer evaluation
VALID_RMSD_MODES = ["ca", "bb3o", "all_atom"]

# Valid folding models
VALID_FOLDING_MODELS = ["esmfold", "colabfold"]


# =============================================================================
# Default Analysis Thresholds
# =============================================================================

# Default monomer designability thresholds
# Format: {mode: {model: {"threshold": float, "op": str}}}
# These are applied to _res_scRMSD_{mode}_{model} columns during analysis
DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS = {
    "ca": {
        "esmfold": {"threshold": 2.0, "op": "<="},
    },
}


# Default monomer codesignability thresholds
# These are applied to _res_co_scRMSD_{mode}_{model} columns during analysis
#
# Note: We have SEPARATE thresholds for CA and all-atom codesignability because
# these are different metrics that users may want to control independently:
#   - CA codesignability: evaluates model-generated sequence via CA RMSD
#   - All-atom codesignability: evaluates model-generated sequence via all-atom RMSD
DEFAULT_MONOMER_CA_CODESIGNABILITY_THRESHOLDS = {
    "ca": {
        "esmfold": {"threshold": 2.0, "op": "<="},
    },
}
DEFAULT_MONOMER_ALL_ATOM_CODESIGNABILITY_THRESHOLDS = {
    "all_atom": {
        "esmfold": {"threshold": 2.0, "op": "<="},
    },
}

# Combined default codesignability thresholds (for backwards compatibility)
# This merges both CA and all-atom defaults
DEFAULT_MONOMER_CODESIGNABILITY_THRESHOLDS = {
    **DEFAULT_MONOMER_CA_CODESIGNABILITY_THRESHOLDS,
    **DEFAULT_MONOMER_ALL_ATOM_CODESIGNABILITY_THRESHOLDS,
}


# =============================================================================
# Column Name Utilities
# =============================================================================


def build_monomer_column_name(
    metric_type: str,
    mode: str,
    model: str,
) -> str:
    """Build the full column name for a monomer metric.

    Args:
        metric_type: Type of metric ("designability", "single_designability", or "codesignability")
        mode: RMSD mode ("ca", "bb3o", "all_atom")
        model: Folding model name ("esmfold", "colabfold", etc.)

    Returns:
        Full column name like "_res_scRMSD_ca_esmfold", "_res_scRMSD_single_ca_esmfold",
        or "_res_co_scRMSD_ca_esmfold"
    """
    if metric_type == "designability":
        return f"_res_scRMSD_{mode}_{model}"
    elif metric_type == "single_designability":
        return f"_res_scRMSD_single_{mode}_{model}"
    elif metric_type == "codesignability":
        return f"_res_co_scRMSD_{mode}_{model}"
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# Suffix appended to monomer columns when merged into motif results
MONOMER_COLUMN_SUFFIX = "_monomer"


def resolve_monomer_column(
    col_name: str,
    available_columns,
) -> str | None:
    """Resolve a monomer column name with ``_monomer`` fallback.

    When monomer results are merged into motif results, conflicting columns
    receive a ``_monomer`` suffix.  This function checks for the canonical
    name first, then the suffixed variant.

    Args:
        col_name: Canonical column name (e.g. ``_res_scRMSD_ca_esmfold``)
        available_columns: Iterable of column names (e.g. ``df.columns``)

    Returns:
        The resolved column name, or ``None`` if neither variant exists.
    """
    cols = set(available_columns) if not isinstance(available_columns, set) else available_columns
    if col_name in cols:
        return col_name
    fallback = f"{col_name}{MONOMER_COLUMN_SUFFIX}"
    if fallback in cols:
        return fallback
    return None


def detect_monomer_folding_models(
    df_columns: list[str],
    metric_type: str = "designability",
) -> dict[str, list[str]]:
    """Detect available folding models and modes from column names.

    Args:
        df_columns: List of column names from the dataframe
        metric_type: Type of metric ("designability", "single_designability", or "codesignability")

    Returns:
        Dictionary mapping modes to list of available models
        e.g., {"ca": ["esmfold", "colabfold"], "all_atom": ["esmfold"]}
    """
    if metric_type == "designability":
        prefix = "_res_scRMSD_"
    elif metric_type == "single_designability":
        prefix = "_res_scRMSD_single_"
    else:
        prefix = "_res_co_scRMSD_"

    result = {}

    for col in df_columns:
        # Skip columns ending with "_all" (aggregated columns), but NOT "all_atom" mode
        if not col.startswith(prefix) or col.endswith("_all"):
            continue

        # Skip _monomer suffixed columns (merged from monomer eval into motif results)
        if col.endswith(MONOMER_COLUMN_SUFFIX):
            continue

        # For designability, skip single_ columns to avoid double-matching
        if metric_type == "designability" and "_res_scRMSD_single_" in col:
            continue

        # Extract mode and model from column name
        suffix = col.replace(prefix, "")

        for mode in VALID_RMSD_MODES:
            if suffix.startswith(f"{mode}_"):
                model = suffix.replace(f"{mode}_", "")
                if model:  # Non-empty model name
                    if mode not in result:
                        result[mode] = []
                    if model not in result[mode]:
                        result[mode].append(model)
                break

    return result


# =============================================================================
# Threshold Normalization
# =============================================================================


def normalize_monomer_thresholds(
    thresholds: dict,
    df_columns: list[str] = None,
) -> dict:
    """Normalize monomer thresholds to ensure consistent format.

    If thresholds only specify a threshold value (float), converts to full spec format.
    Optionally auto-detects available modes/models from dataframe columns.

    Args:
        thresholds: Input threshold dictionary
        df_columns: Optional list of column names for auto-detection

    Returns:
        Normalized threshold dictionary
    """
    normalized = {}

    for mode, models in thresholds.items():
        normalized[mode] = {}
        for model, spec in models.items():
            if isinstance(spec, (int, float)):
                # Convert simple threshold to full spec
                normalized[mode][model] = {"threshold": float(spec), "op": "<="}
            elif isinstance(spec, dict):
                normalized[mode][model] = parse_threshold_spec(spec)
            else:
                raise ValueError(f"Invalid threshold spec for {mode}/{model}: {spec}")

    return normalized


def get_thresholds_for_metric_type(
    thresholds: dict = None,
    metric_type: str = "designability",
    codesignability_mode: str = None,
) -> dict:
    """Get appropriate thresholds for the given metric type.

    Args:
        thresholds: User-provided thresholds (may be None)
        metric_type: "designability", "single_designability", or "codesignability"
        codesignability_mode: For codesignability, specify "ca" or "all_atom" to get
            specific defaults. If None, returns combined defaults.

    Returns:
        Threshold dictionary to use
    """
    if thresholds is not None:
        return thresholds

    if metric_type == "single_designability":
        return DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS.copy()
    if metric_type == "codesignability":
        if codesignability_mode == "ca":
            return DEFAULT_MONOMER_CA_CODESIGNABILITY_THRESHOLDS.copy()
        elif codesignability_mode == "all_atom":
            return DEFAULT_MONOMER_ALL_ATOM_CODESIGNABILITY_THRESHOLDS.copy()
        # Return combined defaults if no specific mode requested
        return DEFAULT_MONOMER_CODESIGNABILITY_THRESHOLDS.copy()
    return DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS.copy()


def get_codesignability_thresholds(
    ca_thresholds: dict = None,
    allatom_thresholds: dict = None,
) -> dict:
    """Get combined codesignability thresholds from separate CA and all-atom configs.

    This function combines CA and all-atom codesignability thresholds into a single
    dictionary that can be used by the analysis functions.

    Args:
        ca_thresholds: CA codesignability thresholds (may be None for defaults)
        allatom_thresholds: All-atom codesignability thresholds (may be None for defaults)

    Returns:
        Combined threshold dictionary with both CA and all-atom thresholds
    """
    combined = {}

    # Add CA thresholds
    if ca_thresholds is not None:
        combined.update(ca_thresholds)
    else:
        combined.update(DEFAULT_MONOMER_CA_CODESIGNABILITY_THRESHOLDS.copy())

    # Add all-atom thresholds
    if allatom_thresholds is not None:
        combined.update(allatom_thresholds)
    else:
        combined.update(DEFAULT_MONOMER_ALL_ATOM_CODESIGNABILITY_THRESHOLDS.copy())

    return combined


def build_thresholds_from_detected(
    detected: dict[str, list[str]],
    default_threshold: float = 2.0,
    default_op: str = "<=",
) -> dict:
    """Build threshold dict from detected modes/models with default values.

    Args:
        detected: Dictionary mapping modes to list of models
        default_threshold: Default threshold value (default: 2.0)
        default_op: Default comparison operator (default: "<=")

    Returns:
        Threshold dictionary in format {mode: {model: {threshold, op}}}
    """
    thresholds = {}
    for mode, models in detected.items():
        thresholds[mode] = {}
        for model in models:
            thresholds[mode][model] = {"threshold": default_threshold, "op": default_op}
    return thresholds


# =============================================================================
# Threshold Evaluation Helpers
# =============================================================================


def check_monomer_passes_threshold(
    value: Any,
    threshold: float,
    op: str = "<=",
) -> bool:
    """Check if a monomer metric value passes the threshold.

    Args:
        value: The metric value to check
        threshold: Threshold value
        op: Comparison operator ("<=", "<", ">=", ">", "==")

    Returns:
        True if passes, False otherwise
    """
    return evaluate_threshold(value, threshold, op, scale=1.0)


def check_monomer_passes_all_thresholds(
    metric_values: dict[str, Any],
    thresholds: dict,
    metric_type: str = "designability",
) -> bool:
    """Check if a sample passes all specified monomer thresholds.

    Args:
        metric_values: Dictionary of column_name -> value
        thresholds: Normalized threshold dictionary {mode: {model: spec}}
        metric_type: "designability" or "codesignability"

    Returns:
        True if all thresholds pass, False otherwise
    """
    for mode, models in thresholds.items():
        for model, spec in models.items():
            col_name = build_monomer_column_name(metric_type, mode, model)

            if col_name not in metric_values:
                return False

            value = metric_values[col_name]
            if not check_monomer_passes_threshold(value, spec["threshold"], spec["op"]):
                return False

    return True


def check_monomer_passes_any_threshold(
    metric_values: dict[str, Any],
    thresholds: dict,
    metric_type: str = "designability",
) -> bool:
    """Check if a sample passes any of the specified monomer thresholds.

    Args:
        metric_values: Dictionary of column_name -> value
        thresholds: Normalized threshold dictionary {mode: {model: spec}}
        metric_type: "designability" or "codesignability"

    Returns:
        True if any threshold passes, False otherwise
    """
    for mode, models in thresholds.items():
        for model, spec in models.items():
            col_name = build_monomer_column_name(metric_type, mode, model)

            if col_name not in metric_values:
                continue

            value = metric_values[col_name]
            if check_monomer_passes_threshold(value, spec["threshold"], spec["op"]):
                return True

    return False
