"""
Shared result-analysis utilities used across all evaluation types.

This module contains domain-agnostic helpers for post-processing evaluation
DataFrames (e.g. column filtering for CSV export, threshold parsing,
aggregate statistics).  Domain-specific utilities live in their own
``*_analysis_utils.py`` modules.
"""

import ast
import os
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# =============================================================================
# Shared Constants
# =============================================================================

# Canonical sequence types in evaluation order.
# Used across all analysis modules for iteration and column detection.
SEQUENCE_TYPES = ["self", "mpnn", "mpnn_fixed"]

# =============================================================================
# CSV Formatting Constants
# =============================================================================

FLOAT_FORMAT_PD = "%.6f"
SEP_CSV_PD = ","


# =============================================================================
# CSV Column Filtering
# =============================================================================

# Patterns to match for columns to drop (uses substring matching).
# This catches both direct columns and generated_* prefixed versions.
# Note: We keep "generated_binder_*", "generated_n_*", "generated_total_*" as actual metrics.
COLUMNS_TO_DROP_PATTERNS: list[str] = [
    # Config flags
    "dryrun",
    "show_progress",
    "result_type",
    # Settings
    "ncpus_",
    "gen_njobs",
    "seed",
    # Aggregation config
    "aggregation_",
    # Dataloader config
    "generation_dataloader_",
    # Target dict config (very large)
    "target_dict_cfg",
    # Path-related (only for aggregated analysis, not raw results)
    "root_path",
    "results_dir",
    # LoRA config (not needed in results)
    "lora_",
]


def filter_columns_for_csv(
    df: "pd.DataFrame",
    log_dropped: bool = False,
) -> "pd.DataFrame":
    """Filter out config/metadata columns from a DataFrame before saving to CSV.

    Uses substring matching against :data:`COLUMNS_TO_DROP_PATTERNS` to catch
    both direct columns (e.g. ``"dryrun"``) and prefixed versions (e.g.
    ``"generated_dryrun"``).

    Args:
        df: DataFrame with all columns.
        log_dropped: If True, log info about dropped columns.

    Returns:
        DataFrame with non-metric columns removed.
    """
    cols_to_drop = []

    for col in df.columns:
        for pattern in COLUMNS_TO_DROP_PATTERNS:
            if pattern in col:
                cols_to_drop.append(col)
                break

    if cols_to_drop:
        if log_dropped:
            logger.info(f"Dropping {len(cols_to_drop)} non-metric columns from CSV output")
            logger.debug(f"Dropped columns: {cols_to_drop[:10]}...")
        df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# =============================================================================
# Threshold Parsing and Evaluation
# =============================================================================


def parse_threshold_spec(spec: int | float | dict | list | tuple) -> dict:
    """Parse a threshold specification into a standardized format.

    Args:
        spec: Can be:
            - A float/int (uses defaults for op and scale based on metric)
            - A dict with keys: threshold, op (optional), scale (optional),
              column_prefix (optional)
            - A tuple: (threshold,) or (threshold, op) or
              (threshold, op, scale) or (threshold, op, scale, column_prefix)

    Returns:
        Standardized dict with threshold, op, scale, column_prefix.
    """
    if isinstance(spec, (int, float)):
        return {
            "threshold": float(spec),
            "op": "<=",
            "scale": 1.0,
            "column_prefix": "complex",
        }
    elif isinstance(spec, dict):
        return {
            "threshold": float(spec.get("threshold", spec.get("value", 0.0))),
            "op": spec.get("op", spec.get("operator", "<=")),
            "scale": float(spec.get("scale", 1.0)),
            "column_prefix": spec.get("column_prefix", "complex"),
        }
    elif isinstance(spec, (list, tuple)):
        return {
            "threshold": float(spec[0]),
            "op": spec[1] if len(spec) > 1 else "<=",
            "scale": float(spec[2]) if len(spec) > 2 else 1.0,
            "column_prefix": spec[3] if len(spec) > 3 else "complex",
        }
    else:
        raise ValueError(f"Invalid threshold specification: {spec}")


def evaluate_threshold(
    value: Any,
    threshold: float,
    op: str,
    scale: float = 1.0,
) -> bool:
    """Evaluate if a value passes a threshold comparison.

    Args:
        value: The value to compare.
        threshold: The threshold value.
        op: Comparison operator (``"<="``, ``"<"``, ``">="``, ``">"``, ``"=="``).
        scale: Scale factor to apply to value before comparison.

    Returns:
        True if the comparison passes, False otherwise.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False

    scaled_value = value * scale

    if op == "<=":
        return scaled_value <= threshold
    elif op == "<":
        return scaled_value < threshold
    elif op == ">=":
        return scaled_value >= threshold
    elif op == ">":
        return scaled_value > threshold
    elif op == "==":
        return scaled_value == threshold
    else:
        raise ValueError(f"Unknown operator: {op}")


# =============================================================================
# Aggregate Statistic Helpers
# =============================================================================


def _filter_valid_values(values: list[Any]) -> list[float]:
    """Filter list to only valid numeric values (not None or NaN)."""
    if not values or not isinstance(values, list):
        return []
    return [v for v in values if v is not None and not np.isnan(v)]


def _count_passing(valid: list[float], threshold: float, op: str) -> int:
    """Count values that pass threshold with given operator."""
    ops = {
        "<=": lambda v: v <= threshold,
        "<": lambda v: v < threshold,
        ">=": lambda v: v >= threshold,
        ">": lambda v: v > threshold,
        "==": lambda v: v == threshold,
    }
    compare = ops.get(op, ops["<="])
    return sum(1 for v in valid if compare(v))


def compute_pass_rate_for_values(
    values: list[Any],
    threshold: float,
    op: str = "<=",
) -> float:
    """Compute pass rate for a list of values against a threshold.

    Args:
        values: List of metric values.
        threshold: Threshold value.
        op: Comparison operator.

    Returns:
        Pass rate (fraction of valid values that pass).
    """
    valid = _filter_valid_values(values)
    if not valid:
        return 0.0
    return _count_passing(valid, threshold, op) / len(valid)


def compute_n_passed_for_values(
    values: list[Any],
    threshold: float,
    op: str = "<=",
) -> int:
    """Count how many values pass a threshold.

    Args:
        values: List of metric values.
        threshold: Threshold value.
        op: Comparison operator.

    Returns:
        Number of values that pass.
    """
    valid = _filter_valid_values(values)
    return _count_passing(valid, threshold, op) if valid else 0


def compute_mean_for_values(values: list[Any]) -> float:
    """Compute mean of valid values.

    Args:
        values: List of metric values.

    Returns:
        Mean of valid values, or inf if no valid values.
    """
    valid = _filter_valid_values(values)
    return np.mean(valid) if valid else float("inf")


def compute_std_for_values(values: list[Any]) -> float:
    """Compute standard deviation of valid values.

    Args:
        values: List of metric values.

    Returns:
        Std of valid values, or 0.0 if insufficient values.
    """
    valid = _filter_valid_values(values)
    return np.std(valid) if len(valid) > 1 else 0.0


# =============================================================================
# Sample Labeling
# =============================================================================


def get_sample_label(row: pd.Series) -> str:
    """Build a human-readable sample label from available identifying columns.

    Args:
        row: A single row from an evaluation DataFrame.

    Returns:
        Compact label string like ``"id=3 | sample_3 | PDL1"``.
    """
    parts = []
    if "id_gen" in row.index and pd.notna(row.get("id_gen")):
        parts.append(f"id={int(row['id_gen'])}")
    if "pdb_path" in row.index and pd.notna(row.get("pdb_path")):
        pdb_name = os.path.basename(str(row["pdb_path"]))
        if pdb_name.endswith(".pdb"):
            pdb_name = pdb_name[:-4]
        parts.append(pdb_name)
    if "task_name" in row.index and pd.notna(row.get("task_name")):
        parts.append(str(row["task_name"]))
    return " | ".join(parts) if parts else f"row_{row.name}"


# =============================================================================
# List Aggregation for GroupBy
# =============================================================================


def keep_lists_separate(
    series: pd.Series,
    expected_len: int | None = None,
) -> list:
    """Aggregation function that preserves per-sample list boundaries.

    Designed for use in ``DataFrame.groupby(...).agg(...)`` where each cell may
    contain a list of per-redesign metric values. The function:

    * Replaces NaN/None *elements* inside a list with ``float("nan")`` so that
      list length (and therefore redesign index alignment) is preserved across
      columns.
    * Converts scalar NaN/None entries to empty lists ``[]`` for positional
      alignment across columns.
    * Parses string representations of lists (``ast.literal_eval``).
    * Optionally validates list length (e.g. ``expected_len=20`` for aa_counts).

    Args:
        series: A pandas Series from a groupby aggregation.
        expected_len: If set, lists whose length != expected_len are replaced
            with ``[]`` and a warning is logged.

    Returns:
        A list of lists, one per group member.
    """
    result = []
    for v in series:
        if isinstance(v, list):
            cleaned = [(float("nan") if isinstance(x, (int, float, type(None))) and pd.isna(x) else x) for x in v]
            if expected_len is not None and len(cleaned) != expected_len:
                logger.warning(f"Invalid list length: {len(cleaned)}, expected {expected_len}")
                cleaned = []
            result.append(cleaned)
            continue
        elif pd.isna(v):
            result.append([])
            continue

        try:
            v = ast.literal_eval(v) if isinstance(v, str) else v
            if isinstance(v, list):
                if expected_len is not None and len(v) != expected_len:
                    logger.warning(f"Invalid list length: {len(v)}, expected {expected_len}")
                    result.append([])
                else:
                    result.append(v)
            else:
                if expected_len is not None:
                    logger.warning(f"Expected list but got {type(v)}")
                    result.append([])
                else:
                    result.append([v])
        except (ValueError, SyntaxError) as e:
            if expected_len is not None:
                logger.warning(f"Error parsing list value: {e}")
            result.append([])
    return result


# =============================================================================
# Filtered CSV Export
# =============================================================================


def save_filtered_csv(
    df_filtered: pd.DataFrame,
    path_store_results: str,
    filter_name: str,
    seq_type: str,
) -> None:
    """Save filtered DataFrame to CSV, dropping problematic columns.

    Drops any column whose name contains ``"dataset_target_dict"`` (large
    nested config objects that break CSV readability) and writes to
    ``<path_store_results>/all_successes_<filter_name>_<seq_type>.csv``.

    Args:
        df_filtered: DataFrame of successful samples.
        path_store_results: Directory to write the CSV into.
        filter_name: Label for the filter (e.g. threshold set name).
        seq_type: Sequence type label (e.g. ``"self"``, ``"mpnn"``).
    """
    df_out = df_filtered.copy()
    bad_columns = [c for c in df_out.columns if "dataset_target_dict" in c]
    if bad_columns:
        logger.warning(f"Dropping columns: {bad_columns}")
        df_out.drop(columns=bad_columns, inplace=True)

    output_path = os.path.join(path_store_results, f"all_successes_{filter_name}_{seq_type}.csv")
    df_out.to_csv(
        output_path,
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    logger.debug(f"Saved filtered results to {output_path}")


# =============================================================================
# List Coercion for Row-Level Metric Extraction
# =============================================================================


def coerce_to_list(value: Any, col_name: str = "") -> list:
    """Parse a cell value into a list, handling string representations.

    Handles the common pattern where DataFrame cells may contain actual lists,
    string representations of lists (from CSV round-trips), or scalar values.

    Args:
        value: Raw cell value (list, str, scalar, or None).
        col_name: Column name for log messages.

    Returns:
        A list of values.
    """
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse list value for column '{col_name}': {value!r} {e}")
            return []
    if isinstance(value, list):
        return value
    return [value]


def extract_list_values(
    row: pd.Series,
    col_map: dict[str, str],
) -> dict[str, list]:
    """Extract list-valued columns from a DataFrame row.

    For each entry in *col_map* (``metric_name -> column_name``), reads the
    cell from *row*, parses it with :func:`coerce_to_list`, and returns a dict
    of ``metric_name -> list[values]``.

    Args:
        row: A single DataFrame row.
        col_map: Mapping of logical metric name to DataFrame column name.

    Returns:
        Dict of metric_name -> list of values.
    """
    result = {}
    for metric_name, col_name in col_map.items():
        result[metric_name] = coerce_to_list(row[col_name], col_name)
    return result
