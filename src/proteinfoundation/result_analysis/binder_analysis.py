"""
Binder success analysis functions and constants.

This module contains all the functions and constants related to binder success
criteria evaluation, including threshold configuration, filtering, and pass rate
computation for both protein and ligand binders.
"""

import json
import os

import numpy as np
import pandas as pd
from loguru import logger

# Import utilities from binder_analysis_utils
from proteinfoundation.result_analysis.analysis_utils import (
    FLOAT_FORMAT_PD,
    SEP_CSV_PD,
    SEQUENCE_TYPES,
    evaluate_threshold,
    extract_list_values,
    get_sample_label,
    keep_lists_separate,
    parse_threshold_spec,
    save_filtered_csv,
)
from proteinfoundation.result_analysis.binder_analysis_utils import (
    DEFAULT_LIGAND_BINDER_THRESHOLDS,
    DEFAULT_PROTEIN_BINDER_THRESHOLDS,
    build_column_name,
    check_redesign_passes_all_thresholds,
    check_sample_has_passing_redesign,
    count_passing_redesigns,
    normalize_threshold_dict,
)


def save_success_criteria_json(
    success_thresholds: dict,
    path_store_results: str,
    filter_name: str = "binder_success",
    seq_type: str | None = None,
) -> str:
    """Save success criteria thresholds to a JSON file for reproducibility.

    Args:
        success_thresholds: Dictionary specifying success criteria
        path_store_results: Directory path to store the JSON file
        filter_name: Name for the filter (e.g., "binder_success", "ligand_binder_success")
        seq_type: Optional sequence type to include in filename

    Returns:
        Path to the saved JSON file
    """
    # Normalize the thresholds before saving
    normalized_thresholds = normalize_threshold_dict(success_thresholds)

    # Parse each threshold to ensure consistent format
    parsed_thresholds = {}
    for metric_name, spec in normalized_thresholds.items():
        parsed_thresholds[metric_name] = parse_threshold_spec(spec)

    # Build the output data structure
    output_data = {
        "filter_name": filter_name,
        "thresholds": parsed_thresholds,
    }
    if seq_type is not None:
        output_data["seq_type"] = seq_type

    # Build filename
    if seq_type is not None:
        filename = f"success_criteria_{filter_name}_{seq_type}.json"
    else:
        filename = f"success_criteria_{filter_name}.json"

    output_path = os.path.join(path_store_results, filename)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.debug(f"Saved success criteria to {output_path}")
    return output_path


def save_combined_success_criteria_json(
    success_thresholds: dict,
    path_store_results: str,
    filter_name: str,
    sequence_types: list[str],
) -> str:
    """Save success criteria for all sequence types to a single JSON file.

    Creates a combined JSON file with thresholds shared across all sequence types,
    and lists which sequence types were evaluated.

    Args:
        success_thresholds: Dictionary specifying success criteria (same for all seq types)
        path_store_results: Directory path to store the JSON file
        filter_name: Name for the filter (e.g., "protein_binder", "ligand_binder")
        sequence_types: List of sequence types evaluated (e.g., ["self", "mpnn"])

    Returns:
        Path to the saved JSON file
    """
    # Normalize the thresholds before saving
    normalized_thresholds = normalize_threshold_dict(success_thresholds)

    # Parse each threshold to ensure consistent format
    parsed_thresholds = {}
    for metric_name, spec in normalized_thresholds.items():
        parsed_thresholds[metric_name] = parse_threshold_spec(spec)

    # Build the output data structure with all sequence types
    output_data = {
        "filter_name": filter_name,
        "thresholds": parsed_thresholds,
        "sequence_types": sequence_types,
    }

    filename = f"success_criteria_{filter_name}.json"
    output_path = os.path.join(path_store_results, filename)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.debug(f"Saved combined success criteria to {output_path}")
    return output_path


def load_success_criteria_json(json_path: str) -> dict:
    """Load success criteria thresholds from a JSON file.

    Args:
        json_path: Path to the JSON file containing success criteria

    Returns:
        Dictionary with success thresholds that can be passed to filtering functions
    """
    with open(json_path) as f:
        data = json.load(f)

    thresholds = data.get("thresholds", {})
    logger.debug(f"Loaded success criteria from {json_path}")
    logger.debug(f"Filter name: {data.get('filter_name', 'unknown')}")
    if "seq_type" in data:
        logger.debug(f"Sequence type: {data['seq_type']}")

    return thresholds


def compute_grouped_pass_rate(
    row: pd.Series,
    seq_type: str,
    success_thresholds: dict,
    max_samples: int | None = None,
) -> tuple[float, list[int]]:
    """Compute pass rate for a grouped row.

    For each sample, checks whether ANY of its redesigns passes ALL criteria.
    A sample is a "success" if at least one redesign meets every threshold.

    Pass rate = number of successful samples / total samples.

    Args:
        row: DataFrame row containing grouped metric columns.
            Each metric column holds a list of lists: samples -> redesigns -> values.
        seq_type: Sequence type ("self", "mpnn", "mpnn_fixed")
        success_thresholds: Threshold dictionary in the standard format
        max_samples: Maximum number of samples to consider (None = all)

    Returns:
        Tuple of (pass_rate, per_sample_pass) where per_sample_pass is a list
        of 1/0 indicating whether each sample passed (e.g. [1, 0, 0, 1]).
    """
    # Normalize and parse thresholds
    normalized_thresholds = normalize_threshold_dict(success_thresholds)
    parsed_thresholds = {}
    for metric_name, spec in normalized_thresholds.items():
        parsed_thresholds[metric_name] = parse_threshold_spec(spec)

    # Build column names and extract data from row
    grouped_data = {}  # metric_name -> list of lists (samples -> redesigns)
    for metric_name, spec in parsed_thresholds.items():
        col_name = build_column_name(seq_type, spec["column_prefix"], metric_name)
        if col_name in row.index:
            grouped_data[metric_name] = row[col_name]

    # If we don't have all required metrics, return 0
    if len(grouped_data) != len(parsed_thresholds):
        return 0.0, []

    # Get number of samples
    first_metric = list(grouped_data.keys())[0]
    if max_samples is not None:
        n_samples = min(len(grouped_data[first_metric]), max_samples)
    else:
        n_samples = len(grouped_data[first_metric])
    if n_samples == 0:
        return 0.0, []

    # Per-sample pass: 1 if ANY redesign meets ALL criteria, 0 otherwise
    per_sample_pass = []
    for sample_idx in range(n_samples):
        sample_metric_values = {}
        for metric_name in parsed_thresholds:
            sample_metric_values[metric_name] = grouped_data[metric_name][sample_idx]

        passed = check_sample_has_passing_redesign(sample_metric_values, parsed_thresholds)
        per_sample_pass.append(1 if passed else 0)

    pass_rate = sum(per_sample_pass) / n_samples
    return pass_rate, per_sample_pass


def add_success_rate_columns(
    df_grouped: pd.DataFrame,
    seq_type: str,
    success_thresholds: dict,
    criteria_name: str,
    metric_suffix: str,
    max_samples: int | None = None,
) -> None:
    """Add success rate columns to a grouped DataFrame for a given criteria set.

    Adds a pass_rate column (float) and a per_sample_pass column (list of 1/0)
    for downstream filtering/aggregation.

    Args:
        df_grouped: Grouped DataFrame to add columns to (modified in place)
        seq_type: Sequence type ("self", "mpnn", "mpnn_fixed")
        success_thresholds: Threshold dictionary in the standard format
        criteria_name: Name for the criteria (e.g., "success", "rfdiffusion")
        metric_suffix: Suffix for metric names
        max_samples: Maximum number of samples to consider
    """
    # Check if we have the required columns
    normalized_thresholds = normalize_threshold_dict(success_thresholds)
    required_cols = []
    for metric_name, spec in normalized_thresholds.items():
        parsed = parse_threshold_spec(spec)
        col_name = build_column_name(seq_type, parsed["column_prefix"], metric_name)
        required_cols.append(col_name)

    if not all(col in df_grouped.columns for col in required_cols):
        return

    # Compute pass rates for each row
    pass_rate_col = f"_res_{seq_type}_pass_rate_{criteria_name}_{metric_suffix}"
    per_sample_col = f"_res_{seq_type}_per_sample_pass_{criteria_name}_{metric_suffix}"

    results = df_grouped.apply(
        lambda row: compute_grouped_pass_rate(row, seq_type, success_thresholds, max_samples),
        axis=1,
    )

    df_grouped[pass_rate_col] = results.apply(lambda x: x[0])
    df_grouped[per_sample_col] = results.apply(lambda x: x[1])


# =============================================================================
# Filtering Functions
# =============================================================================


def filter_by_success_thresholds(
    df: pd.DataFrame,
    seq_type: str,
    success_thresholds: dict | None = None,
    path_store_results: str | None = None,
    filter_name: str = "binder_success",
    save_json: bool = True,
) -> pd.DataFrame:
    """Filter dataframe using flexible success threshold criteria.

    This is the main flexible filtering function that accepts a dictionary of
    threshold specifications. Each metric in the dictionary is checked, and a
    sample is considered successful if ALL metrics pass their thresholds for
    at least one redesign.

    Args:
        df: Input dataframe containing evaluation results
        seq_type: Sequence type to evaluate ("mpnn", "mpnn_fixed", or "self")
        success_thresholds: Dictionary specifying success criteria. Format:
            {
                "metric_suffix": {
                    "threshold": float,     # The threshold value
                    "op": str,              # Operator: "<=", "<", ">=", ">", "=="
                    "scale": float,         # Scale factor (value * scale compared to threshold)
                    "column_prefix": str,   # "complex", "binder", "ligand"
                }
            }
            Metric names are case-normalized (e.g., "plddt" -> "pLDDT").
            If None, uses DEFAULT_PROTEIN_BINDER_THRESHOLDS.
        path_store_results: Optional path to store successful samples CSV
        filter_name: Name for logging and output files
        save_json: If True, save per-sequence-type JSON. Set to False when using
            save_combined_success_criteria_json to avoid redundant files.

    Returns:
        Filtered DataFrame containing only successful samples

    Example:
        # Add i_pTM to success criteria:
        thresholds = {
            "i_pAE": {"threshold": 7.0, "op": "<=", "scale": 31, "column_prefix": "complex"},
            "pLDDT": {"threshold": 0.9, "op": ">=", "scale": 1.0, "column_prefix": "complex"},
            "scRMSD": {"threshold": 1.5, "op": "<", "scale": 1.0, "column_prefix": "binder"},
            "i_pTM": {"threshold": 0.8, "op": ">=", "scale": 1.0, "column_prefix": "complex"},
        }
        df_filtered = filter_by_success_thresholds(df, "self", thresholds)
    """
    # Use default if not provided
    if success_thresholds is None:
        success_thresholds = DEFAULT_PROTEIN_BINDER_THRESHOLDS.copy()

    # Normalize metric names (handle lowercase inputs)
    normalized_thresholds = normalize_threshold_dict(success_thresholds)

    # Parse all threshold specifications
    parsed_thresholds = {}
    for metric_name, spec in normalized_thresholds.items():
        parsed_thresholds[metric_name] = parse_threshold_spec(spec)

    logger.debug(f"Filtering by {filter_name} for sequence type: {seq_type}")
    logger.debug("Thresholds:")
    for metric_name, spec in parsed_thresholds.items():
        scale_str = f"*{spec['scale']}" if spec["scale"] != 1.0 else ""
        logger.debug(f"  {spec['column_prefix']}_{metric_name}{scale_str} {spec['op']} {spec['threshold']}")

    # Build column names and check they exist
    column_mapping = {}  # metric_name -> column_name
    missing_cols = []

    for metric_name, spec in parsed_thresholds.items():
        col_name = build_column_name(seq_type, spec["column_prefix"], metric_name)
        if col_name in df.columns:
            column_mapping[metric_name] = col_name
        else:
            missing_cols.append(col_name)

    if missing_cols:
        logger.warning(f"Missing columns for {seq_type}: {missing_cols}. Will skip these metrics in filtering.")

    if not column_mapping:
        logger.warning(f"No valid columns found for {seq_type}, returning empty dataframe")
        return pd.DataFrame()

    # Create a mask for successful samples
    success_mask = []

    # Evaluate each sample using the shared helper function
    for idx, row in df.iterrows():
        try:
            sample_metric_values = extract_list_values(row, column_mapping)

            # Check if ANY redesign passes ALL criteria using shared helper
            success = check_sample_has_passing_redesign(sample_metric_values, parsed_thresholds)
            n_passing = count_passing_redesigns(sample_metric_values, parsed_thresholds)
            success_mask.append(success)

            # Per-sample logging with per-redesign breakdown
            sample_id = get_sample_label(row)
            _log_binder_sample_redesigns(
                seq_type=seq_type,
                sample_id=sample_id,
                passed=success,
                n_passing=n_passing,
                sample_metric_values=sample_metric_values,
                parsed_thresholds=parsed_thresholds,
            )

        except Exception as e:
            logger.warning(f"Error evaluating sample {idx} for {seq_type}: {e}")
            success_mask.append(False)

    # Apply the filter
    df_filtered = df[success_mask]

    # Log results
    successful_count = len(df_filtered)
    total_count = len(df)
    if total_count > 0:
        logger.info(
            f"  [{seq_type}] Result: {successful_count}/{total_count} samples passed "
            f"({successful_count / total_count * 100:.0f}%)"
        )
    else:
        logger.info(f"  [{seq_type}] Result: 0/0 samples (no data)")

    if path_store_results is not None:
        save_filtered_csv(df_filtered, path_store_results, filter_name, seq_type)

        # Save the success criteria as JSON for reproducibility (if not using combined save)
        if save_json:
            save_success_criteria_json(
                success_thresholds=success_thresholds,
                path_store_results=path_store_results,
                filter_name=filter_name,
                seq_type=seq_type,
            )

    return df_filtered


def filter_by_binder_success(
    df: pd.DataFrame,
    seq_type: str,
    path_store_results: str | None = None,
    success_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Filter dataframe to keep only samples that pass protein binder success criteria.

    Args:
        df: Input dataframe containing binder evaluation results
        seq_type: Sequence type to evaluate ("mpnn", "mpnn_fixed", or "self")
        path_store_results: Path to store results
        success_thresholds: Optional flexible thresholds dictionary. If not provided,
            uses DEFAULT_PROTEIN_BINDER_THRESHOLDS.

    Returns:
        Filtered DataFrame containing only successful samples
    """
    thresholds = success_thresholds if success_thresholds is not None else DEFAULT_PROTEIN_BINDER_THRESHOLDS.copy()

    return filter_by_success_thresholds(
        df=df,
        seq_type=seq_type,
        success_thresholds=thresholds,
        path_store_results=path_store_results,
        filter_name="binder_success",
    )


def filter_by_ligand_binder_success(
    df: pd.DataFrame,
    seq_type: str,
    path_store_results: str | None = None,
    success_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Filter dataframe to keep only samples that pass ligand binder success criteria.

    Args:
        df: Input dataframe containing binder evaluation results
        seq_type: Sequence type to evaluate ("mpnn", "mpnn_fixed", or "self")
        path_store_results: Path to store results
        success_thresholds: Optional flexible thresholds dictionary. If not provided,
            uses DEFAULT_LIGAND_BINDER_THRESHOLDS.

    Returns:
        Filtered DataFrame containing only successful samples
    """
    thresholds = success_thresholds if success_thresholds is not None else DEFAULT_LIGAND_BINDER_THRESHOLDS.copy()

    return filter_by_success_thresholds(
        df=df,
        seq_type=seq_type,
        success_thresholds=thresholds,
        path_store_results=path_store_results,
        filter_name="ligand_binder_success",
    )


# =============================================================================
# Pass Rate Computation Functions
# =============================================================================


def compute_filter_pass_rate(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    success_thresholds: dict | None = None,
    result_type: str = "protein_binder",
) -> pd.DataFrame:
    """Compute filter pass rate for different sequence types using provided success criteria.

    Works for both protein and ligand binder results. The *result_type*
    selects the default thresholds and output filename when
    *success_thresholds* is not provided.

    Args:
        df: Input DataFrame containing results.
        groupby_cols: Columns to group results by.
        path_store_results: Directory to store result CSV.
        metric_suffix: Suffix appended to metric column names.
        success_thresholds: Optional threshold dictionary.  Falls back to
            ``DEFAULT_PROTEIN_BINDER_THRESHOLDS`` or
            ``DEFAULT_LIGAND_BINDER_THRESHOLDS`` based on *result_type*.
        result_type: ``"protein_binder"`` or ``"ligand_binder"``.

    Returns:
        DataFrame with success rates for the given criteria.
    """
    is_ligand = result_type == "ligand_binder"
    default_thresholds = DEFAULT_LIGAND_BINDER_THRESHOLDS if is_ligand else DEFAULT_PROTEIN_BINDER_THRESHOLDS
    thresholds = success_thresholds if success_thresholds is not None else default_thresholds.copy()
    thresholds = normalize_threshold_dict(thresholds)

    label = "ligand " if is_ligand else ""
    logger.debug(f"Computing {label}filter pass rate with thresholds:")
    for metric_name, spec in thresholds.items():
        parsed = parse_threshold_spec(spec)
        scale_str = f"*{parsed['scale']}" if parsed.get("scale", 1.0) != 1.0 else ""
        logger.debug(
            f"  {parsed.get('column_prefix', 'complex')}_{metric_name}{scale_str} {parsed.get('op', '<=')} {parsed.get('threshold')}"
        )

    sequence_types = SEQUENCE_TYPES
    logger.debug(f"Analyzing sequence types: {sequence_types}")

    all_columns = []
    for seq_type in sequence_types:
        for metric_name, spec in thresholds.items():
            parsed = parse_threshold_spec(spec)
            col_name = build_column_name(seq_type, parsed["column_prefix"], metric_name)
            all_columns.append(col_name)
        all_columns.append(f"{seq_type}_complex_pdb_path")

    agg_dict = {col: keep_lists_separate for col in all_columns if col in df.columns}

    existing_columns = [col for col in all_columns if col in df.columns]
    if not existing_columns:
        logger.warning("No sequence type columns found in dataframe")
        return pd.DataFrame()

    df_grouped = df.groupby(groupby_cols, dropna=False)[existing_columns].agg(agg_dict).reset_index()

    for seq_type in sequence_types:
        add_success_rate_columns(df_grouped, seq_type, thresholds, "success", metric_suffix)

        first_col = f"{seq_type}_complex_pdb_path"
        df_grouped[f"_res_{seq_type}_original_samples_{metric_suffix}"] = df_grouped.apply(
            lambda row, fc=first_col: len(row[fc]) if fc in row.index else 0, axis=1
        )
        df_grouped[f"_res_{seq_type}_total_redesigns_{metric_suffix}"] = df_grouped.apply(
            lambda row, fc=first_col: sum(len(sample) for sample in row[fc]) if fc in row.index else 0,
            axis=1,
        )

    columns_to_drop = [col for col in df_grouped.columns if col in existing_columns]
    df_grouped.drop(columns=columns_to_drop, inplace=True)

    csv_prefix = "res_filter_ligand_pass_" if is_ligand else "res_filter_binder_pass_"
    csv_path = os.path.join(path_store_results, f"{csv_prefix}{metric_suffix}.csv")
    df_grouped.to_csv(csv_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
    logger.debug(f"Saved results to {csv_path}")

    save_success_criteria_json(
        success_thresholds=thresholds,
        path_store_results=path_store_results,
        filter_name=f"{csv_prefix}{metric_suffix}",
    )

    return df_grouped


def compute_filter_ligand_pass_rate(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    success_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: calls :func:`compute_filter_pass_rate` with ``result_type="ligand_binder"``."""
    return compute_filter_pass_rate(
        df,
        groupby_cols,
        path_store_results,
        metric_suffix,
        success_thresholds=success_thresholds,
        result_type="ligand_binder",
    )


# =============================================================================
# Per-Sample Logging Helpers
# =============================================================================


def _log_binder_sample_redesigns(
    seq_type: str,
    sample_id: str,
    passed: bool,
    n_passing: int,
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> None:
    """Log per-redesign metric values with pass/fail status for a binder sample.

    For self (1 redesign) produces a single-line log.  For mpnn/mpnn_fixed
    (multiple redesigns) produces a header line + one indented line per
    redesign so it's easy to see which redesign drives success.
    """
    # Determine number of redesigns
    lengths = [len(v) for v in sample_metric_values.values() if isinstance(v, list)]
    n_redesigns = min(lengths) if lengths else 0

    status_icon = "PASS" if passed else "FAIL"

    if n_redesigns <= 1:
        # Single redesign: compact one-liner
        redesign_passed = _check_binder_redesign_i(
            0,
            sample_metric_values,
            parsed_thresholds,
        )
        icon = "+" if redesign_passed else "-"
        metrics_str = _format_binder_redesign_metrics(
            0,
            sample_metric_values,
            parsed_thresholds,
        )
        logger.info(f"  [{seq_type}] {status_icon}  {sample_id}  [{icon}] {metrics_str}")
    else:
        # Multiple redesigns: header + per-redesign lines
        logger.info(f"  [{seq_type}] {status_icon}  {sample_id}  ({n_passing}/{n_redesigns} redesigns pass)")
        for i in range(n_redesigns):
            redesign_passed = _check_binder_redesign_i(
                i,
                sample_metric_values,
                parsed_thresholds,
            )
            icon = "+" if redesign_passed else "-"
            metrics_str = _format_binder_redesign_metrics(
                i,
                sample_metric_values,
                parsed_thresholds,
            )
            logger.info(f"             [{icon}] seq_{i}: {metrics_str}")


def _check_binder_redesign_i(
    i: int,
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> bool:
    """Check if binder redesign *i* passes all threshold criteria."""
    redesign_values = {}
    for metric_name in parsed_thresholds:
        if metric_name in sample_metric_values:
            values = sample_metric_values[metric_name]
            if isinstance(values, list) and i < len(values):
                redesign_values[metric_name] = values[i]
    return check_redesign_passes_all_thresholds(redesign_values, parsed_thresholds)


def _format_binder_redesign_metrics(
    i: int,
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> str:
    """Format binder metric values for redesign *i* into a compact string.

    Each metric shows its value and (ok)/(X) to indicate pass/fail.
    """
    parts = []
    for metric_name, spec in parsed_thresholds.items():
        if metric_name in sample_metric_values:
            values = sample_metric_values[metric_name]
            if isinstance(values, list) and i < len(values):
                value = values[i]
                if isinstance(value, (int, float)) and not (
                    isinstance(value, float) and (np.isnan(value) or np.isinf(value))
                ):
                    ok = evaluate_threshold(value, spec["threshold"], spec["op"], spec["scale"])
                    mark = "ok" if ok else "X"
                    parts.append(f"{metric_name}={value:.3f}({mark})")
                else:
                    parts.append(f"{metric_name}=NaN(X)")
    return "  ".join(parts)
