"""
Monomer success analysis functions and constants.

This module contains all the functions and constants related to monomer success
criteria evaluation, including threshold configuration, filtering, and pass rate
computation for monomer designability and codesignability metrics.
"""

import json
import os
from functools import reduce

import numpy as np
import pandas as pd
from loguru import logger

# Import shared utilities from binder_analysis_utils
from proteinfoundation.result_analysis.analysis_utils import (
    FLOAT_FORMAT_PD,
    SEP_CSV_PD,
    compute_mean_for_values,
    compute_n_passed_for_values,
    compute_pass_rate_for_values,
    compute_std_for_values,
    parse_threshold_spec,
)

# Import monomer analysis utilities and thresholds
from proteinfoundation.result_analysis.monomer_analysis_utils import (
    DEFAULT_MONOMER_CODESIGNABILITY_THRESHOLDS,
    DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS,
    build_monomer_column_name,
    normalize_monomer_thresholds,
    resolve_monomer_column,
)

# =============================================================================
# Length-Based Analysis Functions
# =============================================================================


def compute_pass_rates_by_length(
    df: pd.DataFrame,
    thresholds: dict | None = None,
    metric_type: str = "designability",
    length_column: str = "L",
    path_store_results: str | None = None,
    additional_groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute designability/codesignability pass rates grouped by config columns.

    Groups samples by the provided groupby columns (e.g. run_name, checkpoint,
    sampling params) and computes what percentage of samples in each group pass
    the specified RMSD thresholds. Length (L) is NOT used as a groupby column;
    samples of all lengths are aggregated together within each config group.

    Args:
        df: Input dataframe containing monomer evaluation results
        thresholds: Dictionary specifying thresholds. If None, uses defaults.
        metric_type: Type of metric ("designability" or "codesignability")
        length_column: Name of the column containing sample lengths (unused for groupby,
            kept for API compatibility)
        path_store_results: Optional path to store results
        additional_groupby_cols: Columns to group by (e.g. config/hyperparameter columns)

    Returns:
        DataFrame with groupby columns plus:
            - n_samples: Number of samples in this group
            - _res_{metric_type}_{mode}_{model}: Pass rate for each threshold
            - _res_{metric_type}_mean_scrmsd_{mode}_{model}: Mean RMSD
            - _res_{metric_type}_std_scrmsd_{mode}_{model}: Std of RMSD
    """
    # Use default thresholds if not provided
    if thresholds is None:
        if metric_type == "designability":
            thresholds = DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS.copy()
        else:
            thresholds = DEFAULT_MONOMER_CODESIGNABILITY_THRESHOLDS.copy()

    # Normalize thresholds
    thresholds = normalize_monomer_thresholds(thresholds)

    logger.debug(f"Computing {metric_type} pass rates")
    for mode, models in thresholds.items():
        for model, spec in models.items():
            logger.debug(f"  {mode}/{model}: {spec['op']} {spec['threshold']}")

    # Build groupby columns from additional_groupby_cols only (no length groupby)
    groupby_cols = list(additional_groupby_cols) if additional_groupby_cols else []

    if not groupby_cols:
        logger.warning("No groupby columns provided, computing global pass rates")
        # Create a dummy column so groupby still works
        df = df.copy()
        df["_dummy_group"] = "all"
        groupby_cols = ["_dummy_group"]

    results = []

    # For each mode/model combination, compute pass rates
    for mode, models in thresholds.items():
        for model, spec in models.items():
            canonical = build_monomer_column_name(metric_type, mode, model)
            col_name = resolve_monomer_column(canonical, df.columns)

            if col_name is None:
                logger.warning(f"Column {canonical} not found, skipping")
                continue

            threshold = spec["threshold"]
            op = spec["op"]

            # Group by config columns
            df_grouped = df.groupby(groupby_cols, dropna=False).agg({col_name: list}).reset_index()

            col_suffix = f"{mode}_{model}"

            df_grouped[f"_res_{metric_type}_pass_rate_{col_suffix}"] = df_grouped[col_name].apply(
                lambda v: compute_pass_rate_for_values(v, threshold, op)
            )
            df_grouped[f"_res_{metric_type}_n_passed_{col_suffix}"] = df_grouped[col_name].apply(
                lambda v: compute_n_passed_for_values(v, threshold, op)
            )
            df_grouped[f"_res_{metric_type}_mean_scrmsd_{col_suffix}"] = df_grouped[col_name].apply(
                compute_mean_for_values
            )
            df_grouped[f"_res_{metric_type}_std_scrmsd_{col_suffix}"] = df_grouped[col_name].apply(
                compute_std_for_values
            )
            df_grouped[f"_res_{metric_type}_n_samples_{col_suffix}"] = df_grouped[col_name].apply(len)

            # Drop the list column
            df_grouped = df_grouped.drop(columns=[col_name], errors="ignore")
            results.append(df_grouped)

    if not results:
        logger.warning(f"No {metric_type} pass rates computed")
        return pd.DataFrame()

    # Merge all results
    result = results[0]
    for r in results[1:]:
        # Only merge metric columns, avoid duplicating base columns
        metric_cols = [c for c in r.columns if c.startswith("_res_")]
        result = pd.merge(
            result,
            r[groupby_cols + metric_cols],
            on=groupby_cols,
            how="outer",
        )

    result = result.reset_index(drop=True)

    # Remove dummy column if it was added
    if "_dummy_group" in result.columns:
        result = result.drop(columns=["_dummy_group"])

    if path_store_results is not None:
        output_path = os.path.join(path_store_results, f"res_monomer_{metric_type}_by_length.csv")
        result.to_csv(output_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
        logger.debug(f"Saved {metric_type} pass rates to {output_path}")

    return result


def compute_designability_by_length(
    df: pd.DataFrame,
    thresholds: dict | None = None,
    length_column: str = "L",
    path_store_results: str | None = None,
    additional_groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute designability pass rates aggregated by exact sample length.

    Convenience wrapper for compute_pass_rates_by_length with metric_type="designability".

    Args:
        df: Input dataframe containing monomer evaluation results
        thresholds: Dictionary specifying thresholds. If None, uses defaults.
        length_column: Name of the column containing sample lengths
        path_store_results: Optional path to store results
        additional_groupby_cols: Additional columns to group by besides length

    Returns:
        DataFrame with pass rates by exact length
    """
    return compute_pass_rates_by_length(
        df=df,
        thresholds=thresholds,
        metric_type="designability",
        length_column=length_column,
        path_store_results=path_store_results,
        additional_groupby_cols=additional_groupby_cols,
    )


def compute_codesignability_by_length(
    df: pd.DataFrame,
    thresholds: dict | None = None,
    length_column: str = "L",
    path_store_results: str | None = None,
    additional_groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute codesignability pass rates aggregated by exact sample length.

    Convenience wrapper for compute_pass_rates_by_length with metric_type="codesignability".

    Args:
        df: Input dataframe containing monomer evaluation results
        thresholds: Dictionary specifying thresholds. If None, uses defaults.
        length_column: Name of the column containing sample lengths
        path_store_results: Optional path to store results
        additional_groupby_cols: Additional columns to group by besides length

    Returns:
        DataFrame with pass rates by exact length
    """
    return compute_pass_rates_by_length(
        df=df,
        thresholds=thresholds,
        metric_type="codesignability",
        length_column=length_column,
        path_store_results=path_store_results,
        additional_groupby_cols=additional_groupby_cols,
    )


def compute_single_designability_by_length(
    df: pd.DataFrame,
    thresholds: dict | None = None,
    length_column: str = "L",
    path_store_results: str | None = None,
    additional_groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute single-MPNN designability pass rates aggregated by exact sample length.

    Like designability_by_length but uses only the first ProteinMPNN sequence
    rather than the best over all redesigned sequences.

    Convenience wrapper for compute_pass_rates_by_length with metric_type="single_designability".

    Args:
        df: Input dataframe containing monomer evaluation results
        thresholds: Dictionary specifying thresholds. If None, uses designability defaults.
        length_column: Name of the column containing sample lengths
        path_store_results: Optional path to store results
        additional_groupby_cols: Additional columns to group by besides length

    Returns:
        DataFrame with pass rates by exact length
    """
    return compute_pass_rates_by_length(
        df=df,
        thresholds=thresholds,
        metric_type="single_designability",
        length_column=length_column,
        path_store_results=path_store_results,
        additional_groupby_cols=additional_groupby_cols,
    )


def compute_length_statistics(
    df: pd.DataFrame,
    length_column: str = "L",
    groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute basic statistics about sample lengths.

    Args:
        df: Input dataframe
        length_column: Name of the column containing sample lengths
        groupby_cols: Optional columns to group by

    Returns:
        DataFrame with length statistics (count, mean, std, min, max, quartiles)
    """
    if length_column not in df.columns:
        logger.warning(f"Length column '{length_column}' not found")
        return pd.DataFrame()

    if groupby_cols:
        stats = (
            df.groupby(groupby_cols, dropna=False)[length_column]
            .agg(
                [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "max",
                    lambda x: x.quantile(0.25),
                    lambda x: x.quantile(0.50),
                    lambda x: x.quantile(0.75),
                ]
            )
            .reset_index()
        )
        stats.columns = list(groupby_cols) + [
            "n_samples",
            "mean_length",
            "std_length",
            "min_length",
            "max_length",
            "q25_length",
            "median_length",
            "q75_length",
        ]
    else:
        stats = (
            df[length_column]
            .agg(
                [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
            )
            .to_frame()
            .T
        )
        stats.columns = [
            "n_samples",
            "mean_length",
            "std_length",
            "min_length",
            "max_length",
        ]
        stats["q25_length"] = df[length_column].quantile(0.25)
        stats["median_length"] = df[length_column].quantile(0.50)
        stats["q75_length"] = df[length_column].quantile(0.75)

    return stats


# =============================================================================
# Filtering Functions
# =============================================================================


def filter_monomer_by_designability(
    df: pd.DataFrame,
    thresholds: dict | None = None,
    metric_type: str = "designability",
    require_all: bool = False,
    path_store_results: str | None = None,
    filter_name: str | None = None,
) -> pd.DataFrame:
    """Filter monomer samples by designability/codesignability thresholds.

    This function filters monomer samples based on scRMSD thresholds for different
    modes and folding models.

    Args:
        df: Input dataframe containing monomer evaluation results
        thresholds: Dictionary specifying thresholds. Format:
            {
                "mode": {
                    "model": {"threshold": float, "op": str}
                }
            }
            If None, uses DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS.
        metric_type: Type of metric to filter by ("designability" or "codesignability")
        require_all: If True, sample must pass ALL specified thresholds.
                    If False, sample must pass ANY specified threshold.
        path_store_results: Optional path to store filtered samples CSV
        filter_name: Name for logging and output files (auto-generated if None)

    Returns:
        Filtered DataFrame containing only samples that pass the thresholds

    Example:
        # Filter by CA designability at 2.0Å
        thresholds = {
            "ca": {"esmfold": {"threshold": 2.0, "op": "<="}}
        }
        df_filtered = filter_monomer_by_designability(df, thresholds)

        # Filter requiring BOTH ca and all_atom to pass
        thresholds = {
            "ca": {"esmfold": {"threshold": 2.0, "op": "<="}},
            "all_atom": {"esmfold": {"threshold": 2.5, "op": "<="}}
        }
        df_filtered = filter_monomer_by_designability(df, thresholds, require_all=True)
    """
    # Use defaults if not provided
    if thresholds is None:
        if metric_type == "designability":
            thresholds = DEFAULT_MONOMER_DESIGNABILITY_THRESHOLDS.copy()
        else:
            thresholds = DEFAULT_MONOMER_CODESIGNABILITY_THRESHOLDS.copy()

    # Normalize thresholds
    thresholds = normalize_monomer_thresholds(thresholds)

    if filter_name is None:
        filter_name = f"monomer_{metric_type}"

    logger.debug(f"Filtering monomers by {metric_type}")
    logger.debug(f"Require all thresholds: {require_all}")

    # Build column mapping and log thresholds
    column_specs = []  # List of (column_name, threshold, op)
    for mode, models in thresholds.items():
        for model, spec in models.items():
            canonical = build_monomer_column_name(metric_type, mode, model)
            col_name = resolve_monomer_column(canonical, df.columns)

            if col_name is not None:
                column_specs.append((col_name, spec["threshold"], spec["op"]))
                logger.debug(f"  {col_name} {spec['op']} {spec['threshold']}")
            else:
                logger.warning(f"  Column {canonical} not found in dataframe")

    if not column_specs:
        logger.warning(f"No valid columns found for {metric_type}, returning empty dataframe")
        return pd.DataFrame()

    # Create filter mask
    if require_all:
        # Must pass ALL thresholds
        combined_mask = pd.Series([True] * len(df), index=df.index)
        for col_name, threshold, op in column_specs:
            if op == "<=":
                combined_mask = combined_mask & (df[col_name] <= threshold)
            elif op == "<":
                combined_mask = combined_mask & (df[col_name] < threshold)
            elif op == ">=":
                combined_mask = combined_mask & (df[col_name] >= threshold)
            elif op == ">":
                combined_mask = combined_mask & (df[col_name] > threshold)
            elif op == "==":
                combined_mask = combined_mask & (df[col_name] == threshold)
    else:
        # Must pass ANY threshold
        combined_mask = pd.Series([False] * len(df), index=df.index)
        for col_name, threshold, op in column_specs:
            if op == "<=":
                combined_mask = combined_mask | (df[col_name] <= threshold)
            elif op == "<":
                combined_mask = combined_mask | (df[col_name] < threshold)
            elif op == ">=":
                combined_mask = combined_mask | (df[col_name] >= threshold)
            elif op == ">":
                combined_mask = combined_mask | (df[col_name] > threshold)
            elif op == "==":
                combined_mask = combined_mask | (df[col_name] == threshold)

    df_filtered = df[combined_mask]

    # Log results
    logic_type = "ALL" if require_all else "ANY"
    logger.debug(f"{filter_name} filtering ({logic_type} logic): {len(df_filtered)}/{len(df)} samples passed")

    # Save results if path provided
    if path_store_results is not None:
        output_path = os.path.join(path_store_results, f"monomer_filtered_samples_{filter_name}.csv")
        df_filtered.to_csv(
            output_path,
            sep=SEP_CSV_PD,
            index=False,
            float_format=FLOAT_FORMAT_PD,
        )
        logger.debug(f"Saved filtered samples to {output_path}")

        # Save thresholds as JSON for reproducibility
        save_monomer_thresholds_json(
            thresholds=thresholds,
            metric_type=metric_type,
            require_all=require_all,
            path_store_results=path_store_results,
            filter_name=filter_name,
        )

    return df_filtered


def filter_monomer_by_single_threshold(
    df: pd.DataFrame,
    mode: str,
    model: str,
    threshold: float,
    metric_type: str = "designability",
    op: str = "<=",
) -> pd.DataFrame:
    """Filter monomer samples by a single mode/model threshold.

    Convenience function for filtering by just one threshold.

    Args:
        df: Input dataframe containing monomer evaluation results
        mode: RMSD mode ("ca", "bb3o", "all_atom")
        model: Folding model name ("esmfold", "colabfold", etc.)
        threshold: RMSD threshold value
        metric_type: Type of metric ("designability" or "codesignability")
        op: Comparison operator ("<=", "<", ">=", ">", "==")

    Returns:
        Filtered DataFrame
    """
    canonical = build_monomer_column_name(metric_type, mode, model)
    col_name = resolve_monomer_column(canonical, df.columns)

    if col_name is None:
        logger.warning(f"Column {canonical} not found, returning empty dataframe")
        return pd.DataFrame()

    if op == "<=":
        mask = df[col_name] <= threshold
    elif op == "<":
        mask = df[col_name] < threshold
    elif op == ">=":
        mask = df[col_name] >= threshold
    elif op == ">":
        mask = df[col_name] > threshold
    else:
        mask = df[col_name] == threshold

    df_filtered = df[mask]
    logger.debug(f"Filtered by {col_name} {op} {threshold}: {len(df_filtered)}/{len(df)} samples passed")

    return df_filtered


# =============================================================================
# JSON Save/Load Functions
# =============================================================================


def save_monomer_thresholds_json(
    thresholds: dict,
    metric_type: str,
    require_all: bool,
    path_store_results: str,
    filter_name: str,
) -> str:
    """Save monomer thresholds to a JSON file for reproducibility.

    Args:
        thresholds: Dictionary specifying thresholds
        metric_type: Type of metric ("designability" or "codesignability")
        require_all: Whether all thresholds must pass
        path_store_results: Directory path to store the JSON file
        filter_name: Name for the filter

    Returns:
        Path to the saved JSON file
    """
    output_data = {
        "filter_name": filter_name,
        "metric_type": metric_type,
        "require_all": require_all,
        "thresholds": {},
    }

    # Parse all thresholds
    for mode, models in thresholds.items():
        output_data["thresholds"][mode] = {}
        for model, spec in models.items():
            if isinstance(spec, dict):
                output_data["thresholds"][mode][model] = spec
            else:
                output_data["thresholds"][mode][model] = parse_threshold_spec(spec)

    filename = f"monomer_thresholds_{filter_name}.json"
    output_path = os.path.join(path_store_results, filename)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.debug(f"Saved monomer thresholds to {output_path}")
    return output_path


def load_monomer_thresholds_json(json_path: str) -> dict:
    """Load monomer thresholds from a JSON file.

    Args:
        json_path: Path to the JSON file containing monomer thresholds

    Returns:
        Dictionary with thresholds and metadata
    """
    with open(json_path) as f:
        data = json.load(f)

    logger.debug(f"Loaded monomer thresholds from {json_path}")
    logger.debug(f"Filter name: {data.get('filter_name', 'unknown')}")
    logger.debug(f"Metric type: {data.get('metric_type', 'unknown')}")
    logger.debug(f"Require all: {data.get('require_all', False)}")

    return data


# =============================================================================
# Pass Rate Computation Functions
# =============================================================================


def compute_monomer_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    thresholds: dict,
    metric_type: str = "designability",
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Compute pass rates for monomer samples based on thresholds.

    For each mode/model combination (e.g. ca/esmfold), groups samples by
    groupby_cols and computes:
        - pass_rate: fraction of samples passing the threshold
        - mean_scrmsd: mean scRMSD across all samples
        - n_samples: number of samples in the group

    Output columns follow the pattern:
        _res_{metric_type}_pass_rate_{mode}_{model}
        _res_{metric_type}_mean_scrmsd_{mode}_{model}
        _res_{metric_type}_n_samples_{mode}_{model}

    Args:
        df: Input dataframe containing monomer evaluation results
        groupby_cols: Columns to group results by
        thresholds: Dictionary specifying thresholds
        metric_type: Type of metric ("designability", "single_designability",
            or "codesignability")
        path_store_results: Optional path to store results

    Returns:
        DataFrame with pass rates for each threshold category
    """
    logger.debug(f"Computing monomer {metric_type} pass rates")

    # Normalize thresholds
    thresholds = normalize_monomer_thresholds(thresholds)

    results = []

    # For each mode/model combination, compute pass rates
    for mode, models in thresholds.items():
        for model, spec in models.items():
            canonical = build_monomer_column_name(metric_type, mode, model)
            col_name = resolve_monomer_column(canonical, df.columns)

            if col_name is None:
                logger.warning(f"Column {canonical} not found, skipping")
                continue

            # Group and compute pass rate
            df_grouped = df.groupby(groupby_cols, dropna=False)[col_name].agg(list).reset_index()

            threshold = spec["threshold"]
            op = spec["op"]
            col_suffix = f"{mode}_{model}"

            df_grouped[f"_res_{metric_type}_pass_rate_{col_suffix}"] = df_grouped[col_name].apply(
                lambda v: compute_pass_rate_for_values(v, threshold, op)
            )
            df_grouped[f"_res_{metric_type}_mean_scrmsd_{col_suffix}"] = df_grouped[col_name].apply(
                compute_mean_for_values
            )
            df_grouped[f"_res_{metric_type}_n_samples_{col_suffix}"] = df_grouped[col_name].apply(len)

            # Drop the list column
            df_grouped = df_grouped.drop(columns=[col_name])
            results.append(df_grouped)

    if not results:
        logger.warning(f"No {metric_type} pass rates computed")
        return pd.DataFrame()

    # Merge all results
    result = reduce(lambda left, right: pd.merge(left, right, on=groupby_cols, how="outer"), results)

    if path_store_results is not None:
        output_path = os.path.join(path_store_results, f"res_monomer_{metric_type}_pass_rates.csv")
        result.to_csv(output_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
        logger.debug(f"Saved {metric_type} pass rates to {output_path}")

    return result


def compute_monomer_combined_pass_rate(
    df: pd.DataFrame,
    groupby_cols: list[str],
    thresholds: dict,
    metric_type: str = "designability",
    require_all: bool = True,
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Compute combined pass rates where samples must pass multiple thresholds.

    Args:
        df: Input dataframe containing monomer evaluation results
        groupby_cols: Columns to group results by
        thresholds: Dictionary specifying thresholds
        metric_type: Type of metric ("designability" or "codesignability")
        require_all: If True, sample must pass ALL thresholds. If False, ANY.
        path_store_results: Optional path to store results

    Returns:
        DataFrame with combined pass rates
    """
    logger.debug(f"Computing monomer combined {metric_type} pass rates")
    logic_type = "ALL" if require_all else "ANY"
    logger.debug(f"Logic: {logic_type}")

    # Normalize thresholds
    thresholds = normalize_monomer_thresholds(thresholds)

    # Build list of column specs
    column_specs = []
    for mode, models in thresholds.items():
        for model, spec in models.items():
            canonical = build_monomer_column_name(metric_type, mode, model)
            col_name = resolve_monomer_column(canonical, df.columns)
            if col_name is not None:
                column_specs.append((col_name, spec["threshold"], spec["op"], f"{mode}_{model}"))
            else:
                logger.warning(f"Column {canonical} not found, skipping")

    if not column_specs:
        logger.warning(f"No valid columns found for combined {metric_type} pass rate")
        return pd.DataFrame()

    # Get all required columns
    required_cols = [cs[0] for cs in column_specs]

    # Group by and aggregate
    df_grouped = df.groupby(groupby_cols, dropna=False)[required_cols].agg(list).reset_index()

    def compute_combined_pass_rate(row):
        """Compute pass rate for a grouped row."""
        # Get number of samples from first column
        n_samples = len(row[required_cols[0]])

        passed_count = 0
        for i in range(n_samples):
            passes_all = True
            passes_any = False

            for col_name, threshold, op, _ in column_specs:
                value = row[col_name][i]
                if value is None or np.isnan(value):
                    passes_all = False
                    continue

                if op == "<=":
                    passes = value <= threshold
                elif op == "<":
                    passes = value < threshold
                elif op == ">=":
                    passes = value >= threshold
                elif op == ">":
                    passes = value > threshold
                else:
                    passes = value == threshold

                if passes:
                    passes_any = True
                else:
                    passes_all = False

            if (require_all and passes_all) or (not require_all and passes_any):
                passed_count += 1

        return passed_count / n_samples if n_samples > 0 else 0.0

    # Compute combined pass rate
    df_grouped[f"_res_{metric_type}_combined_pass_rate_{logic_type.lower()}"] = df_grouped.apply(
        compute_combined_pass_rate, axis=1
    )
    df_grouped[f"_res_{metric_type}_combined_n_samples"] = df_grouped[required_cols[0]].apply(len)

    # Drop the list columns
    df_grouped = df_grouped.drop(columns=required_cols)

    if path_store_results is not None:
        output_path = os.path.join(path_store_results, f"res_monomer_{metric_type}_combined_pass_rates.csv")
        df_grouped.to_csv(output_path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
        logger.debug(f"Saved combined {metric_type} pass rates to {output_path}")

    return df_grouped
