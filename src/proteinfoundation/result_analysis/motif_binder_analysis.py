"""
Motif binder success analysis: filtering and pass rate computation.

Combines binder success criteria (ipAE, pLDDT, scRMSD) with motif-specific
criteria (motif RMSD, sequence recovery, optional clash detection) so that
a sample is "successful" only when at least one redesign passes ALL criteria
simultaneously.

This module parallels ``binder_analysis.py`` but uses the joint evaluation
logic from ``motif_binder_analysis_utils.py``.  Individual motif metric
pass rates (motif RMSD pred, seq recovery, clashes) are also computed here.

YAML override: ``aggregation.motif_binder_success_thresholds``
"""

import json
import os

import pandas as pd
from loguru import logger

from proteinfoundation.result_analysis.analysis_utils import (
    FLOAT_FORMAT_PD,
    SEP_CSV_PD,
    SEQUENCE_TYPES,
    coerce_to_list,
    compute_mean_for_values,
    compute_n_passed_for_values,
    compute_pass_rate_for_values,
    compute_std_for_values,
    evaluate_threshold,
    extract_list_values,
    get_sample_label,
    keep_lists_separate,
    parse_threshold_spec,
    save_filtered_csv,
)
from proteinfoundation.result_analysis.binder_analysis_utils import build_column_name
from proteinfoundation.result_analysis.motif_binder_analysis_utils import (
    check_redesign_passes_binder_and_motif,
    check_sample_has_passing_redesign,
    count_passing_redesigns,
    format_success_criteria_for_logging,
    get_default_motif_binder_success,
    parse_motif_binder_success,
)

# =============================================================================
# Filtering Functions
# =============================================================================


def filter_by_motif_binder_success(
    df: pd.DataFrame,
    seq_type: str,
    success_thresholds: dict | None = None,
    result_type: str = "motif_protein_binder",
    path_store_results: str | None = None,
    filter_name: str = "motif_binder_success",
    save_json: bool = True,
) -> pd.DataFrame:
    """Filter samples where ANY redesign passes ALL binder AND motif criteria.

    For each row, the binder ``_all`` columns and motif ``_all`` columns are
    indexed jointly: redesign *i* must pass every binder threshold AND every
    motif criterion simultaneously.

    Args:
        df: DataFrame with per-sample evaluation columns.
        seq_type: Sequence type (``"self"``, ``"mpnn"``, ``"mpnn_fixed"``).
        success_thresholds: Combined ``{binder: ..., motif: ...}`` dict.
            If None, uses defaults for *result_type*.
        result_type: Used to select defaults when *success_thresholds* is None.
        path_store_results: Optional directory to save filtered CSV.
        filter_name: Label for logging and output filenames.
        save_json: Whether to save criteria JSON alongside CSV.

    Returns:
        Filtered DataFrame (only successful samples).
    """
    if success_thresholds is None:
        success_thresholds = get_default_motif_binder_success(result_type)

    parsed_binder, resolved_motif = parse_motif_binder_success(success_thresholds, seq_type)

    logger.debug(f"Filtering by {filter_name} for seq_type={seq_type}")
    logger.debug(format_success_criteria_for_logging(success_thresholds, seq_type))

    # Map binder metric names to DataFrame column names
    binder_col_map = _build_binder_column_mapping(parsed_binder, seq_type, df.columns)
    if not binder_col_map:
        logger.warning(f"No valid binder columns for {seq_type}, returning empty df")
        return pd.DataFrame()

    # Map motif criteria columns (check existence)
    motif_col_map = _build_motif_column_mapping(resolved_motif, df.columns)

    # Evaluate each sample
    success_mask = []
    for idx, row in df.iterrows():
        try:
            binder_lists = extract_list_values(row, binder_col_map)
            motif_lists = _extract_motif_values(row, motif_col_map, binder_lists)
            passed = check_sample_has_passing_redesign(
                binder_lists,
                motif_lists,
                parsed_binder,
                resolved_motif,
            )
            n_passing = count_passing_redesigns(
                binder_lists,
                motif_lists,
                parsed_binder,
                resolved_motif,
            )
            success_mask.append(passed)

            # Per-sample logging with per-redesign breakdown
            sample_id = get_sample_label(row)
            _log_sample_redesigns(
                seq_type=seq_type,
                sample_id=sample_id,
                passed=passed,
                n_passing=n_passing,
                binder_lists=binder_lists,
                motif_lists=motif_lists,
                parsed_binder=parsed_binder,
                resolved_motif=resolved_motif,
            )
        except Exception as e:
            logger.warning(f"Error evaluating sample {idx} for {seq_type}: {e}")
            success_mask.append(False)

    df_filtered = df[success_mask]
    n_passed = len(df_filtered)
    n_total = len(df)
    if n_total > 0:
        logger.info(f"  [{seq_type}] Result: {n_passed}/{n_total} samples passed ({n_passed / n_total * 100:.0f}%)")
    else:
        logger.info(f"  [{seq_type}] Result: 0/0 samples (no data)")

    if path_store_results is not None:
        save_filtered_csv(df_filtered, path_store_results, filter_name, seq_type)
        if save_json:
            save_motif_binder_success_json(
                success_thresholds=success_thresholds,
                path_store_results=path_store_results,
                filter_name=filter_name,
                seq_type=seq_type,
            )

    return df_filtered


# =============================================================================
# Pass Rate Computation
# =============================================================================


def compute_motif_binder_pass_rate(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str,
    success_thresholds: dict | None = None,
    result_type: str = "motif_protein_binder",
) -> pd.DataFrame:
    """Compute combined binder+motif pass rates grouped by config columns.

    For each group and sequence type, computes what fraction of samples have
    at least one redesign passing ALL binder AND motif criteria jointly.

    Args:
        df: Input DataFrame with evaluation results.
        groupby_cols: Columns to group by (e.g. run_name, checkpoint).
        path_store_results: Directory to save result CSV.
        metric_suffix: Suffix appended to output column names.
        success_thresholds: Combined ``{binder: ..., motif: ...}`` dict.
            If None, uses defaults for *result_type*.
        result_type: Used to select defaults.

    Returns:
        DataFrame with ``_res_{seq_type}_pass_rate_motif_binder_{suffix}``
        and sample count columns per group.
    """
    if success_thresholds is None:
        success_thresholds = get_default_motif_binder_success(result_type)

    # Build the list of all columns we need for groupby aggregation
    all_columns = set()
    for seq_type in SEQUENCE_TYPES:
        parsed_binder, resolved_motif = parse_motif_binder_success(success_thresholds, seq_type)
        for metric_name, spec in parsed_binder.items():
            col = build_column_name(seq_type, spec["column_prefix"], metric_name)
            all_columns.add(col)
        for criterion in resolved_motif:
            all_columns.add(criterion["column"])
        # Sample count column
        all_columns.add(f"{seq_type}_complex_pdb_path")

    existing_columns = sorted(col for col in all_columns if col in df.columns)
    if not existing_columns:
        logger.warning("No motif binder metric columns found in dataframe")
        return pd.DataFrame()

    # Group and aggregate: keep lists separate per sample
    agg_dict = dict.fromkeys(existing_columns, keep_lists_separate)
    df_grouped = df.groupby(groupby_cols, dropna=False)[existing_columns].agg(agg_dict).reset_index()

    # Compute pass rates per sequence type
    for seq_type in SEQUENCE_TYPES:
        parsed_binder, resolved_motif = parse_motif_binder_success(success_thresholds, seq_type)

        # Check required columns exist in grouped df
        binder_col_map = _build_binder_column_mapping(parsed_binder, seq_type, df_grouped.columns)
        motif_col_map = _build_motif_column_mapping(resolved_motif, df_grouped.columns)
        if not binder_col_map:
            continue

        pass_rate_col = f"_res_{seq_type}_pass_rate_motif_binder_{metric_suffix}"
        per_sample_col = f"_res_{seq_type}_per_sample_pass_motif_binder_{metric_suffix}"

        results = df_grouped.apply(
            lambda row: _compute_grouped_row_pass_rate(
                row,
                binder_col_map,
                motif_col_map,
                parsed_binder,
                resolved_motif,
            ),
            axis=1,
        )
        df_grouped[pass_rate_col] = results.apply(lambda x: x[0])
        df_grouped[per_sample_col] = results.apply(lambda x: x[1])

        # Sample and redesign counts
        count_col = f"{seq_type}_complex_pdb_path"
        if count_col in df_grouped.columns:
            df_grouped[f"_res_{seq_type}_original_samples_{metric_suffix}"] = df_grouped[count_col].apply(
                lambda v: len(v) if isinstance(v, list) else 0
            )
            df_grouped[f"_res_{seq_type}_total_redesigns_{metric_suffix}"] = df_grouped[count_col].apply(
                lambda v: sum(len(s) for s in v) if isinstance(v, list) else 0
            )

    # Drop raw metric columns, keep only _res_ results
    columns_to_drop = [c for c in df_grouped.columns if c in existing_columns]
    df_grouped.drop(columns=columns_to_drop, inplace=True)

    output_path = os.path.join(
        path_store_results,
        f"res_filter_motif_binder_pass_{metric_suffix}.csv",
    )
    df_grouped.to_csv(
        output_path,
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    logger.debug(f"Saved motif binder pass rates to {output_path}")

    return df_grouped


# =============================================================================
# Individual Motif Metric Pass Rates
# =============================================================================


def compute_motif_pred_metric_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    path_store_results: str,
    metric_suffix: str = "motif_binder",
    is_ligand: bool = False,
    success_thresholds: dict | None = None,
    result_type: str = "motif_protein_binder",
) -> pd.DataFrame:
    """Compute individual pass rates for motif-specific predicted metrics.

    Computes per-group pass rates for:
      - Motif RMSD (predicted) -- ``{seq_type}_motif_rmsd_pred``
      - Motif sequence recovery -- ``{seq_type}_motif_seq_rec``
      - Ligand clashes (if *is_ligand*) -- ``{seq_type}_has_ligand_clashes``

    Args:
        df: Input DataFrame.
        groupby_cols: Columns to group by.
        path_store_results: Directory to save CSV.
        metric_suffix: Suffix for output column names.
        is_ligand: If True, also compute clash-free rates.
        success_thresholds: Combined ``{binder: ..., motif: ...}`` dict.
            If None, uses defaults for *result_type*.
        result_type: Used to select defaults when *success_thresholds* is None.

    Returns:
        DataFrame with per-group pass rates and statistics.
    """
    if success_thresholds is None:
        success_thresholds = get_default_motif_binder_success(result_type)
    metric_defs = _motif_criteria_to_scalar_defs(success_thresholds, is_ligand)

    all_results = []

    for seq_type in SEQUENCE_TYPES:
        for col_template, threshold, op, stat_label in metric_defs:
            col = col_template.format(seq_type=seq_type)
            if col not in df.columns:
                continue

            grouped = df.groupby(groupby_cols, dropna=False)[col].agg(list).reset_index()

            prefix = f"_res_{seq_type}_{stat_label}_{metric_suffix}"
            grouped[f"{prefix}_pass_rate"] = grouped[col].apply(
                lambda v: compute_pass_rate_for_values(v, threshold, op)
            )
            grouped[f"{prefix}_n_passed"] = grouped[col].apply(lambda v: compute_n_passed_for_values(v, threshold, op))
            grouped[f"{prefix}_mean"] = grouped[col].apply(compute_mean_for_values)
            grouped[f"{prefix}_std"] = grouped[col].apply(compute_std_for_values)
            grouped[f"{prefix}_n_samples"] = grouped[col].apply(len)

            grouped = grouped.drop(columns=[col], errors="ignore")
            all_results.append(grouped)

    if not all_results:
        logger.warning("No motif pred metric columns found")
        return pd.DataFrame()

    # Merge all results on groupby columns
    result = all_results[0]
    for r in all_results[1:]:
        metric_cols = [c for c in r.columns if c.startswith("_res_")]
        result = pd.merge(
            result,
            r[groupby_cols + metric_cols],
            on=groupby_cols,
            how="outer",
        )

    output_path = os.path.join(
        path_store_results,
        f"res_motif_binder_pred_metrics_{metric_suffix}.csv",
    )
    result.to_csv(
        output_path,
        sep=SEP_CSV_PD,
        index=False,
        float_format=FLOAT_FORMAT_PD,
    )
    logger.debug(f"Saved motif pred metric pass rates to {output_path}")

    return result


# =============================================================================
# Per-Task Pass Rates
# =============================================================================


def compute_per_task_motif_binder_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    success_thresholds: dict,
    result_type: str = "motif_protein_binder",
    task_column: str = "task_name",
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Compute combined motif binder pass rates grouped by task + config columns.

    Like binder per-target analysis but using joint binder+motif criteria.

    Args:
        df: DataFrame with evaluation results.
        groupby_cols: Config columns to group by.
        success_thresholds: Combined ``{binder: ..., motif: ...}`` dict.
        result_type: Result type for default selection.
        task_column: Column containing task/target names.
        path_store_results: Optional directory to save CSV.

    Returns:
        DataFrame with per-task pass rates.
    """
    if task_column not in df.columns:
        logger.warning(f"'{task_column}' not found, skipping per-task analysis")
        return pd.DataFrame()

    task_groupby = [task_column] + [c for c in groupby_cols if c != task_column]

    result = compute_motif_binder_pass_rate(
        df=df,
        groupby_cols=task_groupby,
        path_store_results=path_store_results or ".",
        metric_suffix="per_task",
        success_thresholds=success_thresholds,
        result_type=result_type,
    )

    if result.empty:
        return pd.DataFrame()

    if path_store_results:
        output_path = os.path.join(
            path_store_results,
            "res_motif_binder_per_task_pass_rates.csv",
        )
        result.to_csv(
            output_path,
            sep=SEP_CSV_PD,
            index=False,
            float_format=FLOAT_FORMAT_PD,
        )
        logger.debug(f"Saved per-task pass rates to {output_path}")

    return result


# =============================================================================
# JSON Persistence
# =============================================================================


def save_motif_binder_success_json(
    success_thresholds: dict,
    path_store_results: str,
    filter_name: str = "motif_binder_success",
    seq_type: str | None = None,
    sequence_types: list[str] | None = None,
) -> str:
    """Save motif binder success criteria to JSON for reproducibility.

    Args:
        success_thresholds: Combined ``{binder: ..., motif: ...}`` dict.
        path_store_results: Directory to write the JSON file.
        filter_name: Label for the filter.
        seq_type: If provided, include in filename (per-seq-type save).
        sequence_types: If provided, include in output (combined save).

    Returns:
        Path to the saved JSON file.
    """
    # Parse binder thresholds for clean output
    raw_binder = success_thresholds.get("binder", {})
    parsed_binder = {}
    for metric_name, spec in raw_binder.items():
        parsed_binder[metric_name] = parse_threshold_spec(spec)

    output_data = {
        "filter_name": filter_name,
        "binder_thresholds": parsed_binder,
        "motif_criteria": success_thresholds.get("motif", []),
    }
    if seq_type is not None:
        output_data["seq_type"] = seq_type
    if sequence_types is not None:
        output_data["sequence_types"] = sequence_types

    suffix = f"_{seq_type}" if seq_type else ""
    filename = f"success_criteria_{filter_name}{suffix}.json"
    output_path = os.path.join(path_store_results, filename)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.debug(f"Saved motif binder success criteria to {output_path}")
    return output_path


# =============================================================================
# Internal Helpers
# =============================================================================


def _motif_criteria_to_scalar_defs(
    success_thresholds: dict,
    is_ligand: bool,
) -> list[tuple]:
    """Convert motif success criteria to (column_template, threshold, op, stat_label) tuples.

    The success criteria reference ``_all`` list columns (per-redesign).
    This function strips the ``_all`` suffix to produce scalar best-of
    column names used for individual pass-rate computation.
    """
    metric_defs = []
    for criterion in success_thresholds.get("motif", []):
        col_template = criterion["column"]
        if not is_ligand and "ligand" in col_template:
            continue
        scalar_template = col_template.replace("_all", "")
        # correct_motif_sequence -> motif_seq_rec (the scalar column name)
        scalar_template = scalar_template.replace("correct_motif_sequence", "motif_seq_rec")
        suffix = scalar_template.replace("{seq_type}_", "")
        stat_label = "no_ligand_clashes" if "ligand_clashes" in suffix else suffix
        metric_defs.append(
            (
                scalar_template,
                criterion["threshold"],
                criterion["op"],
                stat_label,
            )
        )
    return metric_defs


def _build_binder_column_mapping(
    parsed_binder: dict,
    seq_type: str,
    available_columns: pd.Index,
) -> dict[str, str]:
    """Map binder metric names to DataFrame column names.

    Returns:
        Dict of metric_name -> column_name for columns that exist.
    """
    mapping = {}
    available = set(available_columns)
    for metric_name, spec in parsed_binder.items():
        col = build_column_name(seq_type, spec["column_prefix"], metric_name)
        if col in available:
            mapping[metric_name] = col
        else:
            logger.debug(f"Binder column {col} not found, skipping")
    return mapping


def _build_motif_column_mapping(
    resolved_motif: list[dict],
    available_columns: pd.Index,
) -> dict[str, dict]:
    """Map motif criteria to DataFrame columns.

    Returns:
        Dict of column_name -> criterion dict for columns that exist.
    """
    mapping = {}
    available = set(available_columns)
    for criterion in resolved_motif:
        col = criterion["column"]
        if col in available:
            mapping[col] = criterion
        else:
            logger.warning(
                f"Motif criterion column '{col}' not found in DataFrame — all samples will fail this criterion"
            )
    return mapping


def _extract_motif_values(
    row: pd.Series,
    motif_col_map: dict[str, dict],
    binder_lists: dict[str, list],
) -> dict[str, list]:
    """Extract motif criterion values, broadcasting scalars to list length.

    Motif columns ending with ``_all`` are expected to be lists; others
    (e.g. ``correct_motif_sequence``) are scalar values broadcast to match
    the number of redesigns from binder columns.

    Returns:
        Dict of column_name -> list of values (one per redesign).
    """
    # Determine expected list length from binder data
    n_redesigns = 0
    for values in binder_lists.values():
        if isinstance(values, list) and len(values) > n_redesigns:
            n_redesigns = len(values)

    result = {}
    for col, criterion in motif_col_map.items():
        value = row.get(col)
        if value is None:
            continue
        parsed = coerce_to_list(value, col)
        if len(parsed) == 1 and n_redesigns > 1:
            parsed = parsed * n_redesigns
        elif len(parsed) != 1 and n_redesigns > 0 and len(parsed) != n_redesigns:
            logger.warning(
                f"Motif column {col} has {len(parsed)} values but expected "
                f"{n_redesigns} (from binder lists). Evaluation will be truncated to min length."
            )
        result[col] = parsed

    return result


def _compute_grouped_row_pass_rate(
    row: pd.Series,
    binder_col_map: dict[str, str],
    motif_col_map: dict[str, dict],
    parsed_binder: dict,
    resolved_motif: list[dict],
) -> tuple[float, list[int]]:
    """Compute pass rate for a grouped row (list of lists per sample).

    Each cell contains a list of lists: ``[sample0_redesigns, sample1_redesigns, ...]``.
    For each sample, checks if any redesign passes all criteria jointly.

    Returns:
        Tuple of ``(pass_rate, per_sample_pass_list)``.
    """
    # Determine number of samples from first binder column
    if not binder_col_map:
        return 0.0, []

    first_binder_col = next(iter(binder_col_map.values()))
    if first_binder_col not in row.index:
        return 0.0, []

    samples_data = row[first_binder_col]
    if not isinstance(samples_data, list):
        return 0.0, []

    n_samples = len(samples_data)
    if n_samples == 0:
        return 0.0, []

    per_sample_pass = []
    for sample_idx in range(n_samples):
        # Extract binder values for this sample (list of redesign values)
        binder_lists = {}
        for metric_name, col_name in binder_col_map.items():
            if col_name in row.index:
                sample_values = row[col_name]
                if isinstance(sample_values, list) and sample_idx < len(sample_values):
                    val = sample_values[sample_idx]
                    binder_lists[metric_name] = val if isinstance(val, list) else [val]
                else:
                    binder_lists[metric_name] = []

        # Extract motif values for this sample
        motif_lists = {}
        n_redesigns = max((len(v) for v in binder_lists.values()), default=0)
        for col in motif_col_map:
            if col in row.index:
                sample_values = row[col]
                if isinstance(sample_values, list) and sample_idx < len(sample_values):
                    val = sample_values[sample_idx]
                    if isinstance(val, list):
                        motif_lists[col] = val
                    else:
                        motif_lists[col] = [val] * max(n_redesigns, 1)
                else:
                    motif_lists[col] = []

        passed = check_sample_has_passing_redesign(
            binder_lists,
            motif_lists,
            parsed_binder,
            resolved_motif,
        )
        per_sample_pass.append(1 if passed else 0)

    pass_rate = sum(per_sample_pass) / n_samples
    return pass_rate, per_sample_pass


def _log_sample_redesigns(
    seq_type: str,
    sample_id: str,
    passed: bool,
    n_passing: int,
    binder_lists: dict[str, list],
    motif_lists: dict[str, list],
    parsed_binder: dict,
    resolved_motif: list[dict],
) -> None:
    """Log per-redesign metric values with pass/fail status for a sample.

    For self (1 redesign) produces a single-line log.  For mpnn/mpnn_fixed
    (multiple redesigns) produces a header line + one indented line per
    redesign so it's easy to see which redesign drives success.
    """
    # Determine number of redesigns
    all_lengths = [len(v) for v in binder_lists.values() if isinstance(v, list)] + [
        len(v) for v in motif_lists.values() if isinstance(v, list)
    ]
    n_redesigns = min(all_lengths) if all_lengths else 0

    status_icon = "PASS" if passed else "FAIL"

    if n_redesigns <= 1:
        # Single redesign: compact one-liner
        metrics_str = _format_redesign_metrics(
            0,
            binder_lists,
            motif_lists,
            parsed_binder,
            resolved_motif,
            seq_type,
        )
        redesign_passed = _check_redesign_i(
            0,
            binder_lists,
            motif_lists,
            parsed_binder,
            resolved_motif,
        )
        icon = "+" if redesign_passed else "-"
        logger.info(f"  [{seq_type}] {status_icon}  {sample_id}  [{icon}] {metrics_str}")
    else:
        # Multiple redesigns: header + per-redesign lines
        logger.info(f"  [{seq_type}] {status_icon}  {sample_id}  ({n_passing}/{n_redesigns} redesigns pass)")
        for i in range(n_redesigns):
            redesign_passed = _check_redesign_i(
                i,
                binder_lists,
                motif_lists,
                parsed_binder,
                resolved_motif,
            )
            icon = "+" if redesign_passed else "-"
            metrics_str = _format_redesign_metrics(
                i,
                binder_lists,
                motif_lists,
                parsed_binder,
                resolved_motif,
                seq_type,
            )
            logger.info(f"             [{icon}] seq_{i}: {metrics_str}")


def _check_redesign_i(
    i: int,
    binder_lists: dict[str, list],
    motif_lists: dict[str, list],
    parsed_binder: dict,
    resolved_motif: list[dict],
) -> bool:
    """Check if redesign *i* passes all binder AND motif criteria."""
    binder_values = {
        metric: values[i] for metric, values in binder_lists.items() if isinstance(values, list) and i < len(values)
    }
    motif_values = {
        col: values[i] for col, values in motif_lists.items() if isinstance(values, list) and i < len(values)
    }
    return check_redesign_passes_binder_and_motif(
        binder_values,
        motif_values,
        parsed_binder,
        resolved_motif,
    )


def _format_redesign_metrics(
    i: int,
    binder_lists: dict[str, list],
    motif_lists: dict[str, list],
    parsed_binder: dict,
    resolved_motif: list[dict],
    seq_type: str,
) -> str:
    """Format metric values for redesign *i* into a compact string.

    Each metric shows its value and a check/cross to indicate pass/fail
    against the threshold.
    """
    parts = []

    # Binder metrics
    for metric_name, spec in parsed_binder.items():
        if metric_name in binder_lists and i < len(binder_lists[metric_name]):
            value = binder_lists[metric_name][i]
            if isinstance(value, (int, float)):
                ok = evaluate_threshold(value, spec["threshold"], spec["op"], spec["scale"])
                mark = "ok" if ok else "X"
                parts.append(f"{metric_name}={value:.3f}({mark})")

    # Motif metrics
    for criterion in resolved_motif:
        col = criterion["column"]
        if col in motif_lists and i < len(motif_lists[col]):
            value = motif_lists[col][i]
            # Short column name (strip seq_type prefix)
            short_name = col
            if col.startswith(f"{seq_type}_"):
                short_name = col[len(f"{seq_type}_") :]
            if isinstance(value, bool):
                ok = evaluate_threshold(
                    float(value),
                    criterion["threshold"],
                    criterion["op"],
                )
                mark = "ok" if ok else "X"
                parts.append(f"{short_name}={value}({mark})")
            elif isinstance(value, (int, float)):
                ok = evaluate_threshold(value, criterion["threshold"], criterion["op"])
                mark = "ok" if ok else "X"
                parts.append(f"{short_name}={value:.3f}({mark})")

    return "  ".join(parts)
