"""
Motif analysis: individual pass rates, combined success filtering, and persistence.

Most metrics (full-structure designability/codesignability) use the *same*
column names as monomer evaluation, so we reuse ``compute_monomer_pass_rates``
and ``filter_monomer_by_single_threshold`` directly.

This file adds motif-unique functions:
  - Motif RMSD pass rates (no model dimension, just mode)
  - Motif sequence recovery pass rates
  - Motif-region designability/codesignability pass rates (different column prefix)
  - Per-task pass rates (grouped by task_name, like binder per-target analysis)
  - **Combined success filtering** using preset or custom criteria
  - Filtering by direct motif RMSD
  - Threshold / criteria JSON save/load
"""

import json
import os
from functools import reduce

import pandas as pd
from loguru import logger

from proteinfoundation.result_analysis.analysis_utils import (
    FLOAT_FORMAT_PD,
    SEP_CSV_PD,
    compute_mean_for_values,
    compute_n_passed_for_values,
    compute_pass_rate_for_values,
)
from proteinfoundation.result_analysis.monomer_analysis_utils import normalize_monomer_thresholds
from proteinfoundation.result_analysis.motif_analysis_utils import (
    DEFAULT_MOTIF_RMSD_THRESHOLDS,
    DEFAULT_MOTIF_SEQ_REC_THRESHOLD,
    build_motif_region_column_name,
    normalize_motif_rmsd_thresholds,
    resolve_success_criteria,
)

# =============================================================================
# Combined success filtering (like binder filter_by_success_thresholds)
# =============================================================================


def filter_by_motif_success(
    df: pd.DataFrame,
    criteria: list[dict],
    model: str = "esmfold",
) -> pd.DataFrame:
    """Filter to samples where ALL success criteria pass simultaneously.

    This is the motif equivalent of binder ``filter_by_success_thresholds``.

    Args:
        df: DataFrame with per-sample metric columns
        criteria: List of ``{column, threshold, op}`` dicts.
            Column names may contain ``{model}`` placeholder.
        model: Folding model name for ``{model}`` resolution.

    Returns:
        Filtered DataFrame (only successful samples)
    """
    resolved = resolve_success_criteria(criteria, model)

    rules = " AND ".join(f"{c['column']} {c['op']} {c['threshold']}" for c in resolved)

    mask = pd.Series(True, index=df.index)
    for c in resolved:
        col = c["column"]
        if col not in df.columns:
            logger.warning(f"Success column '{col}' not found -- all samples marked as failed")
            return df.iloc[0:0]
        mask &= _compare(df[col], c["threshold"], c["op"])

    filtered = df[mask]
    logger.debug(f"Motif success filter: {len(filtered)}/{len(df)} passed  [{rules}]")
    return filtered


def compute_motif_success_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    criteria: list[dict],
    model: str = "esmfold",
    suffix: str = "motif_success",
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Compute combined pass rates where ALL criteria must be met per sample.

    This is the motif equivalent of binder ``compute_filter_pass_rate``.

    Args:
        df: DataFrame with per-sample metric columns
        groupby_cols: Columns to group by
        criteria: List of ``{column, threshold, op}`` dicts
        model: Folding model for ``{model}`` resolution
        suffix: Name for the success criterion (used in column names)
        path_store_results: Optional directory to save CSV

    Returns:
        DataFrame with ``_res_{suffix}_pass_rate``, ``_res_{suffix}_n_passed``,
        ``_res_{suffix}_n_samples`` columns per group.
    """
    resolved = resolve_success_criteria(criteria, model)

    # Verify all required columns exist
    for c in resolved:
        if c["column"] not in df.columns:
            logger.warning(f"Column '{c['column']}' not found -- cannot compute {suffix} pass rates")
            return pd.DataFrame()

    # Per-sample boolean: passes all criteria?
    passes = pd.Series(True, index=df.index)
    for c in resolved:
        passes &= _compare(df[c["column"]], c["threshold"], c["op"])

    df_tmp = df[groupby_cols].copy()
    df_tmp[f"_pass_{suffix}"] = passes

    g = df_tmp.groupby(groupby_cols, dropna=False)[f"_pass_{suffix}"].agg(["sum", "count"]).reset_index()
    g[f"_res_{suffix}_pass_rate"] = g["sum"] / g["count"]
    g[f"_res_{suffix}_n_passed"] = g["sum"].astype(int)
    g[f"_res_{suffix}_n_samples"] = g["count"].astype(int)
    g = g.drop(columns=["sum", "count"])

    if path_store_results:
        path = os.path.join(path_store_results, f"res_{suffix}_pass_rates.csv")
        g.to_csv(path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)
        logger.debug(f"Saved {suffix} pass rates to {path}")

    return g


# =============================================================================
# Motif RMSD pass rates (no model dimension)
# =============================================================================


def compute_motif_rmsd_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    thresholds: dict | None = None,
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Pass rates for direct motif RMSD (generated vs ground-truth motif)."""
    thresholds = normalize_motif_rmsd_thresholds(thresholds or DEFAULT_MOTIF_RMSD_THRESHOLDS)

    parts = []
    for mode, spec in thresholds.items():
        col = f"_res_motif_rmsd_{mode}"
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue

        thresh, op = spec["threshold"], spec["op"]
        g = df.groupby(groupby_cols, dropna=False)[col].agg(list).reset_index()
        g[f"_res_motif_rmsd_pass_rate_{mode}"] = g[col].apply(lambda v: compute_pass_rate_for_values(v, thresh, op))
        g[f"_res_motif_rmsd_n_passed_{mode}"] = g[col].apply(lambda v: compute_n_passed_for_values(v, thresh, op))
        g[f"_res_motif_rmsd_mean_{mode}"] = g[col].apply(compute_mean_for_values)
        g[f"_res_motif_rmsd_n_samples_{mode}"] = g[col].apply(len)
        parts.append(g.drop(columns=[col], errors="ignore"))

    if not parts:
        return pd.DataFrame()

    result = parts[0]
    for p in parts[1:]:
        metric_cols = [c for c in p.columns if c.startswith("_res_")]
        result = pd.merge(result, p[groupby_cols + metric_cols], on=groupby_cols, how="outer")

    if path_store_results:
        path = os.path.join(path_store_results, "res_motif_rmsd_pass_rates.csv")
        result.to_csv(path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)

    return result


# =============================================================================
# Motif sequence recovery pass rates
# =============================================================================


def compute_motif_seq_rec_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    threshold: dict | None = None,
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Pass rates for motif sequence recovery."""
    threshold = threshold or DEFAULT_MOTIF_SEQ_REC_THRESHOLD
    col = "_res_motif_seq_rec"
    if col not in df.columns:
        logger.warning(f"{col} not found, skipping")
        return pd.DataFrame()

    thresh, op = threshold["threshold"], threshold["op"]
    g = df.groupby(groupby_cols, dropna=False)[col].agg(list).reset_index()
    g["_res_motif_seq_rec_pass_rate"] = g[col].apply(lambda v: compute_pass_rate_for_values(v, thresh, op))
    g["_res_motif_seq_rec_mean"] = g[col].apply(compute_mean_for_values)
    g["_res_motif_seq_rec_n_samples"] = g[col].apply(len)
    g = g.drop(columns=[col], errors="ignore")

    if path_store_results:
        path = os.path.join(path_store_results, "res_motif_seq_rec_pass_rates.csv")
        g.to_csv(path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)

    return g


# =============================================================================
# Motif-region pass rates (designability / codesignability on motif residues)
# =============================================================================


def compute_motif_region_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    thresholds: dict,
    metric_type: str,
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Pass rates for motif-region designability or codesignability.

    Args:
        metric_type: "motif_designability" or "motif_codesignability"
    """
    thresholds = normalize_monomer_thresholds(thresholds)
    parts = []
    for mode, models in thresholds.items():
        for model, spec in models.items():
            col = build_motif_region_column_name(metric_type, mode, model)
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            thresh, op = spec["threshold"], spec["op"]
            suffix = f"{mode}_{model}"
            g = df.groupby(groupby_cols, dropna=False)[col].agg(list).reset_index()
            g[f"_res_{metric_type}_pass_rate_{suffix}"] = g[col].apply(
                lambda v: compute_pass_rate_for_values(v, thresh, op)
            )
            g[f"_res_{metric_type}_n_passed_{suffix}"] = g[col].apply(
                lambda v: compute_n_passed_for_values(v, thresh, op)
            )
            g[f"_res_{metric_type}_mean_{suffix}"] = g[col].apply(compute_mean_for_values)
            g[f"_res_{metric_type}_n_samples_{suffix}"] = g[col].apply(len)
            parts.append(g.drop(columns=[col], errors="ignore"))

    if not parts:
        return pd.DataFrame()

    result = reduce(lambda l, r: pd.merge(l, r, on=groupby_cols, how="outer"), parts)

    if path_store_results:
        path = os.path.join(path_store_results, f"res_motif_{metric_type}_pass_rates.csv")
        result.to_csv(path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)

    return result


# =============================================================================
# Per-task pass rates (like binder per-target)
# =============================================================================


def compute_per_task_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    motif_rmsd_thresholds: dict | None = None,
    task_column: str = "task_name",
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Motif RMSD pass rates grouped by task name + groupby_cols."""
    if task_column not in df.columns:
        logger.warning(f"'{task_column}' not found, skipping per-task analysis")
        return pd.DataFrame()

    task_groupby = [task_column] + [c for c in groupby_cols if c != task_column]
    result = compute_motif_rmsd_pass_rates(
        df,
        task_groupby,
        motif_rmsd_thresholds,
        path_store_results=None,
    )
    if result.empty:
        return pd.DataFrame()

    if path_store_results:
        path = os.path.join(path_store_results, "res_motif_per_task_pass_rates.csv")
        result.to_csv(path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)

    return result


# =============================================================================
# Per-task combined success pass rates
# =============================================================================


def compute_per_task_success_pass_rates(
    df: pd.DataFrame,
    groupby_cols: list[str],
    criteria: list[dict],
    model: str = "esmfold",
    suffix: str = "motif_success",
    task_column: str = "task_name",
    path_store_results: str | None = None,
) -> pd.DataFrame:
    """Combined success pass rates grouped by task name + groupby_cols."""
    if task_column not in df.columns:
        logger.warning(f"'{task_column}' not found, skipping per-task success analysis")
        return pd.DataFrame()

    task_groupby = [task_column] + [c for c in groupby_cols if c != task_column]
    result = compute_motif_success_pass_rates(
        df,
        task_groupby,
        criteria,
        model=model,
        suffix=f"per_task_{suffix}",
        path_store_results=None,
    )
    if result.empty:
        return pd.DataFrame()

    if path_store_results:
        path = os.path.join(path_store_results, f"res_motif_per_task_{suffix}_pass_rates.csv")
        result.to_csv(path, sep=SEP_CSV_PD, index=False, float_format=FLOAT_FORMAT_PD)

    return result


# =============================================================================
# Filtering by motif RMSD
# =============================================================================


def filter_by_motif_rmsd(
    df: pd.DataFrame,
    thresholds: dict | None = None,
    require_all: bool = True,
) -> pd.DataFrame:
    """Filter samples by direct motif RMSD thresholds."""
    thresholds = normalize_motif_rmsd_thresholds(thresholds or DEFAULT_MOTIF_RMSD_THRESHOLDS)

    masks = []
    for mode, spec in thresholds.items():
        col = f"_res_motif_rmsd_{mode}"
        if col not in df.columns:
            continue
        masks.append(_compare(df[col], spec["threshold"], spec["op"]))

    if not masks:
        logger.warning("No motif RMSD columns found for filtering")
        return pd.DataFrame()

    combined = masks[0]
    for m in masks[1:]:
        combined = (combined & m) if require_all else (combined | m)

    filtered = df[combined]
    logger.debug(f"Motif RMSD filter ({'ALL' if require_all else 'ANY'}): {len(filtered)}/{len(df)} passed")
    return filtered


def _compare(series: pd.Series, threshold: float, op: str) -> pd.Series:
    ops = {
        "<=": series.__le__,
        "<": series.__lt__,
        ">=": series.__ge__,
        ">": series.__gt__,
        "==": series.__eq__,
    }
    return ops.get(op, series.__eq__)(threshold)


# =============================================================================
# Threshold / criteria JSON persistence
# =============================================================================


def save_motif_thresholds_json(
    motif_rmsd_thresholds: dict,
    designability_thresholds: dict,
    codesignability_thresholds: dict,
    motif_region_designability_thresholds: dict,
    motif_region_codesignability_thresholds: dict,
    motif_seq_rec_threshold: dict,
    success_criteria: dict[str, list[dict]] | None = None,
    folding_model: str = "esmfold",
    path_store_results: str = "",
    filter_name: str = "motif",
) -> str:
    """Save all motif thresholds and success criteria to a single JSON file.

    One file documents every success criterion used in this run (defaults plus
    any custom). Each criterion is labeled by name (e.g. motif_success,
    refolded_motif_success, custom_motif_success) with resolved column names,
    thresholds, and human-readable descriptions so the JSON is self-documenting.
    """

    # Descriptions for known column patterns
    _COLUMN_DESCRIPTIONS = {
        "_res_motif_seq_rec": "Motif sequence recovery (fraction of motif residues with correct AA)",
        "_res_motif_rmsd_ca": "Direct motif CA RMSD: generated structure vs ground-truth motif (Angstrom)",
        "_res_motif_rmsd_all_atom": "Direct motif all-atom RMSD: generated structure vs ground-truth motif (Angstrom)",
        "_res_co_scRMSD_all_atom_{model}": "Full-structure all-atom codesignability scRMSD: refold PDB sequence, compare to generated (Angstrom)",
        "_res_co_motif_scRMSD_ca_{model}": "Motif-region CA codesignability scRMSD: refold PDB sequence, compare motif residues to generated (Angstrom)",
        "_res_co_motif_scRMSD_all_atom_{model}": "Motif-region all-atom codesignability scRMSD: refold PDB sequence, compare motif residues to generated (Angstrom)",
    }

    _PRESET_DESCRIPTIONS = {
        "motif_success": (
            "Default motif success: ALL four criteria must pass. "
            "Steps 2-3 compare the GENERATED motif to the GROUND-TRUTH reference."
        ),
        "refolded_motif_success": (
            "Refolded motif success: ALL four criteria must pass. "
            "Steps 2-3 compare the REFOLDED structure to the GENERATED structure at motif residues."
        ),
        "custom_motif_success": "User-defined custom motif success criteria.",
    }

    def _annotate_criteria(name: str, criteria: list[dict]) -> dict:
        """Build a self-documenting dict for one success preset."""
        resolved = resolve_success_criteria(criteria, model=folding_model)
        annotated_criteria = []
        for i, c in enumerate(criteria):
            entry = {
                "step": i + 1,
                "column": resolved[i]["column"],
                "threshold": c["threshold"],
                "op": c["op"],
                "description": _COLUMN_DESCRIPTIONS.get(c["column"], ""),
            }
            annotated_criteria.append(entry)
        return {
            "description": _PRESET_DESCRIPTIONS.get(name, ""),
            "folding_model": folding_model,
            "logic": "ALL criteria must pass (AND)",
            "criteria": annotated_criteria,
        }

    data = {
        "filter_name": filter_name,
        "individual_thresholds": {
            "motif_rmsd": motif_rmsd_thresholds,
            "motif_seq_rec": motif_seq_rec_threshold,
            "designability": designability_thresholds,
            "codesignability": codesignability_thresholds,
            "motif_region_designability": motif_region_designability_thresholds,
            "motif_region_codesignability": motif_region_codesignability_thresholds,
        },
    }
    if success_criteria:
        data["success_criteria"] = {name: _annotate_criteria(name, crit) for name, crit in success_criteria.items()}

    path = os.path.join(path_store_results, f"motif_thresholds_{filter_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved motif thresholds to {path}")
    return path
