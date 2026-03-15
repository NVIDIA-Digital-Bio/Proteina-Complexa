"""
Motif binder analysis utilities and default success criteria.

This module contains:
- Default success criteria for protein and ligand motif binders
- Column name helpers for motif binder metrics
- Joint redesign evaluation logic (binder AND motif criteria on same redesign)

A motif binder sample is "successful" when at least one redesign passes ALL
binder criteria (ipAE, pLDDT, scRMSD) AND ALL motif criteria (motif RMSD,
sequence recovery, optional clash detection) simultaneously.

The success criteria can be overridden from YAML config under
``aggregation.motif_binder_success_thresholds``.
"""

from typing import Any

import numpy as np

from proteinfoundation.result_analysis.analysis_utils import evaluate_threshold, parse_threshold_spec
from proteinfoundation.result_analysis.binder_analysis_utils import normalize_threshold_dict

# =============================================================================
# Default Success Criteria
# =============================================================================
#
# Each result type has a standalone criteria dict with two keys:
#   "binder"  - dict of binder metric thresholds (same format as
#               DEFAULT_PROTEIN_BINDER_THRESHOLDS in binder_analysis_utils)
#   "motif"   - list of {column, threshold, op} dicts for motif-specific
#               metrics.  Column names use {seq_type} placeholder resolved
#               at analysis time.
#
# The _all suffix on motif columns (e.g. motif_rmsd_pred_all) indicates a
# list column with one value per redesign, enabling joint per-redesign
# evaluation.  Scalar columns (e.g. correct_motif_sequence) are broadcast
# to all redesigns.

DEFAULT_MOTIF_PROTEIN_BINDER_SUCCESS: dict = {
    "binder": {
        "i_pAE": {
            "threshold": 7.0,
            "op": "<=",
            "scale": 31.0,
            "column_prefix": "complex",
        },
        "pLDDT": {
            "threshold": 0.8,
            "op": ">=",
            "scale": 1.0,
            "column_prefix": "complex",
        },
        "scRMSD_ca": {
            "threshold": 2.0,
            "op": "<",
            "scale": 1.0,
            "column_prefix": "binder",
        },
    },
    "motif": [
        {"column": "{seq_type}_motif_rmsd_pred_all", "threshold": 2.0, "op": "<"},
        {
            "column": "{seq_type}_correct_motif_sequence_all",
            "threshold": 1.0,
            "op": ">=",
        },
    ],
}

# AME Success Criteria
DEFAULT_MOTIF_LIGAND_BINDER_SUCCESS: dict = {
    "binder": {
        "scRMSD_bb3": {
            "threshold": 2.0,
            "op": "<=",
            "scale": 1.0,
            "column_prefix": "binder",
        },
    },
    "motif": [
        {"column": "{seq_type}_motif_rmsd_pred_all", "threshold": 1.5, "op": "<="},
        {
            "column": "{seq_type}_correct_motif_sequence_all",
            "threshold": 1.0,
            "op": ">=",
        },
        {"column": "{seq_type}_has_ligand_clashes_all", "threshold": 0.5, "op": "<"},
    ],
}


def get_default_motif_binder_success(result_type: str) -> dict:
    """Return the appropriate default success criteria for a motif binder result type.

    Args:
        result_type: One of ``"motif_protein_binder"`` or ``"motif_ligand_binder"``.

    Returns:
        Deep copy of the default success criteria dict.

    Raises:
        ValueError: If *result_type* is not a motif binder type.
    """
    if result_type == "motif_ligand_binder":
        return _deep_copy_criteria(DEFAULT_MOTIF_LIGAND_BINDER_SUCCESS)
    if result_type == "motif_protein_binder":
        return _deep_copy_criteria(DEFAULT_MOTIF_PROTEIN_BINDER_SUCCESS)
    raise ValueError(
        f"Unknown motif binder result type: {result_type!r}.  Expected 'motif_protein_binder' or 'motif_ligand_binder'."
    )


# =============================================================================
# Column Name Helpers
# =============================================================================


def resolve_motif_binder_criteria(
    criteria: list[dict],
    seq_type: str,
) -> list[dict]:
    """Resolve ``{seq_type}`` placeholders in motif criteria column names.

    Args:
        criteria: List of ``{column, threshold, op}`` dicts.
        seq_type: Sequence type to substitute (e.g. ``"self"``, ``"mpnn"``).

    Returns:
        New list with resolved column names.
    """
    return [{**c, "column": c["column"].format(seq_type=seq_type)} for c in criteria]


# =============================================================================
# Joint Redesign Evaluation
# =============================================================================


def check_redesign_passes_binder_and_motif(
    binder_values: dict[str, Any],
    motif_values: dict[str, Any],
    parsed_binder_thresholds: dict,
    motif_criteria: list[dict],
) -> bool:
    """Check if a single redesign passes ALL binder AND ALL motif criteria.

    Args:
        binder_values: Metric name -> value for one redesign (binder metrics).
            E.g. ``{"i_pAE": 0.15, "pLDDT": 0.92, "scRMSD": 1.2}``.
        motif_values: Column name -> value for one redesign (motif metrics).
            E.g. ``{"self_motif_rmsd_pred_all": 1.3, ...}``.
        parsed_binder_thresholds: Parsed binder threshold specs (from
            :func:`parse_threshold_spec`).
        motif_criteria: Resolved motif criteria list (``{seq_type}`` already
            substituted via :func:`resolve_motif_binder_criteria`).

    Returns:
        True if this redesign passes every criterion, False otherwise.
    """
    # Check binder thresholds
    for metric_name, spec in parsed_binder_thresholds.items():
        if metric_name not in binder_values:
            return False
        value = binder_values[metric_name]
        if not isinstance(value, (int, float)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return False
        if not evaluate_threshold(value, spec["threshold"], spec["op"], spec["scale"]):
            return False

    # Check motif criteria
    for criterion in motif_criteria:
        col = criterion["column"]
        if col not in motif_values:
            return False
        value = motif_values[col]
        if not isinstance(value, (int, float)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return False
        if not evaluate_threshold(value, criterion["threshold"], criterion["op"]):
            return False

    return True


def check_sample_has_passing_redesign(
    binder_metric_lists: dict[str, list],
    motif_metric_lists: dict[str, list],
    parsed_binder_thresholds: dict,
    motif_criteria: list[dict],
) -> bool:
    """Check if ANY redesign in a sample passes ALL binder AND motif criteria.

    Iterates over redesigns by index; the same index ``i`` is used for both
    binder ``_all`` columns and motif ``_all`` columns so the criteria are
    evaluated jointly on the same predicted structure.

    Args:
        binder_metric_lists: Metric name -> list of values (one per redesign).
        motif_metric_lists: Column name -> list of values (one per redesign).
            Scalar motif columns (e.g. correct_motif_sequence) should be
            broadcast to a list of identical values before calling.
        parsed_binder_thresholds: Parsed binder threshold specs.
        motif_criteria: Resolved motif criteria list.

    Returns:
        True if at least one redesign passes all criteria.
    """
    n_redesigns = _get_redesign_count(binder_metric_lists, motif_metric_lists)
    for i in range(n_redesigns):
        binder_values = {metric: values[i] for metric, values in binder_metric_lists.items() if i < len(values)}
        motif_values = {col: values[i] for col, values in motif_metric_lists.items() if i < len(values)}
        if check_redesign_passes_binder_and_motif(
            binder_values,
            motif_values,
            parsed_binder_thresholds,
            motif_criteria,
        ):
            return True
    return False


def count_passing_redesigns(
    binder_metric_lists: dict[str, list],
    motif_metric_lists: dict[str, list],
    parsed_binder_thresholds: dict,
    motif_criteria: list[dict],
) -> int:
    """Count how many redesigns pass ALL binder AND motif criteria.

    Same joint-index logic as :func:`check_sample_has_passing_redesign`.

    Args:
        binder_metric_lists: Metric name -> list of values (one per redesign).
        motif_metric_lists: Column name -> list of values (one per redesign).
        parsed_binder_thresholds: Parsed binder threshold specs.
        motif_criteria: Resolved motif criteria list.

    Returns:
        Number of redesigns that pass all criteria.
    """
    n_redesigns = _get_redesign_count(binder_metric_lists, motif_metric_lists)
    count = 0
    for i in range(n_redesigns):
        binder_values = {metric: values[i] for metric, values in binder_metric_lists.items() if i < len(values)}
        motif_values = {col: values[i] for col, values in motif_metric_lists.items() if i < len(values)}
        if check_redesign_passes_binder_and_motif(
            binder_values,
            motif_values,
            parsed_binder_thresholds,
            motif_criteria,
        ):
            count += 1
    return count


# =============================================================================
# Threshold Parsing Helpers
# =============================================================================


def parse_motif_binder_success(
    success_thresholds: dict,
    seq_type: str,
) -> tuple:
    """Parse a motif binder success criteria dict into ready-to-evaluate components.

    Normalises binder metric names, parses threshold specs, and resolves
    ``{seq_type}`` placeholders in motif criteria.

    Args:
        success_thresholds: Dict with ``"binder"`` and ``"motif"`` keys.
        seq_type: Sequence type to resolve in motif column names.

    Returns:
        Tuple of ``(parsed_binder_thresholds, resolved_motif_criteria)`` where:
          - *parsed_binder_thresholds* is a dict of normalised metric names ->
            parsed threshold specs (output of :func:`parse_threshold_spec`).
          - *resolved_motif_criteria* is a list of ``{column, threshold, op}``
            dicts with ``{seq_type}`` resolved.
    """
    # Parse binder thresholds
    raw_binder = success_thresholds.get("binder", {})
    normalized_binder = normalize_threshold_dict(raw_binder)
    parsed_binder = {name: parse_threshold_spec(spec) for name, spec in normalized_binder.items()}

    # Parse motif criteria
    raw_motif = success_thresholds.get("motif", [])
    resolved_motif = resolve_motif_binder_criteria(raw_motif, seq_type)

    return parsed_binder, resolved_motif


def format_success_criteria_for_logging(
    success_thresholds: dict,
    seq_type: str = "self",
) -> str:
    """Format success criteria into a compact human-readable string for logging.

    Args:
        success_thresholds: Dict with ``"binder"`` and ``"motif"`` keys.
        seq_type: Sequence type for resolving motif column placeholders.

    Returns:
        Multi-line string describing all criteria.
    """
    parsed_binder, resolved_motif = parse_motif_binder_success(success_thresholds, seq_type)
    parts = []
    for metric_name, spec in parsed_binder.items():
        scale_str = f"*{spec['scale']}" if spec["scale"] != 1.0 else ""
        parts.append(f"  {spec['column_prefix']}_{metric_name}{scale_str} {spec['op']} {spec['threshold']}")
    for c in resolved_motif:
        parts.append(f"  {c['column']} {c['op']} {c['threshold']}")
    return "\n".join(parts)


# =============================================================================
# Internal Helpers
# =============================================================================


def _deep_copy_criteria(criteria: dict) -> dict:
    """Deep copy a success criteria dict (binder dict + motif list)."""
    return {
        "binder": {k: dict(v) for k, v in criteria["binder"].items()},
        "motif": [dict(c) for c in criteria["motif"]],
    }


def _get_redesign_count(
    binder_metric_lists: dict[str, list],
    motif_metric_lists: dict[str, list],
) -> int:
    """Determine number of redesigns from available metric lists.

    Uses the minimum length across all metric lists to avoid index errors
    when binder and motif lists have mismatched lengths.
    """
    lengths = []
    for values in binder_metric_lists.values():
        if isinstance(values, list):
            lengths.append(len(values))
    for values in motif_metric_lists.values():
        if isinstance(values, list):
            lengths.append(len(values))
    return min(lengths) if lengths else 0
