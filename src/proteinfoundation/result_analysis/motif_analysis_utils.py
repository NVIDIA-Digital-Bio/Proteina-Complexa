"""
Motif analysis utilities: default thresholds, success criteria, column helpers.

Motif analysis reuses the monomer threshold/pass-rate infrastructure for
full-structure designability and codesignability (same column names).
This file adds only:
  - Default thresholds for *motif-specific* individual metrics
  - Combined success criteria presets (motif_success, refolded_motif_success, etc.)
  - Column-name builder for motif-region metrics
  - Threshold normalisation for motif RMSD (no model dimension)

Folding-model independence:
  All thresholds and success criteria that reference a folding model use a
  ``{model}`` placeholder (resolved at analysis time via ``resolve_thresholds``
  or ``resolve_success_criteria``).  This keeps defaults model-agnostic so
  switching from esmfold to colabfold requires only changing the ``folding_model``
  config key, not every threshold definition.

All shared helpers (normalize_monomer_thresholds, compute_pass_rate_for_values,
etc.) are imported from monomer_analysis_utils and re-exported for convenience.
"""

# Re-export shared stat helpers so motif_analysis.py can import from one place
from proteinfoundation.result_analysis.analysis_utils import (  # noqa: F401
    compute_mean_for_values,
    compute_n_passed_for_values,
    compute_pass_rate_for_values,
    compute_std_for_values,
    evaluate_threshold,
    parse_threshold_spec,
)

# =============================================================================
# Default individual thresholds (for per-metric pass rate computation)
#
# Thresholds that depend on a folding model use {model} placeholders.
# Call ``resolve_thresholds(thresholds, model)`` before using them.
# =============================================================================

# Direct motif RMSD: generated structure vs ground-truth motif (no model needed)
DEFAULT_MOTIF_RMSD_THRESHOLDS = {
    "ca": {"threshold": 1.0, "op": "<"},
    "all_atom": {"threshold": 2.0, "op": "<"},
}

# Motif sequence recovery (100% = perfect, no model needed)
DEFAULT_MOTIF_SEQ_REC_THRESHOLD = {"threshold": 1.0, "op": ">="}

# Full-structure designability (same metrics as monomer, but model-agnostic)
# Uses {model} placeholder -- resolved at analysis time
DEFAULT_MOTIF_DESIGNABILITY_THRESHOLDS = {
    "ca": {"{model}": {"threshold": 2.0, "op": "<="}},
}

# Full-structure codesignability (same metrics as monomer, but model-agnostic)
# Uses {model} placeholder -- resolved at analysis time
DEFAULT_MOTIF_CODESIGNABILITY_THRESHOLDS = {
    "ca": {"{model}": {"threshold": 2.0, "op": "<="}},
    "all_atom": {"{model}": {"threshold": 2.0, "op": "<="}},
}

# Motif-region designability (refolded structure, motif residues only)
# Uses {model} placeholder -- resolved at analysis time
DEFAULT_MOTIF_REGION_DESIGNABILITY_THRESHOLDS = {
    "ca": {"{model}": {"threshold": 1.0, "op": "<"}},
}

# Motif-region codesignability
# Uses {model} placeholder -- resolved at analysis time
DEFAULT_MOTIF_REGION_CODESIGNABILITY_THRESHOLDS = {
    "ca": {"{model}": {"threshold": 1.0, "op": "<"}},
    "all_atom": {"{model}": {"threshold": 2.0, "op": "<"}},
}


# =============================================================================
# Combined success criteria presets
#
# Each preset is a list of {column, threshold, op} dicts.
# Column names may contain {model} which is resolved at runtime
# (default: "esmfold"). A sample is "successful" when ALL criteria pass.
# =============================================================================

# Motif success (generated motif vs ground-truth reference):
#   1. 100% motif sequence recovery
#   2. Motif CA RMSD < 1 Å   (generated vs true motif)
#   3. Motif all-atom RMSD < 2 Å  (generated vs true motif)
#   4. Full-structure all-atom codesignability scRMSD < 2 Å
#
# For tasks with atom_selection_mode != "all_atom" (e.g. tip_atoms), the
# evaluation layer auto-fills CA-level motif metrics with 0.0 so the columns
# always exist and criterion #2 passes automatically.
DEFAULT_MOTIF_SUCCESS_CRITERIA = [
    {"column": "_res_motif_seq_rec", "threshold": 1.0, "op": ">="},
    {"column": "_res_motif_rmsd_ca", "threshold": 1.0, "op": "<"},
    {"column": "_res_motif_rmsd_all_atom", "threshold": 2.0, "op": "<"},
    {"column": "_res_co_scRMSD_all_atom_{model}", "threshold": 2.0, "op": "<"},
]

# Refolded motif success (refolded structure vs generated structure at motif region):
#   1. 100% motif sequence recovery
#   2. Motif-region CA scRMSD < 1 Å   (refolded vs generated, motif residues)
#   3. Motif-region all-atom scRMSD < 2 Å  (refolded vs generated, motif residues)
#   4. Full-structure all-atom codesignability scRMSD < 2 Å
#
# Same auto-fill logic: CA motif-region scRMSD is 0.0 for non-all_atom tasks.
DEFAULT_REFOLDED_MOTIF_SUCCESS_CRITERIA = [
    {"column": "_res_motif_seq_rec", "threshold": 1.0, "op": ">="},
    {"column": "_res_co_motif_scRMSD_ca_{model}", "threshold": 1.0, "op": "<"},
    {"column": "_res_co_motif_scRMSD_all_atom_{model}", "threshold": 2.0, "op": "<"},
    {"column": "_res_co_scRMSD_all_atom_{model}", "threshold": 2.0, "op": "<"},
]

# Registry of presets (name -> criteria list)
# One set of presets works for all atom_selection_modes because the evaluation
# layer auto-fills CA metrics with 0.0 when the motif doesn't contain CA atoms.
MOTIF_SUCCESS_PRESETS = {
    "motif_success": DEFAULT_MOTIF_SUCCESS_CRITERIA,
    "refolded_motif_success": DEFAULT_REFOLDED_MOTIF_SUCCESS_CRITERIA,
}


# =============================================================================
# Resolution helpers (substitute {model} placeholders)
# =============================================================================


def resolve_success_criteria(
    criteria: list[dict],
    model: str = "esmfold",
) -> list[dict]:
    """Resolve ``{model}`` placeholders in criteria column names.

    Args:
        criteria: List of ``{column, threshold, op}`` dicts
        model: Folding model name to substitute

    Returns:
        New list with resolved column names
    """
    return [{**c, "column": c["column"].format(model=model)} for c in criteria]


def resolve_thresholds(thresholds: dict, model: str = "esmfold") -> dict:
    """Resolve ``{model}`` placeholders in threshold dicts that have a model dimension.

    Handles the ``{mode: {"{model}": spec}}`` pattern used by motif-region
    designability/codesignability thresholds.  Plain thresholds without a
    model key (like motif RMSD) pass through unchanged.

    Args:
        thresholds: Threshold dict, possibly containing ``{model}`` keys
        model: Folding model name to substitute

    Returns:
        New dict with ``{model}`` replaced by the actual model name
    """
    out = {}
    for key, value in thresholds.items():
        if isinstance(value, dict):
            resolved_inner = {}
            for inner_key, inner_value in value.items():
                resolved_key = inner_key.format(model=model) if isinstance(inner_key, str) else inner_key
                resolved_inner[resolved_key] = inner_value
            out[key] = resolved_inner
        else:
            out[key] = value
    return out


# =============================================================================
# Column-name helpers for motif-region metrics
# =============================================================================


def build_motif_region_column_name(metric_type: str, mode: str, model: str) -> str:
    """Column name for motif-region designability/codesignability.

    Args:
        metric_type: "motif_designability" or "motif_codesignability"
        mode: RMSD mode (ca, bb3o, all_atom)
        model: Folding model (esmfold, colabfold, ...)

    Returns:
        e.g. "_res_des_motif_scRMSD_ca_esmfold"
    """
    if metric_type == "motif_designability":
        return f"_res_des_motif_scRMSD_{mode}_{model}"
    elif metric_type == "motif_codesignability":
        return f"_res_co_motif_scRMSD_{mode}_{model}"
    else:
        raise ValueError(f"Unknown motif-region metric type: {metric_type}")


# =============================================================================
# Motif RMSD threshold normalisation
# =============================================================================


def normalize_motif_rmsd_thresholds(thresholds: dict) -> dict:
    """Normalise motif RMSD thresholds (no model dimension).

    Accepts either ``{mode: float}`` or ``{mode: {"threshold": float, "op": str}}``.
    """
    out = {}
    for mode, spec in thresholds.items():
        if isinstance(spec, (int, float)):
            out[mode] = {"threshold": float(spec), "op": "<"}
        elif isinstance(spec, dict):
            out[mode] = parse_threshold_spec(spec)
        else:
            raise ValueError(f"Invalid motif RMSD threshold for {mode}: {spec}")
    return out
