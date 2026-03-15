"""
Motif evaluation utilities: data classes, constants, and column name patterns.

Motif evaluation = monomer evaluation + motif-specific metrics.
Shared constants (RMSD modes, folding models, ProteinMPNN defaults)
live in monomer_eval_utils and are imported directly by motif_eval.

Data classes (motif-specific):
  - MotifInfo:                  parsed motif task config + ground-truth tensors
  - MotifAlignmentResult:       motif aligned into a generated structure
  - MotifSelfConsistencyResult: full + motif-region scRMSD from fold-and-compare

Column name patterns (written by motif_eval.compute_motif_metrics):
  Direct motif RMSD:
    _res_motif_rmsd_{mode}                      generated vs ground truth motif
    _res_motif_seq_rec                           motif sequence recovery

  Designability (ProteinMPNN + refold):
    _res_scRMSD_{mode}_{model}                  best full-structure scRMSD (at motif argmin)
    _res_scRMSD_{mode}_{model}_all              all full-structure scRMSD values
    _res_des_motif_scRMSD_{mode}_{model}        best motif-region scRMSD
    _res_des_motif_scRMSD_{mode}_{model}_all    all motif-region scRMSD values

  Codesignability (PDB seq + refold):
    _res_co_scRMSD_{mode}_{model}               best full-structure scRMSD (at motif argmin)
    _res_co_scRMSD_{mode}_{model}_all           all full-structure scRMSD values
    _res_co_motif_scRMSD_{mode}_{model}         best motif-region scRMSD
    _res_co_motif_scRMSD_{mode}_{model}_all     all motif-region scRMSD values
"""

import os
import shutil
from dataclasses import dataclass, field

import torch
from loguru import logger
from omegaconf import DictConfig

# =============================================================================
# Default motif RMSD modes (superset of monomer defaults)
# =============================================================================

DEFAULT_MOTIF_RMSD_MODES = ["ca", "all_atom"]


# =============================================================================
# Data Classes (motif-specific)
# =============================================================================


@dataclass
class MotifInfo:
    """Parsed motif task config + extracted ground-truth tensors."""

    task_name: str
    contig_string: str
    motif_pdb_path: str
    motif_only: bool = True
    motif_min_length: int = 0
    motif_max_length: int = 999
    segment_order: str = "A"
    atom_selection_mode: str = "all_atom"
    # Populated by load_motif_info
    motif_mask: torch.Tensor | None = None
    x_motif: torch.Tensor | None = None
    residue_type: torch.Tensor | None = None


@dataclass
class MotifAlignmentResult:
    """Result from aligning ground-truth motif into a generated structure."""

    motif_mask_full: torch.Tensor  # (n_full, 37)
    x_motif_full: torch.Tensor  # (n_full, 37, 3)
    residue_type_full: torch.Tensor  # (n_full,)
    motif_index: list[str] = field(default_factory=list)  # 1-indexed for ProteinMPNN
    motif_residue_indices: list[int] = field(default_factory=list)  # 0-indexed for tensors
    motif_sequence_mask: torch.Tensor | None = None  # (n_full,) bool


@dataclass
class MotifSelfConsistencyResult:
    """Result from motif designability/codesignability evaluation.

    Stores both full-structure and motif-region RMSD values, with best
    indices selected by argmin of motif-region RMSD.
    """

    # Full-structure RMSD
    rmsd_values: dict[str, dict[str, list[float]]]  # mode -> model -> list of full rmsds
    best_rmsd: dict[str, dict[str, float]]  # mode -> model -> best full rmsd (at motif argmin idx)
    # Motif-region RMSD
    motif_rmsd_values: dict[str, dict[str, list[float]]]  # mode -> model -> list of motif rmsds
    best_motif_rmsd: dict[str, dict[str, float]]  # mode -> model -> best motif rmsd
    # Index of best (argmin of motif RMSD)
    best_indices: dict[str, dict[str, int]]  # mode -> model -> index
    folded_paths: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)  # MPNN/PDB sequences used


# =============================================================================
# Metric storage helpers (used by motif_eval.compute_motif_metrics)
# =============================================================================


def append_metric_defaults(metrics: dict[str, list]) -> None:
    """Append safe defaults for all metrics when a sample fails (e.g. alignment error).

    List columns (ending with '_all' or '_sequences') get [].
    Sequence recovery columns get 0.0, best sequence gets "".
    Everything else (RMSD scalars) gets inf.
    """
    for key, vals in metrics.items():
        if key.endswith("_all") or key.endswith("_sequences"):
            vals.append([])
        elif "seq_rec" in key:
            vals.append(0.0)
        elif key.endswith("_best_sequence"):
            vals.append("")
        else:
            vals.append(float("inf"))


def store_scrmsd_results(
    metrics: dict[str, list],
    result: "MotifSelfConsistencyResult",
    modes: list[str],
    models: list[str],
    do_motif: bool,
    full_prefix: str,
    motif_prefix: str,
) -> None:
    """Store full + motif-region scRMSD from a MotifSelfConsistencyResult.

    For each (mode, model) pair appends:
      - {full_prefix}_{m}_{model}:       best full scRMSD (at motif argmin)
      - {full_prefix}_{m}_{model}_all:   all full scRMSD values
      - {motif_prefix}_{m}_{model}:      best motif-region scRMSD  [if do_motif]
      - {motif_prefix}_{m}_{model}_all:  all motif-region scRMSD   [if do_motif]

    Shared by designability and codesignability storage.
    """
    _INF = float("inf")
    for model in models:
        for m in modes:
            metrics[f"{full_prefix}_{m}_{model}"].append(result.best_rmsd[m].get(model, _INF))
            metrics[f"{full_prefix}_{m}_{model}_all"].append(result.rmsd_values[m].get(model, [_INF]) or [_INF])
            if do_motif:
                metrics[f"{motif_prefix}_{m}_{model}"].append(result.best_motif_rmsd[m].get(model, _INF))
                metrics[f"{motif_prefix}_{m}_{model}_all"].append(
                    result.motif_rmsd_values[m].get(model, [_INF]) or [_INF]
                )


def append_scrmsd_defaults(
    metrics: dict[str, list],
    modes: list[str],
    models: list[str],
    do_motif: bool,
    full_prefix: str,
    motif_prefix: str,
) -> None:
    """Append inf defaults for full + motif scRMSD columns.

    Shared by designability and codesignability defaults.
    """
    _INF = float("inf")
    for model in models:
        for m in modes:
            metrics[f"{full_prefix}_{m}_{model}"].append(_INF)
            metrics[f"{full_prefix}_{m}_{model}_all"].append([_INF])
            if do_motif:
                metrics[f"{motif_prefix}_{m}_{model}"].append(_INF)
                metrics[f"{motif_prefix}_{m}_{model}_all"].append([_INF])


def store_des_results(
    metrics: dict[str, list],
    result: "MotifSelfConsistencyResult",
    mpnn_seqs: list[str],
    des_modes: list[str],
    des_models: list[str],
    do_motif_des: bool,
) -> None:
    """Store designability RMSD values + MPNN sequences."""
    store_scrmsd_results(
        metrics,
        result,
        des_modes,
        des_models,
        do_motif_des,
        full_prefix="_res_scRMSD",
        motif_prefix="_res_des_motif_scRMSD",
    )
    # MPNN sequences + best sequence (selected by motif argmin of first mode/model)
    metrics["_res_mpnn_sequences"].append(mpnn_seqs)
    first_mode = des_modes[0] if des_modes else None
    first_model = des_models[0] if des_models else None
    best_idx = result.best_indices.get(first_mode, {}).get(first_model, 0) if first_mode and first_model else 0
    if mpnn_seqs:
        best_idx = min(best_idx, len(mpnn_seqs) - 1)
        metrics["_res_mpnn_best_sequence"].append(mpnn_seqs[best_idx])
    else:
        metrics["_res_mpnn_best_sequence"].append("")


def append_des_defaults(
    metrics: dict[str, list],
    des_modes: list[str],
    des_models: list[str],
    do_motif_des: bool,
) -> None:
    """Append defaults for all designability columns."""
    append_scrmsd_defaults(
        metrics,
        des_modes,
        des_models,
        do_motif_des,
        full_prefix="_res_scRMSD",
        motif_prefix="_res_des_motif_scRMSD",
    )
    metrics["_res_mpnn_sequences"].append([])
    metrics["_res_mpnn_best_sequence"].append("")


def store_codes_results(
    metrics: dict[str, list],
    result: "MotifSelfConsistencyResult",
    codes_modes: list[str],
    codes_models: list[str],
    do_motif_codes: bool,
) -> None:
    """Store codesignability RMSD values."""
    store_scrmsd_results(
        metrics,
        result,
        codes_modes,
        codes_models,
        do_motif_codes,
        full_prefix="_res_co_scRMSD",
        motif_prefix="_res_co_motif_scRMSD",
    )


def append_codes_defaults(
    metrics: dict[str, list],
    codes_modes: list[str],
    codes_models: list[str],
    do_motif_codes: bool,
) -> None:
    """Append defaults for all codesignability columns."""
    append_scrmsd_defaults(
        metrics,
        codes_modes,
        codes_models,
        do_motif_codes,
        full_prefix="_res_co_scRMSD",
        motif_prefix="_res_co_motif_scRMSD",
    )


def compute_and_store_ss(metrics: dict[str, list], pdb_path: str) -> None:
    """Compute secondary structure fractions and append to metrics."""
    from proteinfoundation.metrics.structural_metric_ss_ca_ca import compute_ss_metrics

    try:
        ss = compute_ss_metrics(pdb_path)
        metrics["_res_ss_alpha"].append(ss["biot_alpha"])
        metrics["_res_ss_beta"].append(ss["biot_beta"])
        metrics["_res_ss_coil"].append(ss["biot_coil"])
    except Exception as e:
        logger.warning(f"SS failed for {pdb_path}: {e}")
        metrics["_res_ss_alpha"].append(0.0)
        metrics["_res_ss_beta"].append(0.0)
        metrics["_res_ss_coil"].append(1.0)


# =============================================================================
# Config Extraction
# =============================================================================


def get_motif_dataset_config(cfg: DictConfig) -> tuple[DictConfig, str, bool]:
    """Extract motif dataset config, task name, and unindexed flag from config.

    Supports two layouts (mirroring the binder ``get_target_info`` dual-path):

    1. **Evaluate config** (``evaluate_motif.yaml``):
       ``cfg.dataset.motif_dict_cfg``, ``cfg.dataset.motif_task_name``,
       ``cfg.dataset.unindexed``

    2. **Legacy / generation config**:
       ``cfg.generation.dataset.motif_dict_cfg``,
       ``cfg.generation.dataset.motif_task_name``,
       ``cfg.generation.args.unindexed``

    Returns:
        Tuple of (motif_dict_cfg, motif_task_name, unindexed).
    """
    # New layout: cfg.dataset.motif_dict_cfg  (same level as binder targets)
    if "dataset" in cfg and "motif_dict_cfg" in cfg.dataset:
        ds = cfg.dataset
        motif_dict_cfg = ds.motif_dict_cfg
        motif_task_name = ds.get("motif_task_name", None)
        unindexed = ds.get("unindexed", True)
    # Legacy layout: cfg.generation.dataset.motif_dict_cfg
    elif "generation" in cfg:
        gen = cfg.generation
        ds = gen.get("dataset", gen.get("dataloader", {}).get("dataset", {}))
        motif_dict_cfg = ds.get("motif_dict_cfg", None)
        motif_task_name = ds.get("motif_task_name", None)
        unindexed = gen.get("args", {}).get("unindexed", True)
    else:
        raise ValueError(
            "Cannot find motif_dict_cfg in config. Expected either "
            "cfg.dataset.motif_dict_cfg or cfg.generation.dataset.motif_dict_cfg"
        )

    if motif_dict_cfg is None:
        raise ValueError("motif_dict_cfg is missing from config")
    if motif_task_name is None:
        raise ValueError("motif_task_name is missing from config")
    if motif_task_name not in motif_dict_cfg:
        raise ValueError(
            f"motif_task_name '{motif_task_name}' not found in motif_dict_cfg. "
            f"Available tasks: {list(motif_dict_cfg.keys())[:10]}"
        )

    return motif_dict_cfg, motif_task_name, unindexed


# =============================================================================
# Motif CSV Helpers
# =============================================================================


def copy_motif_csvs(
    sample_paths: list[str],
    output_dir: str,
    input_mode: str,
    task_name: str,
) -> None:
    """Copy motif_info CSVs matching *task_name* from the source location into *output_dir*.

    After ``generate.py`` runs, the motif_info CSV sits alongside the sample
    directories (e.g. ``inference/my_run/{task_name}_0_motif_info.csv``).
    When we copy the individual sample dirs into the evaluation output
    directory, the CSV is left behind.  This helper ensures it is copied so
    that the output directory is fully self-contained and the contig CSV can
    be auto-discovered by ``motif_eval._resolve_and_load_motif_csv``.

    Only CSVs whose filename contains ``{task_name}`` and ends with
    ``_motif_info.csv`` are copied, preventing unnecessary I/O when the
    source directory holds CSVs for many tasks.

    For **generated** mode the CSV lives in the *parent* of the sample dirs.
    For **pdb_dir** mode it may live *next to* or *inside* the PDB directory.
    """
    source_dirs: set[str] = set()

    if input_mode == "pdb_dir":
        for p in sample_paths:
            source_dirs.add(os.path.dirname(p))
            parent = os.path.dirname(os.path.dirname(p))
            if parent:
                source_dirs.add(parent)
    else:
        for p in sample_paths:
            source_dirs.add(os.path.dirname(p))

    copied = 0
    for src_dir in source_dirs:
        if not os.path.isdir(src_dir):
            continue
        for fname in os.listdir(src_dir):
            if fname.endswith("_motif_info.csv") and task_name in fname:
                src = os.path.join(src_dir, fname)
                dst = os.path.join(output_dir, fname)
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copy2(src, dst)
                    copied += 1
    if copied:
        logger.info(f"Copied {copied} motif_info CSV(s) to {output_dir}")
