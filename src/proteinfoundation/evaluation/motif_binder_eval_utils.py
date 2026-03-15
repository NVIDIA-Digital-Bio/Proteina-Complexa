"""
Motif binder evaluation utilities: ranking and config helpers.

This module provides:
- Default ranking criteria for selecting best refolded sample
- Dataclass for parsed task configuration (MotifBinderTaskConfig)
- Config parsing for the combined motif+target dict (motif_target_dict_cfg)
- Target info extraction (get_motif_binder_target_info)

Eval only computes and stores raw metrics.  Success/failure criteria
belong in the analysis step (result_analysis), not here.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from proteinfoundation.evaluation.binder_eval_utils import validate_ranking_criteria

# =============================================================================
# Default Ranking Criteria (extends binder defaults with motif RMSD)
# =============================================================================

DEFAULT_MOTIF_PROTEIN_BINDER_RANKING = {
    "motif_rmsd_pred": {"scale": 1.0, "direction": "minimize"},
}

DEFAULT_MOTIF_LIGAND_BINDER_RANKING = {
    "motif_rmsd_pred": {"scale": 1.0, "direction": "minimize"},
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MotifBinderTaskConfig:
    """Parsed motif binder task configuration.

    Holds both target info (PDB path, ligand names, chain IDs) and motif info
    (contig_atoms or contig_string, motif PDB path) extracted from a single
    motif_target_dict_cfg entry.

    Mirrors the shared target definition convention from ``target_dict_cfg``
    (source, target_filename, hotspot_residues, target_path) with additional
    motif-specific fields (contig_atoms, contig_string).
    """

    task_name: str
    target_pdb_path: str  # Resolved from target_path > target_pdb_path > source+filename
    is_target_ligand: bool
    ligand_names: list[str] = field(default_factory=list)
    # Motif specification (atom-level for ligand targets, residue-level for protein)
    contig_atoms: str | None = None
    contig_string: str | None = None
    motif_pdb_path: str | None = None
    atom_selection_mode: str = "all_atom"
    motif_only: bool = True
    # Shared target fields (same convention as target_dict_cfg / ligand_targets_dict)
    hotspot_residues: list[str] = field(default_factory=list)
    target_pdb_chain: list[str] = field(default_factory=lambda: ["A"])


# =============================================================================
# Config Parsing
# =============================================================================


def _parse_task_config(task_name: str, task_cfg: DictConfig) -> MotifBinderTaskConfig:
    """Parse a single motif_target_dict_cfg entry into a MotifBinderTaskConfig.

    Determines ``is_target_ligand`` from the presence of a ``ligand`` field.
    Parses shared target fields (hotspot_residues, target chains, etc.)
    using the same conventions as ``target_dict_cfg``.

    Args:
        task_name: Task identifier (e.g. "M0024_1nzy").
        task_cfg: DictConfig for one task entry.

    Returns:
        Parsed MotifBinderTaskConfig.
    """
    # Target PDB path — same resolution as get_target_info / get_motif_binder_target_info:
    #   1. target_path (shared convention)
    #   2. target_pdb_path (legacy)
    #   3. Construct from source + target_filename + .pdb
    target_pdb_path = task_cfg.get("target_path", "") or task_cfg.get("target_pdb_path", "")
    if not target_pdb_path:
        target_filename = task_cfg.get("target_filename", "")
        data_path = os.environ.get("DATA_PATH", "")
        source = task_cfg.get("source", "")
        if target_filename and data_path and source:
            target_pdb_path = os.path.join(data_path, f"target_data/{source}/{target_filename}.pdb")

    # Determine ligand status
    ligand_field = task_cfg.get("ligand", None)
    if ligand_field is not None:
        is_target_ligand = True
        if isinstance(ligand_field, str):
            ligand_names = [ligand_field]
        else:
            ligand_names = list(OmegaConf.to_object(ligand_field))
    else:
        is_target_ligand = False
        ligand_names = []

    # Motif specification
    contig_atoms = task_cfg.get("contig_atoms", None)
    contig_string = task_cfg.get("contig_string", None)
    # Resolve motif PDB: motif_pdb_path > target_path > target_pdb_path
    motif_pdb_path = task_cfg.get("motif_pdb_path", None)
    if motif_pdb_path is None and contig_atoms is not None:
        motif_pdb_path = target_pdb_path

    atom_selection_mode = task_cfg.get("atom_selection_mode", "all_atom")
    motif_only = task_cfg.get("motif_only", False if contig_atoms else True)

    # Target chains
    target_input = task_cfg.get("target_input", None)
    if target_input and not is_target_ligand:
        target_pdb_chain = sorted(set(x[0] for x in target_input.split(",")))
    else:
        target_pdb_chain = ["A"]

    # Hotspot residues — same convention as target_dict_cfg.
    # Format: ["A33", "A95", ...] for protein targets, [null] for ligand targets.
    # Filter out null/None entries so downstream code gets a clean list.
    raw_hotspots = task_cfg.get("hotspot_residues", None)
    if raw_hotspots is not None:
        hotspot_residues = [h for h in list(OmegaConf.to_object(raw_hotspots)) if h is not None]
    else:
        hotspot_residues = []

    return MotifBinderTaskConfig(
        task_name=task_name,
        target_pdb_path=target_pdb_path,
        is_target_ligand=is_target_ligand,
        ligand_names=ligand_names,
        contig_atoms=contig_atoms,
        contig_string=contig_string,
        motif_pdb_path=motif_pdb_path,
        atom_selection_mode=atom_selection_mode,
        motif_only=motif_only,
        hotspot_residues=hotspot_residues,
        target_pdb_chain=target_pdb_chain,
    )


def get_ranking_criteria(
    is_target_ligand: bool,
    overrides: dict[str, Any] | DictConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Get ranking criteria for best sample selection.

    Args:
        is_target_ligand: Whether the target is a ligand.
        overrides: Optional custom ranking criteria from config.

    Returns:
        Validated ranking criteria dictionary.
    """
    if overrides is not None:
        return validate_ranking_criteria(overrides)
    return DEFAULT_MOTIF_LIGAND_BINDER_RANKING if is_target_ligand else DEFAULT_MOTIF_PROTEIN_BINDER_RANKING


# =============================================================================
# Target Info Extraction
# =============================================================================


def get_motif_binder_target_info(cfg: DictConfig) -> tuple[str, str, list[str], bool]:
    """Extract target info from ``motif_target_dict_cfg`` (motif binder tasks).

    Mirrors ``get_target_info`` but reads from the combined
    ``motif_target_dict_cfg`` instead of ``target_dict_cfg``.  Delegates
    path/ligand/chain resolution to ``_parse_task_config``.

    Returns:
        Tuple of (task_name, target_pdb_path, target_pdb_chain, is_target_ligand).
    """
    # Locate dataset + motif_target_dict_cfg (mirrors get_target_info pattern).
    # In standalone eval mode, both live under cfg.dataset.
    # In pipeline mode, task_name is in cfg.generation.dataloader.dataset but
    # motif_target_dict_cfg is at cfg.generation (top-level of the generation config).
    if "dataset" in cfg and "motif_target_dict_cfg" in cfg.get("dataset", {}):
        cfg_dataset = cfg.dataset
        motif_target_dict_cfg = cfg.dataset.motif_target_dict_cfg
    elif "generation" in cfg:
        gen = cfg.generation
        cfg_dataset = gen.get("dataset", gen.get("dataloader", {}).get("dataset", {}))
        motif_target_dict_cfg = gen.get("motif_target_dict_cfg", None)
    else:
        raise ValueError("Cannot find motif_target_dict_cfg in config")

    task_name = cfg_dataset.get("task_name", None)
    if motif_target_dict_cfg is None:
        raise ValueError("motif_target_dict_cfg is missing from config")
    if task_name is None:
        raise ValueError("task_name is missing from config")
    if task_name not in motif_target_dict_cfg:
        raise ValueError(
            f"task_name '{task_name}' not found in motif_target_dict_cfg. "
            f"Available: {list(motif_target_dict_cfg.keys())[:10]}"
        )

    parsed = _parse_task_config(task_name, motif_target_dict_cfg[task_name])
    return (
        task_name,
        parsed.target_pdb_path,
        parsed.target_pdb_chain,
        parsed.is_target_ligand,
    )
