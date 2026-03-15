"""
Monomer evaluation utilities: data classes and default configuration.

Data classes:
  - FoldingResult:        output of a single structure prediction run
  - DesignabilityResult:  full-structure scRMSD values from fold-and-compare

Default constants:
  - RMSD modes, folding models, ProteinMPNN parameters

Column name patterns (written by monomer_eval.compute_monomer_metrics):
  Designability (ProteinMPNN + refold):
    _res_scRMSD_{mode}_{model}           best scRMSD (min over sequences)
    _res_scRMSD_{mode}_{model}_all       all scRMSD values (list)
    _res_scRMSD_single_{mode}_{model}    first ProteinMPNN sequence only
  Codesignability (PDB seq + refold):
    _res_co_scRMSD_{mode}_{model}        best scRMSD
    _res_co_scRMSD_{mode}_{model}_all    all scRMSD values (list)

Note: Thresholds for filtering/analysis are in monomer_analysis_utils.py
"""

from dataclasses import dataclass, field

# =============================================================================
# Folding Configuration Constants
# =============================================================================

VALID_RMSD_MODES = ["ca", "bb3o", "all_atom"]
VALID_FOLDING_MODELS = ["esmfold", "colabfold"]

# Default folding configuration
DEFAULT_DESIGNABILITY_MODES = ["ca"]
DEFAULT_DESIGNABILITY_FOLDING_MODELS = ["esmfold"]
DEFAULT_CODESIGNABILITY_MODES = ["ca", "all_atom"]
DEFAULT_CODESIGNABILITY_FOLDING_MODELS = ["esmfold"]

# ProteinMPNN default parameters
DEFAULT_NUM_SEQ_PER_TARGET = 8
DEFAULT_PMPNN_SAMPLING_TEMP = 0.1


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FoldingResult:
    """Result from a single structure prediction run."""

    pdb_path: str | None  # Path to folded structure, None if failed
    sequence: str
    model_name: str
    success: bool = True
    error: str | None = None


@dataclass
class DesignabilityResult:
    """Full-structure scRMSD values from fold-and-compare (monomer evaluation)."""

    rmsd_values: dict[str, dict[str, list[float]]]  # mode -> model -> list of rmsds
    best_rmsd: dict[str, dict[str, float]]  # mode -> model -> best rmsd
    folded_paths: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
