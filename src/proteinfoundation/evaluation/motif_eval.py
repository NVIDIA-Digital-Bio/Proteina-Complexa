"""
Motif evaluation: monomer evaluation + motif-specific metrics.

This module layers motif-aware metrics on top of the standard monomer pipeline:
  1. Load motif config, extract ground-truth motif coordinates
  2. Per sample: align motif -> compute direct motif RMSD & seq recovery
  3. ProteinMPNN (with motif positions *fixed*) -> fold -> full & motif-region scRMSD
  4. Codesignability: fold PDB sequence -> full & motif-region scRMSD

All shared logic (fold_sequences, scRMSD, SS, etc.) is imported from monomer_eval.
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from openfold.np.residue_constants import restype_num, restype_order

from proteinfoundation.evaluation.monomer_eval import fold_sequences
from proteinfoundation.evaluation.monomer_eval_utils import (
    DEFAULT_CODESIGNABILITY_MODES,
    DEFAULT_DESIGNABILITY_FOLDING_MODELS,
    DEFAULT_DESIGNABILITY_MODES,
    DEFAULT_NUM_SEQ_PER_TARGET,
    DEFAULT_PMPNN_SAMPLING_TEMP,
    FoldingResult,
)
from proteinfoundation.evaluation.motif_eval_utils import (
    DEFAULT_MOTIF_RMSD_MODES,
    MotifAlignmentResult,
    MotifInfo,
    MotifSelfConsistencyResult,
    append_codes_defaults,
    append_des_defaults,
    append_metric_defaults,
    compute_and_store_ss,
    get_motif_dataset_config,
    store_codes_results,
    store_des_results,
)
from proteinfoundation.evaluation.utils import maybe_tqdm, parse_cfg_for_table
from proteinfoundation.metrics.inverse_folding_models import run_proteinmpnn
from proteinfoundation.metrics.metric_utils import rmsd_metric
from proteinfoundation.utils.motif_utils import (
    extract_motif_from_pdb,
    pad_motif_to_full_length,
    pad_motif_to_full_length_unindexed,
)
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb, load_pdb, pdb_name_from_path

# CA-based RMSD modes that need auto-filling when motif uses selected atoms
# (these modes have no atoms to compare when atom_selection_mode != "all_atom")
_CA_MODES = {"ca", "bb3o"}


# =============================================================================
# Motif setup helpers
# =============================================================================


def load_motif_info(motif_task_cfg: DictConfig, task_name: str) -> MotifInfo:
    """Load motif config and extract ground-truth tensors.

    Uses the task's ``atom_selection_mode`` (e.g. "all_atom", "tip_atoms")
    to determine which atoms appear in the motif mask.  All downstream RMSD
    computations (direct motif RMSD, motif-region scRMSD) use the mask to
    decide which atoms participate, so this setting controls what gets compared.

    When ``atom_selection_mode`` is not "all_atom", CA-level motif metrics
    (direct motif RMSD, motif-region scRMSD) are auto-filled with 0.0 so
    that the columns always exist and naturally pass ``< threshold`` checks.

    Args:
        motif_task_cfg: DictConfig for one motif task entry from the motif
            dictionary.  Expected keys: ``contig_string``, ``motif_pdb_path``,
            ``motif_only``, ``atom_selection_mode``, ``motif_min_length``,
            ``motif_max_length``, ``segment_order``.
        task_name: Human-readable task identifier (e.g. "1QJG_AA_NATIVE"),
            used for logging and stored in the returned ``MotifInfo``.

    Returns:
        A ``MotifInfo`` dataclass containing the task metadata, ground-truth
        motif coordinates (``x_motif``), atom mask (``motif_mask``), and
        residue types (``residue_type``).
    """
    contig_string = motif_task_cfg.get("contig_string", "")
    motif_pdb_path = motif_task_cfg.get("motif_pdb_path", "")
    motif_only = motif_task_cfg.get("motif_only", True)
    atom_selection_mode = motif_task_cfg.get("atom_selection_mode", "all_atom")

    # Extract using the task's atom selection mode so the mask only contains
    # the atoms that the task cares about (e.g. tip atoms for tip scaffolding).
    # coors_to_nm=False keeps coordinates in Angstroms to match load_pdb output.
    motif_mask, x_motif, residue_type = extract_motif_from_pdb(
        motif_pdb_path,
        position=contig_string,
        motif_only=motif_only,
        atom_selection_mode=atom_selection_mode,
        coors_to_nm=False,
        center_motif=True,  # Center motif at origin for unindexed greedy matching
    )

    n_atoms = int(motif_mask.sum())
    logger.info(
        f"Motif task '{task_name}': {motif_mask.shape[0]} residues, "
        f"{n_atoms} atoms (atom_selection_mode={atom_selection_mode}), "
        f"contig={contig_string}"
    )
    return MotifInfo(
        task_name=task_name,
        contig_string=contig_string,
        motif_pdb_path=motif_pdb_path,
        motif_only=motif_only,
        motif_min_length=motif_task_cfg.get("motif_min_length", 0),
        motif_max_length=motif_task_cfg.get("motif_max_length", 999),
        segment_order=motif_task_cfg.get("segment_order", "A"),
        atom_selection_mode=atom_selection_mode,
        motif_mask=motif_mask,
        x_motif=x_motif,
        residue_type=residue_type,
    )


def align_motif_to_sample(
    motif_info: MotifInfo,
    contig_string: str,
    unindexed: bool = True,
    gen_coors: torch.Tensor | None = None,
    gen_mask: torch.Tensor | None = None,
    gen_aa_type: torch.Tensor | None = None,
) -> MotifAlignmentResult:
    """Pad motif to full length and build indices for ProteinMPNN fix_pos.

    Two modes:
      - **Indexed** (default): The contig string specifies exact positions.
        Uses ``pad_motif_to_full_length`` to place the motif at known indices.
      - **Unindexed** (``unindexed=True``): The model does not encode position,
        so we greedily match each motif residue to the closest residue in the
        generated structure (by coordinate RMSD + residue type).
        Requires ``gen_coors``, ``gen_mask``, and ``gen_aa_type``.

    Args:
        motif_info: Ground-truth motif data (mask, coordinates, residue types)
            as returned by ``load_motif_info``.
        contig_string: Per-sample contig string defining scaffold/motif segments
            (e.g. ``"10-20/A1-7/10-20/A28-79/10-20"``).  Used in indexed mode
            to determine where motif residues are placed.
        unindexed: If True, use greedy coordinate matching instead of contig
            positions.  Defaults to True (unindexed mode).
        gen_coors: Generated structure coordinates ``(L, 37, 3)``.  Required
            when ``unindexed=True``.
        gen_mask: Generated structure atom mask ``(L, 37)``.  Required when
            ``unindexed=True``.
        gen_aa_type: Generated residue types ``(L,)`` as integer indices.
            Required when ``unindexed=True``.

    Returns:
        A ``MotifAlignmentResult`` containing the full-length motif mask,
        coordinates, residue types, ProteinMPNN fix_pos indices
        (``motif_index``), and 0-indexed residue positions
        (``motif_residue_indices``).

    Raises:
        ValueError: If ``unindexed=True`` and the required tensors are not
            provided, or if greedy matching fails to align all motif residues.
    """
    if unindexed:
        if gen_coors is None or gen_mask is None or gen_aa_type is None:
            raise ValueError("Unindexed motif alignment requires gen_coors, gen_mask, and gen_aa_type")
        motif_mask_full, x_motif_full, residue_type_full = pad_motif_to_full_length_unindexed(
            motif_mask=motif_info.motif_mask,
            x_motif=motif_info.x_motif,
            residue_type=motif_info.residue_type,
            gen_coors=gen_coors,
            gen_mask=gen_mask,
            gen_aa_type=gen_aa_type,
        )
        if motif_mask_full is None:
            raise ValueError(
                "Unindexed motif alignment failed: could not match all motif "
                "residues to generated structure after both AA-type and "
                "coordinate-only passes"
            )
    else:
        motif_mask_full, x_motif_full, residue_type_full = pad_motif_to_full_length(
            motif_info.motif_mask,
            motif_info.x_motif,
            motif_info.residue_type,
            contig_string,
        )

    motif_sequence_mask = motif_mask_full.any(dim=1)

    motif_index, motif_residue_indices = [], []
    for idx in motif_sequence_mask.nonzero():
        i = idx.item()
        motif_index.append(f"A{i + 1}")  # 1-indexed for ProteinMPNN
        motif_residue_indices.append(i)  # 0-indexed for tensors

    return MotifAlignmentResult(
        motif_mask_full=motif_mask_full,
        x_motif_full=x_motif_full,
        residue_type_full=residue_type_full,
        motif_index=motif_index,
        motif_residue_indices=motif_residue_indices,
        motif_sequence_mask=motif_sequence_mask,
    )


# =============================================================================
# Indexed-mode contig CSV helpers
# =============================================================================


def _resolve_and_load_motif_csv(
    cfg: DictConfig,
    cfg_metric: DictConfig,
    motif_task_name: str,
    job_id: int,
    root_path: str,
) -> tuple[pd.DataFrame, dict[str, str] | None]:
    """Resolve, validate, and load the per-sample motif_info CSV.

    Called **only** in indexed mode.  Auto-discovers the CSV next to
    ``sample_storage_path`` when no explicit path is configured.

    Discovery order:
      1. Explicit ``motif_info_csv`` from the metric config section.
      2. ``{root_path}/{task}_{job_id}_motif_info.csv`` (evaluation output dir).
      3. ``{sample_storage_path}/{task}_{job_id}_motif_info.csv``.
      4. ``{parent of sample_storage_path}/{task}_{job_id}_motif_info.csv``.

    Args:
        cfg: Top-level evaluation config.  Used to read ``sample_storage_path``
            for auto-discovery.
        cfg_metric: Metric sub-config.  Used to read an explicit
            ``motif_info_csv`` path when provided.
        motif_task_name: Name of the motif task (e.g. ``"1QJG_AA_NATIVE"``),
            used to construct the expected CSV filename.
        job_id: Job index used to construct the expected CSV filename
            (``{task}_{job_id}_motif_info.csv``).
        root_path: Evaluation output directory — the first place checked
            during auto-discovery (CSVs are copied here by
            ``copy_motif_csvs``).

    Returns:
        A tuple of ``(motif_info_df, contig_by_filename)``.
        ``motif_info_df`` is the loaded DataFrame with at least ``contig``
        and ``sample_num`` columns.
        ``contig_by_filename`` is a dict mapping extensionless PDB stem to
        contig string when the CSV has a ``filename`` column, else ``None``.

    Raises:
        FileNotFoundError: If no CSV can be found — indexed evaluation
            cannot proceed without per-sample contigs.
    """
    # 1. Explicit path from config (top-level → metric section → None)
    motif_csv = cfg_metric.get("motif_info_csv", None)

    # 2. Auto-discover when not explicitly provided
    if motif_csv is None:
        sample_storage = cfg.get("sample_storage_path", None)
        csv_name = f"{motif_task_name}_{job_id}_motif_info.csv"
        candidates = [
            # Evaluation output dir (copied here by copy_motif_csvs)
            os.path.join(root_path, csv_name),
        ]
        if sample_storage:
            # Inside the sample directory
            candidates.append(os.path.join(sample_storage, csv_name))
            # Next to the sample directory (one level up)
            candidates.append(os.path.join(os.path.dirname(sample_storage), csv_name))

        for candidate in candidates:
            if os.path.exists(candidate):
                motif_csv = candidate
                logger.info(f"Auto-discovered motif_info CSV: {motif_csv}")
                break

    # 3. Validate — indexed mode *requires* the CSV
    if motif_csv is None or not os.path.exists(str(motif_csv)):
        path_info = f" (path: {motif_csv})" if motif_csv else ""
        raise FileNotFoundError(
            f"Indexed motif mode requires a motif_info CSV with per-sample "
            f"contigs, but none was found{path_info}. "
            f"Set 'motif_info_csv' in the config or place the CSV next to "
            f"sample_storage_path following the naming convention "
            f"{{task_name}}_{{job_id}}_motif_info.csv."
        )

    # 4. Load
    motif_info_df = pd.read_csv(motif_csv)
    logger.info(f"Loaded motif info CSV: {motif_csv} ({len(motif_info_df)} rows)")

    # 5. Build filename lookup dict (if column exists)
    contig_by_filename: dict[str, str] | None = None
    filename_col = "filename"
    if filename_col in motif_info_df.columns:
        contig_by_filename = {
            os.path.splitext(os.path.basename(str(f)))[0]: c
            for f, c in zip(motif_info_df[filename_col], motif_info_df["contig"], strict=False)
        }
        logger.info(
            f"motif_info CSV has '{filename_col}' column — matching by filename ({len(contig_by_filename)} entries)"
        )
    else:
        logger.info(
            "motif_info CSV has no 'filename' column — matching by row order. "
            "Ensure CSV rows are in the same order as the PDB files."
        )

    return motif_info_df, contig_by_filename


def _lookup_contig(
    pdb_path: str,
    sample_idx: int,
    default_contig: str,
    motif_info_df: pd.DataFrame | None,
    contig_by_filename: dict[str, str] | None,
) -> str:
    """Look up the per-sample contig string from a loaded motif_info CSV.

    Lookup priority:
      1. **Filename-column match** — most reliable; matches the PDB stem
         (with ``tmp_`` prefix stripped) against the CSV ``filename`` column.
      2. **Regex sample_num extraction** — parses ``_id_N_motif_`` from the
         PDB filename and looks up ``sample_num == N`` in the DataFrame.
      3. **Row-order fallback** — uses ``sample_idx`` to index into the
         DataFrame by row position.

    Falls back to *default_contig* (the task-level contig) when no match is
    found.  Only called in indexed mode.

    Args:
        pdb_path: Path to the generated PDB file.  The filename stem is used
            for matching (e.g. ``tmp_job_0_id_5_motif_1QJG_AA_NATIVE.pdb``).
        sample_idx: Loop index of this sample in the evaluation batch.  Used
            as a last-resort row-order fallback.
        default_contig: Task-level contig string returned when no CSV match
            is found (contains length ranges like ``10-20``).
        motif_info_df: Loaded motif_info DataFrame, or ``None`` when running
            in unindexed mode (returns *default_contig* immediately).
        contig_by_filename: Dict mapping extensionless PDB stems to contig
            strings (built from the CSV ``filename`` column), or ``None``
            if the column is absent.

    Returns:
        The per-sample contig string for this PDB, with exact scaffold lengths
        (e.g. ``"15/A1-7/12/A28-79/18"``).
    """
    if motif_info_df is None:
        return default_contig

    # Strip the "tmp_" prefix that pdb_dir mode adds when copying
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    clean_stem = stem.removeprefix("tmp_")

    # 1. Filename-column lookup (most reliable, works for any mode)
    if contig_by_filename is not None:
        if stem in contig_by_filename:
            return contig_by_filename[stem]
        if clean_stem in contig_by_filename:
            return contig_by_filename[clean_stem]

    # 2. Parse sample_num from the filename pattern job_X_id_N_motif_TASK
    #    e.g. "job_0_id_10_motif_1QJG_AA_NATIVE" → sample_num = 10
    m = re.search(r"_id_(\d+)_motif_", clean_stem)
    if m and "sample_num" in motif_info_df.columns:
        sid = int(m.group(1))
        # Compare with numeric column (CSV may have read sample_num as float)
        snum = pd.to_numeric(motif_info_df["sample_num"], errors="coerce")
        match = motif_info_df[snum == sid]
        if len(match) > 0:
            if len(match) > 1:
                logger.debug(f"Multiple CSV rows with sample_num={sid}; using first")
            return match["contig"].values[0]

    # 3. Row-order fallback (only if nothing else matched)
    if sample_idx < len(motif_info_df):
        return motif_info_df.iloc[sample_idx]["contig"]

    return default_contig


# =============================================================================
# Motif-specific metric functions
# =============================================================================


def compute_direct_motif_rmsd(
    gen_coors: torch.Tensor,
    gen_mask: torch.Tensor,
    alignment: MotifAlignmentResult,
    rmsd_modes: list[str],
    atom_selection_mode: str = "all_atom",
) -> dict[str, float]:
    """RMSD between generated structure and ground-truth motif at motif positions.

    When ``atom_selection_mode`` is not "all_atom" (e.g. tip_atoms), the motif
    mask only contains the selected atoms, so CA-based RMSD would have an empty
    intersection.  In that case CA modes are auto-filled with 0.0 (always passes
    threshold checks) and a log message is emitted.

    Args:
        gen_coors: Generated structure atom coordinates ``(L, 37, 3)`` in
            Angstroms.
        gen_mask: Generated structure atom mask ``(L, 37)`` (boolean).
        alignment: Motif alignment result containing the full-length motif
            mask and coordinates to compare against.
        rmsd_modes: RMSD computation modes to evaluate (e.g.
            ``["ca", "all_atom"]``).
        atom_selection_mode: Which atoms the motif task uses.  When not
            ``"all_atom"``, CA-based modes are auto-filled with 0.0.

    Returns:
        Dict mapping each RMSD mode name to its computed value (float).
        Failed computations return ``float("inf")``.
    """
    # When the motif doesn't contain all atoms, CA-based modes have no atoms
    # to compare and should be zero-filled.
    auto_fill_modes = _CA_MODES if atom_selection_mode != "all_atom" else set()

    combined_mask = gen_mask * alignment.motif_mask_full
    results = {}
    for mode in rmsd_modes:
        if mode in auto_fill_modes:
            results[mode] = 0.0
            continue
        try:
            val = rmsd_metric(
                coors_1_atom37=gen_coors,
                coors_2_atom37=alignment.x_motif_full,
                mask_atom_37=combined_mask,
                mode=mode,
            )
            results[mode] = val.item() if torch.is_tensor(val) else float(val)
        except Exception as e:
            logger.warning(f"Motif RMSD ({mode}) failed: {e}")
            results[mode] = float("inf")
    return results


def compute_motif_sequence_recovery(gen_seq: str, alignment: MotifAlignmentResult) -> float:
    """Fraction of motif residues with correct amino acid in generated sequence.

    Args:
        gen_seq: One-letter amino acid sequence of the generated structure.
        alignment: Motif alignment result containing the full-length residue
            types and the motif sequence mask identifying which positions
            belong to the motif.

    Returns:
        Sequence recovery as a float in ``[0.0, 1.0]``.
    """
    gen_types = torch.as_tensor([restype_order.get(r, restype_num) for r in gen_seq])
    matches = (gen_types == alignment.residue_type_full)[alignment.motif_sequence_mask]
    return matches.float().mean().item()


def _run_mpnn_with_fixed_motif(
    pdb_path: str,
    motif_index: list[str],
    num_seq: int,
    temp: float,
    tmp_path: str,
) -> list[str]:
    """Run ProteinMPNN sequence design with motif positions fixed.

    Args:
        pdb_path: Path to the generated PDB structure to redesign.
        motif_index: List of 1-indexed chain-position strings to fix
            (e.g. ``["A1", "A2", "A28"]``).
        num_seq: Number of sequences to sample.
        temp: ProteinMPNN sampling temperature.
        tmp_path: Directory for intermediate ProteinMPNN files.

    Returns:
        List of designed amino acid sequences (one-letter strings), with
        motif positions preserved from the input PDB.
    """
    seqs = run_proteinmpnn(
        pdb_path,
        tmp_path,
        all_chains=["A"],
        pdb_path_chains=["A"],
        fix_pos=motif_index,
        num_seq_per_target=num_seq,
        sampling_temp=temp,
    )
    return [v["seq"] for v in seqs]


def _log_scrmsd_debug(
    basename: str,
    result: "MotifSelfConsistencyResult",
    models: list[str],
    modes: list[str],
    prefix: str,
    do_motif: bool,
) -> None:
    """Log per-sample scRMSD results (debug level)."""
    for model in models:
        for m in modes:
            _full = result.best_rmsd[m].get(model, float("inf"))
            _motif = result.best_motif_rmsd[m].get(model, float("inf")) if do_motif else None
            _motif_str = f", motif={_motif:.3f}" if _motif is not None else ""
            logger.debug(f"{prefix} {basename} [{m}/{model}]: full={_full:.3f}{_motif_str}")


def _log_scrmsd_summary(
    metrics: dict[str, list],
    models: list[str],
    modes: list[str],
    full_key_prefix: str,
    motif_key_prefix: str,
    label: str,
    do_motif: bool,
) -> None:
    """Log aggregate scRMSD stats over all samples (info level)."""
    for model in models:
        for m in modes:
            vals = metrics.get(f"{full_key_prefix}{m}_{model}", [])
            if vals:
                arr = np.array(vals, dtype=float)
                finite = arr[np.isfinite(arr)]
                if len(finite) > 0:
                    logger.info(
                        f"{label} [{m}/{model}] over {len(finite)}/{len(vals)} samples: "
                        f"mean={finite.mean():.3f}, min={finite.min():.3f}, median={np.median(finite):.3f}"
                    )
            if do_motif:
                mvals = metrics.get(f"{motif_key_prefix}{m}_{model}", [])
                if mvals:
                    marr = np.array(mvals, dtype=float)
                    mfinite = marr[np.isfinite(marr)]
                    if len(mfinite) > 0:
                        logger.info(
                            f"{label} motif-region [{m}/{model}] over {len(mfinite)}/{len(mvals)} samples: "
                            f"mean={mfinite.mean():.3f}, min={mfinite.min():.3f}, median={np.median(mfinite):.3f}"
                        )


def _compute_scrmsd_full_and_motif(
    ref_pdb: str,
    folding_results: dict[str, list[FoldingResult]],
    rmsd_modes: list[str],
    motif_residue_indices: list[int] | None = None,
    atom_selection_mode: str = "all_atom",
) -> MotifSelfConsistencyResult:
    """Compute full-structure scRMSD and motif-region scRMSD from folding results.

    Compares each folded structure against the reference PDB at both full and
    motif-region scope.  Best indices are selected by argmin of motif-region
    RMSD so that the "best" designable sequence is the one that best recovers
    the motif.

    When ``atom_selection_mode`` is not "all_atom", CA-based motif-region RMSD
    modes are auto-filled with 0.0 (full-structure RMSD is always computed
    normally since the folded structures have all atoms).

    Args:
        ref_pdb: Path to the reference (generated) PDB to compare against.
        folding_results: Dict mapping folding model name to a list of
            ``FoldingResult`` objects (one per designed sequence).
        rmsd_modes: RMSD computation modes (e.g. ``["ca", "all_atom"]``).
        motif_residue_indices: 0-indexed residue positions of the motif in
            the full structure.  If ``None``, motif-region RMSD is set to
            ``float("inf")``.
        atom_selection_mode: Which atoms the motif task uses.  When not
            ``"all_atom"``, CA-based motif-region modes are auto-filled
            with 0.0.

    Returns:
        A ``MotifSelfConsistencyResult`` containing per-mode, per-model
        full-structure and motif-region RMSD values, best indices (by
        motif-region argmin), and paths to folded PDB files.
    """
    # CA-based motif-region metrics are auto-filled when motif uses selected atoms
    motif_auto_fill_modes = _CA_MODES if atom_selection_mode != "all_atom" else set()

    ref_prot = load_pdb(ref_pdb)
    ref_coors = torch.tensor(ref_prot.atom_positions, dtype=torch.float32)
    ref_mask = torch.tensor(ref_prot.atom_mask, dtype=torch.bool)

    full_rmsd = {m: {} for m in rmsd_modes}
    motif_rmsd = {m: {} for m in rmsd_modes}
    folded_paths = []

    for model_name, results in folding_results.items():
        for m in rmsd_modes:
            full_rmsd[m][model_name] = []
            motif_rmsd[m][model_name] = []

        for res in results:
            if not res.success or res.pdb_path is None:
                for m in rmsd_modes:
                    full_rmsd[m][model_name].append(float("inf"))
                    motif_rmsd[m][model_name].append(float("inf") if m not in motif_auto_fill_modes else 0.0)
                continue

            folded_paths.append(res.pdb_path)
            try:
                fp = load_pdb(res.pdb_path)
                fc = torch.tensor(fp.atom_positions, dtype=torch.float32)
                fm = torch.tensor(fp.atom_mask, dtype=torch.bool)
                mask = ref_mask * fm

                for m in rmsd_modes:
                    # Full structure -- always computed (folded PDBs have all atoms)
                    v = rmsd_metric(ref_coors, fc, mask, mode=m)
                    full_rmsd[m][model_name].append(v.item() if torch.is_tensor(v) else float(v))

                    # Motif region
                    if m in motif_auto_fill_modes:
                        motif_rmsd[m][model_name].append(0.0)
                    elif motif_residue_indices:
                        v2 = rmsd_metric(
                            ref_coors,
                            fc,
                            mask,
                            mode=m,
                            residue_indices=motif_residue_indices,
                        )
                        motif_rmsd[m][model_name].append(v2.item() if torch.is_tensor(v2) else float(v2))
                    else:
                        motif_rmsd[m][model_name].append(float("inf"))
            except Exception as e:
                logger.error(f"RMSD error for {res.pdb_path}: {e}")
                for m in rmsd_modes:
                    full_rmsd[m][model_name].append(float("inf"))
                    motif_rmsd[m][model_name].append(float("inf") if m not in motif_auto_fill_modes else 0.0)

    # Best RMSD per mode/model — selected by argmin of motif-region RMSD
    best_indices = {
        m: {mn: int(np.argmin(motif_rmsd[m][mn])) if motif_rmsd[m][mn] else 0 for mn in motif_rmsd[m]}
        for m in rmsd_modes
    }
    best_motif = {
        m: {mn: (motif_rmsd[m][mn][best_indices[m][mn]] if motif_rmsd[m][mn] else float("inf")) for mn in motif_rmsd[m]}
        for m in rmsd_modes
    }
    best_full = {
        m: {mn: (full_rmsd[m][mn][best_indices[m][mn]] if full_rmsd[m][mn] else float("inf")) for mn in full_rmsd[m]}
        for m in rmsd_modes
    }

    return MotifSelfConsistencyResult(
        rmsd_values=full_rmsd,
        best_rmsd=best_full,
        motif_rmsd_values=motif_rmsd,
        best_motif_rmsd=best_motif,
        best_indices=best_indices,
        folded_paths=folded_paths,
    )


# =============================================================================
# Main entry point
# =============================================================================


def compute_motif_metrics(
    cfg: DictConfig,
    cfg_metric: DictConfig,
    samples_paths: list[str],
    job_id: int,
    ncpus: int,
    root_path: str,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute motif evaluation metrics for a set of generated PDB samples.

    Orchestrates the full motif evaluation pipeline:
      1. Load motif ground-truth from the task config.
      2. For indexed mode, resolve and load the per-sample contig CSV.
      3. Per sample: align motif, compute direct motif RMSD and sequence
         recovery, run ProteinMPNN with fixed motif + fold (designability),
         fold the PDB sequence (codesignability), and compute secondary
         structure fractions.
      4. Aggregate results into a DataFrame with one row per sample.

    Analogous to ``compute_monomer_metrics`` but adds motif-specific columns.

    Args:
        cfg: Top-level evaluation config (contains ``sample_storage_path``,
            ``input_mode``, ``dataset``, etc.).
        cfg_metric: Metric sub-config controlling which evaluations to run
            and their parameters (e.g. ``compute_motif_metrics``,
            ``designability_num_seq``, ``motif_info_csv``).
        samples_paths: List of PDB file paths to evaluate.
        job_id: Job index for this evaluation slice.  Used to construct
            the expected motif_info CSV filename for auto-discovery.
        ncpus: Number of CPUs available (reserved for future parallelism).
        root_path: Evaluation output directory where results and copied
            CSVs are stored.
        show_progress: If True, display a tqdm progress bar over samples.

    Returns:
        A ``pd.DataFrame`` with one row per successfully evaluated sample.
        Columns include config metadata, per-sample contig string, and all
        enabled metric results (motif RMSD, sequence recovery, scRMSD,
        motif-region scRMSD, secondary structure fractions, etc.).
    """

    # --- Config ---
    columns, flat_dict = parse_cfg_for_table(cfg)
    columns += [
        "id_gen",
        "pdb_path",
        "L",
        "task_name",
        "contig_string",
        "atom_selection_mode",
    ]

    # monomer_folding_models is the shared default; per-metric keys override if set.
    shared_models = cfg_metric.get("monomer_folding_models", DEFAULT_DESIGNABILITY_FOLDING_MODELS)
    des_modes = cfg_metric.get("designability_modes", DEFAULT_DESIGNABILITY_MODES)
    des_models = cfg_metric.get("designability_folding_models", shared_models)
    codes_modes = cfg_metric.get("codesignability_modes", DEFAULT_CODESIGNABILITY_MODES)
    codes_models = cfg_metric.get("codesignability_folding_models", shared_models)

    # --- Resolve motif dataset config (mirrors binder's get_target_info) ---
    motif_dict_cfg, motif_task_name, unindexed = get_motif_dataset_config(cfg)

    # Motif RMSD modes: always include both ca and all_atom.  When
    # atom_selection_mode != "all_atom" (e.g. tip_atoms), CA-based motif metrics
    # are auto-filled with 0.0 (the columns still exist and pass thresholds).
    motif_rmsd_modes = cfg_metric.get("motif_rmsd_modes", DEFAULT_MOTIF_RMSD_MODES)
    motif_atom_mode = motif_dict_cfg.get(
        motif_task_name,
        {},
    ).get("atom_selection_mode", "all_atom")
    if motif_atom_mode != "all_atom":
        logger.info(f"atom_selection_mode={motif_atom_mode}: CA-level motif metrics will be auto-filled with 0.0")

    # compute_motif_metrics=True turns on all motif sub-flags by default.
    # Individual sub-flags can still be set to False to disable specific parts.
    motif_on = cfg_metric.get("compute_motif_metrics", False)
    do_motif_rmsd = cfg_metric.get("compute_motif_rmsd", motif_on)
    do_des = cfg_metric.get("compute_designability", motif_on)
    do_codes = cfg_metric.get("compute_codesignability", motif_on)
    do_motif_des = cfg_metric.get("compute_motif_designability", motif_on) and do_des
    do_motif_codes = cfg_metric.get("compute_motif_codesignability", motif_on) and do_codes
    do_ss = cfg_metric.get("compute_ss", True)

    num_seq = cfg_metric.get("designability_num_seq", DEFAULT_NUM_SEQ_PER_TARGET)
    pmpnn_temp = cfg_metric.get("pmpnn_sampling_temp", DEFAULT_PMPNN_SAMPLING_TEMP)

    if unindexed:
        logger.info("Unindexed motif mode: will use greedy coordinate matching")

    # --- Initialise metric columns ---
    metrics: dict[str, list] = {}

    if do_motif_rmsd:
        for m in motif_rmsd_modes:
            metrics[f"_res_motif_rmsd_{m}"] = []
        metrics["_res_motif_seq_rec"] = []

    if do_des:
        for model in des_models:
            for m in des_modes:
                metrics[f"_res_scRMSD_{m}_{model}"] = []
                metrics[f"_res_scRMSD_{m}_{model}_all"] = []
        metrics["_res_mpnn_sequences"] = []
        metrics["_res_mpnn_best_sequence"] = []
        if do_motif_des:
            for model in des_models:
                for m in des_modes:
                    metrics[f"_res_des_motif_scRMSD_{m}_{model}"] = []
                    metrics[f"_res_des_motif_scRMSD_{m}_{model}_all"] = []

    if do_codes:
        for model in codes_models:
            for m in codes_modes:
                metrics[f"_res_co_scRMSD_{m}_{model}"] = []
                metrics[f"_res_co_scRMSD_{m}_{model}_all"] = []
        if do_motif_codes:
            for model in codes_models:
                for m in codes_modes:
                    metrics[f"_res_co_motif_scRMSD_{m}_{model}"] = []
                    metrics[f"_res_co_motif_scRMSD_{m}_{model}_all"] = []

    if do_ss:
        for k in ("_res_ss_alpha", "_res_ss_beta", "_res_ss_coil"):
            metrics[k] = []

    # --- Load motif task ---
    motif_cfg = motif_dict_cfg[motif_task_name]
    motif_info = load_motif_info(motif_cfg, motif_task_name)

    # --- Per-sample contig CSV (indexed mode only) ---
    if unindexed:
        # Unindexed mode uses greedy coordinate matching — no CSV needed.
        motif_info_df = None
        _motif_contig_by_filename = None
    else:
        # Indexed mode: resolve, validate, and load the contig CSV.
        motif_info_df, _motif_contig_by_filename = _resolve_and_load_motif_csv(
            cfg,
            cfg_metric,
            motif_task_name,
            job_id,
            root_path,
        )

    # --- Per-sample loop ---
    rows = []
    for i, pdb_path in enumerate(maybe_tqdm(samples_paths, "Motif eval", show_progress)):
        if not os.path.exists(pdb_path):
            logger.warning(f"Missing: {pdb_path}")
            continue

        try:
            seq = extract_seq_from_pdb(pdb_path)
        except Exception as e:
            logger.error(f"Seq extraction failed for {pdb_path}: {e}")
            continue

        n = len(seq)

        # Per-sample contig (indexed: from CSV, unindexed: task-level default)
        contig = _lookup_contig(
            pdb_path,
            i,
            motif_info.contig_string,
            motif_info_df,
            _motif_contig_by_filename,
        )
        logger.debug(f"Using contig: {contig}")
        rows.append(
            {
                **flat_dict,
                "id_gen": i,
                "pdb_path": pdb_path,
                "L": n,
                "task_name": motif_task_name,
                "contig_string": contig,
                "atom_selection_mode": motif_info.atom_selection_mode,
            }
        )
        tmp_dir = os.path.splitext(pdb_path)[0]
        os.makedirs(tmp_dir, exist_ok=True)

        gen_prot = load_pdb(pdb_path)
        gen_coors = torch.tensor(gen_prot.atom_positions, dtype=torch.float32)
        gen_mask = torch.tensor(gen_prot.atom_mask, dtype=torch.bool)
        gen_aa_type = torch.as_tensor([restype_order.get(r, restype_num) for r in seq])

        # Align motif (indexed: use contig positions, unindexed: greedy matching)
        try:
            alignment = align_motif_to_sample(
                motif_info,
                contig,
                unindexed=unindexed,
                gen_coors=gen_coors if unindexed else None,
                gen_mask=gen_mask if unindexed else None,
                gen_aa_type=gen_aa_type if unindexed else None,
            )
        except Exception as e:
            logger.error(f"Motif alignment failed for {pdb_path}: {e}")
            append_metric_defaults(metrics)
            continue

        # --- 1. Direct motif RMSD + seq recovery ---
        if do_motif_rmsd:
            rmsds = compute_direct_motif_rmsd(
                gen_coors,
                gen_mask,
                alignment,
                motif_rmsd_modes,
                atom_selection_mode=motif_info.atom_selection_mode,
            )
            for m in motif_rmsd_modes:
                metrics[f"_res_motif_rmsd_{m}"].append(rmsds.get(m, float("inf")))
            seq_rec = compute_motif_sequence_recovery(seq, alignment)
            metrics["_res_motif_seq_rec"].append(seq_rec)
            _basename = os.path.basename(pdb_path)
            logger.debug(
                f"Motif RMSD {_basename}: "
                + ", ".join(f"{m}={rmsds.get(m, float('inf')):.3f}" for m in motif_rmsd_modes)
                + f", seq_rec={seq_rec:.3f}"
            )

        # --- 2. Designability (ProteinMPNN w/ fixed motif + fold) ---
        if do_des:
            try:
                mpnn_seqs = _run_mpnn_with_fixed_motif(
                    pdb_path,
                    alignment.motif_index,
                    num_seq,
                    pmpnn_temp,
                    tmp_dir,
                )
                name = pdb_name_from_path(pdb_path)
                fold_res = fold_sequences(
                    mpnn_seqs,
                    tmp_dir,
                    name,
                    des_models,
                    suffix="mpnn_fix_motif",
                    keep_outputs=cfg_metric.get("keep_folding_outputs", True),
                )
                des_result = _compute_scrmsd_full_and_motif(
                    pdb_path,
                    fold_res,
                    des_modes,
                    alignment.motif_residue_indices if do_motif_des else None,
                    atom_selection_mode=motif_info.atom_selection_mode,
                )
                store_des_results(metrics, des_result, mpnn_seqs, des_modes, des_models, do_motif_des)
                _log_scrmsd_debug(
                    os.path.basename(pdb_path),
                    des_result,
                    des_models,
                    des_modes,
                    "Des",
                    do_motif_des,
                )
            except Exception as e:
                logger.error(f"Designability failed for {pdb_path}: {e}")
                append_des_defaults(metrics, des_modes, des_models, do_motif_des)

        # --- 3. Codesignability (PDB seq + fold) ---
        if do_codes:
            try:
                name = pdb_name_from_path(pdb_path)
                fold_res = fold_sequences(
                    [seq],
                    tmp_dir,
                    name,
                    codes_models,
                    suffix="pdb",
                    keep_outputs=cfg_metric.get("keep_folding_outputs", True),
                )
                codes_result = _compute_scrmsd_full_and_motif(
                    pdb_path,
                    fold_res,
                    codes_modes,
                    alignment.motif_residue_indices if do_motif_codes else None,
                    atom_selection_mode=motif_info.atom_selection_mode,
                )
                store_codes_results(metrics, codes_result, codes_modes, codes_models, do_motif_codes)
                _log_scrmsd_debug(
                    os.path.basename(pdb_path),
                    codes_result,
                    codes_models,
                    codes_modes,
                    "Codes",
                    do_motif_codes,
                )
            except Exception as e:
                logger.error(f"Codesignability failed for {pdb_path}: {e}")
                append_codes_defaults(metrics, codes_modes, codes_models, do_motif_codes)

        # --- 4. Secondary structure ---
        if do_ss:
            compute_and_store_ss(metrics, pdb_path)

    # --- Post-loop summaries ---
    if do_motif_rmsd:
        for m in motif_rmsd_modes:
            vals = metrics.get(f"_res_motif_rmsd_{m}", [])
            if vals:
                arr = np.array(vals, dtype=float)
                finite = arr[np.isfinite(arr)]
                if len(finite) > 0:
                    logger.info(
                        f"Motif RMSD [{m}] over {len(finite)}/{len(vals)} samples: "
                        f"mean={finite.mean():.3f}, min={finite.min():.3f}, median={np.median(finite):.3f}"
                    )
        sr = metrics.get("_res_motif_seq_rec", [])
        if sr:
            logger.info(f"Motif seq recovery over {len(sr)} samples: mean={np.nanmean(sr):.3f}")

    if do_des:
        _log_scrmsd_summary(
            metrics,
            des_models,
            des_modes,
            "_res_scRMSD_",
            "_res_des_motif_scRMSD_",
            "Designability",
            do_motif_des,
        )

    if do_codes:
        _log_scrmsd_summary(
            metrics,
            codes_models,
            codes_modes,
            "_res_co_scRMSD_",
            "_res_co_motif_scRMSD_",
            "Codesignability",
            do_motif_codes,
        )

    # --- Build DataFrame ---
    df = pd.DataFrame(rows).reindex(columns=columns)
    for col_name, values in metrics.items():
        df[col_name] = values

    logger.info(f"Motif evaluation complete: {len(df)} samples")
    return df
