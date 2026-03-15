import itertools
import os
import random
import re
from typing import Literal

import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from loguru import logger
from openfold.np.residue_constants import atom_order, restype_3to1, restype_num, restype_order

from proteinfoundation.utils.align_utils import mean_w_mask
from proteinfoundation.utils.constants import AME_ATOMS, DEBUG_ATOMS, SIDECHAIN_TIP_ATOMS
from proteinfoundation.utils.coors_utils import ang_to_nm


def _select_motif_atoms(
    available_atoms: list[int],
    atom_selection_mode: Literal["ca", "bb3o", "all_atom", "tip_atoms", "debug", "ame"] = "ca",
    residue_name: str = None,
) -> list[int]:
    """Select atoms for a residue based on the specified mode.

    Args:
        available_atoms: List of available atom indices for the residue.
        atom_selection_mode: Mode for atom selection:
            - "ca": Only CA atoms
            - "bb3o": Backbone atoms (N, CA, C, O)
            - "all_atom": All available atoms
            - "tip_atoms": Tip atoms of sidechains (requires residue_name)
            - "debug": Debug atom set (requires residue_name)
            - "ame": AME atom set, randomly chosen config (requires residue_name)
        residue_name: Three-letter residue name (required for tip_atoms/debug/ame).

    Returns:
        List of selected atom indices.
    """
    backbone_atoms = [0, 1, 2, 4]  # N, CA, C, O in atom37 format
    ca_index = 1  # CA atom index in atom37 format

    if atom_selection_mode == "ca":
        return [ca_index] if ca_index in available_atoms else []

    elif atom_selection_mode == "bb3o":
        return [i for i in backbone_atoms if i in available_atoms]

    elif atom_selection_mode == "all_atom":
        return available_atoms

    elif atom_selection_mode == "tip_atoms":
        if residue_name is None:
            raise ValueError("residue_name must be provided for tip_atoms mode")
        tip_atom_names = SIDECHAIN_TIP_ATOMS.get(residue_name, [])
        return [
            atom_order[name] for name in tip_atom_names if name in atom_order and atom_order[name] in available_atoms
        ]

    elif atom_selection_mode == "debug":
        if residue_name is None:
            raise ValueError("residue_name must be provided for debug mode")
        tip_atom_names = DEBUG_ATOMS.get(residue_name, [])
        return [
            atom_order[name] for name in tip_atom_names if name in atom_order and atom_order[name] in available_atoms
        ]

    elif atom_selection_mode == "ame":
        if residue_name is None:
            raise ValueError("residue_name must be provided for ame mode")
        tip_atom_names = AME_ATOMS.get(residue_name, [])
        tip_atom_names = random.choice(tip_atom_names)
        return [
            atom_order[name] for name in tip_atom_names if name in atom_order and atom_order[name] in available_atoms
        ]

    else:
        raise ValueError(
            f"Unknown atom selection mode: {atom_selection_mode}. "
            f"Supported modes: ca, bb3o, all_atom, tip_atoms, debug, ame"
        )


def generate_combinations(min_cost, max_cost, ranges):
    result = []
    ranges = [[x] if isinstance(x, int) else range(x[0], x[1] + 1) for x in ranges]
    for combination in itertools.product(*ranges):
        total_cost = sum(combination)
        if min_cost <= total_cost <= max_cost:
            padded_combination = list(combination) + [0] * (len(ranges) - len(combination))
            result.append(padded_combination)
    return result


def generate_motif_indices(
    contig: str,
    min_length: int,
    max_length: int,
    nsamples: int = 1,
) -> tuple[list[int], list[list[int]], list[str]]:
    """Index motif and scaffold positions by contig for sequence redesign.
    Args:
        contig (str): A string containing positions for scaffolds and motifs.

        Details:
        Scaffold parts: Contain a single integer.
        Motif parts: Start with a letter (chain ID) and contain either a single positions (e.g. A33) or a range of positions (e.g. A33-39).
        The numbers following chain IDs corresponds to the motif positions in native backbones, which are used to calculate motif reconstruction later on.
        e.g. "15/A45-65/20/A20-30"
        NOTE: The scaffold part should be DETERMINISTIC in this case as it contains information for the corresponding protein backbones.

    Raises:
        ValueError: Once a "-" is detected in scaffold parts, throws an error for the aforementioned reason.

    Returns:
        A Tuple containing:
            - overall_lengths (List[int]): Total length of the sequence defined by the contig.
            - motif_indices (List[List[int]]): List of indices where motifs are located.
            - out_strs (List[str]): String of motif indices and scaffold lengths.
    """
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    components = contig.split("/")
    ranges = []
    motif_length = 0
    for part in components:
        if not part:
            continue
        if part[0] in ALPHABET:
            # Motif part
            if "-" in part:
                start, end = map(int, part[1:].split("-"))
            else:
                start = end = int(part[1:])
            length = end - start + 1
            motif_length += length
        else:
            # Scaffold part
            if "-" in part:
                bounds = part.split("-")
                assert int(bounds[0]) <= int(bounds[-1])
                ranges.append((int(bounds[0]), int(bounds[-1])))
            else:
                length = int(part)
                ranges.append(length)
    combinations = generate_combinations(min_length - motif_length, max_length - motif_length, ranges)
    if len(combinations) == 0:
        raise ValueError("No Motif combinations to sample from please update the max and min lengths")

    overall_lengths = []
    motif_indices = []
    out_strs = []
    combos = random.choices(combinations, k=nsamples)
    for combo in combos:
        combo_idx = 0
        current_position = 1  # Start positions at 1 for 1-based indexing
        motif_index = []
        output_string = ""
        for part in components:
            if not part:
                continue
            if part[0] in ALPHABET:
                # Motif part
                if "-" in part:
                    start, end = map(int, part[1:].split("-"))
                else:
                    start = end = int(part[1:])
                length = end - start + 1
                motif_index.extend(range(current_position, current_position + length))
                new_part = part[0] + str(current_position)
                if length > 1:
                    new_part += "-" + str(current_position + length - 1)
                output_string += new_part + "/"
            else:
                # Scaffold part
                length = int(combo[combo_idx])
                combo_idx += 1
                output_string += str(length) + "/"
            current_position += length  # Update the current position after processing each part
        overall_lengths.append(current_position - 1)  # current_position is 1 past the last residue
        motif_indices.append(motif_index)
        out_strs.append(output_string[:-1])
    return (overall_lengths, motif_indices, out_strs)


def parse_motif_atom_spec(spec: str):
    """Parse a motif atom specification string into a list of (chain, res_id, [atom_names]).

    Format: ``"B64: [O, C]; B86: [CB, CA, N, C]"``

    Returns:
        List of ``(chain: str, res_id: int, atom_names: list[str])`` tuples.
    """
    motif_atoms = []
    for match in re.finditer(r"([A-Za-z])(\d+): \[([^\]]+)\]", spec):
        chain = match.group(1)
        res_id = int(match.group(2))
        atoms = [a.strip() for a in match.group(3).split(",")]
        motif_atoms.append((chain, res_id, atoms))
    return motif_atoms


def extract_motif_atoms_from_pdb(
    pdb_path: str,
    motif_atom_spec: str,
):
    """Efficiently extract only the specified motif atoms from a PDB using biotite."""
    array = strucio.load_structure(pdb_path, model=1)
    motif_atoms = parse_motif_atom_spec(motif_atom_spec)
    mask = np.zeros(len(array), dtype=bool)
    for chain, res_id, atom_names in motif_atoms:
        mask |= (array.chain_id == chain) & (array.res_id == res_id) & np.isin(array.atom_name, atom_names)
    return array[mask]


def _find_best_match(
    motif_mask_i: torch.Tensor,
    x_motif_i: torch.Tensor,
    aatype_motif_i: int,
    gen_coors: torch.Tensor,
    gen_mask: torch.Tensor,
    gen_aa_type: torch.Tensor,
    claimed: set,
    require_aatype: bool,
) -> tuple[int | None, float]:
    """Find the generated residue closest to a single motif residue.

    Iterates over all generated residues, skipping those already claimed,
    and returns the index with the lowest RMSD over overlapping atoms.

    Args:
        motif_mask_i: Atom mask for this motif residue. ``(37,)``
        x_motif_i: Coordinates for this motif residue. ``(37, 3)``
        aatype_motif_i: Residue type index for this motif residue.
        gen_coors: Generated protein coordinates. ``(n_res, 37, 3)``
        gen_mask: Generated protein atom mask. ``(n_res, 37)``
        gen_aa_type: Generated protein residue types. ``(n_res,)``
        claimed: Set of generated residue indices already assigned.
        require_aatype: If ``True``, only consider residues whose AA type
            matches ``aatype_motif_i``.

    Returns:
        ``(best_index, best_rmsd)``.  ``best_index`` is ``None`` when no
        candidate has overlapping atoms (and matching type, if required).
    """
    best_idx: int | None = None
    best_rmsd = float("inf")

    for j in range(gen_coors.shape[0]):
        if j in claimed:
            continue
        if require_aatype and aatype_motif_i != gen_aa_type[j]:
            continue

        overlap = motif_mask_i & gen_mask[j]  # (37,)
        if overlap.sum() == 0:
            continue

        rmsd = torch.sqrt(torch.sum((x_motif_i[overlap] - gen_coors[j][overlap]) ** 2, dim=1).mean())
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_idx = j

    return best_idx, best_rmsd


def _run_greedy_pass(
    motif_mask: torch.Tensor,
    x_motif: torch.Tensor,
    residue_type: torch.Tensor,
    gen_coors: torch.Tensor,
    gen_mask: torch.Tensor,
    gen_aa_type: torch.Tensor,
    require_aatype: bool,
    motif_index: list[int | None] | None = None,
    claimed: set[int] | None = None,
    only_indices: list[int] | None = None,
) -> tuple[list[int | None], set[int], list[int]]:
    """Run one greedy matching pass over motif residues.

    Args:
        motif_mask / x_motif / residue_type: Motif tensors.
        gen_coors / gen_mask / gen_aa_type: Generated protein tensors.
        require_aatype: Require AA-type match for this pass.
        motif_index: Pre-existing assignment list to update in-place.
            If ``None`` a fresh list of ``None`` is created.
        claimed: Pre-existing set of claimed generated indices.
            If ``None`` an empty set is created.
        only_indices: If given, only process these motif residue indices
            (used for retry passes).  ``None`` means process all.

    Returns:
        ``(motif_index, claimed, failed)`` — updated assignment list,
        updated claimed set, and list of motif indices that failed.
    """
    nres_motif = x_motif.shape[0]
    if motif_index is None:
        motif_index = [None] * nres_motif
    if claimed is None:
        claimed = set()
    indices_to_process = only_indices if only_indices is not None else list(range(nres_motif))

    failed: list[int] = []
    for i in indices_to_process:
        best_idx, _ = _find_best_match(
            motif_mask[i],
            x_motif[i],
            int(residue_type[i]),
            gen_coors,
            gen_mask,
            gen_aa_type,
            claimed=claimed,
            require_aatype=require_aatype,
        )
        if best_idx is not None:
            motif_index[i] = best_idx
            claimed.add(best_idx)
        else:
            failed.append(i)

    return motif_index, claimed, failed


def pad_motif_to_full_length_unindexed(
    motif_mask: torch.Tensor,
    x_motif: torch.Tensor,
    residue_type: torch.Tensor,
    gen_coors: torch.Tensor,
    gen_mask: torch.Tensor,
    gen_aa_type: torch.Tensor,
    match_aatype: bool = True,
    retry_strategy: Literal["restart", "per_residue"] = "restart",
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Greedily match each motif residue to the closest generated residue.

    For each motif residue the algorithm searches all *unclaimed* generated
    residues for the one with the lowest RMSD over overlapping atoms.

    When ``match_aatype=True`` and some residues fail to find an AA-type
    match, the ``retry_strategy`` controls how the fallback works:

    **``"restart"`` (default):**
        Discard all partial matches and re-run the entire greedy sweep
        from scratch without the AA-type constraint.  This gives the
        coordinate-only pass a clean slate with no claimed indices, which
        can sometimes yield a globally better assignment. This was done in La-Proteina paper.

    **``"per_residue"``:**
        Keep all successful AA-type matches locked in, then retry *only*
        the failed residues without the type constraint.  This preserves
        high-quality type-constrained matches and avoids conflicts, at the
        cost of constraining the retry to fewer available generated residues.

    If any residue still has no match after the retry (no overlapping
    atoms with any unclaimed generated residue) the function returns
    ``(None, None, None)`` to signal alignment failure.

    Args:
        motif_mask: Atom37 boolean mask for motif residues. ``(n_motif, 37)``
        x_motif: Motif atom coordinates. ``(n_motif, 37, 3)``
        residue_type: Motif residue type indices. ``(n_motif,)``
        gen_coors: Generated protein coordinates. ``(n_res, 37, 3)``
        gen_mask: Generated protein atom mask. ``(n_res, 37)``
        gen_aa_type: Generated protein residue types. ``(n_res,)``
        match_aatype: Start with AA-type-constrained matching (recommended).
        retry_strategy: ``"restart"`` to re-run from scratch on failure, or
            ``"per_residue"`` to retry only the failed residues.

    Returns:
        ``(motif_mask_full, x_motif_full, residue_type_full)`` each shaped
        ``(n_res, …)``, or ``(None, None, None)`` if alignment fails.
    """
    nres = gen_coors.shape[0]
    nres_motif = x_motif.shape[0]

    # ------------------------------------------------------------------
    # Validate: motif must be centered for raw distance matching
    # ------------------------------------------------------------------
    if not motif_mask.any():
        raise ValueError("Unindexed motif matching requires a non-empty motif mask")
    motif_centroid_norm = x_motif[motif_mask].mean(dim=0).norm().item()
    assert motif_centroid_norm < 1.0, (
        f"Unindexed motif matching requires centered motif coordinates "
        f"(centroid norm = {motif_centroid_norm:.2f} Å, expected < 1.0). "
        f"Ensure center_motif=True when loading the motif."
    )

    # ------------------------------------------------------------------
    # Pass 1: greedy matching (with AA-type constraint if requested)
    # ------------------------------------------------------------------
    motif_index, claimed, failed_indices = _run_greedy_pass(
        motif_mask,
        x_motif,
        residue_type,
        gen_coors,
        gen_mask,
        gen_aa_type,
        require_aatype=match_aatype,
    )

    # ------------------------------------------------------------------
    # Pass 2 (if needed): retry without AA-type constraint
    # ------------------------------------------------------------------
    if failed_indices and match_aatype:
        n_pass1 = nres_motif - len(failed_indices)
        logger.debug(
            f"Pass 1 matched {n_pass1}/{nres_motif} with AA-type constraint; "
            f"retrying {len(failed_indices)} via strategy='{retry_strategy}'"
        )

        if retry_strategy == "restart":
            # Discard everything and re-run from scratch without AA-type
            motif_index, claimed, failed_indices = _run_greedy_pass(
                motif_mask,
                x_motif,
                residue_type,
                gen_coors,
                gen_mask,
                gen_aa_type,
                require_aatype=False,
            )
        elif retry_strategy == "per_residue":
            # Keep pass-1 matches locked; retry only the failed residues
            still_failed: list[int] = []
            for i in failed_indices:
                best_idx, best_rmsd = _find_best_match(
                    motif_mask[i],
                    x_motif[i],
                    int(residue_type[i]),
                    gen_coors,
                    gen_mask,
                    gen_aa_type,
                    claimed=claimed,
                    require_aatype=False,
                )
                if best_idx is not None:
                    motif_index[i] = best_idx
                    claimed.add(best_idx)
                    logger.debug(
                        f"  Motif residue {i}: coordinate-only fallback → "
                        f"gen residue {best_idx} (RMSD={best_rmsd:.3f} Å)"
                    )
                else:
                    still_failed.append(i)
            failed_indices = still_failed
        else:
            raise ValueError(f"Unknown retry_strategy='{retry_strategy}'. Must be 'restart' or 'per_residue'.")

    # ------------------------------------------------------------------
    # Final check: any unmatched residues → signal failure
    # ------------------------------------------------------------------
    n_matched = sum(1 for idx in motif_index if idx is not None)

    if failed_indices:
        logger.warning(
            f"Unindexed motif matching failed for {len(failed_indices)}/{nres_motif} "
            f"residues (no overlapping atoms with any unclaimed generated residue): "
            f"motif positions {failed_indices}"
        )
        return None, None, None

    logger.debug(f"Unindexed motif matching: {n_matched}/{nres_motif} matched")

    # ------------------------------------------------------------------
    # Build full-length tensors
    # ------------------------------------------------------------------
    motif_mask_full = torch.zeros((nres, 37), dtype=torch.bool)
    x_motif_full = torch.zeros((nres, 37, 3), dtype=torch.float)
    residue_type_full = torch.ones((nres,), dtype=torch.int64) * restype_num
    motif_mask_full[motif_index] = motif_mask
    x_motif_full[motif_index] = x_motif
    residue_type_full[motif_index] = residue_type
    return motif_mask_full, x_motif_full, residue_type_full


def extract_motif_from_pdb(
    pdb_path: str,
    position: str | None = None,
    motif_only: bool = False,
    motif_atom_spec: str | None = None,
    atom_selection_mode: Literal["ca", "bb3o", "all_atom", "tip_atoms", "debug", "ame"] = "ca",
    coors_to_nm: bool = True,
    center_motif: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract motif residue coordinates from a PDB structure.

    Two extraction modes:

    1. **Atom-level** (``motif_atom_spec`` provided): Only the explicitly
       listed atoms are included.  ``atom_selection_mode`` is ignored.
       ``position`` is not required.
    2. **Residue/range-based** (default): ``position`` specifies residue
       ranges (e.g. ``"A1-7/A28-79"``), and ``atom_selection_mode``
       controls which atoms per residue are kept.

    Args:
        pdb_path: Path to the input PDB file.
        position: Contig-style motif specification, e.g. ``"A1-7/A28-79"``.
            Required when ``motif_atom_spec`` is not provided.
        motif_only: If ``True``, select all residues from each chain that
            appears in ``position`` (ignore residue numbers).
        motif_atom_spec: Atom-level specification string, e.g.
            ``"A64: [O, CG]; B12: [N, CA]"``.  When set, range-based
            mode is not used and ``position`` is ignored.
        atom_selection_mode: Which atoms to keep per residue in range-based
            mode.  One of ``"ca"``, ``"bb3o"``, ``"all_atom"``, ``"tip_atoms"``,
            ``"debug"``, ``"ame"``.
        coors_to_nm: Convert coordinates from Angstroms to nanometers.
        center_motif: If ``True``, center motif coordinates around their
            masked mean.

    Returns:
        motif_mask: ``(n_motif_res, 37)`` bool — atom37 mask for motif atoms.
        x_motif:    ``(n_motif_res, 37, 3)`` float — atom37 coordinates.
        residue_type: ``(n_motif_res,)`` int64 — residue type indices.
    """
    # --- Mode 1: atom-level specification ---
    if motif_atom_spec is not None:
        spec_lines = "\n  ".join(part.strip() for part in motif_atom_spec.split(";") if part.strip())
        logger.info(f"Using atom-level motif specification:\n  {spec_lines}")
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        array = strucio.load_structure(pdb_path, model=1)
        motif_atoms = parse_motif_atom_spec(motif_atom_spec)

        # Ordered unique (chain, res_id)
        unique_residues = []
        seen = set()
        for chain, res_id, _ in motif_atoms:
            if (chain, res_id) not in seen:
                seen.add((chain, res_id))
                unique_residues.append((chain, res_id))

        n_res = len(unique_residues)
        motif_mask = torch.zeros((n_res, 37), dtype=torch.bool)
        x_motif = torch.zeros((n_res, 37, 3), dtype=torch.float)
        residue_type = torch.ones(n_res, dtype=torch.int64) * restype_num

        for i, (chain_id, res_id) in enumerate(unique_residues):
            atom_names = []
            for c, r, names in motif_atoms:
                if c == chain_id and r == res_id:
                    atom_names.extend(names)

            res_mask = (array.chain_id == chain_id) & (array.res_id == res_id)
            res_atoms = array[res_mask]
            if len(res_atoms) == 0:
                continue

            residue_type[i] = restype_order.get(restype_3to1.get(res_atoms[0].res_name, "UNK"), restype_num)
            for atom in res_atoms:
                if atom.atom_name in atom_names and atom.atom_name in atom_order:
                    idx = atom_order[atom.atom_name]
                    motif_mask[i, idx] = True
                    coord = torch.as_tensor(atom.coord)
                    x_motif[i, idx] = ang_to_nm(coord) if coors_to_nm else coord

    else:
        # --- Mode 2: residue/range-based specification ---
        if position is None:
            raise ValueError("position (contig string) is required when motif_atom_spec is not provided")
        ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        array = strucio.load_structure(pdb_path, model=1)

        # Collect biotite atom arrays for each motif segment
        motif_arrays = []
        seen_chains = set()
        for part in position.split("/"):
            if not part:
                continue
            chain_id = part[0]
            if chain_id not in ALPHABET:
                continue

            atom_mask = (array.chain_id == chain_id) & (array.hetero == False)

            if motif_only:
                if chain_id in seen_chains:
                    continue
                seen_chains.add(chain_id)
            else:
                residue_spec = part[1:]  # strip chain letter
                if "-" not in residue_spec:
                    start = end = int(residue_spec)
                else:
                    start, end = map(int, residue_spec.split("-"))
                atom_mask = atom_mask & (array.res_id >= start) & (array.res_id <= end)

            motif_arrays.append(array[atom_mask])

        # Concatenate all segments
        motif = motif_arrays[0]
        for seg in motif_arrays[1:]:
            motif += seg

        # Ordered unique residues
        unique_residues = []
        seen = set()
        for chain, resid in zip(motif.chain_id, motif.res_id, strict=False):
            if (chain, resid) not in seen:
                seen.add((chain, resid))
                unique_residues.append((chain, resid))
        n_res = len(unique_residues)

        # Build atom37 tensors
        motif_mask = torch.zeros((n_res, 37), dtype=torch.bool)
        x_motif = torch.zeros((n_res, 37, 3), dtype=torch.float)
        residue_type = torch.ones(n_res, dtype=torch.int64) * restype_num

        for i, (chain_id, res_id) in enumerate(unique_residues):
            res_mask = (motif.chain_id == chain_id) & (motif.res_id == res_id)
            res_atoms = motif[res_mask]
            residue_type[i] = restype_order.get(restype_3to1.get(res_atoms[0].res_name, "UNK"), restype_num)

            # Determine available atoms, then filter by selection mode
            available = [atom_order[a.atom_name] for a in res_atoms if a.atom_name in atom_order]
            if not available:
                continue
            selected = set(_select_motif_atoms(available, atom_selection_mode, res_atoms[0].res_name))

            for atom in res_atoms:
                if atom.atom_name in atom_order:
                    idx = atom_order[atom.atom_name]
                    if idx in selected:
                        motif_mask[i, idx] = True
                        coord = torch.as_tensor(atom.coord)
                        x_motif[i, idx] = ang_to_nm(coord) if coors_to_nm else coord

    if center_motif:
        motif_center = mean_w_mask(x_motif.flatten(0, 1), motif_mask.flatten(0, 1)).unsqueeze(0)
        x_motif = (x_motif - motif_center) * motif_mask[..., None]

    return motif_mask, x_motif, residue_type


def pad_motif_to_full_length(
    motif_mask: torch.Tensor,
    x_motif: torch.Tensor,
    residue_type: torch.Tensor,
    contig_string: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad motif to full length.
    Args:
        motif_mask (torch.Tensor): Boolean array for atom37 mask for the motif positions. (n_motif_res, 37)
        x_motif (torch.Tensor): Motif positions in atom37 format. (n_motif_res, 37, 3)
        residue_type (torch.Tensor): Residue types of the motif. (n_motif_res)
        contig_string (str): Contig string containing motif positions.

    Returns:
        motif_mask_full (torch.Tensor): Boolean array for atom37 mask for the motif positions. (n_full_length, 37)
        x_motif_full (torch.Tensor): Motif positions in atom37 format. (n_full_length, 37, 3)
        residue_type_full (torch.Tensor): Residue types of the motif. (n_full_length)
    """
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    components = contig_string.split("/")
    current_position = 1  # Start positions at 1 for 1-based indexing
    motif_index = []
    for part in components:
        if not part:
            continue
        if part[0] in ALPHABET:
            # Motif part
            if "-" in part:
                start, end = map(int, part[1:].split("-"))
            else:
                start = end = int(part[1:])
            length = end - start + 1
            motif_index.extend(range(current_position, current_position + length))
        else:
            # Scaffold part
            length = int(part)
        current_position += length  # Update the current position after processing each part

    # current_position is 1 past the last residue, so subtract 1 for actual length
    actual_length = current_position - 1
    motif_index = torch.tensor(motif_index, dtype=torch.int64) - 1  # Change to 0-based indexing
    motif_mask_full = torch.zeros((actual_length, 37), dtype=torch.bool)
    x_motif_full = torch.zeros((actual_length, 37, 3), dtype=torch.float)
    residue_type_full = torch.ones((actual_length,), dtype=torch.int64) * restype_num
    motif_mask_full[motif_index] = motif_mask
    x_motif_full[motif_index] = x_motif
    residue_type_full[motif_index] = residue_type
    return motif_mask_full, x_motif_full, residue_type_full


def parse_motif(
    motif_pdb_path: str,
    contig_string: str = None,
    nsamples: int = 1,
    motif_only: bool = False,
    motif_min_length: int = None,
    motif_max_length: int = None,
    segment_order: str = None,
    motif_atom_spec: str = None,
    atom_selection_mode: Literal["ca", "bb3o", "all_atom", "tip_atoms"] = "ca",
) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[str]]:
    """
    Extract motif positions from input protein structure and generate motif indices and mask.

    This function supports two modes of motif specification:

    1. **Atom-level specification** (when motif_atom_spec is provided):
       - Allows precise specification of which atoms to include for each residue
       - Format: "A64: [O, CG]; B12: [N, CA]; ..."
       - atom_selection_mode is ignored in this mode

    2. **Residue/range-based specification** (when motif_atom_spec is None):
       - Uses contig_string to specify residue ranges (e.g., "A1-7/A28-79")
       - atom_selection_mode determines which atoms are selected for each residue
       - Options: "ca", "bb3o", "all_atom", "tip_atoms"

    Args:
        motif_pdb_path (str): Path to the input protein structure.
        contig_string (str): Contig string containing motif positions (used in mode 2).
        nsamples (int): Number of samples to generate.
        motif_only (bool): Whether to extract only motif positions.
        motif_min_length (int): Minimum length of the motif.
        motif_max_length (int): Maximum length of the motif.
        segment_order (str): Optional segment order.
        motif_atom_spec (str, optional): Atom-level specification (mode 1).
            Format: "A64: [O, CG]; B12: [N, CA]; ..." If provided, uses atom-level extraction.
        atom_selection_mode (str): Atom selection mode for residue/range-based extraction (mode 2).
            Options:
            - "ca": Select only CA atoms (default, fastest)
            - "bb3o": Select backbone atoms (N, CA, C, O)
            - "all_atom": Select all available atoms (most complete)
            - "tip_atoms": Select tip atoms of sidechains

    Returns:
        lengths (List[int]): List of motif lengths.
        motif_masks (List[torch.Tensor]): List of full motif masks.  (n_res, 37)
        x_motifs (List[torch.Tensor]): List of full motif positions. (n_res, 37, 3)
        residue_types (List[torch.Tensor]): List of full motif residue types. (n_res)
        out_strs (List[str] or None): List of motif indices and scaffold lengths (None for atom-level extraction).

    Example:
        # Mode 1: Atom-level specification
        parse_motif(
            motif_pdb_path="motif.pdb",
            motif_atom_spec="A64: [O, CG]; A65: [N, CA]",
            # atom_selection_mode is ignored
        )

        # Mode 2: Residue/range-based with different atom selection modes
        parse_motif(
            motif_pdb_path="motif.pdb",
            contig_string="A1-7/A28-79",
            atom_selection_mode="tip_atoms"  # or "ca", "bb3o", "all_atom", etc.
        )
    """
    if motif_atom_spec is not None:
        parsed_atoms = parse_motif_atom_spec(motif_atom_spec)
        spec_lines = "\n  ".join(f"{c}{r}: [{', '.join(a)}]" for c, r, a in parsed_atoms)
        # logger.info(f"Using atom-level motif specification:\n  {spec_lines}")
        motif_mask, x_motif, residue_type = extract_motif_from_pdb(motif_pdb_path, motif_atom_spec=motif_atom_spec)
        n_res = motif_mask.shape[0]
        return [n_res], [motif_mask], [x_motif], [residue_type], [None]

    # Validate atom_selection_mode for classic mode
    valid_modes = ["ca", "bb3o", "all_atom", "tip_atoms"]
    if atom_selection_mode not in valid_modes:
        raise ValueError(f"Invalid atom_selection_mode '{atom_selection_mode}'. Must be one of: {valid_modes}")

    logger.info(f"Using residue/range-based motif specification with atom_selection_mode='{atom_selection_mode}'")
    if contig_string:
        logger.info(f"Contig string: {contig_string}")

    # Otherwise, use the old logic
    motif_mask, x_motif, residue_type = extract_motif_from_pdb(
        motif_pdb_path,
        position=contig_string,
        motif_only=motif_only,
        atom_selection_mode=atom_selection_mode,
        # center_motif=True, # we center later
    )
    lengths, motif_indices, out_strs = generate_motif_indices(
        contig_string, motif_min_length, motif_max_length, nsamples
    )
    motif_masks = []
    x_motifs = []
    residue_types = []
    for length, motif_index, _ in zip(lengths, motif_indices, out_strs, strict=False):
        # Construct motif_mask
        cur_mask = torch.zeros((length, 37), dtype=torch.bool)
        assert len(motif_index) == motif_mask.shape[0] == x_motif.shape[0], (
            f"motif_index: {len(motif_index)}, motif_mask: {motif_mask.shape[0]}, x_motif: {x_motif.shape[0]}, lengths don't match"
        )
        motif_index = torch.tensor(motif_index, dtype=torch.int64) - 1  # Change to 0-based indexing
        cur_mask[motif_index] = motif_mask

        # Construct full structure with zero padding for the scaffold
        cur_motif = torch.zeros((length, 37, 3), dtype=x_motif.dtype)
        cur_motif[motif_index] = x_motif
        cur_residue_type = torch.ones((length), dtype=torch.int64) * restype_num
        cur_residue_type[motif_index] = residue_type
        motif_masks.append(cur_mask)
        x_motifs.append(cur_motif)
        residue_types.append(cur_residue_type)
    return lengths, motif_masks, x_motifs, residue_types, out_strs


def save_motif_csv(pdb_path, motif_task_name, contigs, outpath=None, segment_order="A"):
    pdb_name = pdb_path.split("/")[-1].split(".")[0]
    filename = os.path.basename(pdb_path).split(".")[0]
    if filename.startswith("tmp_"):
        filename = filename[4:]
    # Create a list of dictionaries to be converted into a DataFrame
    # Each dictionary represents a row in the CSV file. "filename" is used by
    # motif_eval._lookup_contig for per-sample contig lookup; pdb_name kept for compatibility.
    data = [
        {
            "pdb_name": pdb_name,
            "sample_num": index,
            "contig": value,
            "redesign_positions": " ",  #';'.join([x for x in value.split('/') if 'A' in x or 'B' in x or 'C' in x or 'D' in x]),
            "segment_order": segment_order,
            "filename": filename,
        }
        for index, value in enumerate(contigs)
    ]

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    if outpath is None:
        outpath = f"./{motif_task_name}_motif_info.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(outpath, index=False)
