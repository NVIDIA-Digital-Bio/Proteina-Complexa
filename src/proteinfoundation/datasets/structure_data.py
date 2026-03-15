"""Structure Data Module for protein structures.

This module provides a unified interface for loading protein structures from
PDB/CIF files and converting them to atom37 Data format compatible with existing
transforms like CoordsToNanometers, GlobalRotationTransform, etc.

Supports two modes:
1. Simple mode: PDB/CIF → AtomArray → atomarray_transforms → atom37 Data → atom37_transforms
2. Pipeline mode: PDB/CIF → AtomArray → pipeline_target() → atom37 Data → atom37_transforms

Both modes convert to atom37 at the end and apply atom37_transforms.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.datasets.transforms import Data
from proteinfoundation.utils.constants import UNIFIED_ATOM37_ENCODING


def load_structure(
    path: str,
    use_parse: bool = False,
    cache_dir: str | None = None,
    **parser_args,
):
    """Load a structure file (PDB or CIF) as a biotite AtomArray.

    Args:
        path: Path to structure file (PDB, CIF, or gzipped versions).
        use_parse: If True, use atomworks parse() for full processing.
        cache_dir: Cache directory for parse().
        **parser_args: Additional arguments passed to parse().

    Returns:
        AtomArray object.
    """
    from atomworks.io.utils.io_utils import load_any

    if not os.path.exists(path):
        raise FileNotFoundError(f"Structure file not found: {path}")

    if use_parse:
        from atomworks.io.parser import parse

        # Filter out keys that are explicitly passed to avoid duplicates
        filtered_parser_args = {
            k: v for k, v in parser_args.items() if k not in ("cache_dir", "save_to_cache", "load_from_cache")
        }
        result = parse(
            path,
            cache_dir=cache_dir,
            save_to_cache=cache_dir is not None,
            load_from_cache=cache_dir is not None,
            **filtered_parser_args,
        )
        if result.get("assemblies") and result["assemblies"].get("1"):
            atom_array = result["assemblies"]["1"][0]
        elif result.get("asym_unit"):
            atom_array = result["asym_unit"][0] if isinstance(result["asym_unit"], list) else result["asym_unit"]
        else:
            raise ValueError(f"No structure found in parse result for {path}")
    else:
        atom_array = load_any(path, model=1)

    # Ensure common annotations exist (some loaders don't set them)
    if not hasattr(atom_array, "occupancy") or atom_array.occupancy is None:
        atom_array.set_annotation("occupancy", np.ones(len(atom_array), dtype=np.float32))
    if not hasattr(atom_array, "b_factor") or atom_array.b_factor is None:
        atom_array.set_annotation("b_factor", np.zeros(len(atom_array), dtype=np.float32))

    return atom_array


def _ensure_atomworks_annotations(atom_array):
    """Add required annotations for atomworks transforms if missing."""
    from atomworks.enums import ChainType

    n_atoms = len(atom_array)

    if not hasattr(atom_array, "chain_type") or atom_array.chain_type is None:
        chain_types = np.full(n_atoms, ChainType.POLYPEPTIDE_L, dtype=object)
        if hasattr(atom_array, "hetero") and atom_array.hetero is not None:
            chain_types[atom_array.hetero] = ChainType.NON_POLYMER
        atom_array.set_annotation("chain_type", chain_types)

    if not hasattr(atom_array, "pn_unit_iid") or atom_array.pn_unit_iid is None:
        unique_chains = np.unique(atom_array.chain_id)
        chain_to_iid = {c: i for i, c in enumerate(unique_chains)}
        atom_array.set_annotation("pn_unit_iid", np.array([chain_to_iid[c] for c in atom_array.chain_id]))

    if not hasattr(atom_array, "pn_unit_id") or atom_array.pn_unit_id is None:
        atom_array.set_annotation("pn_unit_id", atom_array.chain_id.copy())

    if not hasattr(atom_array, "atomize") or atom_array.atomize is None:
        atom_array.set_annotation("atomize", np.zeros(n_atoms, dtype=bool))

    if not hasattr(atom_array, "atom_id") or atom_array.atom_id is None:
        atom_array.set_annotation("atom_id", np.arange(n_atoms))

    if not hasattr(atom_array, "is_polymer") or atom_array.is_polymer is None:
        is_polymer = (
            ~atom_array.hetero
            if hasattr(atom_array, "hetero") and atom_array.hetero is not None
            else np.ones(n_atoms, dtype=bool)
        )
        atom_array.set_annotation("is_polymer", is_polymer)

    if not hasattr(atom_array, "occupancy") or atom_array.occupancy is None:
        atom_array.set_annotation("occupancy", np.ones(n_atoms, dtype=np.float32))

    if not hasattr(atom_array, "b_factor") or atom_array.b_factor is None:
        atom_array.set_annotation("b_factor", np.zeros(n_atoms, dtype=np.float32))

    return atom_array


def atomarray_to_atom37(
    atom_array,
    sample_id: str = "unknown",
    database: str = "structure",
    atomworks_data: dict | None = None,
) -> Data:
    """Convert a biotite AtomArray to an atom37 Data object.

    Uses atom_array_to_encoding from atomworks for consistent atom ordering.
    """
    from atomworks.ml.transforms.encoding import atom_array_to_encoding, token_iter

    # Ensure occupancy annotation exists (required by atom_array_to_encoding)
    if not hasattr(atom_array, "occupancy") or atom_array.occupancy is None:
        atom_array.set_annotation("occupancy", np.ones(len(atom_array), dtype=np.float32))

    encoding = UNIFIED_ATOM37_ENCODING
    encoded = atom_array_to_encoding(
        atom_array,
        encoding,
        default_coord=float("nan"),
        extra_annotations=["chain_id"],
        atomize_token="<A>",  # Use atomized token for all atomized atoms
        atomize_atom_name="X",  # Use general X atom name for all atomized atoms
    )
    n_tokens = len(encoded["xyz"])
    if n_tokens == 0:
        logger.warning(f"Empty structure for {sample_id}")
        return Data(
            coords=torch.zeros(0, 37, 3, dtype=torch.float32),
            coord_mask=torch.zeros(0, 37, dtype=torch.bool),
            residues=[],
            chains=torch.zeros(0, dtype=torch.long),
            residue_type=torch.zeros(0, dtype=torch.long),
            id=sample_id,
            seq_pos=torch.zeros(0, 1, dtype=torch.long),
            residue_pdb_idx=torch.zeros(0, dtype=torch.long),
            database=database,
        )

    coords = torch.from_numpy(encoded["xyz"]).float()
    coords = torch.nan_to_num(coords, nan=0.0)
    coord_mask = torch.from_numpy(encoded["mask"]).bool()

    idx_to_token = {v: k for k, v in encoding.token_to_idx.items()}
    residues = [idx_to_token[idx] for idx in encoded["seq"]]

    int_to_chain_id = {v: k for k, v in encoded.get("chain_id_to_int", {}).items()}
    chain_ids = [int_to_chain_id.get(i, "A") for i in encoded.get("chain_id", np.zeros(n_tokens))]

    residue_pdb_idx = [token.res_id[0] for token in token_iter(atom_array)]

    unique_chains = list(dict.fromkeys(chain_ids))
    chain_to_idx = {c: i for i, c in enumerate(unique_chains)}
    chains = torch.tensor([chain_to_idx[c] for c in chain_ids], dtype=torch.long)

    residue_type = torch.tensor(encoded["seq"], dtype=torch.long)

    # Build base Data object
    data = Data(
        coords=coords,
        coord_mask=coord_mask,
        residue_type=residue_type,
        residues=residues,
        chains=chains,
        chain_id=chain_ids,
        chain_names=unique_chains,
        residue_pdb_idx=torch.tensor(residue_pdb_idx, dtype=torch.long),
        seq_pos=torch.arange(n_tokens).unsqueeze(-1),
        id=sample_id,
        database=database,
        num_nodes=n_tokens,
    )

    # If atomworks pipeline data is available, store it for AtomworksLigandFeaturesTransform
    # The transform will process this data and add ligand-specific features
    if atomworks_data is not None:
        data._atomworks_data = atomworks_data  # Contains atom_array already

    return data


class StructureDataset(Dataset):
    """Dataset for loading protein structures from metadata.

    Supports two modes:
    1. Simple mode (atomarray_transforms): List of transforms applied sequentially
    2. Pipeline mode (pipeline_target): A function that builds a complete transform pipeline

    Both modes convert to atom37 at the end and apply atom37_transforms.
    """

    def __init__(
        self,
        metadata_file: str | None = None,
        metadata_df: pd.DataFrame | None = None,
        base_dir: str = "",
        atomarray_transforms: list[Callable] | None = None,
        atom37_transforms: list[Callable] | None = None,
        transforms: list[Callable] | None = None,  # Legacy alias
        pipeline_target: str | None = None,  # e.g. "module.build_pipeline"
        path_column: str = "path",
        id_column: str = "id",
        file_extension: str = "",
        use_parse: bool = False,
        parser_args: dict | None = None,
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        is_inference: bool = False,
        **pipeline_kwargs,
    ):
        self.base_dir = Path(base_dir) if base_dir else None
        self.atom37_transforms = atom37_transforms or transforms or []
        self.path_column = path_column
        self.id_column = id_column
        self.file_extension = file_extension
        self.use_parse = use_parse
        self.parser_args = parser_args or {}
        self.pipeline_kwargs = pipeline_kwargs

        # Build the atomarray transform - either from pipeline or list
        if pipeline_target is not None:
            # Pipeline mode: build transform from target function
            pipeline_fn = hydra.utils.get_method(pipeline_target)
            self.atomarray_transform = pipeline_fn(is_inference=is_inference, **pipeline_kwargs)
            self._use_pipeline_mode = True
        else:
            # Simple mode: use list of transforms
            self.atomarray_transforms_list = atomarray_transforms or []
            self.atomarray_transform = None
            self._use_pipeline_mode = False

        # Load metadata from file or use provided DataFrame
        if metadata_df is not None:
            self.metadata = metadata_df
        elif metadata_file is not None:
            metadata_path = Path(metadata_file)
            if metadata_path.suffix == ".parquet":
                self.metadata = (
                    pd.read_parquet(metadata_path, columns=columns_to_load)
                    if columns_to_load
                    else pd.read_parquet(metadata_path)
                )
            else:
                self.metadata = (
                    pd.read_csv(metadata_path, usecols=columns_to_load)
                    if columns_to_load
                    else pd.read_csv(metadata_path)
                )
        else:
            raise ValueError("Either metadata_file or metadata_df must be provided")

        if filters:
            for f in filters:
                self.metadata = self.metadata.query(f)
            self.metadata = self.metadata.reset_index(drop=True)

        logger.info(f"Loaded {len(self.metadata)} samples")

    def __len__(self) -> int:
        return len(self.metadata)

    def _construct_path(self, row) -> Path:
        path = row[self.path_column]
        if self.file_extension:
            path = str(path) + self.file_extension
        if self.base_dir and not Path(path).is_absolute():
            return self.base_dir / path
        return Path(path)

    def __getitem__(self, idx: int) -> Data | None:
        row = self.metadata.iloc[idx]
        path = self._construct_path(row)
        sample_id = row.get(self.id_column, str(idx))

        try:
            atom_array = load_structure(
                str(path),
                use_parse=self.use_parse,
                **self.parser_args,
            )
        except Exception as e:
            logger.warning(f"StructureDataset:__getitem__: Failed to load {path}: {e}")
            return None

        # Apply transforms based on mode
        try:
            if self._use_pipeline_mode:
                # Pipeline mode: transform expects dict with atom_array
                atom_array = _ensure_atomworks_annotations(atom_array)
                data = {
                    "atom_array": atom_array,
                    "example_id": sample_id,
                    "file_path": str(path),
                }
                # Add metadata columns to data dict
                for col in row.index:
                    if col not in [self.path_column, self.id_column]:
                        data[col] = row[col]

                data = self.atomarray_transform(data)
                atom_array = data.get("atom_array")
                if atom_array is None:
                    logger.warning(f"No atom_array after transform for {path}")
                    return None
            else:
                # Simple mode: apply list of transforms
                atom_array = self._apply_atomarray_transforms(atom_array)
                data = None
        except Exception as e:
            logger.warning(f"Transform failed for {path}: {e}")
            return None
        # Convert to atom37
        data = atomarray_to_atom37(atom_array, sample_id=str(sample_id), atomworks_data=data)
        # Apply atom37 transforms
        for transform in self.atom37_transforms:
            data = transform(data)

        return data

    def _apply_atomarray_transforms(self, atom_array):
        """Apply atomarray transforms list, supporting atomworks-style transforms."""
        data = {"atom_array": atom_array}
        for transform in self.atomarray_transforms_list:
            if hasattr(transform, "forward"):
                # atomworks-style transform
                data = transform(data)
                if data is None or "atom_array" not in data:
                    raise ValueError("Atomarray transform did not return data with atom_array.")
            else:
                # Simple callable transform
                data["atom_array"] = transform(data["atom_array"])
        return data["atom_array"]


def pad_tensor(
    tensor: torch.Tensor,
    dim_sizes: dict[int, int],
    fill_value: float = 0,
) -> torch.Tensor:
    """Pad a tensor along multiple dimensions to specified sizes.

    Args:
        tensor: Input tensor to pad.
        dim_sizes: Dict mapping dimension index to target size, e.g., {0: 100, 1: 50}.
        fill_value: Value to use for padding (default: 0).

    Returns:
        Padded tensor.

    Example:
        # Pad 1D tensor [n] to size 100
        padded = pad_tensor(t, {0: 100})

        # Pad 2D square tensor [n, n] to [100, 100]
        padded = pad_tensor(t, {0: 100, 1: 100})

        # Pad 3D tensor [n, 37, 3] along first dim only
        padded = pad_tensor(t, {0: 100})
    """
    if not dim_sizes:
        return tensor

    # Check if padding is actually needed
    needs_padding = any(tensor.size(dim) < size for dim, size in dim_sizes.items())
    if not needs_padding:
        return tensor

    # Build padding tuple in reverse dim order for torch.nn.functional.pad
    # Format: (dim_n_left, dim_n_right, ..., dim_0_left, dim_0_right)
    padding = []
    for dim in range(tensor.dim() - 1, -1, -1):
        if dim in dim_sizes and tensor.size(dim) < dim_sizes[dim]:
            pad_size = dim_sizes[dim] - tensor.size(dim)
            padding.extend([0, pad_size])  # left=0, right=pad_size
        else:
            padding.extend([0, 0])

    return torch.nn.functional.pad(tensor, tuple(padding), mode="constant", value=fill_value)


def _pad_and_stack(
    key: str,
    value_list: list,
    pad_size: int,
    sample_lengths: list[int] | None = None,
) -> torch.Tensor | list | None:
    """Pad a list of values to pad_size and stack into a batch tensor.

    Args:
        key: Name of the field being collated.
        value_list: List of values (one per sample in the batch).
        pad_size: Target size for the padded dimension.
        sample_lengths: Per-sample sequence lengths (num_nodes). Used to detect
            genuine pair tensors [n, n, ...] vs tensors that coincidentally have
            equal first two dims (e.g. coords [37, 37, 3] when n_residues=37).

    Returns:
        Stacked tensor, plain list, or None if skipped.
    """
    if value_list[0] is None:
        return None

    if isinstance(value_list[0], torch.Tensor):
        ndim = value_list[0].dim()
        if ndim == 0:
            return torch.stack(value_list)

        # Detect genuine pair tensors: both dim 0 and dim 1 must equal
        # num_nodes for ALL samples. This avoids the shape[0]==shape[1]
        # heuristic which breaks when n_residues coincides with a fixed
        # dimension (e.g. n=37 matching atom37 in coords [37, 37, 3]).
        is_pair = (
            ndim >= 2
            and sample_lengths is not None
            and all(
                t.shape[0] == length and t.shape[1] == length
                for t, length in zip(value_list, sample_lengths, strict=False)
            )
        )

        if is_pair:
            padded = [pad_tensor(t, {0: pad_size, 1: pad_size}) for t in value_list]
        else:
            padded = [pad_tensor(t, {0: pad_size}) for t in value_list]
        return torch.stack(padded)

    elif isinstance(value_list[0], (int, float)):
        return torch.tensor(value_list)
    else:
        return value_list


def _discover_groups(
    batch: list[Data],
) -> dict[str, dict[str, Any]]:
    """Discover non-binder token groups registered by extraction transforms.

    Extraction transforms (``ExtractTargetCoordinatesTransform``,
    ``ExtractMotifCoordinatesTransform``) call ``Data.register_token_group()``
    in compact mode to declare which fields belong to each group.  This
    function reads that registry and computes per-sample lengths.

    Returns:
        ``{group_name: {"fields": set[str], "lengths": list[int]}}``
    """
    first = batch[0]
    token_groups: dict[str, list[str]] = getattr(first, "_token_groups", {})

    # Validate that all samples share the same token group structure and fields
    for i, sample in enumerate(batch[1:], 1):
        sample_groups = getattr(sample, "_token_groups", {})
        if set(sample_groups.keys()) != set(token_groups.keys()):
            raise ValueError(
                f"Sample {i} has different token groups "
                f"({set(sample_groups.keys())}) than sample 0 "
                f"({set(token_groups.keys())}). "
                f"All samples in a batch must have the same group structure."
            )
        for group_name in token_groups:
            if set(sample_groups[group_name]) != set(token_groups[group_name]):
                raise ValueError(
                    f"Sample {i} has different fields for group '{group_name}' "
                    f"({set(sample_groups[group_name])}) than sample 0 "
                    f"({set(token_groups[group_name])})."
                )

    groups: dict[str, dict[str, Any]] = {}
    for name, fields in token_groups.items():
        field_set = set(fields)
        rep = next((f for f in fields if hasattr(first, f)), None)
        if rep is None:
            raise ValueError(
                f"Token group '{name}' is registered but none of its fields "
                f"({fields}) exist in the sample. This indicates a bug in the "
                f"extraction transform or data pipeline."
            )
        lengths = [getattr(s, rep).shape[0] for s in batch]
        groups[name] = {"fields": field_set, "lengths": lengths}

    return groups


def _allocate_budget(
    binder_max: int,
    groups: dict[str, dict[str, Any]],
    budget: int,
    priority: list[str] | None,
) -> tuple[int, dict[str, int]]:
    """Allocate a fixed token budget across binder + groups.

    When *priority* is ``None`` (shared/proportional mode), all groups and
    binder are scaled proportionally if the total exceeds the budget.  When
    under budget the surplus is distributed evenly.

    When *priority* is a list, named groups are allocated greedily in order
    (each gets up to its max-in-batch).  Binder gets whatever remains.  An
    empty list means binder-first.

    Returns:
        ``(binder_pad, {group_name: pad_size})``
    """
    group_maxes = {name: max(g["lengths"]) if g["lengths"] else 0 for name, g in groups.items()}

    if priority is None:
        # Proportional allocation
        total = binder_max + sum(group_maxes.values())
        if total <= budget:
            n_parts = 1 + len(group_maxes)
            surplus = budget - total
            per_part = surplus // n_parts
            binder_pad = binder_max + per_part
            group_pads = {n: mx + per_part for n, mx in group_maxes.items()}
            leftover = budget - binder_pad - sum(group_pads.values())
            binder_pad += leftover
        elif total > 0:
            ratio = budget / total
            group_pads = {n: int(mx * ratio) for n, mx in group_maxes.items()}
            binder_pad = budget - sum(group_pads.values())
        else:
            binder_pad = budget
            group_pads = dict.fromkeys(group_maxes, 0)
        return binder_pad, group_pads

    # Priority-based greedy allocation
    remaining = budget
    group_pads: dict[str, int] = {}
    for name in priority:
        if name not in group_maxes:
            continue
        alloc = min(group_maxes[name], remaining)
        group_pads[name] = alloc
        remaining -= alloc

    binder_pad = max(remaining, 0)

    # Groups not in the priority list must be explicitly listed or they're an error.
    for name in group_maxes:
        if name not in group_pads:
            if group_maxes[name] > 0:
                raise ValueError(
                    f"Group '{name}' has {group_maxes[name]} tokens in batch but is not in "
                    f"pad_group_priority list. Add '{name}' to pad_group_priority or remove "
                    f"the extraction transform that creates this group."
                )
            group_pads[name] = 0

    return binder_pad, group_pads


def structure_collate_fn(
    batch: list[Data | None],
    pad_max_total_tokens: int | None = None,
    pad_group_priority: list[str] | None = None,
) -> dict[str, Any]:
    """Collate function with generic group-based padding.

    Fields are assigned to groups (target, motif, etc.) via the
    ``_token_groups`` registry populated by extraction transforms.
    Fields not belonging to any group are treated as binder fields.

    When ``pad_max_total_tokens`` is set, the total budget is distributed
    across binder + groups via ``_allocate_budget``.

    Args:
        batch: List of Data objects (None values are filtered out).
        pad_max_total_tokens: If provided, total padding across all groups
            is capped to this value.  If None, each group is padded to its
            own max in the batch (dynamic).
        pad_group_priority: Controls budget distribution when
            ``pad_max_total_tokens`` is set.  ``None`` = proportional,
            ``["target", "motif"]`` = greedy priority, ``[]`` = binder-first.

    Returns:
        Dictionary with ``nsamples``, ``nres``, per-group ``n_{group}``,
        ``mask`` (binder), and ``{group}_padding_mask`` for each group.
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples in batch failed to load")

    first_sample = batch[0]

    # ---- Discover token groups ----
    groups = _discover_groups(batch)
    grouped_fields: set[str] = set()
    for g in groups.values():
        grouped_fields |= g["fields"]

    # ---- Compute binder lengths ----
    binder_lengths = [getattr(s, "num_nodes", 0) for s in batch]

    # ---- Compute target lengths from a representative target tensor ----
    has_target = hasattr(first_sample, "target_mask") and first_sample.target_mask is not None
    if has_target:
        target_lengths = [s.target_mask.shape[0] for s in batch]
    else:
        target_lengths = [0] * len(batch)

    # ---- Determine pad sizes ----
    max_binder = max(binder_lengths) if binder_lengths else 0

    # ---- Allocate pad sizes ----
    group_pads: dict[str, int] = {}
    if pad_max_total_tokens is not None:
        binder_pad, group_pads = _allocate_budget(
            max_binder,
            groups,
            pad_max_total_tokens,
            pad_group_priority,
        )
    else:
        binder_pad = max_binder
        for name, g in groups.items():
            group_pads[name] = max(g["lengths"]) if g["lengths"] else 0

    logger.debug(
        f"Pad binder={binder_pad} groups={group_pads} (budget={pad_max_total_tokens}, priority={pad_group_priority})"
    )

    # ---- Build result dict ----
    batch_size = len(batch)
    result: dict[str, Any] = {
        "nsamples": batch_size,
        "nres": binder_pad,
    }
    for name, pad in group_pads.items():
        result[f"n_{name}"] = pad

    # ---- Collate each field ----
    for key in first_sample.keys():
        if key == "num_nodes":
            continue

        value_list = [getattr(sample, key) for sample in batch]

        try:
            group_name = next(
                (n for n, g in groups.items() if key in g["fields"]),
                None,
            )
            if group_name is not None:
                ps = group_pads[group_name]
                lengths = groups[group_name]["lengths"]
            else:
                ps = binder_pad
                lengths = binder_lengths

            collated = _pad_and_stack(key, value_list, ps, sample_lengths=lengths)
            if collated is not None:
                result[key] = collated
        except Exception as e:
            logger.warning(f"Collate error for '{key}': {e}")

    # ---- Binder padding mask [B, binder_pad] ----
    mask = torch.zeros(batch_size, binder_pad, dtype=torch.bool)
    for i, length in enumerate(binder_lengths):
        mask[i, : min(length, binder_pad)] = True
    result["mask"] = mask

    # ---- Per-group padding masks [B, group_pad] ----
    for name, g in groups.items():
        pad = group_pads.get(name, 0)
        if pad > 0:
            gmask = torch.zeros(batch_size, pad, dtype=torch.bool)
            for i, length in enumerate(g["lengths"]):
                gmask[i, : min(length, pad)] = True
            result[f"{name}_padding_mask"] = gmask

    return result


def make_collate_fn(
    pad_max_total_tokens: int | None = None,
    pad_group_priority: list[str] | None = None,
):
    """Factory to create a collate function with optional fixed padding size.

    Args:
        pad_max_total_tokens: If provided, total padding across all groups
            is capped to this value.
        pad_group_priority: Controls budget distribution (see
            ``structure_collate_fn`` for details).  ``None`` = proportional,
            ``["target", "motif"]`` = greedy priority, ``[]`` = binder-first.

    Returns:
        A collate function for use with DataLoader.

    Example:
        # Dynamic padding (default)
        collate_fn = make_collate_fn()

        # Fixed 512-token budget, target and motif get priority
        collate_fn = make_collate_fn(
            pad_max_total_tokens=512,
            pad_group_priority=["target", "motif"],
        )

        # Proportional allocation across all groups
        collate_fn = make_collate_fn(
            pad_max_total_tokens=512,
            pad_group_priority=None,
        )
    """

    def collate_fn(batch):
        return structure_collate_fn(
            batch,
            pad_max_total_tokens=pad_max_total_tokens,
            pad_group_priority=pad_group_priority,
        )

    return collate_fn


class StructureDataModule(L.LightningDataModule):
    """Lightning DataModule for structure loading.

    Supports two modes:
    1. Simple mode (atomarray_transforms): List of transforms applied sequentially
    2. Pipeline mode (pipeline_target): A function that builds a complete transform pipeline

    Both modes convert to atom37 at the end and apply atom37_transforms.

    Args:
        metadata_file: Path to CSV or parquet metadata file.
        pipeline_target: Optional path to pipeline build function (e.g. "module.build_pipeline").
        atomarray_transforms: List of transforms for AtomArray (simple mode).
        atom37_transforms: Transforms for atom37 Data (applied after conversion).
    """

    def __init__(
        self,
        metadata_file: str,
        base_dir: str = "",
        batch_size: int = 32,
        num_workers: int = 8,
        atomarray_transforms: list[Callable] | None = None,
        atom37_transforms: list[Callable] | None = None,
        transforms: list[Callable] | None = None,  # Legacy alias
        pipeline_target: str | None = None,
        train_split: float = 0.9,
        path_column: str = "path",
        id_column: str = "id",
        pin_memory: bool = True,
        use_parse: bool = False,
        parser_args: dict | None = None,
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        val_metadata_file: str | None = None,
        val_filters: list[str] | None = None,
        pad_max_total_tokens: int | None = None,
        pad_group_priority: list[str] | None = None,
        **pipeline_kwargs,
    ):
        super().__init__()
        self.metadata_file = metadata_file
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.atomarray_transforms = atomarray_transforms or []
        self.atom37_transforms = atom37_transforms or transforms or []
        self.pipeline_target = pipeline_target
        self.train_split = train_split
        self.path_column = path_column
        self.id_column = id_column
        self.pin_memory = pin_memory
        self.use_parse = use_parse
        self.parser_args = parser_args or {}
        self.filters = filters
        self.columns_to_load = columns_to_load
        self.val_metadata_file = val_metadata_file
        self.val_filters = val_filters
        self.pad_max_total_tokens = pad_max_total_tokens
        self.pad_group_priority = pad_group_priority
        self.pipeline_kwargs = pipeline_kwargs

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None):
        # Load metadata
        metadata_path = Path(self.metadata_file)
        if metadata_path.suffix == ".parquet":
            full_metadata = pd.read_parquet(metadata_path)
        else:
            full_metadata = pd.read_csv(metadata_path)

        if self.filters:
            for f in self.filters:
                full_metadata = full_metadata.query(f)
            full_metadata = full_metadata.reset_index(drop=True)

        # Split train/val
        if self.val_metadata_file:
            train_metadata = full_metadata
            val_path = Path(self.val_metadata_file)
            val_metadata = pd.read_parquet(val_path) if val_path.suffix == ".parquet" else pd.read_csv(val_path)
            if self.val_filters:
                for f in self.val_filters:
                    val_metadata = val_metadata.query(f)
                val_metadata = val_metadata.reset_index(drop=True)
        else:
            n_train = int(len(full_metadata) * self.train_split)
            train_metadata = full_metadata.iloc[:n_train].reset_index(drop=True)
            val_metadata = full_metadata.iloc[n_train:].reset_index(drop=True)

        # Instantiate atom37_transforms if they're config dicts
        atom37_transforms = []
        for t in self.atom37_transforms:
            if isinstance(t, dict) and "_target_" in t:
                atom37_transforms.append(hydra.utils.instantiate(t))
            elif callable(t):
                atom37_transforms.append(t)

        # Create train dataset
        self.train_dataset = StructureDataset(
            metadata_df=train_metadata,
            base_dir=self.base_dir,
            atomarray_transforms=self.atomarray_transforms,
            atom37_transforms=atom37_transforms,
            pipeline_target=self.pipeline_target,
            path_column=self.path_column,
            id_column=self.id_column,
            use_parse=self.use_parse,
            parser_args=self.parser_args,
            is_inference=False,
            **self.pipeline_kwargs,
        )

        # Create val dataset
        if len(val_metadata) > 0:
            self.val_dataset = StructureDataset(
                metadata_df=val_metadata,
                base_dir=self.base_dir,
                atomarray_transforms=self.atomarray_transforms,
                atom37_transforms=atom37_transforms,
                pipeline_target=self.pipeline_target,
                path_column=self.path_column,
                id_column=self.id_column,
                use_parse=self.use_parse,
                parser_args=self.parser_args,
                is_inference=True,  # Val uses inference mode
                **self.pipeline_kwargs,
            )

    def train_dataloader(self):
        collate_fn = make_collate_fn(
            pad_max_total_tokens=self.pad_max_total_tokens,
            pad_group_priority=self.pad_group_priority,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        collate_fn = make_collate_fn(
            pad_max_total_tokens=self.pad_max_total_tokens,
            pad_group_priority=self.pad_group_priority,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def add_validation_dataloader(self, cfg_val_data, n_replicas: int = 1):
        """Compatibility method for validation dataloader from config."""
