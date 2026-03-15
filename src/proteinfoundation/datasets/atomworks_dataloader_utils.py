#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../scripts/shebang/modelhub_exec.sh" "$0" "$@"'
from typing import Any

import torch
from atomworks.ml.transforms._checks import check_contains_keys, check_is_instance
from biotite.structure import AtomArray

# TODO: This file will be cleaned up. padding should just pad. everything else is a transform


def atomworks_pad_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Pads the batch of atomworks data.

    Args:
        batch: List of dictionaries containing atomworks data

    Returns:
        Dictionary containing padded atomworks data
    """
    padder = PadTokenAtom()
    max_num_atoms = max(x["all_coords_nm"].shape[0] for x in batch)
    max_num_tokens = max(x["token_mask"].shape[0] for x in batch)
    max_num_tokens2 = max(x["protein_token_mask"].shape[0] for x in batch)
    batch = [padder.pad(x, max_num_atoms=max_num_atoms, max_num_tokens=max_num_tokens) for x in batch]
    result = concat_tensor_dicts(batch)
    ligand_feats = get_ligand_feats(result)
    protein_feats = get_protein_feats(result)
    # result["target"] = ligand_feats
    # result["binder"] = protein_feats
    new_result = {}
    result = new_result
    result["target_mask"] = ligand_feats["atom_mask"]
    B, N = result["target_mask"].shape
    result["x_target"] = ligand_feats["coords_nm"]
    result["seq_target"] = ligand_feats["atom_types"]
    result["target_charge"] = ligand_feats["atom_charges"]
    result["target_atom_name"] = ligand_feats["atom_names"].reshape(B, N, -1)
    result["target_laplacian_pe"] = ligand_feats["laplacian_pe"]
    result["target_bond_order"] = ligand_feats["bond_order"]
    result["target_bond_mask"] = ligand_feats["atom_bonds"]

    result["coords_nm"] = protein_feats["coords37_nm"]
    result["coords"] = result["coords_nm"] * 10  # needed for angle features
    result["coord_mask"] = protein_feats["mask"]
    result["mask"] = protein_feats["mask"][..., 0]
    lengths = result["mask"].sum(dim=-1)
    test = [x == result["mask"].shape[1] for x in lengths]
    if not any(test):
        raise ValueError(f"Padding error: {lengths} {max_num_tokens2} {max_num_tokens}")
        test = 1
    result["residue_pdb_idx"] = protein_feats["residue_pdb_idx"]
    result["residue_type"] = protein_feats["restype"]
    result["chain_breaks_per_residue"] = protein_feats["chain_breaks_per_residue"]

    return result


def concat_tensor_dicts(dict_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Recursively concatenate torch tensors from a list of dictionaries.

    Args:
        dict_list: List of dictionaries with potentially nested structure containing tensors

    Returns:
        Dictionary with the same structure as input, but with concatenated tensors
    """
    if not dict_list:
        return {}

    # Get the first dict to understand the structure
    first_dict = dict_list[0]
    result = {}

    for key, value in first_dict.items():
        if isinstance(value, torch.Tensor):
            # Concatenate tensors along the first dimension
            tensors = [d[key] for d in dict_list]
            try:
                result[key] = torch.stack(tensors, dim=0)
            except Exception as e:
                print(e)
                print([x.shape for x in tensors])
                print(key)
                raise e
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            nested_dicts = [d[key] for d in dict_list]
            result[key] = concat_tensor_dicts(nested_dicts)
    return result


def pad_tensor(tensor, max_size, dim, fill_value=0):
    if tensor.size(dim) >= max_size:
        return tensor

    pad_size = max_size - tensor.size(dim)
    padding = [0] * (2 * tensor.dim())
    padding[2 * (tensor.dim() - 1 - dim) + 1] = pad_size
    return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)


def pad_tensor_multi_dim(tensor, dim_sizes, fill_value=0):
    """
    Pad tensor along multiple dimensions to specified sizes.

    Args:
        tensor: Input tensor to pad
        dim_sizes: Dict mapping dimension index to max size (e.g., {0: 100, 1: 50})
        fill_value: Value to use for padding (default: 0)
    """
    if not dim_sizes:
        return tensor

    # Check if padding is needed
    needs_padding = False
    for dim, max_size in dim_sizes.items():
        if tensor.size(dim) < max_size:
            needs_padding = True
            break

    if not needs_padding:
        return tensor

    # Create padding tuple in the correct order for torch.nn.functional.pad
    # torch.nn.functional.pad expects: (pad_left_dimN, pad_right_dimN, ..., pad_left_dim0, pad_right_dim0)
    padding = []
    for dim in range(tensor.dim() - 1, -1, -1):  # Start from last dimension
        if dim in dim_sizes and tensor.size(dim) < dim_sizes[dim]:
            pad_size = dim_sizes[dim] - tensor.size(dim)
            padding.extend([0, pad_size])  # pad_left=0, pad_right=pad_size
        else:
            padding.extend([0, 0])  # no padding for this dimension

    return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)


def extract_and_pad_features(result, feature_key, mask_key, feature_dim=None, pair=False, feature_result=None):
    """Generic function to extract and pad features using a mask."""
    # Extract real features for each batch item
    if feature_result is None:
        feature_result = result
    if pair:
        real_features = [
            feature_result[feature_key][i][result[mask_key][i]][:, result[mask_key][i]]
            for i in range(result[feature_key].shape[0])
        ]
    else:
        real_features = [
            feature_result[feature_key][i][result[mask_key][i]] for i in range(feature_result[feature_key].shape[0])
        ]

    # Find maximum size
    max_atoms = max(feats.shape[0] for feats in real_features)
    batch_size = len(real_features)

    if pair:
        padded_features = torch.zeros(batch_size, max_atoms, max_atoms)
    else:
        # Create padded tensor
        if feature_dim is None:
            padded_features = torch.zeros(batch_size, max_atoms)
        else:
            padded_features = torch.zeros(batch_size, max_atoms, *feature_dim)

    # Fill with real data
    for i, feats in enumerate(real_features):
        if pair:
            padded_features[i, : feats.shape[0], : feats.shape[1]] = feats
        else:
            padded_features[i, : feats.shape[0]] = feats

    return padded_features, max_atoms


def get_ligand_feats(result: dict[str, Any]) -> dict[str, Any]:
    """Extract and pad ligand features from the result dictionary."""

    # Extract and pad all features
    padded_coords, max_atoms = extract_and_pad_features(result, "all_coords_nm", "ligand_atom_mask", (3,))
    padded_atom_types, _ = extract_and_pad_features(result, "atom_element", "ligand_atom_mask", (128,))
    padded_atom_names, _ = extract_and_pad_features(result, "atom_name_chars", "ligand_atom_mask", (4, 64))
    padded_atom_charges, _ = extract_and_pad_features(result, "atom_charge", "ligand_atom_mask")
    padded_atom_bonds, _ = extract_and_pad_features(result, "token_bond_mask", "ligand_token_mask", pair=True)
    padded_ligand_bond_order, _ = extract_and_pad_features(result, "atom_bond_order", "ligand_atom_mask", pair=True)
    padded_ligand_laplacian_pe, _ = extract_and_pad_features(result, "token_laplacian_pe", "ligand_token_mask", (32,))
    # Create mask
    batch_size = padded_coords.shape[0]
    padded_mask = torch.zeros(batch_size, max_atoms)
    for i in range(batch_size):
        num_atoms = result["ligand_atom_mask"][i].sum().item()
        padded_mask[i, :num_atoms] = 1
    return {
        "coords_nm": padded_coords,
        "atom_mask": padded_mask,
        "atom_types": padded_atom_types,
        "atom_names": padded_atom_names,
        "atom_charges": padded_atom_charges,
        "atom_bonds": padded_atom_bonds,
        "laplacian_pe": padded_ligand_laplacian_pe,
        "bond_order": padded_ligand_bond_order,
    }


def get_protein_feats(result: dict[str, Any]) -> dict[str, Any]:
    """Extract and pad ligand features from the result dictionary."""

    # Extract and pad all features
    protein_result = result["protein_atom37_feats"]
    result["protein_token_mask"]
    # total_num_residues = protein_token_mask.sum(1).max().item()
    total_num_residues = (
        protein_result["atom37_mask"][..., 0].sum(1).max().item()
    )  #! for plinder I am making this change

    protein_feats = {
        "coords37_nm": protein_result["atom37"][:, :total_num_residues],
        "mask": protein_result["atom37_mask"][:, :total_num_residues],
        "restype": protein_result["restype"][:, :total_num_residues],
        "residue_pdb_idx": result["residue_pdb_idx"][:, :total_num_residues],
        "chain_breaks_per_residue": protein_result["chain_breaks_per_residue"][:, :total_num_residues],
    }
    return protein_feats


class PadTokenAtom:
    """
    Pads the token and atom arrays to the max crop size.
    """

    def __init__(self, max_num_atoms: int = 5000, max_num_tokens: int = 384):
        self.max_num_atoms = max_num_atoms
        self.max_num_tokens = max_num_tokens

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def pad(
        self,
        data: dict[str, Any],
        max_num_atoms: int = None,
        max_num_tokens: int = None,
    ) -> dict[str, Any]:
        if max_num_atoms is None:
            max_num_atoms = self.max_num_atoms
        if max_num_tokens is None:
            max_num_tokens = self.max_num_tokens
        for k, v in data.items():
            if k == "atom_array":
                continue
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.bool:
                    fill_value = False
                else:
                    fill_value = 0
                if "bond" in k:
                    if "atom" in k:
                        pad_dim = {0: max_num_atoms, 1: max_num_atoms}
                    else:
                        pad_dim = {0: max_num_tokens, 1: max_num_tokens}
                    data[k] = self.pad_tensor_multi_dim(v, pad_dim, fill_value)
                elif "atom" in k or k == "all_coords_nm":
                    data[k] = self.pad_tensor(v, max_num_atoms, 0, fill_value)
                else:
                    data[k] = self.pad_tensor(v, max_num_tokens, 0, fill_value)
            else:
                for kk, vv in v.items():
                    if not isinstance(vv, torch.Tensor):
                        continue
                    if vv.dtype == torch.bool:
                        fill_value = False
                    else:
                        fill_value = 0
                    if "bond" in kk:
                        if "atom" in kk:
                            pad_dim = {0: max_num_atoms, 1: max_num_atoms}
                        else:
                            pad_dim = {0: max_num_tokens, 1: max_num_tokens}
                        padded_vv = self.pad_tensor_multi_dim(vv, pad_dim, fill_value)
                    else:
                        if "atom" in kk and kk not in ["rep_atom_idxs"] and "atom37" not in kk:
                            padded_vv = self.pad_tensor(vv, max_num_atoms, 0, fill_value)
                        else:
                            padded_vv = self.pad_tensor(vv, max_num_tokens, 0, fill_value)
                    data[k][kk] = padded_vv
        return data

    def pad_tensor(self, tensor, max_size, dim, fill_value=0):
        if tensor.size(dim) >= max_size:
            return tensor

        pad_size = max_size - tensor.size(dim)
        padding = [0] * (2 * tensor.dim())
        padding[2 * (tensor.dim() - 1 - dim) + 1] = pad_size
        return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)

    def pad_tensor_multi_dim(self, tensor, dim_sizes, fill_value=0):
        """
        Pad tensor along multiple dimensions to specified sizes.

        Args:
            tensor: Input tensor to pad
            dim_sizes: Dict mapping dimension index to max size (e.g., {0: 100, 1: 50})
            fill_value: Value to use for padding (default: 0)
        """
        if not dim_sizes:
            return tensor

        # Check if padding is needed
        needs_padding = False
        for dim, max_size in dim_sizes.items():
            if tensor.size(dim) < max_size:
                needs_padding = True
                break

        if not needs_padding:
            return tensor

        # Create padding tuple in the correct order for torch.nn.functional.pad
        # torch.nn.functional.pad expects: (pad_left_dimN, pad_right_dimN, ..., pad_left_dim0, pad_right_dim0)
        padding = []
        for dim in range(tensor.dim() - 1, -1, -1):  # Start from last dimension
            if dim in dim_sizes and tensor.size(dim) < dim_sizes[dim]:
                pad_size = dim_sizes[dim] - tensor.size(dim)
                padding.extend([0, pad_size])  # pad_left=0, pad_right=pad_size
            else:
                padding.extend([0, 0])  # no padding for this dimension

        return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)
