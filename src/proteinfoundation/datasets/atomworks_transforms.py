import logging
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from atomworks.constants import STANDARD_AA
from atomworks.io.transforms.atom_array import remove_waters
from atomworks.ml.encoding_definitions import TokenEncoding
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from atomworks.ml.utils.token import get_af3_token_representative_idxs, get_token_starts
from biotite.structure import AtomArray

from proteinfoundation.nn.feature_factory.feature_utils import BOND_ORDER_MAP
from proteinfoundation.utils.align_utils import mean_w_mask

# https://www.biotite-python.org/latest/apidoc/biotite.structure.BondType.html#biotite.structure.BondType
# https://baker-laboratory.github.io/atomworks-dev/latest/io/constants.html#atomworks.constants.BIOTITE_BOND_TYPE_TO_BOND_ORDER
# Combination of the two to ensure no failures ever
logger = logging.getLogger("atomworks.io.custom")


class FilterForProteinaLigandComplex(Transform):
    """
    Filters PN units that have all unresolved atoms (i.e., atoms with occupancy 0) from the AtomArray.

    Can be applied before or after croppping, since cropping may lead to PN units with all unresolved atoms that were previously not entirely unresolved.
    At training time, these unresolved PN units provide minimal value and can lead to errors in the model.
    """

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["hetero", "is_polymer"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        atom_array = remove_waters(atom_array)
        # If the atom array has <=2 chain_ids then we do nothing
        if len(np.unique(atom_array.chain_id)) <= 2:
            data["atom_array"] = atom_array
            return data
        elif len(np.unique(atom_array.chain_id)) > 2:
            # If the atom array has more than 2 chains first figure out how many are protein chains, how many are ligand chains
            protein_chain_ids = np.unique(atom_array.chain_id[atom_array.is_polymer == True])
            ligand_chain_ids = np.unique(atom_array.chain_id[atom_array.is_polymer == False])
            if len(protein_chain_ids) > 1:
                # Keep 1 protein chain
                protein_chain_id = np.random.choice(protein_chain_ids)
                keep_chains = np.concatenate([[protein_chain_id], ligand_chain_ids])
                atom_array = atom_array[np.isin(atom_array.chain_id, keep_chains)]
            if len(ligand_chain_ids) > 1:
                # rename the second ligand chain to the first ligand chain and make 1 big ligand
                ligand_chain_id = ligand_chain_ids[0]
                mask = np.isin(atom_array.chain_id, ligand_chain_ids) & (atom_array.chain_id != ligand_chain_id)
                atom_array.chain_id[mask] = ligand_chain_id
        data["atom_array"] = atom_array
        return data


class BreakPointTransform(Transform):
    """
    Breaks the point in the pipeline to allow for custom processing.
    """

    def __init__(self):
        pass

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        import ipdb

        ipdb.set_trace()
        return data


class ProteinaFinalTransform(Transform):
    """
    Final transform for the protein.
    """

    def __init__(self):
        pass

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # coords_nm = ang_to_nm(data["coord_atom_lvl_to_be_noised"])
        # rot = sample_uniform_rotation(dtype=coords_nm.dtype, device=coords_nm.device)
        # data["all_coords_nm"] = torch.matmul(coords_nm, rot)
        data["atom_mask"] = data["ground_truth"]["mask_atom_lvl"]
        data["mask"] = data["token_mask"] = data["ground_truth"]["mask_token_lvl"]
        data["protein_token_mask"] = data["feats"]["is_protein"]
        data["ligand_token_mask"] = data["feats"]["is_ligand"]
        data["protein_atom_mask"] = data["feats"]["is_protein"][data["feats"]["atom_to_token_map"]] * data["atom_mask"]
        data["ligand_atom_mask"] = data["feats"]["is_ligand"][data["feats"]["atom_to_token_map"]] * data["atom_mask"]
        data["residue_pdb_idx"] = data["feats"]["residue_index"]
        is_ligand = data["feats"]["is_ligand"]  #!token level
        if not is_ligand.any():
            return data
        data["atom_element"] = data["feats"]["atom_element"]
        data["atom_name_chars"] = data["feats"]["atom_name_chars"]
        data["atom_charge"] = data["feats"]["atom_charge"]
        data["token_bond_mask"] = data["feats"]["token_bonds"]
        data["residue_type"] = data["ground_truth"]["restype_idx"]
        # data["ligand_laplacian_pe"] = data["feats"]["ligand_laplacian_pe"]
        data["token_laplacian_pe"] = data["feats"]["ligand_laplacian_pe_all"]
        bm = data["atom_array"].bonds.bond_type_matrix()
        bm = np.vectorize(BOND_ORDER_MAP.get)(bm)
        bm = torch.from_numpy(bm).float()
        data["atom_bond_order"] = bm

        #! convert anything else that has slipped through
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v)
        return data


class ProteinaCenteringTransform(Transform):
    """
    Centers the protein on the CA atoms.
    """

    def __init__(
        self,
        center_mode: str = "full",
        data_mode: str = "all-atom",
        variance_perturbation: float = 0,
    ):
        self.center_mode = center_mode
        self.data_mode = data_mode
        assert data_mode in ["all-atom"]
        self.variance_perturbation = variance_perturbation

    # def check_input(self, data: dict[str, Any]) -> None:
    #     check_contains_keys(data, ["atom_array"])
    #     check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:

        atom_mask = data["ground_truth"]["mask_atom_lvl"]
        data["ground_truth"]["mask_token_lvl"]
        p_tokenmask = data["feats"]["is_protein"]
        p_atommask = p_tokenmask[data["feats"]["atom_to_token_map"]] * atom_mask
        if self.center_mode == "full":
            centering_mask = atom_mask
        elif self.center_mode == "ligand":
            centering_mask = (~p_atommask) * atom_mask
        elif self.center_mode == "protein":
            centering_mask = p_atommask
        else:
            raise ValueError(f"Invalid center mode: {self.center_mode}")
        coords = data["coord_atom_lvl_to_be_noised"].clone()
        mean = mean_w_mask(coords, centering_mask.bool(), keepdim=True)
        if self.variance_perturbation > 0:
            translation = torch.normal(
                mean=0.0,
                std=self.variance_perturbation**0.5,
                size=(3,),
                dtype=coords.dtype,
                device=coords.device,
            )
        coords -= mean
        coords = coords * atom_mask[..., None]

        data["coord_atom_lvl_to_be_noised"] = coords
        return data


class PaddingTokenAtomTransform(Transform):
    """
    Pads the token and atom arrays to the max crop size.
    """

    def __init__(self, max_num_atoms: int = 5000, max_num_tokens: int = 384):
        self.max_num_atoms = max_num_atoms
        self.max_num_tokens = max_num_tokens

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(
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
                if "bonds" in k:
                    data[k] = self.pad_tensor_multi_dim(v, {0: max_num_tokens, 1: max_num_tokens}, fill_value)
                elif "atom" in k or k == "all_coords_nm":
                    data[k] = self.pad_tensor(v, max_num_atoms, 0, fill_value)
                else:
                    data[k] = self.pad_tensor(v, max_num_tokens, 0, fill_value)
            else:
                for kk, vv in v.items():
                    # print(kk, vv.shape)
                    if not isinstance(vv, torch.Tensor):
                        continue
                    if vv.dtype == torch.bool:
                        fill_value = False
                    else:
                        fill_value = 0
                    if "bonds" in kk:
                        padded_vv = self.pad_tensor_multi_dim(vv, {0: max_num_tokens, 1: max_num_tokens}, fill_value)
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


class EncodeProteinAtomArray(Transform):
    """Encode an atom array to an arbitrary `TokenEncoding`.

    This will add the following information to the data dict:
        - `encoding` (dict)
            - `xyz`: Atom coordinates (`xyz`)
            - `mask`: Atom mask giving information about which atoms are resolved in the encoded sequence (`mask`)
            - `seq`: Token sequence (`seq`)
            - `token_is_atom`: Token type (atom or residue) (`token_is_atom`)
            - Various other optional annotations such as `chain_id`, `chain_entity`, etc. See `atom_array_to_encoding`
              for more details.
    """

    def __init__(
        self,
        encoding: TokenEncoding,
        default_coord: float | np.ndarray = float("nan"),
        occupancy_threshold: float = 0.0,
        extra_annotations: list[str] = [
            "chain_id",
            "chain_entity",
            "molecule_iid",
            "chain_iid",
            "transformation_id",
        ],
    ):
        """
        Convert an atom array to an encoding.

        Args:
            - `encoding` (TokenEncoding): The encoding to use for encoding the atom array.
            - `default_coord` (float | np.ndarray, optional): Default coordinate value. Defaults to float("nan").
            - `occupancy_threshold` (float, optional): Minimum occupancy for atoms to be considered resolved
                in the mask. Defaults to 0.0 (only completely unresolved atoms are masked).
            - `extra_annotations` (list[str], optional): Extra annotations to encode. These must be `id` style annotations
                like `chain_id` or `molecule_iid`, as the encoding will be generated as `int`s. Each first occurrence
                of a given `id` will be encoded as `0`, and each subsequent occurrence will be encoded as `1`, `2`, etc.
                Defaults to ["chain_id", "chain_entity", "molecule_iid", "chain_iid", "transformation_id"].
        """
        if not isinstance(encoding, TokenEncoding):
            raise ValueError(f"Encoding must be a `TokenEncoding`, but got: {type(encoding)}.")
        self.encoding = encoding
        self.default_coord = default_coord
        self.occupancy_threshold = occupancy_threshold
        self.extra_annotations = extra_annotations

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        if "all_coords_nm" in data:
            coords = data["all_coords_nm"]
            atom_array.coord = coords.numpy()
        p_tokenmask = data["feats"]["is_protein"]
        p_atommask = p_tokenmask[data["feats"]["atom_to_token_map"]] * torch.from_numpy(
            np.isin(atom_array.res_name, STANDARD_AA)
        )

        protein_atom_array = atom_array[p_atommask]
        # protein_atom_array.atomize *= False #! this seems to hack around non polymer protein residues
        encoded = atom_array_to_encoding(
            protein_atom_array,
            encoding=self.encoding,
            default_coord=self.default_coord,
            occupancy_threshold=self.occupancy_threshold,
            extra_annotations=self.extra_annotations,
        )

        data["protein_encoded"] = encoded
        return data


class PackageProteinAtom37Feats(Transform):
    """
    Restructures all the confidence information so it's included in the confidence_feats dictionary.
    Converts sequence to torch tensor. Properly indexes atom_frames to only include atomized tokens.

    Adds:
    - confidence_feats: Dict[str, torch.Tensor]
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeProteinAtomArray",
    ]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(
            data,
            [
                "protein_encoded",
            ],
        )

    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        atom_array = data["atom_array"]  # biotite atom array

        proteina_feats = {
            "atom37": torch.nan_to_num(torch.from_numpy(data["protein_encoded"]["xyz"]), nan=0.0),
            "atom37_mask": torch.from_numpy(data["protein_encoded"]["mask"]),
            "restype": torch.from_numpy(data["protein_encoded"]["seq"]),
        }

        # Check if mask is sparse by verifying total True count - 1 equals index of final True
        if proteina_feats["atom37_mask"].shape[0] < 32:
            # raise ValueError("Skipping all proteins < 32 residues")
            logger.warning("Including a protein with < 32 residues")

        mask = proteina_feats["atom37_mask"][..., 0]  # Shape: [batch_size, seq_len]
        if mask.sum(-1) != mask.shape[0]:
            raise ValueError("Protein sparse_mask is not supported")
        ca_coords = proteina_feats["atom37"][:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        chain_breaks_per_residue = ca_dists > 0.4  #! in nm already 0 #self.chain_break_cutoff
        proteina_feats["chain_breaks_per_residue"] = torch.cat(
            (
                chain_breaks_per_residue,
                torch.tensor([False], dtype=torch.bool),
            )
        )
        data["protein_atom37_feats"] = proteina_feats
        return data


class AggregateFeaturesLikeLaProteina(Transform):
    """
    Aggregates features into the correct places, and shapes with the names for AlphaFold 3.

    This transform combines various features from the input data into the format
    expected by the AlphaFold 3 model. It processes MSA features, ground truth
    structures, and other relevant data.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "AtomizeByCCDName",
        "EncodeAF3TokenLevelFeatures",
    ]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["AggregateFeaturesLikeLaProteina"]

    def check_input(self, data: dict[str, Any]) -> None:
        """
        Checks if the input data contains the required keys and types.

        Args:
            data (Dict[str, Any]): The input data dictionary.

        Raises:
            KeyError: If a required key is missing from the input data.
            TypeError: If a value in the input data is not of the expected type.
        """
        # check_contains_keys(data, ["msa_features", "atom_array"])
        check_contains_keys(data, ["atom_array"])
        # check_is_instance(data, "msa_features", dict)
        check_is_instance(data, "atom_array", AtomArray)

        # Check MSA features
        # msa_features = data["msa_features"]
        # check_contains_keys(msa_features, ["msa_features_per_recycle_dict", "msa_static_features_dict"])
        # check_is_instance(msa_features, "msa_features_per_recycle_dict", dict)
        # check_is_instance(msa_features, "msa_static_features_dict", dict)

        # Check specific MSA feature keys
        # msa_per_recycle = msa_features["msa_features_per_recycle_dict"]
        # check_contains_keys(msa_per_recycle, ["msa", "has_insertion", "insertion_value"])
        # msa_static = msa_features["msa_static_features_dict"]
        # check_contains_keys(msa_static, ["profile", "insertion_mean"])

        # Check atom array annotations
        check_atom_array_annotation(data, ["chain_iid", "occupancy"])  # "coord_to_be_noised",

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Aggregates features into the format expected by AlphaFold 3.

        This method processes the input data, combining MSA features, ground truth
        structures, and other relevant information into a standardized format.

        Args:
            data (Dict[str, Any]): The input data dictionary containing MSA features,
                atom array, and other relevant information.

        Returns:
            Dict[str, Any]: The processed data dictionary with aggregated features.
        """
        # Initialize feats dictionary if not present
        if "feats" not in data:
            data["feats"] = {}

        if "atom_name_chars" in data["feats"]:
            data["feats"]["atom_name_chars"] = F.one_hot(
                torch.from_numpy(data["feats"]["atom_name_chars"]).long(),
                num_classes=64,
            )

        # NOTE: the ref element is one-hot encoded by element number up to 128 (more than the known number of elements)
        if "atom_element" in data["feats"]:
            data["feats"]["atom_element"] = F.one_hot(
                torch.from_numpy(data["feats"]["atom_element"]).long(), num_classes=128
            )  #! .long() 'numpy.ndarray' object has no attribute 'long'

        # handle case where reference conformer was not able to be made and is currently nan
        # if "ref_pos" in data["feats"]:
        #     data["feats"]["ref_pos"] = torch.nan_to_num(torch.from_numpy(data["feats"]["ref_pos"]), nan=0.0)

        # Process ground truth structure
        atom_array = data["atom_array"]
        coord_atom_lvl = atom_array.coord
        mask_atom_lvl = atom_array.occupancy > 0.0

        _token_rep_idxs = get_af3_token_representative_idxs(atom_array)
        coord_token_lvl = atom_array.coord[_token_rep_idxs]
        mask_token_lvl = atom_array.occupancy[_token_rep_idxs] > 0.0

        # ...get chain_iid for each token (needed in validation for scoring)
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        chain_iid_token_lvl = token_level_array.chain_iid

        # (We may already have ground_truth in the data, i.e., during validation, when we pass extra information for evaluation)
        if "ground_truth" not in data:
            data["ground_truth"] = {}

        data["ground_truth"].update(
            {
                "coord_atom_lvl": torch.nan_to_num(torch.tensor(coord_atom_lvl), nan=0.0),  # [n_atoms, 3]
                "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                "coord_token_lvl": torch.nan_to_num(
                    torch.tensor(coord_token_lvl), nan=0.0
                ),  # [n_tokens, 3], using the representative tokens
                "mask_token_lvl": torch.tensor(mask_token_lvl),  # [n_tokens], using the representative tokens
                "chain_iid_token_lvl": chain_iid_token_lvl,  # numpy.ndarray of strings with shape (n_tokens,)
                "rep_atom_idxs": torch.tensor(_token_rep_idxs),  # [n_tokens]
                "restype_idx": torch.from_numpy(data["encoded"]["seq"]),
            }
        )

        # data for symmetry resolution
        if "symmetry_resolution" not in data:
            data["symmetry_resolution"] = {}

        if "crop_info" not in data:
            data["symmetry_resolution"].update(
                {
                    "molecule_entity": torch.tensor(data["atom_array"].molecule_entity),
                    "molecule_iid": torch.tensor(data["atom_array"].molecule_iid),
                    "crop_mask": torch.arange(data["atom_array"].shape[0]),
                    "coord_atom_lvl": torch.tensor(coord_atom_lvl),  # [n_atoms, 3]
                    "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                }
            )
        else:
            token_starts = get_token_starts(data["crop_info"]["atom_array"])
            data["symmetry_resolution"].update(
                {
                    "molecule_entity": torch.tensor(data["crop_info"]["atom_array"].molecule_entity),
                    "molecule_iid": torch.tensor(data["crop_info"]["atom_array"].molecule_iid),
                    "crop_mask": torch.tensor(data["crop_info"]["crop_atom_idxs"]),
                    "coord_atom_lvl": torch.tensor(data["crop_info"]["atom_array"].coord),
                    "mask_atom_lvl": torch.tensor(data["crop_info"]["atom_array"].occupancy > 0.0),
                }
            )

        # # Add atom-level features for noising
        # data["coord_atom_lvl_to_be_noised"] = torch.tensor(
        #     atom_array.coord_to_be_noised
        # )

        #! Here we convert numpy to torch tensors
        for k, v in data["feats"].items():
            if isinstance(v, np.ndarray):
                data["feats"][k] = torch.from_numpy(v)
        return data


class AggregateFeaturesLikeAF3NoMSA(Transform):
    """
    Aggregates features into the correct places, and shapes with the names for AlphaFold 3.

    This transform combines various features from the input data into the format
    expected by the AlphaFold 3 model. It processes MSA features, ground truth
    structures, and other relevant data.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "AtomizeByCCDName",
        # "FeaturizeMSALikeAF3",
        "EncodeAF3TokenLevelFeatures",
    ]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = ["AggregateFeaturesLikeAF3NoMSA"]

    def check_input(self, data: dict[str, Any]) -> None:
        """
        Checks if the input data contains the required keys and types.

        Args:
            data (Dict[str, Any]): The input data dictionary.

        Raises:
            KeyError: If a required key is missing from the input data.
            TypeError: If a value in the input data is not of the expected type.
        """
        # check_contains_keys(data, ["msa_features", "atom_array"])
        check_contains_keys(data, ["atom_array"])
        # check_is_instance(data, "msa_features", dict)
        check_is_instance(data, "atom_array", AtomArray)

        # Check MSA features
        # msa_features = data["msa_features"]
        # check_contains_keys(msa_features, ["msa_features_per_recycle_dict", "msa_static_features_dict"])
        # check_is_instance(msa_features, "msa_features_per_recycle_dict", dict)
        # check_is_instance(msa_features, "msa_static_features_dict", dict)

        # Check specific MSA feature keys
        # msa_per_recycle = msa_features["msa_features_per_recycle_dict"]
        # check_contains_keys(msa_per_recycle, ["msa", "has_insertion", "insertion_value"])
        # msa_static = msa_features["msa_static_features_dict"]
        # check_contains_keys(msa_static, ["profile", "insertion_mean"])

        # Check atom array annotations
        check_atom_array_annotation(data, ["coord_to_be_noised", "chain_iid", "occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Aggregates features into the format expected by AlphaFold 3.

        This method processes the input data, combining MSA features, ground truth
        structures, and other relevant information into a standardized format.

        Args:
            data (Dict[str, Any]): The input data dictionary containing MSA features,
                atom array, and other relevant information.

        Returns:
            Dict[str, Any]: The processed data dictionary with aggregated features.
        """
        # Initialize feats dictionary if not present
        if "feats" not in data:
            data["feats"] = {}

        # # Aggregate and stack MSA features
        # msa_feats = data["msa_features"]

        # msa_stacked_by_recycle = torch.stack(
        #     msa_feats["msa_features_per_recycle_dict"]["msa"]
        # ).float()  # [n_recycles, n_sequences, n_tokens_across_chains, n_types_of_tokens]
        # has_deletion_stacked_by_recycle = torch.stack(
        #     msa_feats["msa_features_per_recycle_dict"]["has_insertion"]
        # )  # [n_recycles, n_sequences, n_tokens_across_chains]
        # deltion_value_stacked_by_recycle = torch.stack(
        #     msa_feats["msa_features_per_recycle_dict"]["insertion_value"]
        # )  # [n_recycles, n_sequences, n_tokens_across_chains]

        # data["feats"]["msa_stack"] = torch.concatenate(
        #     [
        #         msa_stacked_by_recycle,
        #         rearrange(has_deletion_stacked_by_recycle, "... -> ... 1"),
        #         rearrange(deltion_value_stacked_by_recycle, "... -> ... 1"),
        #     ],
        #     dim=-1,
        # )  # [n_recycles, n_msa, n_tokens_across_chains, n_types_of_tokens + 2] (float)

        # data["feats"] |= {
        #     "profile": msa_feats["msa_static_features_dict"]["profile"],
        #     "deletion_mean": msa_feats["msa_static_features_dict"]["insertion_mean"],
        # }
        #! TODO may have to add ligand feats based on the real data
        # NOTE: Each atom name is encoded as `ord(c) - 32`, which shifts the character values to create a
        # more compact one-hot encoding (as the first 32 Unicode characters will not occur in an atom name)
        #! one_hot(): argument 'input' (position 1) must be Tensor, not numpy.ndarray
        data["feats"]["ref_atom_name_chars"] = F.one_hot(
            torch.from_numpy(data["feats"]["ref_atom_name_chars"]).long(),
            num_classes=64,
        )

        # NOTE: the ref element is one-hot encoded by element number up to 128 (more than the known number of elements)
        data["feats"]["ref_element"] = F.one_hot(
            torch.from_numpy(data["feats"]["ref_element"]).long(), num_classes=128
        )  #! .long() 'numpy.ndarray' object has no attribute 'long'

        # handle case where reference conformer was not able to be made and is currently nan
        data["feats"]["ref_pos"] = torch.nan_to_num(torch.from_numpy(data["feats"]["ref_pos"]), nan=0.0)

        # Process ground truth structure
        atom_array = data["atom_array"]
        coord_atom_lvl = atom_array.coord
        mask_atom_lvl = atom_array.occupancy > 0.0

        _token_rep_idxs = get_af3_token_representative_idxs(atom_array)
        coord_token_lvl = atom_array.coord[_token_rep_idxs]
        mask_token_lvl = atom_array.occupancy[_token_rep_idxs] > 0.0

        # ...get chain_iid for each token (needed in validation for scoring)
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        chain_iid_token_lvl = token_level_array.chain_iid

        # (We may already have ground_truth in the data, i.e., during validation, when we pass extra information for evaluation)
        if "ground_truth" not in data:
            data["ground_truth"] = {}

        data["ground_truth"].update(
            {
                "coord_atom_lvl": torch.tensor(coord_atom_lvl),  # [n_atoms, 3]
                "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                "coord_token_lvl": torch.tensor(coord_token_lvl),  # [n_tokens, 3], using the representative tokens
                "mask_token_lvl": torch.tensor(mask_token_lvl),  # [n_tokens], using the representative tokens
                "chain_iid_token_lvl": chain_iid_token_lvl,  # numpy.ndarray of strings with shape (n_tokens,)
                "rep_atom_idxs": torch.tensor(_token_rep_idxs),  # [n_tokens]
            }
        )

        # data for symmetry resolution
        if "symmetry_resolution" not in data:
            data["symmetry_resolution"] = {}

        if "crop_info" not in data:
            data["symmetry_resolution"].update(
                {
                    "molecule_entity": torch.tensor(data["atom_array"].molecule_entity),
                    "molecule_iid": torch.tensor(data["atom_array"].molecule_iid),
                    "crop_mask": torch.arange(data["atom_array"].shape[0]),
                    "coord_atom_lvl": torch.tensor(coord_atom_lvl),  # [n_atoms, 3]
                    "mask_atom_lvl": torch.tensor(mask_atom_lvl),  # [n_atoms]
                }
            )
        else:
            token_starts = get_token_starts(data["crop_info"]["atom_array"])
            data["symmetry_resolution"].update(
                {
                    "molecule_entity": torch.tensor(data["crop_info"]["atom_array"].molecule_entity),
                    "molecule_iid": torch.tensor(data["crop_info"]["atom_array"].molecule_iid),
                    "crop_mask": torch.tensor(data["crop_info"]["crop_atom_idxs"]),
                    "coord_atom_lvl": torch.tensor(data["crop_info"]["atom_array"].coord),
                    "mask_atom_lvl": torch.tensor(data["crop_info"]["atom_array"].occupancy > 0.0),
                }
            )

        # Add atom-level features for noising
        data["coord_atom_lvl_to_be_noised"] = torch.tensor(atom_array.coord_to_be_noised)

        return data
