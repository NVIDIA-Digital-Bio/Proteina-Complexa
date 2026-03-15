import itertools
import logging
from typing import ClassVar

import networkx as nx
import numpy as np
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.atom_array import atom_id_to_atom_idx, atom_id_to_token_idx
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.crop import CropTransformBase, resize_crop_info_if_too_many_atoms
from atomworks.ml.utils.token import get_af3_token_center_coords, get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from scipy.spatial import KDTree

logger = logging.getLogger("atomworks.ml")


class CropSpatialLikeAF3LigandCenter(CropTransformBase):
    """
    A transform that performs spatial cropping similar to AF3 and AF2 Multimer.

    This class implements the spatial cropping procedure as described in AF3. It selects a crop center
    from a spatial region of the atom array and samples a crop around this center.

    WARNING: This transform is probabilistic if the atom array is larger than the crop size!

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf

    Attributes:
        crop_size (int): The maximum number of tokens to crop. Must be greater than 0.
        jitter_scale (float): The scale of the jitter to apply to the crop center. This is to break
            ties between atoms with the same spatial distance. Defaults to 1e-3.
        crop_center_cutoff_distance (float): The cutoff distance to consider for selecting crop
            centers. Measured in Angstroms. Defaults to 15.0.
        keep_uncropped_atom_array (bool): Whether to keep the uncropped atom array in the data.
            If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
            under the key `"atom_array"`. Defaults to `False`.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "AddGlobalAtomIdAnnotation",
        "AtomizeByCCDName",
    ]
    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
        "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence",
    ]

    def __init__(
        self,
        crop_size: int,
        jitter_scale: float = 1e-3,
        crop_center_cutoff_distance: float = 15.0,
        keep_uncropped_atom_array: bool = False,
        force_crop: bool = False,
        max_atoms_in_crop: int | None = None,
        raise_if_missing_query: bool = True,
        force_include_query_tokens: bool = True,
        **kwargs,
    ):
        """Initialize the CropSpatialLikeAF3 transform.

        Args:
            crop_size: The maximum number of tokens to crop. Must be greater than 0.
            jitter_scale: The scale of the jitter to apply to the crop center.
                This is to break ties between atoms with the same spatial distance. Defaults to 1e-3.
            crop_center_cutoff_distance: The cutoff distance to consider for
                selecting crop centers. Measured in Angstroms. Defaults to 15.0.
            keep_uncropped_atom_array: Whether to keep the uncropped atom array in the data.
                If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
                under the key `"atom_array"`. Defaults to `False`.
            force_crop: Whether to force crop even if the atom array is already small enough.
                Defaults to `False`.
            max_atoms_in_crop (int, optional): Maximum number of atoms allowed in a crop. If None, no resizing is performed.
                Defaults to None.
            raise_if_missing_query (bool): Whether to raise an Exception if no crop centers are found, e.g. if the
                query pn_unit(s) are not present due to a previous filtering step. Defaults to `True`. If `False`, a random
                pn_unit will be selected for the crop center.
            force_include_query_tokens (bool): If True, force-include all query/ligand tokens in
                the crop even if they fall outside crop_size. This prevents failures when the
                ligand is spatially spread out. If False, assert that all centers survived (old
                behavior). Defaults to True.
        """
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.jitter_scale = jitter_scale
        self.crop_center_cutoff_distance = crop_center_cutoff_distance
        self.keep_uncropped_atom_array = keep_uncropped_atom_array
        self.force_crop = force_crop
        self.max_atoms_in_crop = max_atoms_in_crop
        self.raise_if_missing_query = raise_if_missing_query
        self.force_include_query_tokens = force_include_query_tokens
        self._validate()

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_iid", "atomize", "atom_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if data.get("query_pn_unit_iids"):
            query_pn_units = data["query_pn_unit_iids"]
        else:
            query_pn_units = np.unique(atom_array.pn_unit_iid)
            logger.info(f"No query PN unit(s) provided for spatial crop. Randomly selecting from {query_pn_units}.")
        try:
            lig_mask = data["feats"]["is_ligand"][data["feats"]["atom_to_token_map"]]
            ligand_qn_iids = np.unique(atom_array[(lig_mask) & (~atom_array.is_polymer)].pn_unit_iid)[0]
        except Exception:
            #! the is ligand mask is wrong here so default to the old
            ligand_qn_iids = query_pn_units[-1]

        crop_info = crop_spatial_like_af3_local(
            atom_array=atom_array,
            query_pn_unit_iids=[ligand_qn_iids],  # [query_pn_units[-1]],
            crop_size=self.crop_size,
            jitter_scale=self.jitter_scale,
            crop_center_cutoff_distance=self.crop_center_cutoff_distance,
            force_crop=self.force_crop,
            raise_if_missing_query=self.raise_if_missing_query,
            force_include_query_tokens=self.force_include_query_tokens,
        )
        crop_info = resize_crop_info_if_too_many_atoms(
            crop_info=crop_info,
            atom_array=atom_array,
            max_atoms=self.max_atoms_in_crop,
        )

        data["crop_info"] = {"type": self.__class__.__name__} | crop_info

        if self.keep_uncropped_atom_array:
            data["crop_info"]["atom_array"] = atom_array

        # Update data with cropped atom array
        data["atom_array"] = atom_array[crop_info["crop_atom_idxs"]]

        clean_up = crop_info["requires_crop"]  # True #! causes weird issue
        if clean_up:
            da = data["atom_array"]
            kept_residues = da[da.pn_unit_iid == "A_1"].res_id
            og_atoms = atom_array[(atom_array.pn_unit_iid == "A_1") & np.isin(atom_array.res_id, kept_residues)]
            try:
                assert og_atoms.shape[0] == len(kept_residues), (
                    "CROP MISTAKE DID NOT KEEP ALL RESIDUES"
                )  #! this is True
            except Exception as e:
                logger.error(f"CROP MISTAKE DID NOT KEEP ALL RESIDUES: {e}")
                raise e
            bonds = data["atom_array"].bonds.as_array()
            G = nx.Graph()
            # Add edges from the first two columns of the bonds array
            G.add_edges_from(bonds[:, :2])
            # Get connected components
            components = list(nx.connected_components(G))
            #! this is only True of SAIR
            if "feats" in data and "is_ligand" in data["feats"] and "atom_to_token_map" in data["feats"]:
                lig_mask = data["feats"]["is_ligand"][data["feats"]["atom_to_token_map"]][crop_info["crop_atom_idxs"]]
                del data["feats"]  # we will remake it
            else:  #! SAIR has clean LIG
                lig_mask = data["atom_array"].res_name == "LIG"

            lig_idx = set([i for i, m in enumerate(lig_mask) if m])
            protein_comp = [c for c in components if len(c.intersection(lig_idx)) == 0]

            bres = []
            for bond in bonds:
                bres.append((da[bond[0]].res_id, da[bond[1]].res_id))
            GG = nx.Graph()
            GG.add_edges_from(bres)
            bcomponents = list(nx.connected_components(GG))

            center_atom = atom_array[crop_info["crop_center_atom_idx"]]
            unneeded_residues = []
            RESIDUE_FRAGMENT_SIZE = 20  # 15 #20 #10
            POCKET_CUTOFF = 15

            # lig_coords = da[da.res_name == "LIG"].coord
            lig_coords = da[lig_mask].coord
            for x in bcomponents:
                if len(x) < RESIDUE_FRAGMENT_SIZE:
                    overall_min_dist = 1e8
                    for rid in x:
                        atom_coords = da[da.res_id == rid].coord
                        use_whole_ligand = True
                        if use_whole_ligand:
                            # Compute pairwise distance matrix: N ligand atoms × M residue atoms
                            # Reshape for broadcasting: (N, 1, 3) - (1, M, 3) = (N, M, 3)
                            lig_coords_reshaped = lig_coords[:, np.newaxis, :]  # (N, 1, 3)
                            atom_coords_reshaped = atom_coords[np.newaxis, :, :]  # (1, M, 3)
                            pair_dist = np.sqrt(
                                ((lig_coords_reshaped - atom_coords_reshaped) ** 2).sum(axis=-1)
                            )  # (N, M)
                            min_dist = pair_dist.min()
                        else:
                            dists = np.sqrt(((atom_coords - center_atom.coord) ** 2).sum(-1))
                            min_dist = dists.min()
                        overall_min_dist = min(min_dist, overall_min_dist)

                    if overall_min_dist > POCKET_CUTOFF:
                        unneeded_residues.extend(x)
                        logger.info(f"Removing {len(x)} residues because they are too far to the crop center.")
                    else:
                        logger.info(
                            f"Keeping {len(x)} residues from pocket because as one atom is within {POCKET_CUTOFF} Å of the crop center."
                        )

            # Apply filtering only for A_1, keep all other pn_unit_iids
            # mask = (da.pn_unit_iid == "A_1") & (~np.isin(da.res_id, unneeded_residues))
            # mask |= (da.pn_unit_iid != "A_1")  # Keep all non-A_1 atoms
            # cut_atom_array = da[mask]
            # logger.info(f"Cutting {len(unneeded_residues)} residues")
            # data["atom_array"] = cut_atom_array
            mask = (~np.isin(da.res_id, unneeded_residues)) & (da.pn_unit_iid != ligand_qn_iids)
            mask |= da.pn_unit_iid == ligand_qn_iids
            cut_atom_array = da[mask]
            logger.info(f"Cutting {len(unneeded_residues)} residues")
            data["atom_array"] = cut_atom_array

        return data


def get_spatial_crop_center_local(
    atom_array: AtomArray,
    query_pn_unit_iids: list[str],
    cutoff_distance: float = 15.0,
    raise_if_missing_query: bool = True,
) -> np.ndarray:
    """
    Sample a crop center from a spatial region of the atom array.

    Implements the selection of a crop center as described in AF3.
    ```
        In this procedure, polymer residues and ligand atoms are selected that
        are within close spatial distance of an interface atom. The interface
        atom is selected at random from the set of token centre atoms (defined
        in subsection 2.6) with a distance under 15 Å to another chain's token
        centre atom. For examples coming out of the Weighted PDB or Disordered
        protein PDB complex datasets, where a preferred chain or interface is
        provided (subsection 2.5), the reference atom is selected at random
        from interfacial token centre atoms that exist within this chain or
        interface.
    ```

    Args:
        atom_array (AtomArray): The array containing atom information.
        query_pn_unit_iids (list[str]): List of PN unit instance IDs to query.
        cutoff_distance (float, optional): The distance cutoff to consider for spatial proximity. Defaults to 15.0.
        raise_if_missing_query (bool): Whether to raise an Exception if no crop centers are found, e.g. if the
            query pn_unit(s) are not present due to a previous filtering step. Defaults to `True`. If `False`, a random
            pn_unit will be selected for the crop center.

    Returns:
        np.ndarray: A boolean mask indicating the crop center.
    """
    # ... get mask for query polymer/non-polymer unit
    is_query_pn_unit = np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)
    # ... get mask for occupied atoms
    is_occupied = atom_array.occupancy > 0

    # ... optionally provide a fallback when not all query pn_units are present
    if not raise_if_missing_query:
        available_query_pn_unit_iids = np.unique(atom_array.pn_unit_iid[is_query_pn_unit])

        # If only one of the query pn_units is present, we will just use that
        if len(available_query_pn_unit_iids) == 1 and len(query_pn_unit_iids) > 1:
            query_pn_unit_iids = available_query_pn_unit_iids
            logger.warning(
                f"Falling back to only available query pn_unit ({query_pn_unit_iids[0]}) for the crop center."
            )

        # If none of the query pn_units are present, we will randomly select one
        elif len(available_query_pn_unit_iids) == 0:
            all_available_pn_unit_iids = np.unique(atom_array.pn_unit_iid)
            query_pn_unit_iids = np.random.choice(all_available_pn_unit_iids, size=1)
            logger.warning(f"Falling back to randomly-selected pn_unit ({query_pn_unit_iids[0]}) for the crop center.")

        # Update the mask for query pn_unit
        is_query_pn_unit = np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)

    if len(query_pn_unit_iids) == 1:
        # If there's only one query unit, we don't need to check for spatial proximity,
        # so we can just return the mask for the query unit.
        can_be_crop_center = is_query_pn_unit & is_occupied
        assert np.any(can_be_crop_center), (
            f"No crop center found! It appears `query_pn_unit_iid` {query_pn_unit_iids} is not in the atom array or unresolved."
        )

        return can_be_crop_center

    # ... get mask for ligands of interest
    is_at_interface = np.zeros_like(is_query_pn_unit, dtype=bool)
    for pn_unit_1_iid, pn_unit_2_iid in itertools.combinations(query_pn_unit_iids, 2):
        # ... get mask, indices, and kdtree for pn_unit_1
        pn_unit_1_mask = (atom_array.pn_unit_iid == pn_unit_1_iid) & is_occupied
        pn_unit_1_indices = np.where(pn_unit_1_mask)[0]
        _tree1 = KDTree(atom_array.coord[pn_unit_1_mask])

        # ... get mask, indices, and kdtree for pn_unit_2
        pn_unit_2_mask = (atom_array.pn_unit_iid == pn_unit_2_iid) & is_occupied
        pn_unit_2_indices = np.where(pn_unit_2_mask)[0]
        _tree2 = KDTree(atom_array.coord[pn_unit_2_mask])

        dists = _tree1.sparse_distance_matrix(_tree2, max_distance=cutoff_distance, output_type="coo_matrix")

        # ... update the interface mask (by converting the local idxs to the global idxs)
        is_at_interface[pn_unit_1_indices[np.unique(dists.row)]] = True
        is_at_interface[pn_unit_2_indices[np.unique(dists.col)]] = True

    # ... assemble final crop mask
    can_be_crop_center = is_query_pn_unit & is_at_interface & is_occupied
    assert np.any(can_be_crop_center), "No crop center found!"
    return can_be_crop_center


def crop_spatial_like_af3_local(
    atom_array: AtomArray,
    query_pn_unit_iids: list[str],
    crop_size: int,
    jitter_scale: float = 1e-3,
    crop_center_cutoff_distance: float = 15.0,
    force_crop: bool = False,
    raise_if_missing_query: bool = True,
    do_not_crop_target: bool = False,
    force_include_query_tokens: bool = True,
) -> dict:
    """Crop spatial tokens around a given `crop_center` by keeping the `crop_size` nearest neighbors (with jitter).

    Args:
        - atom_array (AtomArray): The atom array to crop.
        - query_pn_unit_iids (list[str]): List of query polymer/non-polymer unit instance IDs.
        - crop_size (int): The maximum number of tokens to crop.
        - jitter_scale (float, optional): Scale of jitter to apply when calculating distances.
            Defaults to 1e-3.
        - crop_center_cutoff_distance (float, optional): Maximum distance from query units to
            consider for crop center. Defaults to 15.0 Angstroms.
        - force_crop (bool, optional): Whether to force crop even if the atom array is already small enough.
            Defaults to False.
        - raise_if_missing_query (bool): Whether to raise an Exception if no crop centers are found, e.g. if the
            query pn_unit(s) are not present due to a previous filtering step. Defaults to `True`. If `False`, a random
            pn_unit will be selected for the crop center.

    Returns:
        dict: A dictionary containing crop information, including:
            - requires_crop (bool): Whether cropping was necessary.
            - crop_center_atom_id (int or np.nan): ID of the atom chosen as crop center.
            - crop_center_atom_idx (int or np.nan): Index of the atom chosen as crop center.
            - crop_center_token_idx (int or np.nan): Index of the token containing the crop center.
            - crop_token_idxs (np.ndarray): Indices of tokens included in the crop.
            - crop_atom_idxs (np.ndarray): Indices of atoms included in the crop.

    Note:
        This function implements the spatial cropping procedure as described in AlphaFold 3 and AlphaFold 2 Multimer.

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf
    """
    token_segments = get_token_starts(atom_array, add_exclusive_stop=True)
    n_tokens = len(token_segments) - 1
    requires_crop = n_tokens > crop_size

    # ... get possible crop centers
    can_be_crop_center = get_spatial_crop_center_local(
        atom_array,
        query_pn_unit_iids,
        crop_center_cutoff_distance,
        raise_if_missing_query=raise_if_missing_query,
    )
    possible_centers = atom_array[can_be_crop_center].atom_id
    # crop_center_atom_ids = [atom_id_to_atom_idx(atom_array, atom_id) for atom_id in possible_centers]
    crop_center_token_idxs = [atom_id_to_token_idx(atom_array, atom_id) for atom_id in possible_centers]
    # ... sample crop center atom
    crop_center_atom_id = np.random.choice(atom_array[can_be_crop_center].atom_id)
    crop_center_atom_idx = atom_id_to_atom_idx(atom_array, crop_center_atom_id)
    crop_center_token_idx = atom_id_to_token_idx(atom_array, crop_center_atom_id)
    atom_array = atom_array.copy()
    # ... sample crop
    if force_crop or requires_crop:
        # Broadcast the center coordinates to all possible centers
        center_coord = atom_array[crop_center_atom_id].coord
        token_coords = get_af3_token_center_coords(atom_array)
        # token_coords[-np.sum(can_be_crop_center):] = np.tile(center_coord, (np.sum(can_be_crop_center), 1))
        token_coords[crop_center_token_idxs] = np.tile(center_coord, (np.sum(can_be_crop_center), 1))
        if do_not_crop_target:
            # Exclude query/target tokens from spatial crop, reduce budget accordingly
            n_query_tokens = len(set(crop_center_token_idxs))
            protein_crop_size = max(1, crop_size - n_query_tokens)
            min_q, max_q = min(crop_center_token_idxs), max(crop_center_token_idxs)
            sub_token_coords = np.concatenate([token_coords[:min_q], token_coords[max_q + 1 :]])
            is_token_in_crop = get_spatial_crop_mask_local(
                token_coords,
                crop_center_token_idx,
                crop_size=protein_crop_size,
                jitter_scale=jitter_scale,
                atom_array=atom_array,
                sub_coord=sub_token_coords,
            )
            # Re-insert query tokens as always-included
            is_token_in_crop = np.concatenate(
                [
                    is_token_in_crop[:min_q],
                    np.ones(max_q - min_q + 1, dtype=bool),
                    is_token_in_crop[min_q:],
                ]
            )
        else:
            is_token_in_crop = get_spatial_crop_mask_local(  #! added atom array
                token_coords,
                crop_center_token_idx,
                crop_size=crop_size,
                jitter_scale=jitter_scale,
                atom_array=atom_array,
            )
        # ... spread token-level crop mask to atom-level
        is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop, token_starts=token_segments)
    else:
        # ... no need to crop since the atom array is already small enough
        is_atom_in_crop = np.ones(len(atom_array), dtype=bool)
        is_token_in_crop = np.ones(n_tokens, dtype=bool)

    final_crop_info = {
        "requires_crop": requires_crop,  # whether cropping was necessary
        "crop_center_atom_id": crop_center_atom_id,  # atom_id of crop center
        "crop_center_atom_idx": crop_center_atom_idx,  # atom_idx of crop center
        "crop_center_token_idx": crop_center_token_idx,  # token_idx of crop center
        "crop_token_idxs": np.where(is_token_in_crop)[0],  # token_idxs in crop
        "crop_atom_idxs": np.where(is_atom_in_crop)[0],  # atom_idxs in crop
    }

    # Check that all possible center atoms are included in the final crop.
    # BUGFIX: The old code compared atom_id (annotation values like 0,1,2...) against
    # crop_atom_idxs (positional indices). These are different things and would cause
    # false positives - triggering the force-include warning even when all atoms were
    # actually in the crop. Fix: compare positional indices against positional indices.
    center_atom_idxs = np.where(can_be_crop_center)[0]
    centers_in_crop = np.isin(center_atom_idxs, final_crop_info["crop_atom_idxs"])
    if not np.all(centers_in_crop):
        if force_include_query_tokens:
            # Force-include all query/center tokens so ligand atoms are never dropped
            # This may slightly exceed crop_size but guarantees the ligand is complete
            can_be_crop_center_token = np.zeros(n_tokens, dtype=bool)
            for atom_idx in np.where(can_be_crop_center)[0]:
                token_idx = atom_id_to_token_idx(atom_array, atom_array[atom_idx].atom_id)
                can_be_crop_center_token[token_idx] = True

            n_missing = int(np.sum(can_be_crop_center_token & ~is_token_in_crop))
            logger.warning(
                f"Force-including {n_missing} query tokens that fell outside crop_size={crop_size}. "
                f"Crop will have {int(np.sum(is_token_in_crop)) + n_missing} tokens."
            )
            is_token_in_crop = is_token_in_crop | can_be_crop_center_token
            is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop, token_starts=token_segments)

            final_crop_info["crop_token_idxs"] = np.where(is_token_in_crop)[0]
            final_crop_info["crop_atom_idxs"] = np.where(is_atom_in_crop)[0]
        else:
            # Old behavior: assert and fail
            raise AssertionError(
                f"Not all possible centers survived cropping. Missing {np.sum(~centers_in_crop)} centers"
            )

    return final_crop_info


def get_spatial_crop_mask_local(
    coord: np.ndarray,
    crop_center_idx: int,
    crop_size: int,
    jitter_scale: float = 1e-3,
    atom_array: AtomArray = None,
    sub_coord: np.ndarray = None,
) -> np.ndarray:
    """
    Crop spatial tokens around a given `crop_center` by keeping the `crop_size` nearest neighbors (with jitter).

    Implements the `crop_spatial` (algorithm 2 in section 7.2.1) of AF2 Multimer and AF3

    Args:
        coord (np.ndarray): A 2D numpy array of shape (N, 3) representing the 3D token-level coordinates.
            Coordinates are expected to be in Angstroms.
        crop_center_idx (int): The index of the token to be used as the center of the crop.
        crop_size (int): The number of nearest neighbors to include in the crop.
        jitter_scale (float): The scale of the jitter to add to the coordinates.

    Returns:
        crop_mask (np.ndarray): A boolean mask of shape (N,) where True indicates that the token is within the crop.

    References:
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf

    Example:
        >>> coord = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        >>> crop_center_idx = 1
        >>> crop_size = 2
        >>> crop_mask = get_spatial_crop_mask(coord, crop_center_idx, crop_size)
        >>> print(crop_mask)
        [ True  True False False]
    """
    assert coord.ndim == 2, f"Expected coord to be 2-dimensional, got {coord.ndim} dimensions"
    assert coord.shape[1] == 3, f"Expected coord to have 3 coordinates per point, got {coord.shape[1]}"
    assert crop_center_idx < coord.shape[0], (
        f"Crop center index {crop_center_idx} is out of bounds for coord array of length {coord.shape[0]}"
    )
    assert crop_size > 0, f"Crop size must be positive, got {crop_size}"
    assert jitter_scale >= 0, f"Jitter scale must be non-negative, got {jitter_scale}"

    # Add small jitter to coordinates to break ties
    if jitter_scale > 0:
        coord = coord + np.random.normal(scale=jitter_scale, size=coord.shape)

    # ... get query center
    query_center = coord[crop_center_idx]

    # ... extract a mask for valid coordiantes (i.e. no `nan`'s, which indicate unknown token centers)
    #     including including unoccupied tokens in the crop
    if sub_coord is not None:
        is_valid = np.isfinite(sub_coord).all(axis=1)
        # ... build a KDTree for efficient querying, excluding invalid coordinates
        tree = KDTree(sub_coord[is_valid])
    else:
        is_valid = np.isfinite(coord).all(axis=1)

        # ... build a KDTree for efficient querying, excluding invalid coordinates
        tree = KDTree(coord[is_valid])

    # ... query the `crop_size` nearest neighbors of the crop center
    _, nearest_neighbor_idxs = tree.query(query_center, k=crop_size, p=2)
    # ... filter out missing neighbours (index equal to `tree.n`)
    nearest_neighbor_idxs = nearest_neighbor_idxs[nearest_neighbor_idxs < tree.n]

    # ... crop mask is True for the `crop_size` nearest neighbors of the crop center
    if sub_coord is not None:
        crop_mask = np.zeros(sub_coord.shape[0], dtype=bool)
    else:
        crop_mask = np.zeros(coord.shape[0], dtype=bool)
    is_valid_and_in_crop_idxs = np.where(is_valid)[0][nearest_neighbor_idxs]
    crop_mask[is_valid_and_in_crop_idxs] = True

    return crop_mask
