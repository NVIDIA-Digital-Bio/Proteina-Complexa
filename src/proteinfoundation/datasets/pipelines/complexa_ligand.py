from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from atomworks.constants import AF3_EXCLUDED_LIGANDS, STANDARD_AA
from atomworks.io.utils.selection import get_annotation_categories
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    add_global_token_id_annotation,
    compute_atom_to_token_map,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    Identity,
    RandomRoute,
    SubsetToKeys,
    Transform,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from atomworks.ml.transforms.crop import CropContiguousLikeAF3
from atomworks.ml.transforms.featurize_unresolved_residues import MaskPolymerResiduesWithUnresolvedFrameAtoms
from atomworks.ml.transforms.filters import (
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedTokens,
)
from atomworks.ml.utils import nested_dict

from proteinfoundation.datasets.atomworks_crop_transforms import CropSpatialLikeAF3LigandCenter
from proteinfoundation.datasets.atomworks_ligand_transforms import (
    EncodeAF3TokenLevelFeatures,
    GetAF3MoleculeFeatures,
    ProteinaLigandTransform,
)
from proteinfoundation.datasets.atomworks_transforms import (
    AggregateFeaturesLikeLaProteina,
    FilterForProteinaLigandComplex,
    ProteinaFinalTransform,
)

# ---------------------------------------------------------------------------
# Helper transforms
# ---------------------------------------------------------------------------

# Local versions of atomworks transforms to avoid incompatible_previous_transforms
# checks that would conflict when the same transform is used at multiple stages
# (e.g., AddGlobalTokenIdAnnotation before and after cropping).


class AddGlobalTokenIdAnnotation(Transform):
    """Adds `token_id` annotation. Overwrites if present (needed after cropping).

    Local version to bypass atomworks' incompatible_previous_transforms checks.
    """

    def check_input(self, data: dict) -> None:
        if "token_id" in set(get_annotation_categories(data["atom_array"], n_body=1)):
            data["atom_array"].del_annotation("token_id")
        check_atom_array_annotation(data, required=[], forbidden=["token_id"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = add_global_token_id_annotation(data["atom_array"])
        return data


class PlinderPreCropAnnotation(Transform):
    """Same as AddGlobalTokenIdAnnotation but with a different class name.

    Used before cropping so atomworks doesn't flag it as incompatible with
    the post-crop AddGlobalTokenIdAnnotation.
    """

    def check_input(self, data: dict) -> None:
        if "token_id" in set(get_annotation_categories(data["atom_array"], n_body=1)):
            data["atom_array"].del_annotation("token_id")
        check_atom_array_annotation(data, required=[], forbidden=["token_id"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = add_global_token_id_annotation(data["atom_array"])
        return data


class PlinderCropFeats(EncodeAF3TokenLevelFeatures):
    """Pre-crop version of EncodeAF3TokenLevelFeatures.

    Separate class name to avoid incompatible_previous_transforms conflict
    with the post-crop EncodeAF3TokenLevelFeatures.
    """

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return super().forward(data)


class ComputePlinderAtomToTokenMap(Transform):
    """Add `feats.atom_to_token_map` mapping each atom to its token index.

    Local version with relaxed requires_previous_transforms.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["PlinderPreCropAnnotation"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["token_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_to_token_map = compute_atom_to_token_map(data["atom_array"])
        nested_dict.set(data, ("feats", "atom_to_token_map"), atom_to_token_map)
        return data


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def build_protein_ligand_transform_pipeline(
    *,
    is_inference: bool,
    # Crop params
    crop_size: int = 384,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    max_atoms_in_crop: int | None = None,
    # Filtering
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    use_element_for_atom_names_of_atomized_tokens: bool = False,
    return_atom_array: bool = True,
    b_factor_min: float | None = None,
    b_factor_max: float | None = None,
    **kwargs,
) -> Transform:
    """Build the protein-ligand transform pipeline.

    Pipeline: clean structure -> annotate -> atomize -> crop -> encode -> featurize.

    Args:
        is_inference: If True, skip cropping.
        crop_size: Max number of tokens after cropping.
        crop_center_cutoff_distance: Spatial crop cutoff distance (Angstroms).
        crop_contiguous_probability: Probability of contiguous cropping.
        crop_spatial_probability: Probability of spatial cropping.
        max_atoms_in_crop: Max atoms allowed in crop (None = no limit).
        undesired_res_names: Residue names to remove.
        use_element_for_atom_names_of_atomized_tokens: Use element symbols for atom names.
        return_atom_array: Keep atom_array in output.
        b_factor_min: Min b-factor for occupancy masking.
        b_factor_max: Max b-factor for occupancy masking.

    Returns:
        Composed transform pipeline.
    """
    if (crop_contiguous_probability > 0 or crop_spatial_probability > 0) and not is_inference:
        assert np.isclose(crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6), (
            "Crop probabilities must sum to 1.0"
        )
        assert crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    af3_sequence_encoding = AF3SequenceEncoding()

    # --- Stage 1: Clean and annotate ---
    transforms = [
        AddData({"is_inference": is_inference}),
        RemoveHydrogens(),
        # Use this for debugging
        # BreakPointTransform(),
        RemoveTerminalOxygen(),
        SetOccToZeroOnBfactor(b_factor_min, b_factor_max),
        RemoveUnresolvedTokens(),
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        # NOTE: For inference, we must keep UNL to support ligands that are not in the CCD
        HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names),  # e.g., non-standard residues
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(allow_overwrite=True),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
        # Remove unresolved tokens again - some non-ligand atoms with occupancy 0
        # can appear as feats[is_ligand] but not in the ligand atom map
        RemoveUnresolvedTokens(),
        FilterForProteinaLigandComplex(),
        # Pre-crop: use Plinder-prefixed versions to avoid incompatible_previous_transforms
        # checks conflicting with the same transforms used again post-crop
        PlinderPreCropAnnotation(),
        PlinderCropFeats(sequence_encoding=af3_sequence_encoding),
        ComputePlinderAtomToTokenMap(),
    ]

    # --- Stage 2: Crop (skipped during inference via ConditionalRoute) ---
    if crop_size is not None:
        cropping_transform = RandomRoute(
            transforms=[
                CropContiguousLikeAF3(
                    crop_size=crop_size,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                ),
                CropSpatialLikeAF3LigandCenter(
                    crop_size=crop_size,
                    crop_center_cutoff_distance=crop_center_cutoff_distance,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                    do_not_crop_target=True,
                ),
            ],
            probs=[crop_contiguous_probability, crop_spatial_probability],
        )
        transforms.append(
            ConditionalRoute(
                condition_func=lambda data: data.get("is_inference", False),
                transform_map={
                    True: Identity(),  # No cropping during inference
                    False: cropping_transform,  # Crop during training
                },
            )
        )

    # --- Stage 3: Post-crop encoding and featurization ---
    transforms += [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
        GetAF3MoleculeFeatures(
            use_element_for_atom_names_of_atomized_tokens=use_element_for_atom_names_of_atomized_tokens,
        ),
        ComputeAtomToTokenMap(),
        AddAF3TokenBondFeatures(),
        AggregateFeaturesLikeLaProteina(),
    ]

    # --- Stage 4: Ligand-specific features and final packaging ---
    keys_to_keep = ["feats", "ground_truth"]
    if return_atom_array:
        keys_to_keep.append("atom_array")

    transforms += [
        ProteinaLigandTransform(use_raw_file=True, use_rdkit_from_smiles=False, use_openbabel=False),
        # AddAF3TokenBondFeatures(),  # only redo if use rdkit or openbabel
        # Subset to only keys necessary
        SubsetToKeys(keys_to_keep),  # also has RemoveKey
        ProteinaFinalTransform(),
    ]
    # For debugging uncomment the below
    # transforms.append(BreakPointTransform())
    # ... compose final pipeline
    pipeline = Compose(transforms)

    return pipeline
