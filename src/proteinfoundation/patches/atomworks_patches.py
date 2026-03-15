"""
Patches for atomworks library.
Import this module early to apply patches.

Usage:
    import proteinfoundation.patches.atomworks_patches  # noqa: F401
"""

from __future__ import annotations

import os

# Set before importing atomworks so it does not print "Environment variable ... not set"
print("Loading Atomworks Patches")
if "CCD_MIRROR_PATH" not in os.environ:
    os.environ["CCD_MIRROR_PATH"] = ""
if "PDB_MIRROR_PATH" not in os.environ:
    os.environ["PDB_MIRROR_PATH"] = ""
if "LOCAL_MSA_DIRS" not in os.environ:
    os.environ["LOCAL_MSA_DIRS"] = ""

import logging

import numpy as np
import pandas as pd
import toolz
from atomworks.common import KeyToIntMapper, exists
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.enums import ChainType
from atomworks.io.utils.ccd import get_std_to_alt_atom_name_map
from atomworks.ml.encoding_definitions import TokenEncoding
from atomworks.ml.utils.token import get_token_count, token_iter
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock

logger = logging.getLogger("atomworks.io")


# =============================================================================
# Patch 1: initialize_chain_info_from_category
# Handles missing pdbx_strand_id in CIF files
# =============================================================================


def category_to_df(cif_block: CIFBlock, category: str) -> pd.DataFrame | None:
    """Convert a CIF block category to a pandas DataFrame."""
    return pd.DataFrame(category_to_dict(cif_block, category)) if category in cif_block else None


def category_to_dict(cif_block: CIFBlock, category: str) -> dict[str, np.ndarray]:
    """Convert a CIF block category to a dictionary of numpy arrays."""
    if exists(cif_block.get(category)):
        return toolz.valmap(lambda x: x.as_array(), dict(cif_block[category]))
    return {}


def patched_initialize_chain_info_from_category(cif_block: CIFBlock, atom_array: AtomArray) -> dict:
    """Patched version that handles missing pdbx_strand_id.

    Extracts chain entity-level information from the CIF block.
    Requires the categories 'entity' and 'entity_poly' to be present in the CIF block.
    """
    assert "entity" in cif_block, "entity category not found in CIF block."
    assert "entity_poly" in cif_block, "entity_poly category not found in CIF block."

    chain_info_dict = {}

    # Step 1: Build a mapping of chain id to entity id from the atom_site
    chain_ids = atom_array.get_annotation("chain_id")
    rcsb_entities = atom_array.get_annotation("label_entity_id").astype(str)
    unique_chain_entity_map = dict(zip(chain_ids, rcsb_entities, strict=True))

    # Step 2: Load additional chain information from entity category
    rcsb_entity_df = category_to_df(cif_block, "entity")
    rcsb_entity_df["id"] = rcsb_entity_df["id"].astype(str)
    rcsb_entity_df.rename(columns={"type": "entity_type", "pdbx_ec": "ec_numbers"}, inplace=True)
    rcsb_entity_dict = rcsb_entity_df.set_index("id").to_dict(orient="index")

    # Step 3: Load polymer information from entity_poly
    polymer_df = category_to_df(cif_block, "entity_poly")

    # Handle missing pdbx_strand_id by creating it from chain-entity mapping
    if "pdbx_strand_id" not in polymer_df.columns:
        entity_to_chains = {}
        for chain_id, entity_id in unique_chain_entity_map.items():
            if entity_id not in entity_to_chains:
                entity_to_chains[entity_id] = []
            entity_to_chains[entity_id].append(chain_id)

        polymer_df["pdbx_strand_id"] = (
            polymer_df["entity_id"].astype(str).map(lambda eid: ",".join(sorted(entity_to_chains.get(eid, []))))
        )

    required_columns = ["entity_id", "type", "pdbx_strand_id"]
    optional_columns = ["pdbx_seq_one_letter_code", "pdbx_seq_one_letter_code_can"]
    polymer_df = polymer_df[required_columns + [col for col in optional_columns if col in polymer_df.columns]]

    rename_map = {
        "type": "polymer_type",
        "pdbx_seq_one_letter_code": "non_canonical_sequence",
        "pdbx_seq_one_letter_code_can": "canonical_sequence",
    }
    polymer_df.rename(columns=rename_map, inplace=True)
    polymer_df["entity_id"] = polymer_df["entity_id"].astype(str)
    polymer_dict = polymer_df.set_index("entity_id").to_dict(orient="index")

    # Step 4: Merge information into final dictionary
    for chain_id, rscb_entity in unique_chain_entity_map.items():
        chain_info = rcsb_entity_dict.get(rscb_entity, {})
        polymer_info = polymer_dict.get(rscb_entity, {})

        if chain_info.get("ec_numbers", "?") != "?":
            ec_numbers = [ec.strip() for ec in chain_info.get("ec_numbers", "").split(",")]
        else:
            ec_numbers = []

        chain_type = ChainType.as_enum(polymer_info.get("polymer_type", chain_info.get("entity_type", "non-polymer")))

        chain_info_dict[chain_id] = {
            "rcsb_entity": rscb_entity,
            "chain_type": chain_type,
            "unprocessed_entity_canonical_sequence": polymer_info.get("canonical_sequence", "").replace("\n", ""),
            "unprocessed_entity_non_canonical_sequence": polymer_info.get("non_canonical_sequence", "").replace(
                "\n", ""
            ),
            "is_polymer": chain_type.is_polymer(),
            "ec_numbers": ec_numbers,
        }

    return chain_info_dict


# =============================================================================
# Patch 2: atom_array_to_encoding
# Adds optional atomize_token parameter for unified ligand handling
# =============================================================================


def patched_atom_array_to_encoding(
    atom_array: AtomArray,
    encoding: TokenEncoding,
    default_coord: np.ndarray | float = float("nan"),
    occupancy_threshold: float = 0.0,
    extra_annotations: list[str] = [
        "chain_id",
        "chain_entity",
        "molecule_iid",
        "chain_iid",
        "transformation_id",
    ],
    coord_annotation: str = "coord",
    atomize_token: str | None = None,
    atomize_atom_name: str | None = None,
) -> dict:
    """
    Encode an atom array using a specified `TokenEncoding`.

    This function processes an `AtomArray` to generate encoded representations, including coordinates, masks,
    sequences, and additional annotations. The encoded data comes in numpy arrays which can readily be converted
    to tensors and used in machine learning tasks

    NOTE:
        - `n_token` refers to the number of tokens in the atom array.
        - `n_atoms_per_token` indicates the number of atoms associated with each token in the `encoding`.
          The number of atoms in a token corresponds to the number of residues in the atom array, unless
          the atom array has the `atomize` annotation, in which case the number of tokens may exceed the
          number of residues.

    TODO: Refactor so that `atom_array_to_encoding` uses `atom_array_to_encoded_resnames` internally.
    TODO: Vectorize

    Args:
        - atom_array (AtomArray): The atom array containing polymer information. If the atom array has the
          `atomize` annotation (True for atoms that should be atomized), the number of tokens will differ
          from the number of residues.
        - encoding (TokenEncoding): The encoding scheme to apply to the atom array.
        - default_coord (np.ndarray | float, optional): Default coordinate value to use for uninitialized
          coordinates. Defaults to float("nan").
        - occupancy_threshold (float, optional): Minimum occupancy for atoms to be considered resolved
          in the mask. Defaults to 0.0 (only completely unresolved atoms are masked).
        - extra_annotations (list[str], optional): A list of additional annotations to encode. These must
          be `id` style annotations (e.g., `chain_id`, `molecule_iid`). The encoding will be generated as
          integers, where the first occurrence of a given ID is encoded as `0`, and subsequent occurrences
          are encoded as `1`, `2`, etc. Defaults to
          ["chain_id", "chain_entity", "molecule_iid", "chain_iid", "transformation_id"].
        - coord_annotation (str, optional): The annotation of the AtomArray containing the coordinates to encode.
          Defaults to "coord".

    Returns:
        - dict: A dictionary containing the following keys:
            - `xyz` (np.ndarray): Encoded coordinates of shape [n_token, n_atoms_per_token, 3].
            - `mask` (np.ndarray): Encoded mask of shape [n_token, n_atoms_per_token], indicating which
              atoms are resolved in the encoded sequence.
            - `seq` (np.ndarray): Encoded sequence of shape [n_token].
            - `token_is_atom` (np.ndarray): Boolean array of shape [n_token] indicating whether each token
              corresponds to an atom.
            - Various additional annotations encoded as extra keys in the dictionary. Each extra annotation
                that gets exposed is results in 2 keys in the dictionary. One for the encoded annotation itself
                and one mapping the annotation to integers if e.g. the original annotation was strings.
                For example, the defaults above result in:
                - `chain_id` (np.ndarray): Encoded chain IDs of shape [n_token].
                - `chain_id_to_int` (dict): Mapping of chain IDs to integers in the `chain_id` array.
                - `chain_entity` (np.ndarray): Encoded entity IDs of shape [n_token].
                - `chain_entity_to_int` (dict): Mapping of entity IDs to integers in the `chain_entity` array.
    """
    # Extract atom array information
    n_token = get_token_count(atom_array)

    # Init encoded arrays
    encoded_coord = np.full(
        (n_token, encoding.n_atoms_per_token, 3),
        fill_value=default_coord,
        dtype=np.float32,
    )  # [n_token, n_atoms_per_token, 3] (float)

    encoded_mask = np.zeros((n_token, encoding.n_atoms_per_token), dtype=bool)  # [n_token, n_atoms_per_token] (bool)
    encoded_seq = np.empty(n_token, dtype=int)  # [n_token] (int)
    encoded_token_is_atom = np.empty(n_token, dtype=bool)  # [n_token] (bool)

    # init additional annotation
    extra_annot_counters = {}
    extra_annot_encoded = {}
    for key in extra_annotations:
        if key in atom_array.get_annotation_categories():
            extra_annot_counters[key] = KeyToIntMapper()
            extra_annot_encoded[key] = []

    # Iterate over residues and encode (# TODO: Speed up by vectorizing if necessary)
    # ... record whether the atom array has the `atomize` annotation to deal with atomized residues
    has_atomize = "atomize" in atom_array.get_annotation_categories()
    for i, token in enumerate(token_iter(atom_array)):
        # ... extract token name
        # ... case 1: atom tokens (e.g. 6 - for carbon)
        if (has_atomize and token.atomize[0]) or len(token) == 1:
            if atomize_token is not None:
                token_name = atomize_token
                true_token_name = (
                    token.atomic_number[0]
                    if "atomic_number" in token.get_annotation_categories()
                    else ELEMENT_NAME_TO_ATOMIC_NUMBER[token.element[0].upper()]
                )
            else:
                token_name = (
                    token.atomic_number[0]
                    if "atomic_number" in token.get_annotation_categories()
                    else ELEMENT_NAME_TO_ATOMIC_NUMBER[token.element[0].upper()]
                )
            token_is_atom = True
        # ... case 2: residue tokens (e.g. "ALA")
        else:
            token_name = token.res_name[0]
            token_is_atom = False

        if token_name not in encoding.token_to_idx:
            token_name = encoding.resolve_unknown_token_name(token_name, token_is_atom)
            assert token_name in encoding.token_to_idx, f"Unknown token name: {token_name}"

        # Encode sequence
        encoded_seq[i] = encoding.token_to_idx[token_name]

        # Encode if token is an `atom-level` token or a `residue-level` token
        encoded_token_is_atom[i] = token_is_atom

        # Encode coords
        for atom in token:
            if atomize_token is not None and token_is_atom:
                atom_name = atomize_atom_name if atomize_atom_name is not None else "X"  # true_token_name
            else:
                atom_name = str(token_name) if token_is_atom else atom.atom_name
            # (token_name, atom_name) is e.g.
            #  ... ('ALA', 'CA') if  token_is_atom=False
            #  ... ('UNK', whatever) if token_is_atom=False but we had to resolve an unknown token
            #  ... (6, '6') if token_is_atom=True

            # ... case 1: atom name is in the encoding
            if (token_name, atom_name) in encoding.atom_to_idx:
                to_idx = encoding.atom_to_idx[(token_name, atom_name)]
                encoded_coord[i, to_idx, :] = getattr(atom, coord_annotation)
                encoded_mask[i, to_idx] = atom.occupancy > occupancy_threshold

            # ... case 2: atom name does not exist for token, but token is an `unknown` token,
            #  so it's `ok` to not match
            elif token_name in encoding.unknown_tokens:
                continue

            # ... case 3: atom name is not in encoding, but token is, and try_matching_alt_atom_name_if_fails is True
            elif not token_is_atom:
                alt_to_std = get_std_to_alt_atom_name_map(token_name)
                alt_atom_name = alt_to_std.get(atom_name, None)
                if exists(alt_atom_name) and (token_name, alt_atom_name) in encoding.atom_to_idx:
                    to_idx = encoding.atom_to_idx[(token_name, alt_atom_name)]
                    encoded_coord[i, to_idx, :] = getattr(atom, coord_annotation)

            # ... case 4: failed to find the relevant atom_name for this token when we should, so we raise an error
            else:
                msg = f"Atom ({token_name}, {atom_name}) not in encoding for token `{token_name}`"
                msg += "\nProblematic atom:\n"
                msg += f"{atom}"
                raise ValueError(msg)

        # Encode additional annotation
        for key in extra_annot_counters:
            annot = token.get_annotation(key)[0]
            extra_annot_encoded[key].append(extra_annot_counters[key](annot))

    return {
        "xyz": encoded_coord,  # [n_token_in_atom_array, n_atoms_per_token, 3] (float)
        "mask": encoded_mask,  # [n_token_in_atom_array, n_atoms_per_token] (bool)
        "seq": encoded_seq,  # [n_token_in_atom_array] (int)
        "token_is_atom": encoded_token_is_atom,  # [n_token_in_atom_array] (bool)
        **{annot: np.array(extra_annot_encoded[annot], dtype=np.int16) for annot in extra_annot_encoded},
        **{annot + "_to_int": extra_annot_counters[annot].key_to_id for annot in extra_annot_counters},
    }


# =============================================================================
# Apply patches
# =============================================================================

import atomworks.io.parser as parser_module

parser_module.initialize_chain_info_from_category = patched_initialize_chain_info_from_category

import atomworks.ml.transforms.encoding as encoding_module

encoding_module.atom_array_to_encoding = patched_atom_array_to_encoding

print("[atomworks_patches] Patched initialize_chain_info_from_category")
print("[atomworks_patches] Patched atom_array_to_encoding (added atomize_token parameter)")
