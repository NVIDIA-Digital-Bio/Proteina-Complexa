# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Protein data type."""

import io
import os
import re
from collections.abc import Mapping
from typing import Any

import biotite
import numpy as np
import torch
from atomworks.io.utils.io_utils import load_any
from atomworks.io.utils.selection import AtomSelectionStack
from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_from_encoding, atom_array_to_encoding
from Bio.PDB import PDBParser, Superimposer
from biotite.structure import AtomArray
from biotite.structure.io import load_structure, save_structure
from openfold.np import residue_constants
from openfold.np.protein import Protein

from proteinfoundation.utils.coors_utils import ang_to_nm

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


def pdb_name_from_path(pdb_file_path: str) -> str:
    """Extracts the PDB filename without extension from a file path.

    Args:
        pdb_file_path: Full path to the PDB file.

    Returns:
        The PDB filename without the .pdb extension.
    """
    return pdb_file_path.strip(os.sep).split(os.sep)[-1][:-4]


def extract_seq_from_pdb(fname: str, chain_id: str = None) -> str:
    """Extracts the amino acid sequence from a PDB file.

    Args:
        fname: Path to the PDB file.
        chain_id: Chain ID to extract. If None, all chains are extracted.

    Returns:
        Single-letter amino acid sequence as a string.
    """
    protein = load_pdb(fname, chain_id=chain_id)
    seq = []
    for aa in protein.aatype:
        seq.append(residue_constants.restypes[aa])
    return "".join(seq)


def extract_binder_from_pdb(
    pdb_file_path: str,
    binder_chain: str,
    output_path: str,
) -> str:
    """Extracts the binder chain (CA atoms only) from the PDB file and saves to a new file.

    Args:
        pdb_file_path: Path to the input PDB file.
        binder_chain: Chain ID of the binder to extract.
        output_path: Path to save the extracted binder.

    Returns:
        The output path where the binder was saved.
    """
    struct = load_structure(pdb_file_path)
    binder_pdb = struct[struct.chain_id == binder_chain]
    binder_pdb = binder_pdb[binder_pdb.atom_name == "CA"]
    save_structure(output_path, binder_pdb)
    return output_path


def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    chain_index=None,
    aatype=None,
    b_factors=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    if chain_index is None:
        chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    aatype: np.ndarray = None,
    chain_index: np.ndarray = None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx + 1}.pdb"
    else:
        save_path = file_path
    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37,
                    atom37_mask,
                    chain_index=chain_index,
                    aatype=aatype,
                    b_factors=b_factors,
                )
                pdb_prot = to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos,
                atom37_mask,
                chain_index=chain_index,
                aatype=aatype,
                b_factors=b_factors,
            )
            pdb_prot = to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path


def to_pdb(prot: Protein, model=1, add_end=True) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(int) + 1  # to start from 1
    chain_index = prot.chain_index.astype(int)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    last_chain_index = chain_index[0]
    chain_residue_index_offset = 1
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            chain_residue_index_offset = residue_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i], strict=False
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i] - chain_residue_index_offset + 1:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    if add_end:
        pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} {chain_name:>1}{residue_index:>4}"


def from_pdb_file(pdb_file: str, chain_id: str | None = None) -> Protein:
    """Takes a PDB file and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_file: Path to the PDB file
      chain_id: If chain_id is specified (e.g. A), then only that chain is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    with open(pdb_file) as f:
        pdb_str = f.read()
    return from_pdb_string(pdb_str=pdb_str, chain_id=chain_id)


def from_pdb_string(pdb_str: str, chain_id: str | None = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def load_pdb(fname: str, chain_id: str = None) -> str:
    """Returns pdb stored in input file as string."""
    with open(fname) as f:
        return from_pdb_string(f.read(), chain_id=chain_id)


def get_chain_ids_from_pdb(pdb_path: str) -> tuple[str, str]:
    """Returns the target and binder chain IDs from a PDB file."""
    # Quickly extract all chain IDs from the PDB file by parsing lines that start with "ATOM" or "HETATM"
    # Assume all PDB files have the same chain ID set
    chain_ids = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or (line.startswith("HETATM") and len(line) >= 22):
                chain_id = line[21].strip()
                if chain_id:
                    chain_ids.add(chain_id)
    chain_ids = sorted(chain_ids)
    target_chain = ",".join(chain_ids[:-1])
    binder_chain = chain_ids[-1]
    return target_chain, binder_chain


def load_atoms_from_pdb(pdb_file: str, chain_id: str | None = None) -> AtomArray:
    """
    Parses a PDB file and returns an AtomArray for a specified chain.

    Args:
        pdb_file: The path to the PDB file.
        chain_id: The ID of the chain to extract (e.g., 'A').
            If None, all chains are extracted.

    Returns:
        AtomArray containing atoms from the specified chain(s).

    Raises:
        FileNotFoundError: If the PDB file does not exist.
        ValueError: If the specified chain is not found in the PDB file.
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"Error: PDB file not found at '{pdb_file}'")

    try:
        struct = load_any(pdb_file)[0]
    except Exception as e:
        raise RuntimeError(f"An error occurred while parsing {pdb_file}: {e}")

    # Filter to specified chain
    if chain_id is not None:
        chain_mask = struct.chain_id == chain_id
    else:
        chain_mask = np.ones(len(struct), dtype=bool)
    if not chain_mask.any():
        available_chains = sorted(set(struct.chain_id.tolist()))
        raise ValueError(f"Error: Chain '{chain_id}' not found in '{pdb_file}'. Available chains: {available_chains}")

    ligand_atoms = struct[chain_mask]
    if len(ligand_atoms) == 0:
        raise ValueError(f"Error: No atoms found in chains '{chain_id}' of '{pdb_file}'")

    return ligand_atoms


def calculate_rmsd_from_atoms(atoms1, atoms2):
    """
    Calculates the Root Mean Square Deviation (RMSD) between two sets of atoms.
    It first aligns the atoms using the Superimposer to find the minimum RMSD.

    Args:
        atoms1 (list): A list of Bio.PDB.Atom objects.
        atoms2 (list): A list of Bio.PDB.Atom objects.

    Returns:
        float: The RMSD value in Angstroms.

    Raises:
        ValueError: If the number of atoms in the two lists is not equal.
    """
    if len(atoms1) != len(atoms2):
        raise ValueError("Error: Cannot calculate RMSD. The number of atoms is not equal.")
    # Initialize the Superimposer
    super_imposer = Superimposer()
    # Set the coordinates to be superimposed
    super_imposer.set_atoms(atoms1, atoms2)
    # The RMSD is automatically calculated upon setting the atoms
    rmsd = super_imposer.rms

    return rmsd


def write_prot_ligand_to_pdb(coors: torch.Tensor, residue_type: torch.Tensor, ligand: AtomArray, pdb_path: str):
    """
    Writes a protein and ligand to a PDB file.

    Args:
        coors: torch.Tensor, shape (n_res, 37, 3)
        residue_type: torch.Tensor, shape (n_res,)
        ligand: biotite.structure.AtomArray
        pdb_path: str
    """
    atom37_mask = torch.sum(torch.abs(coors), axis=-1) > 1e-7
    prot = atom_array_from_encoding(
        encoded_coord=coors,
        encoded_mask=atom37_mask,
        encoded_seq=residue_type.int(),
        encoding=AF2_ATOM37_ENCODING,
    ).copy()
    res_names = np.array([name for name in prot.res_name], dtype="U5")
    prot.del_annotation("res_name")
    prot.set_annotation("res_name", res_names)
    lig = ligand.copy()
    chain_b = np.array(["B" for chain in prot.chain_id], dtype=lig.chain_id.dtype)
    prot.del_annotation("chain_id")
    prot.set_annotation("chain_id", chain_b)
    complex_array = biotite.structure.concatenate([prot, lig])
    biotite.structure.io.save_structure(pdb_path, complex_array)


def sort_AtomArray_by_chain_id(atom_array: biotite.structure.AtomArray):
    """
    Sorts an AtomArray by chain ID in alphabetical order.
    Args:
        atom_array: biotite.structure.AtomArray
    Returns:
        biotite.structure.AtomArray: The sorted AtomArray.
    """
    chain_ids = sorted(set(atom_array.chain_id))
    chains = []
    for chain_id in chain_ids:
        chains.append(atom_array[atom_array.chain_id == chain_id])
    return biotite.structure.concatenate(chains)


def load_target_from_pdb(target_spec, pdb_path, target_hotspots=None, convert_ang_to_nm=True):
    """
    Loads a target from a PDB file and extracts the target mask, structure, residue type, chain and hotspots mask.
    Args:
        target_spec: str, the target specification
        pdb_path: str, the path to the PDB file
        target_hotspots: list, the target hotspots
            Default is None.
        convert_ang_to_nm: bool, whether to convert the target structure from Angstrom to nanometers.
            Default is True.
    Returns:
        target_mask: torch.Tensor, the target mask
        target_structure: torch.Tensor, the target structure
        target_residue_type: torch.Tensor, the target residue type
        target_hotspots_mask: torch.Tensor, the target hotspots mask
        target_chain: torch.Tensor, the target chain
    """
    struct = load_any(pdb_path, model=1)
    if not hasattr(struct, "occupancy"):
        struct.set_annotation("occupancy", np.ones(len(struct), dtype=np.float32))

    select = AtomSelectionStack.from_contig(target_spec)
    mask = select.get_mask(struct)
    struct = struct[mask]
    ca_struct = struct[struct.atom_name == "CA"]

    d = atom_array_to_encoding(
        struct,
        encoding=AF2_ATOM37_ENCODING,
        default_coord=0.0,
    )
    target_mask = torch.from_numpy(d["mask"]).bool()
    if convert_ang_to_nm:
        target_structure = ang_to_nm(torch.from_numpy(d["xyz"]))
    else:
        target_structure = torch.from_numpy(d["xyz"])
    target_residue_type = torch.from_numpy(d["seq"]).long()
    target_chain = torch.from_numpy(d["chain_id"]).long()
    target_hotspots_mask = torch.zeros(len(ca_struct), dtype=torch.bool)
    if target_hotspots is not None:
        for idx, atom in enumerate(ca_struct):
            if f"{atom.chain_id}{atom.res_id}" in target_hotspots:
                target_hotspots_mask[idx] = True

    return (
        target_mask,
        target_structure,
        target_residue_type,
        target_hotspots_mask,
        target_chain,
    )
