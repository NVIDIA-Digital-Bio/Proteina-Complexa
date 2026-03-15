####################################
################ BioPython functions
####################################
### Import dependencies
import gc
import math
import os
import stat
from collections import defaultdict

import numpy as np
from Bio.PDB import DSSP, PDBIO, PDBParser, Selection, Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial import cKDTree

# Global cache for DSSP results to reduce redundant calculations
_dssp_cache = {}


def safe_dssp_calculation(model, pdb_file, dssp_path, max_retries=3):
    """
    Safely calculate DSSP with proper subprocess cleanup and retry logic.
    Uses caching to avoid redundant calculations on the same file.
    Returns DSSP object or None if all attempts fail.
    """
    # Create a cache key based on the PDB file path
    cache_key = pdb_file

    # Check if we already have this result cached
    if cache_key in _dssp_cache:
        return _dssp_cache[cache_key]

    for attempt in range(max_retries):
        dssp = None
        try:
            # Ensure provided dssp_path is executable if it is a file path
            if isinstance(dssp_path, str) and os.path.isfile(dssp_path):
                try:
                    if not os.access(dssp_path, os.X_OK):
                        st = os.stat(dssp_path)
                        os.chmod(
                            dssp_path,
                            st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
                        )
                except Exception:
                    pass

            # Primary attempt with configured path
            try:
                dssp = DSSP(model, pdb_file, dssp=dssp_path)
            except Exception as primary_error:
                # Fallback to system-installed mkdssp/dssp if available on PATH
                last_error = primary_error
                for alt_cmd in ("mkdssp", "dssp"):
                    try:
                        dssp = DSSP(model, pdb_file, dssp=alt_cmd)
                        last_error = None
                        break
                    except Exception as e_alt:
                        last_error = e_alt
                if dssp is None and last_error is not None:
                    raise last_error
            # Cache the successful result
            _dssp_cache[cache_key] = dssp
            return dssp
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"DSSP attempt {attempt + 1} failed for {pdb_file}: {e}. Retrying...")
                gc.collect()  # Force cleanup before retry
            else:
                print(f"DSSP calculation failed after {max_retries} attempts for {pdb_file}: {e}")
                # Cache the failure to avoid repeated attempts
                _dssp_cache[cache_key] = None
                return None
        finally:
            # Ensure any partial DSSP objects are cleaned up
            if dssp is not None and attempt < max_retries - 1:
                try:
                    del dssp
                except Exception:
                    pass
            gc.collect()
    return None


def clear_dssp_cache():
    """Clear the DSSP cache to free memory."""
    global _dssp_cache
    _dssp_cache.clear()
    gc.collect()


# analyze sequence composition of design
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []

    # Check if protein contains clashes after relaxation
    if num_clashes > 0:
        note_array.append("Relaxed structure contains clashes.")

    # Check if the sequence contains disallowed amino acids
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(",")
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append("Contains: " + restricted_AA + "!")

    # Analyze the protein
    analysis = ProteinAnalysis(sequence)

    # Calculate the reduced extinction coefficient per 1% solution
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # Check if the absorption is high enough
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(
            f"Absorption value is {extinction_coefficient_reduced_1}, consider adding tryptophane to design."
        )

    # Join the notes into a single string
    notes = " ".join(note_array)

    return notes


# temporary function, calculate RMSD of input PDB and trajectory target
def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure("trajectory", trajectory_pdb)
    structure_starting = parser.get_structure("starting", starting_pdb)

    # Extract chain A from trajectory_pdb
    chain_trajectory = structure_trajectory[0]["A"]

    # Extract the specified chains from starting_pdb
    chain_ids = chain_ids_string.split(",")
    residues_starting = []
    for chain_id in chain_ids:
        chain_id = chain_id.strip()
        chain = structure_starting[0][chain_id]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)

    # Extract residues from chain A in trajectory_pdb
    residues_trajectory = [residue for residue in chain_trajectory if is_aa(residue, standard=True)]

    # Ensure that both structures have the same number of residues
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]

    # Collect CA atoms from the two sets of residues
    atoms_starting = [residue["CA"] for residue in residues_starting if "CA" in residue]
    atoms_trajectory = [residue["CA"] for residue in residues_trajectory if "CA" in residue]

    # Calculate RMSD using structural alignment
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms

    return round(rmsd, 2)


# detect C alpha clashes for deformed trajectories
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    atoms = []
    atom_info = []  # Detailed atom info for debugging and processing

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == "H":  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != "CA":
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for i, j in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)


three_to_one_map = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", target_chain="A", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], "A")
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0][target_chain], "A")
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues


# calculate secondary structure percentage of design
def calc_ss_percentage(
    pdb_file,
    advanced_settings,
    chain_id="B",
    target_chain="A",
    atom_distance_cutoff=4.0,
):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model with proper cleanup
    dssp = safe_dssp_calculation(model, pdb_file, advanced_settings["dssp_path"])
    if dssp is None:
        print(f"Warning: DSSP calculation failed for {pdb_file}, returning default values")
        # Return default values if DSSP fails: helix%, beta%, loop%, interface_helix%, interface_beta%, interface_loop%, i_plddt, ss_plddt
        return 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 0.0, 0.0

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(hotspot_residues(pdb_file, chain_id, target_chain, atom_distance_cutoff).keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = "loop"
            if ss in ["H", "G", "I"]:
                ss_type = "helix"
            elif ss == "E":
                ss_type = "sheet"

            ss_counts[ss_type] += 1

            if ss_type != "loop":
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts["helix"], ss_counts["sheet"])
    interface_percentages = calculate_percentages(
        total_interface_residues,
        ss_interface_counts["helix"],
        ss_interface_counts["sheet"],
    )

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    # Explicitly clean up references to help with garbage collection
    del dssp, structure, model, parser
    gc.collect()

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)


def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100, 2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100, 2) if total > 0 else 0
    loop_percentage = round(((total - helix - sheet) / total) * 100, 2) if total > 0 else 0

    return helix_percentage, sheet_percentage, loop_percentage


# PyRosetta-free implementation of align_pdbs using Biopython
def biopython_align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Aligns the align_pdb to the reference_pdb using Biopython and overwrites
    the align_pdb file with the aligned structure.

    Args:
        reference_pdb: Path to the reference PDB file
        align_pdb: Path to the PDB file to be aligned
        reference_chain_id: Chain ID of the reference structure to use for alignment
        align_chain_id: Chain ID of the structure to be aligned
    """
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    reference_structure = parser.get_structure("reference", reference_pdb)
    align_structure = parser.get_structure("align", align_pdb)

    # If the chain IDs contain commas, split them and only take the first value
    reference_chain_id = reference_chain_id.split(",")[0]
    align_chain_id = align_chain_id.split(",")[0]

    # Get the specified chains
    reference_chain = reference_structure[0][reference_chain_id]
    align_chain = align_structure[0][align_chain_id]

    # Extract CA atoms for alignment
    reference_atoms = []
    align_atoms = []

    for residue in reference_chain:
        if is_aa(residue) and "CA" in residue:
            reference_atoms.append(residue["CA"])

    for residue in align_chain:
        if is_aa(residue) and "CA" in residue:
            align_atoms.append(residue["CA"])

    # Use min length to ensure comparable sets
    min_length = min(len(reference_atoms), len(align_atoms))
    reference_atoms = reference_atoms[:min_length]
    align_atoms = align_atoms[:min_length]

    # Align structures
    sup = Superimposer()
    sup.set_atoms(reference_atoms, align_atoms)

    # Apply rotation/translation to all atoms in the structure
    sup.apply(align_structure.get_atoms())

    # Save the aligned structure
    io = PDBIO()
    io.set_structure(align_structure)
    io.save(align_pdb)

    # # Clean the aligned PDB to maintain consistency
    from proteinfoundation.utils.pr_alternative_utils import clean_pdb

    clean_pdb(align_pdb)


# PyRosetta-free implementation of unaligned_rmsd using Biopython
def biopython_unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Calculate RMSD between two PDB structures without aligning them first.

    Args:
        reference_pdb: Path to the reference PDB file
        align_pdb: Path to the PDB file to compare
        reference_chain_id: Chain ID of the reference structure
        align_chain_id: Chain ID of the structure to compare

    Returns:
        float: RMSD value
    """
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    reference_structure = parser.get_structure("reference", reference_pdb)
    align_structure = parser.get_structure("align", align_pdb)

    # If the chain IDs contain commas, split them and only take the first value
    reference_chain_id = reference_chain_id.split(",")[0]
    align_chain_id = align_chain_id.split(",")[0]

    # Get the specified chains
    reference_chain = reference_structure[0][reference_chain_id]
    align_chain = align_structure[0][align_chain_id]

    # Extract CA atoms for RMSD calculation
    reference_atoms = []
    align_atoms = []

    for residue in reference_chain:
        if is_aa(residue) and "CA" in residue:
            reference_atoms.append(residue["CA"])

    for residue in align_chain:
        if is_aa(residue) and "CA" in residue:
            align_atoms.append(residue["CA"])

    # Use min length to ensure comparable sets
    min_length = min(len(reference_atoms), len(align_atoms))
    reference_atoms = reference_atoms[:min_length]
    align_atoms = align_atoms[:min_length]

    # Calculate RMSD without performing alignment
    squared_sum = 0.0
    for ref_atom, align_atom in zip(reference_atoms, align_atoms, strict=False):
        squared_sum += sum((ref_atom.coord - align_atom.coord) ** 2)

    rmsd = math.sqrt(squared_sum / len(reference_atoms))

    return round(rmsd, 2)


def biopython_align_all_ca(reference_pdb_path: str, pdb_to_align_path: str):
    """
    Aligns the pdb_to_align_path to the reference_pdb_path using all C-alpha atoms
    and overwrites the pdb_to_align_path file with the aligned structure.

    Args:
        reference_pdb_path: Path to the reference PDB file.
        pdb_to_align_path: Path to the PDB file to be aligned. This file will be overwritten.
    """
    parser = PDBParser(QUIET=True)
    try:
        reference_structure = parser.get_structure("reference", reference_pdb_path)
        structure_to_align = parser.get_structure("to_align", pdb_to_align_path)
    except Exception as e:
        print(f"Error parsing PDB files for alignment: {e}")
        # Consider whether to raise or simply return if parsing fails
        return

    ref_atoms = []
    align_atoms = []

    # Collect CA atoms from all chains in reference structure
    for model in reference_structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and "CA" in residue:
                    ref_atoms.append(residue["CA"])

    # Collect CA atoms from all chains in structure to align
    for model in structure_to_align:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and "CA" in residue:
                    align_atoms.append(residue["CA"])

    if not ref_atoms or not align_atoms:
        print("Warning: No C-alpha atoms found for alignment in one or both structures. Skipping alignment.")
        return

    # Ensure an equal number of atoms are used for superimposition
    min_len = min(len(ref_atoms), len(align_atoms))
    if min_len == 0:
        print("Warning: Zero common C-alpha atoms for alignment. Skipping alignment.")
        return

    ref_atoms = ref_atoms[:min_len]
    align_atoms = align_atoms[:min_len]

    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, align_atoms)

    # Apply the rotation and translation to all atoms in the structure_to_align
    super_imposer.apply(structure_to_align.get_atoms())

    # Save the aligned structure, overwriting the original file
    io = PDBIO()
    io.set_structure(structure_to_align)
    try:
        io.save(pdb_to_align_path)
        # print(f"Successfully aligned {pdb_to_align_path} to {reference_pdb_path} based on all CA atoms.")
        # Clean the PDB after alignment
        # from .generic_utils import clean_pdb  # Local import to avoid circular dependency issues at module load time
        from proteinfoundation.utils.pr_alternative_utils import clean_pdb

        clean_pdb(pdb_to_align_path)
    except Exception as e:
        print(f"Error saving aligned PDB file {pdb_to_align_path}: {e}")
