#!/usr/bin/env python3
"""
Script to visualize protein interfaces and hydrogen bonds between binder and target proteins.

This script provides comprehensive interface visualization and analysis:
1. Interface residue identification and visualization
2. Hydrogen bond detection and display
3. Polarity analysis of interface residues
4. Multiple visualization modes (overview, detailed, hydrogen bonds, basic)
5. Batch processing capabilities
6. Statistics export to CSV
7. Example demonstrations
8. Integration with existing protein foundation models codebase

Usage Examples:
    # Basic visualization of a single PDB file
    python script_utils/visualize_interfaces_comprehensive.py --pdb_file path/to/complex.pdb --output_dir visualizations

    # Advanced visualization with multiple modes
    python script_utils/visualize_interfaces_comprehensive.py --pdb_file complex.pdb --modes overview,detailed,hbonds --output_dir visualizations --save_stats

    # Batch processing with statistics
    python script_utils/visualize_interfaces_comprehensive.py --pdb_dir path/to/pdb_files --output_dir visualizations --save_stats --limit 10

    # Run example demonstrations
    python script_utils/visualize_interfaces_comprehensive.py --run_examples

    # Custom analysis parameters
    python script_utils/visualize_interfaces_comprehensive.py --pdb_file complex.pdb --distance_threshold 10.0 --hbond_distance 3.5

Arguments:
    --pdb_file: Path to a single PDB file to visualize
    --pdb_dir: Directory containing PDB files to visualize
    --output_dir: Output directory for visualization images (default: interface_visualizations)
    --mode: Visualization mode - overview, detailed, hbonds, or all (default: detailed)
    --modes: Multiple visualization modes (comma-separated)
    --distance_threshold: Distance threshold for interface definition in Angstroms (default: 8.0)
    --hbond_distance: Distance threshold for hydrogen bonds in Angstroms (default: 3.5)
    --high_res: Generate high resolution images (default: False)
    --save_session: Save PyMOL session files (default: False)
    --save_stats: Save interface statistics to CSV (default: False)
    --chain_binder: Chain ID for binder protein (default: auto-detect)
    --chain_target: Chain ID for target protein (default: auto-detect)
    --limit: Maximum number of files to process
    --run_examples: Run demonstration examples
    --analysis_only: Only compute and save statistics without visualization
"""

import argparse
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import ammolite
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from proteinfoundation.utils.constants import AA_CHARACTER_PROTORP


def categorize_contact_by_polarity(residue_name: str) -> str:
    """Categorize amino acid by polarity based on AA_CHARACTER_PROTORP.

    Args:
        residue_name: Three-letter amino acid code (e.g., 'ALA', 'GLU')

    Returns:
        Polarity category - 'A' (apolar), 'P' (polar), 'C' (charged), or 'Unknown'
    """
    return AA_CHARACTER_PROTORP.get(residue_name, "Unknown")


def get_polarity_color(polarity: str) -> str:
    """Get color for polarity category.

    Args:
        polarity: Polarity category ('A', 'P', 'C')

    Returns:
        Color name for PyMOL
    """
    color_map = {
        "A": "orange",  # Apolar - orange
        "P": "cyan",  # Polar - cyan
        "C": "magenta",  # Charged - magenta
    }
    return color_map.get(polarity, "white")


def convert_ss_annotations(ss_array: np.ndarray) -> np.ndarray:
    """Convert secondary structure annotations from Biotite to PyMOL format.

    Args:
        ss_array: Array of secondary structure annotations from Biotite.

    Returns:
        Array of secondary structure annotations in PyMOL format.
    """
    ss_mapping = {"c": "L", "a": "H", "b": "S"}
    return np.vectorize(ss_mapping.get)(ss_array)


def setup_clean_visualization(pymol_obj: ammolite.PyMOLObject) -> None:
    """Setup clean visualization with basic cartoon representation.

    Args:
        pymol_obj: PyMOL object to setup
    """
    # Show cartoon representation
    ammolite.cmd.show("cartoon", pymol_obj.name)
    ammolite.cmd.set("cartoon_transparency", 0.6, pymol_obj.name)


def apply_highres_settings() -> None:
    """Apply settings for high resolution rendering in PyMOL.
    Inherited from visual_utils.py.
    """
    ammolite.cmd.set("cartoon_discrete_colors", 1)
    ammolite.cmd.set("cartoon_oval_length", 1.0)
    ammolite.cmd.set("ray_trace_mode", 1)
    ammolite.cmd.set("ambient", 0.1)


def load_structure(pdb_path: str) -> struc.AtomArray:
    """Load protein structure from PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Biotite structure object
    """
    try:
        if pdb_path.endswith(".mmtf"):
            mmtf_file = mmtf.MMTFFile.read(pdb_path)
            structure = mmtf.get_structure(mmtf_file, model=1, include_bonds=True)
        else:
            pdb_file = pdb.PDBFile.read(pdb_path)
            structure = pdb_file.get_structure(model=1, include_bonds=True)

        # Remove water and ions
        structure = structure[~struc.filter_solvent(structure) & ~struc.filter_monoatomic_ions(structure)]

        # Add hydrogen atoms for better hydrogen bond detection
        # Note: biotite doesn't have a direct add_hydrogen_atoms function
        # We'll use the structure as-is and handle missing hydrogens in hydrogen bond detection
        logger.info(f"Structure loaded with {len(structure)} atoms")

        # Check if structure already has hydrogen atoms
        has_hydrogens = np.any(structure.element == "H")
        if has_hydrogens:
            logger.info(f"Structure contains {np.sum(structure.element == 'H')} hydrogen atoms")
        else:
            logger.info("No hydrogen atoms found - hydrogen bond detection may be limited")

        return structure
    except Exception as e:
        logger.error(f"Failed to load structure from {pdb_path}: {e}")
        raise


def identify_chains(
    structure: struc.AtomArray,
    chain_binder: str | None = None,
    chain_target: str | None = None,
) -> tuple[str, str]:
    """Identify binder and target chains.

    Args:
        structure: Protein structure
        chain_binder: Optional binder chain ID
        chain_target: Optional target chain ID

    Returns:
        Tuple of (binder_chain, target_chain)
    """
    unique_chains = np.unique(structure.chain_id)

    if len(unique_chains) < 2:
        raise ValueError(f"Structure must have at least 2 chains, found {len(unique_chains)}")

    if chain_binder and chain_target:
        if chain_binder not in unique_chains or chain_target not in unique_chains:
            raise ValueError(f"Specified chains {chain_binder}, {chain_target} not found in structure")
        return chain_binder, chain_target

    # Auto-detect: assume first chain is binder, second is target
    # In practice, you might want more sophisticated logic here
    binder_chain = unique_chains[0]
    target_chain = unique_chains[1]

    logger.info(f"Auto-detected chains: binder={binder_chain}, target={target_chain}")
    return binder_chain, target_chain


def compute_interface_analysis(
    structure: struc.AtomArray,
    binder_chain: str,
    target_chain: str,
    distance_threshold: float = 8.0,
    hbond_distance: float = 3.5,
) -> dict[str, Any]:
    """Compute comprehensive interface analysis.

    Args:
        structure: Protein structure
        binder_chain: Binder chain ID
        target_chain: Target chain ID
        distance_threshold: Distance threshold for interface definition
        hbond_distance: Distance threshold for hydrogen bonds

    Returns:
        Dictionary with comprehensive interface statistics
    """

    # Split structure into binder and target
    binder_mask = structure.chain_id == binder_chain
    target_mask = structure.chain_id == target_chain

    binder_structure = structure[binder_mask]
    target_structure = structure[target_mask]

    # Get CA coordinates
    binder_ca_mask = binder_structure.atom_name == "CA"
    target_ca_mask = target_structure.atom_name == "CA"

    binder_ca_coords = binder_structure.coord[binder_ca_mask]
    target_ca_coords = target_structure.coord[target_ca_mask]

    # Compute pairwise distances
    distances = np.linalg.norm(binder_ca_coords[:, None, :] - target_ca_coords[None, :, :], axis=-1)

    # Find interface residues
    interface_mask = distances < distance_threshold

    binder_interface_mask = interface_mask.any(axis=1)
    target_interface_mask = interface_mask.any(axis=0)

    # Log interface detection results
    logger.info(f"Distance threshold: {distance_threshold}Å")
    logger.info(f"Binder CA atoms: {len(binder_ca_coords)}")
    logger.info(f"Target CA atoms: {len(target_ca_coords)}")
    logger.info(f"Binder interface residues: {np.sum(binder_interface_mask)}")
    logger.info(f"Target interface residues: {np.sum(target_interface_mask)}")
    logger.info(f"Min distance between chains: {distances.min():.2f}Å")
    logger.info(f"Max distance between chains: {distances.max():.2f}Å")

    # Get interface residue information
    binder_interface_indices = np.where(binder_interface_mask)[0]
    target_interface_indices = np.where(target_interface_mask)[0]

    # Get the CA atoms that are at the interface
    binder_interface_ca_atoms = binder_structure[binder_ca_mask][binder_interface_indices]
    target_interface_ca_atoms = target_structure[target_ca_mask][target_interface_indices]

    # Extract unique residue IDs from the interface CA atoms
    binder_interface_res_ids = np.unique(binder_interface_ca_atoms.res_id)
    target_interface_res_ids = np.unique(target_interface_ca_atoms.res_id)

    # Create unique identifiers for interface residues
    binder_interface_identifiers = [(binder_chain, res_id) for res_id in binder_interface_res_ids]
    target_interface_identifiers = [(target_chain, res_id) for res_id in target_interface_res_ids]

    # Analyze polarity of interface residues
    binder_polarity_counts = defaultdict(int)
    target_polarity_counts = defaultdict(int)

    for res_id in binder_interface_res_ids:
        res_mask = binder_structure.res_id == res_id
        res_name = binder_structure.res_name[res_mask][0]
        polarity = categorize_contact_by_polarity(res_name)
        binder_polarity_counts[polarity] += 1

    for res_id in target_interface_res_ids:
        res_mask = target_structure.res_id == res_id
        res_name = target_structure.res_name[res_mask][0]
        polarity = categorize_contact_by_polarity(res_name)
        target_polarity_counts[polarity] += 1

    # Find hydrogen bonds
    hydrogen_bonds = find_hydrogen_bonds(structure, binder_chain, target_chain, hbond_distance)

    # Compute interface statistics
    interface_distances = distances[interface_mask]

    stats = {
        "binder_interface_residues": len(binder_interface_res_ids),
        "target_interface_residues": len(target_interface_res_ids),
        "total_interface_residues": len(binder_interface_res_ids) + len(target_interface_res_ids),
        "hydrogen_bonds": len(hydrogen_bonds),
        "min_distance": distances.min() if len(distances) > 0 else float("inf"),
        "max_distance": distances.max() if len(distances) > 0 else 0.0,
        "avg_distance": distances.mean() if len(distances) > 0 else 0.0,
        "interface_min_distance": (interface_distances.min() if len(interface_distances) > 0 else float("inf")),
        "interface_max_distance": (interface_distances.max() if len(interface_distances) > 0 else 0.0),
        "interface_avg_distance": (interface_distances.mean() if len(interface_distances) > 0 else 0.0),
        "distance_threshold": distance_threshold,
        "hbond_distance": hbond_distance,
        "binder_polarity_counts": dict(binder_polarity_counts),
        "target_polarity_counts": dict(target_polarity_counts),
        "binder_interface_res_ids": binder_interface_res_ids.tolist(),
        "target_interface_res_ids": target_interface_res_ids.tolist(),
        "binder_interface_identifiers": binder_interface_identifiers,
        "target_interface_identifiers": target_interface_identifiers,
        "hydrogen_bonds": hydrogen_bonds,
    }

    return stats


def find_hydrogen_bonds(
    structure: struc.AtomArray,
    binder_chain: str,
    target_chain: str,
    hbond_distance: float = 3.5,
) -> list[tuple[int, int, int, float]]:
    """Find hydrogen bonds between binder and target chains with distances.

    Args:
        structure: Protein structure
        binder_chain: Binder chain ID
        target_chain: Target chain ID
        hbond_distance: Distance threshold for hydrogen bonds

    Returns:
        List of (donor, hydrogen, acceptor, distance) tuples
    """

    # Create atom selections for binder and target chains
    binder_mask = structure.chain_id == binder_chain
    target_mask = structure.chain_id == target_chain

    # Check if we have hydrogen atoms
    has_hydrogens = np.any(structure.element == "H")
    if not has_hydrogens:
        logger.info("No explicit hydrogen atoms found - biotite will infer hydrogen positions for bond detection")
        logger.info("This may limit the accuracy of hydrogen bond detection")

    # Use biotite's hbond function with atom selections
    # This finds hydrogen bonds between atoms in selection1 and selection2
    try:
        triplets = struc.hbond(
            structure,
            selection1=binder_mask,
            selection2=target_mask,
            selection1_type="both",  # Consider both donors and acceptors in binder
            cutoff_dist=hbond_distance,
            cutoff_angle=120,  # Default angle cutoff
            donor_elements=("O", "N", "S"),  # Default donor elements
            acceptor_elements=("O", "N", "S"),  # Default acceptor elements
        )

        # Convert triplets to list of tuples with distances
        cross_chain_bonds = []
        for donor_idx, hydrogen_idx, acceptor_idx in triplets:
            # Calculate hydrogen-acceptor distance
            h_acceptor_dist = np.linalg.norm(structure.coord[hydrogen_idx] - structure.coord[acceptor_idx])
            cross_chain_bonds.append((donor_idx, hydrogen_idx, acceptor_idx, h_acceptor_dist))

        # Sort by distance
        cross_chain_bonds.sort(key=lambda x: x[3])

        logger.info(f"Found {len(cross_chain_bonds)} cross-chain hydrogen bonds")

        # Log some details about the hydrogen bonds found
        if cross_chain_bonds:
            for i, (donor, hydrogen, acceptor, distance) in enumerate(cross_chain_bonds[:3]):  # Show first 3
                donor_res = structure.res_name[donor]
                acceptor_res = structure.res_name[acceptor]
                logger.info(f"HBond {i + 1}: {donor_res} -> {acceptor_res} (distance: {distance:.2f}Å)")

        return cross_chain_bonds

    except Exception as e:
        logger.error(f"Error detecting hydrogen bonds: {e}")
        return []


def create_overview_visualization(
    structure: struc.AtomArray,
    stats: dict[str, Any],
    binder_chain: str,
    target_chain: str,
    output_path: str,
    high_res: bool = False,
) -> None:
    """Create overview visualization showing both proteins and interface."""

    ammolite.reset()
    pymol_obj = ammolite.PyMOLObject.from_structure(structure)

    # Setup clean visualization
    setup_clean_visualization(pymol_obj)

    # Color chains (binder vs target)
    binder_mask = structure.chain_id == binder_chain
    target_mask = structure.chain_id == target_chain

    pymol_obj.color("salmon", binder_mask)
    pymol_obj.color("lightblue", target_mask)

    # Highlight interface residues
    interface_mask = np.zeros(len(structure), dtype=bool)
    for chain_id, res_id in stats["binder_interface_identifiers"] + stats["target_interface_identifiers"]:
        interface_mask |= (structure.chain_id == chain_id) & (structure.res_id == res_id)

    # Color interface residues yellow to highlight them
    pymol_obj.color("yellow", interface_mask)

    # Zoom to show both proteins with better angle
    pymol_obj.zoom("all")

    # Rotate to get a better view of the complex
    ammolite.cmd.rotate("x", 20)
    ammolite.cmd.rotate("y", 30)

    # Save image (inherited settings from visual_utils.py)
    if high_res:
        apply_highres_settings()
        ammolite.cmd.png(f"{output_path}_overview.png", width=3000, height=2000, dpi=300, ray=1)
    else:
        ammolite.cmd.png(f"{output_path}_overview.png", width=500, height=330, dpi=100, ray=0)


def create_detailed_visualization(
    structure: struc.AtomArray,
    stats: dict[str, Any],
    binder_chain: str,
    target_chain: str,
    output_path: str,
    high_res: bool = False,
) -> None:
    """Create detailed visualization with polarity coloring and hydrogen bonds."""

    ammolite.reset()
    pymol_obj = ammolite.PyMOLObject.from_structure(structure)

    # Setup clean visualization
    setup_clean_visualization(pymol_obj)

    # Color chains (binder vs target)
    binder_mask = structure.chain_id == binder_chain
    target_mask = structure.chain_id == target_chain

    pymol_obj.color("salmon", binder_mask)
    pymol_obj.color("lightblue", target_mask)

    # Highlight interface residues with a different color
    interface_mask = np.zeros(len(structure), dtype=bool)
    for chain_id, res_id in stats["binder_interface_identifiers"] + stats["target_interface_identifiers"]:
        interface_mask |= (structure.chain_id == chain_id) & (structure.res_id == res_id)

    # Show elegant side chains for interface residues
    pymol_obj.show("sticks", interface_mask & (structure.atom_name != "CA"))
    pymol_obj.set("stick_radius", 0.3, interface_mask & (structure.atom_name != "CA"))
    pymol_obj.color("yellow", interface_mask & (structure.atom_name != "CA"))

    # Show hydrogen bonds with elegant styling
    if stats["hydrogen_bonds"]:
        for i, (donor, hydrogen, acceptor, distance) in enumerate(stats["hydrogen_bonds"]):
            # Show hydrogen bond atoms as spheres
            pymol_obj.show("spheres", [donor, hydrogen, acceptor])
            pymol_obj.set("sphere_scale", 0.4, [donor, hydrogen, acceptor])
            pymol_obj.color("red", [donor, hydrogen, acceptor])

            # Create elegant hydrogen bond line
            pymol_obj.distance(f"hbond_{i}", hydrogen, acceptor, show_label=True)
            ammolite.cmd.set("dash_gap", 0.0, f"hbond_{i}")
            ammolite.cmd.set("dash_width", 3.0, f"hbond_{i}")
            ammolite.cmd.set("dash_color", "red", f"hbond_{i}")

            # Add distance labels
            ammolite.cmd.set("label_size", 14, f"hbond_{i}")
            ammolite.cmd.set("label_color", "black", f"hbond_{i}")
    else:
        logger.info("No hydrogen bonds detected - this may be due to missing hydrogen atoms or distance threshold")

    # Set better camera angle for interface viewing
    pymol_obj.zoom(interface_mask, buffer=8.0)

    # Rotate to get a better view of the interface
    ammolite.cmd.rotate("x", 30)
    ammolite.cmd.rotate("y", 45)

    # Adjust the view to focus on the interface plane
    ammolite.cmd.orient()

    # Set label properties
    ammolite.cmd.set("label_color", "black")
    ammolite.cmd.set("label_size", 16)

    # Save image (inherited settings from visual_utils.py)
    if high_res:
        apply_highres_settings()
        ammolite.cmd.png(f"{output_path}_detailed.png", width=3000, height=2000, dpi=300, ray=1)
    else:
        ammolite.cmd.png(f"{output_path}_detailed.png", width=500, height=330, dpi=100, ray=0)


def create_hbonds_visualization(
    structure: struc.AtomArray,
    stats: dict[str, Any],
    binder_chain: str,
    target_chain: str,
    output_path: str,
    high_res: bool = False,
) -> None:
    """Create hydrogen bonds focused visualization."""

    ammolite.reset()
    pymol_obj = ammolite.PyMOLObject.from_structure(structure)

    # Setup clean visualization
    setup_clean_visualization(pymol_obj)

    # Color chains (binder vs target)
    binder_mask = structure.chain_id == binder_chain
    target_mask = structure.chain_id == target_chain

    pymol_obj.color("salmon", binder_mask)
    pymol_obj.color("lightblue", target_mask)

    # Show only interface residues and hydrogen bonds
    interface_mask = np.zeros(len(structure), dtype=bool)
    for chain_id, res_id in stats["binder_interface_identifiers"] + stats["target_interface_identifiers"]:
        interface_mask |= (structure.chain_id == chain_id) & (structure.res_id == res_id)

    # Make non-interface regions more transparent
    pymol_obj.set("cartoon_transparency", 0.8, ~interface_mask)

    # Show elegant side chains for interface residues
    pymol_obj.show("sticks", interface_mask & (structure.atom_name != "CA"))
    pymol_obj.set("stick_radius", 0.3, interface_mask & (structure.atom_name != "CA"))
    pymol_obj.color("yellow", interface_mask & (structure.atom_name != "CA"))

    # Show hydrogen bonds with elegant styling
    if stats["hydrogen_bonds"]:
        for i, (donor, hydrogen, acceptor, distance) in enumerate(stats["hydrogen_bonds"]):
            # Show hydrogen bond atoms as spheres
            pymol_obj.show("spheres", [donor, hydrogen, acceptor])
            pymol_obj.set("sphere_scale", 0.4, [donor, hydrogen, acceptor])
            pymol_obj.color("red", [donor, hydrogen, acceptor])

            # Create elegant hydrogen bond line
            pymol_obj.distance(f"hbond_{i}", hydrogen, acceptor, show_label=True)
            ammolite.cmd.set("dash_gap", 0.0, f"hbond_{i}")
            ammolite.cmd.set("dash_width", 3.0, f"hbond_{i}")
            ammolite.cmd.set("dash_color", "red", f"hbond_{i}")

            # Add distance labels
            ammolite.cmd.set("label_size", 14, f"hbond_{i}")
            ammolite.cmd.set("label_color", "black", f"hbond_{i}")
    else:
        logger.info("No hydrogen bonds detected - this may be due to missing hydrogen atoms or distance threshold")

    # Zoom to interface with better angle
    pymol_obj.zoom(interface_mask, buffer=3.0)

    # Rotate to get a better view of the interface
    ammolite.cmd.rotate("x", 25)
    ammolite.cmd.rotate("y", 40)

    # Orient to focus on the interface
    ammolite.cmd.orient()

    # Set label properties
    ammolite.cmd.set("label_color", "black")
    ammolite.cmd.set("label_size", 16)

    # Save image (inherited settings from visual_utils.py)
    if high_res:
        apply_highres_settings()
        ammolite.cmd.png(f"{output_path}_hbonds.png", width=3000, height=2000, dpi=300, ray=1)
    else:
        ammolite.cmd.png(f"{output_path}_hbonds.png", width=500, height=330, dpi=100, ray=0)


def create_publication_style_visualization(
    structure: struc.AtomArray,
    stats: dict[str, Any],
    binder_chain: str,
    target_chain: str,
    output_path: str,
    high_res: bool = False,
) -> None:
    """Create publication-style interface visualization mimicking professional figures.

    This follows the hybrid approach used in structural biology publications:
    - Cartoon for overall protein structure (orange/teal)
    - Sticks for interface residues with atom-type coloring
    - Blue dashed lines for hydrogen bonds
    - Focused zoom on interface
    """

    ammolite.reset()
    pymol_obj = ammolite.PyMOLObject.from_structure(structure)

    # Setup clean visualization
    setup_clean_visualization(pymol_obj)

    # Color chains (binder vs target)
    binder_mask = structure.chain_id == binder_chain
    target_mask = structure.chain_id == target_chain

    pymol_obj.color("salmon", binder_mask)
    pymol_obj.color("lightblue", target_mask)

    # Get interface residue identifiers
    binder_interface_identifiers = stats["binder_interface_identifiers"]
    target_interface_identifiers = stats["target_interface_identifiers"]

    # Create interface mask
    interface_mask = np.zeros(len(structure), dtype=bool)
    for chain_id, res_id in binder_interface_identifiers + target_interface_identifiers:
        interface_mask |= (structure.chain_id == chain_id) & (structure.res_id == res_id)

    # Publication-style interface visualization mimicking the reference figure

    # 1. Show sticks for interface residues (excluding CA atoms)
    pymol_obj.show("sticks", interface_mask & (structure.atom_name != "CA"))
    pymol_obj.set("stick_radius", 0.2, interface_mask & (structure.atom_name != "CA"))

    # 2. Apply atom-type coloring like the reference figure
    # Carbon atoms = tan/brown
    carbon_mask = interface_mask & (structure.atom_name != "CA") & (structure.element == "C")
    pymol_obj.color("brown", carbon_mask)

    # Oxygen atoms = red
    oxygen_mask = interface_mask & (structure.atom_name != "CA") & (structure.element == "O")
    pymol_obj.color("red", oxygen_mask)

    # Nitrogen atoms = blue
    nitrogen_mask = interface_mask & (structure.atom_name != "CA") & (structure.element == "N")
    pymol_obj.color("blue", nitrogen_mask)

    # Sulfur atoms = yellow
    sulfur_mask = interface_mask & (structure.atom_name != "CA") & (structure.element == "S")
    pymol_obj.color("yellow", sulfur_mask)

    # 3. Show hydrogen bonds with blue dashed lines and small spheres
    if stats["hydrogen_bonds"]:
        for i, (donor, hydrogen, acceptor, distance) in enumerate(stats["hydrogen_bonds"]):
            # Small spheres at hydrogen bond endpoints
            pymol_obj.show("spheres", [donor, hydrogen, acceptor])
            pymol_obj.set("sphere_scale", 0.3, [donor, hydrogen, acceptor])

            # Color spheres by atom type
            if structure.element[donor] == "O":
                pymol_obj.color("red", [donor])
            elif structure.element[donor] == "N":
                pymol_obj.color("blue", [donor])

            if structure.element[acceptor] == "O":
                pymol_obj.color("red", [acceptor])
            elif structure.element[acceptor] == "N":
                pymol_obj.color("blue", [acceptor])

            # Blue dashed lines for hydrogen bonds
            pymol_obj.distance(f"hbond_{i}", hydrogen, acceptor, show_label=False)
            ammolite.cmd.set("dash_gap", 0.0, f"hbond_{i}")
            ammolite.cmd.set("dash_width", 2.0, f"hbond_{i}")
            ammolite.cmd.set("dash_color", "blue", f"hbond_{i}")
    else:
        logger.info("No hydrogen bonds detected between binder and target")

    # Zoom to interface with better angle
    pymol_obj.zoom(interface_mask, buffer=5.0)

    # Rotate to get a better view of the interface
    ammolite.cmd.rotate("x", 30)
    ammolite.cmd.rotate("y", 45)

    # Orient to focus on the interface
    ammolite.cmd.orient()

    # Set label properties
    ammolite.cmd.set("label_color", "black")
    ammolite.cmd.set("label_size", 20)

    # Save image (inherited settings from visual_utils.py)
    if high_res:
        apply_highres_settings()
        ammolite.cmd.png(f"{output_path}_interface.png", width=3000, height=2000, dpi=300, ray=1)
    else:
        ammolite.cmd.png(f"{output_path}_interface.png", width=500, height=330, dpi=100, ray=0)


def visualize_interface(
    pdb_path: str,
    output_dir: str,
    modes: list[str] = ["detailed"],
    distance_threshold: float = 8.0,
    hbond_distance: float = 3.5,
    high_res: bool = False,
    save_session: bool = False,
    analysis_only: bool = False,
    chain_binder: str | None = None,
    chain_target: str | None = None,
) -> dict[str, Any]:
    """Interface visualization with multiple modes.

    Args:
        pdb_path: Path to PDB file
        output_dir: Output directory for images
        modes: List of visualization modes
        distance_threshold: Distance threshold for interface definition
        hbond_distance: Distance threshold for hydrogen bonds
        high_res: Generate high resolution images
        save_session: Save PyMOL session files
        analysis_only: Only compute statistics without visualization
        chain_binder: Optional binder chain ID
        chain_target: Optional target chain ID

    Returns:
        Dictionary with interface statistics
    """

    logger.info(f"Visualizing interface for {pdb_path}")

    # Load structure
    structure = load_structure(pdb_path)

    # Identify chains
    binder_chain, target_chain = identify_chains(structure, chain_binder, chain_target)

    # Compute interface analysis
    stats = compute_interface_analysis(structure, binder_chain, target_chain, distance_threshold, hbond_distance)

    # Add file information
    stats["pdb_file"] = pdb_path
    stats["binder_chain"] = binder_chain
    stats["target_chain"] = target_chain

    # Create visualizations if not analysis_only
    if not analysis_only:
        pdb_name = Path(pdb_path).stem
        output_path = Path(output_dir) / pdb_name

        for mode in modes:
            if mode == "overview":
                create_overview_visualization(
                    structure,
                    stats,
                    binder_chain,
                    target_chain,
                    str(output_path),
                    high_res,
                )
            elif mode == "detailed":
                create_detailed_visualization(
                    structure,
                    stats,
                    binder_chain,
                    target_chain,
                    str(output_path),
                    high_res,
                )
            elif mode == "hbonds":
                create_hbonds_visualization(
                    structure,
                    stats,
                    binder_chain,
                    target_chain,
                    str(output_path),
                    high_res,
                )
            elif mode == "basic":
                create_basic_visualization(
                    structure,
                    stats,
                    binder_chain,
                    target_chain,
                    str(output_path),
                    high_res,
                )
            elif mode == "publication":
                create_publication_style_visualization(
                    structure,
                    stats,
                    binder_chain,
                    target_chain,
                    str(output_path),
                    high_res,
                )

        # Save PyMOL session if requested
        if save_session:
            ammolite.cmd.save(f"{output_path}_session.pse")

    logger.info(f"Interface statistics: {stats}")
    return stats


def visualize_multiple_pdbs(
    pdb_dir: str,
    output_dir: str,
    modes: list[str] = ["detailed"],
    distance_threshold: float = 8.0,
    hbond_distance: float = 3.5,
    high_res: bool = False,
    save_session: bool = False,
    analysis_only: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Visualize multiple PDB files with progress tracking."""

    pdb_files = list(Path(pdb_dir).glob("*.pdb")) + list(Path(pdb_dir).glob("*.mmtf"))

    if limit:
        pdb_files = pdb_files[:limit]

    all_stats = []

    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        try:
            stats = visualize_interface(
                str(pdb_file),
                output_dir,
                modes,
                distance_threshold,
                hbond_distance,
                high_res,
                save_session,
                analysis_only,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Failed to visualize {pdb_file}: {e}")
            continue

    return all_stats


def save_interface_statistics(stats_list: list[dict[str, Any]], output_dir: str) -> None:
    """Save interface statistics to CSV file."""

    # Flatten statistics for CSV
    flattened_stats = []
    for stats in stats_list:
        flat_stats = {
            "pdb_file": stats["pdb_file"],
            "binder_chain": stats["binder_chain"],
            "target_chain": stats["target_chain"],
            "binder_interface_residues": stats["binder_interface_residues"],
            "target_interface_residues": stats["target_interface_residues"],
            "total_interface_residues": stats["total_interface_residues"],
            "hydrogen_bonds": stats["hydrogen_bonds"],
            "min_distance": stats["min_distance"],
            "max_distance": stats["max_distance"],
            "avg_distance": stats["avg_distance"],
            "interface_min_distance": stats["interface_min_distance"],
            "interface_max_distance": stats["interface_max_distance"],
            "interface_avg_distance": stats["interface_avg_distance"],
            "distance_threshold": stats["distance_threshold"],
            "hbond_distance": stats["hbond_distance"],
        }

        # Add polarity counts
        for polarity in ["A", "P", "C"]:
            flat_stats[f"binder_{polarity}_count"] = stats["binder_polarity_counts"].get(polarity, 0)
            flat_stats[f"target_{polarity}_count"] = stats["target_polarity_counts"].get(polarity, 0)

        flattened_stats.append(flat_stats)

    # Create DataFrame and save
    df = pd.DataFrame(flattened_stats)
    output_path = Path(output_dir) / "interface_statistics.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Interface statistics saved to {output_path}")


def download_sample_structure(pdb_id: str = "2RTG") -> str:
    """Download a sample protein structure from PDB.

    Args:
        pdb_id: PDB ID to download (default: 2RTG - streptavidin-biotin complex)

    Returns:
        Path to downloaded PDB file
    """
    logger.info(f"Downloading structure {pdb_id} from PDB")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    pdb_path = os.path.join(temp_dir, f"{pdb_id}.pdb")

    # Download structure
    pdb_file = rcsb.fetch(pdb_id, "pdb")
    structure = pdb.PDBFile.read(pdb_file)

    # Save to temporary file
    pdb.PDBFile.write(pdb_path, structure)

    logger.info(f"Structure saved to {pdb_path}")
    return pdb_path


def create_synthetic_complex() -> str:
    """Create a synthetic protein complex for demonstration.

    Returns:
        Path to synthetic PDB file
    """
    logger.info("Creating synthetic protein complex")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    pdb_path = os.path.join(temp_dir, "synthetic_complex.pdb")

    # Create a simple two-chain structure
    # Chain A: Small protein (binder)
    # Chain B: Larger protein (target)

    # Generate coordinates for chain A (binder)
    n_binder = 50
    binder_coords = np.random.rand(n_binder, 3) * 20  # 20Å cube
    binder_coords[:, 0] += 30  # Offset in x direction

    # Generate coordinates for chain B (target)
    n_target = 100
    target_coords = np.random.rand(n_target, 3) * 30  # 30Å cube
    target_coords[:, 0] -= 10  # Offset in x direction

    # Create atom arrays
    binder_atoms = []
    target_atoms = []

    # Add CA atoms for binder
    for i in range(n_binder):
        atom = struc.Atom(
            coord=binder_coords[i],
            atom_name="CA",
            res_name="ALA",
            res_id=i + 1,
            chain_id="A",
        )
        binder_atoms.append(atom)

    # Add CA atoms for target
    for i in range(n_target):
        atom = struc.Atom(
            coord=target_coords[i],
            atom_name="CA",
            res_name="GLY",
            res_id=i + 1,
            chain_id="B",
        )
        target_atoms.append(atom)

    # Combine into structure
    structure = struc.AtomArray(binder_atoms + target_atoms)

    # Save to file
    pdb.PDBFile.write(pdb_path, structure)

    logger.info(f"Synthetic complex saved to {pdb_path}")
    return pdb_path


def run_examples():
    """Run demonstration examples."""
    logger.info("=== Running Interface Visualization Examples ===")

    # Example 1: Basic visualization
    logger.info("Example 1: Basic Interface Visualization")
    pdb_path = download_sample_structure("2RTG")
    try:
        output_dir = "example_visualizations"
        os.makedirs(output_dir, exist_ok=True)

        stats = visualize_interface(
            pdb_path=pdb_path,
            output_dir=output_dir,
            modes=["basic"],
            distance_threshold=8.0,
            high_res=False,
            save_session=True,
        )
        logger.info(f"Basic visualization completed. Statistics: {stats}")
    finally:
        os.remove(pdb_path)
        os.rmdir(os.path.dirname(pdb_path))

    # Example 2: Advanced visualization
    logger.info("Example 2: Advanced Interface Visualization")
    pdb_path = download_sample_structure("2RTG")
    try:
        output_dir = "example_advanced_visualizations"
        os.makedirs(output_dir, exist_ok=True)

        stats = visualize_interface(
            pdb_path=pdb_path,
            output_dir=output_dir,
            modes=["overview", "detailed", "hbonds"],
            distance_threshold=8.0,
            hbond_distance=3.5,
            high_res=False,
            save_session=True,
        )
        logger.info(f"Advanced visualization completed. Statistics: {stats}")
    finally:
        os.remove(pdb_path)
        os.rmdir(os.path.dirname(pdb_path))

    # Example 3: Interface analysis only
    logger.info("Example 3: Interface Analysis Only")
    pdb_path = download_sample_structure("2RTG")
    try:
        structure = load_structure(pdb_path)
        binder_chain, target_chain = identify_chains(structure)
        stats = compute_interface_analysis(structure, binder_chain, target_chain)

        logger.info("Interface Analysis Results:")
        logger.info(f"  Binder interface residues: {stats['binder_interface_residues']}")
        logger.info(f"  Target interface residues: {stats['target_interface_residues']}")
        logger.info(f"  Total interface residues: {stats['total_interface_residues']}")
        logger.info(f"  Hydrogen bonds: {stats['hydrogen_bonds']}")
        logger.info(f"  Average interface distance: {stats['interface_avg_distance']:.2f} Å")
        logger.info(f"  Binder polarity counts: {stats['binder_polarity_counts']}")
        logger.info(f"  Target polarity counts: {stats['target_polarity_counts']}")
    finally:
        os.remove(pdb_path)
        os.rmdir(os.path.dirname(pdb_path))

    # Example 4: Synthetic complex
    logger.info("Example 4: Synthetic Complex Visualization")
    pdb_path = create_synthetic_complex()
    try:
        output_dir = "example_synthetic_visualizations"
        os.makedirs(output_dir, exist_ok=True)

        stats = visualize_interface(
            pdb_path=pdb_path,
            output_dir=output_dir,
            modes=["overview", "detailed"],
            distance_threshold=10.0,  # Larger threshold for synthetic data
            hbond_distance=3.5,
            high_res=False,
            save_session=True,
        )
        logger.info(f"Synthetic complex visualization completed. Statistics: {stats}")
    finally:
        os.remove(pdb_path)
        os.rmdir(os.path.dirname(pdb_path))

    logger.info("All examples completed!")
    logger.info("Check the 'example_*_visualizations' directories for output images")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Protein interface visualization")
    parser.add_argument("--pdb_file", type=str, help="Path to single PDB file")
    parser.add_argument("--pdb_dir", type=str, help="Directory containing PDB files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="interface_visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="detailed",
        choices=["overview", "detailed", "hbonds", "basic"],
        help="Visualization mode",
    )
    parser.add_argument("--modes", type=str, help="Multiple visualization modes (comma-separated)")
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=8.0,
        help="Distance threshold for interface definition (Angstroms)",
    )
    parser.add_argument(
        "--hbond_distance",
        type=float,
        default=3.5,
        help="Distance threshold for hydrogen bonds (Angstroms)",
    )
    parser.add_argument("--high_res", action="store_true", help="Generate high resolution images")
    parser.add_argument("--save_session", action="store_true", help="Save PyMOL session files")
    parser.add_argument("--save_stats", action="store_true", help="Save interface statistics to CSV")
    parser.add_argument(
        "--analysis_only",
        action="store_true",
        help="Only compute and save statistics without visualization",
    )
    parser.add_argument("--chain_binder", type=str, help="Chain ID for binder protein")
    parser.add_argument("--chain_target", type=str, help="Chain ID for target protein")
    parser.add_argument("--limit", type=int, help="Maximum number of files to process")
    parser.add_argument("--run_examples", action="store_true", help="Run demonstration examples")

    args = parser.parse_args()

    # Run examples if requested
    if args.run_examples:
        run_examples()
        return

    # Validate arguments
    if not args.pdb_file and not args.pdb_dir:
        parser.error("Either --pdb_file, --pdb_dir, or --run_examples must be specified")

    # Parse modes
    if args.modes:
        modes = [mode.strip() for mode in args.modes.split(",")]
    else:
        modes = [args.mode]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process files
    if args.pdb_file:
        if not os.path.exists(args.pdb_file):
            logger.error(f"PDB file not found: {args.pdb_file}")
            return

        stats = visualize_interface(
            args.pdb_file,
            args.output_dir,
            modes,
            args.distance_threshold,
            args.hbond_distance,
            args.high_res,
            args.save_session,
            args.analysis_only,
            args.chain_binder,
            args.chain_target,
        )
        logger.info(f"Visualization completed for {args.pdb_file}")

    elif args.pdb_dir:
        if not os.path.exists(args.pdb_dir):
            logger.error(f"PDB directory not found: {args.pdb_dir}")
            return

        all_stats = visualize_multiple_pdbs(
            args.pdb_dir,
            args.output_dir,
            modes,
            args.distance_threshold,
            args.hbond_distance,
            args.high_res,
            args.save_session,
            args.analysis_only,
            args.limit,
        )
        logger.info(f"Visualization completed for {len(all_stats)} files")

        # Save statistics if requested
        if args.save_stats and all_stats:
            save_interface_statistics(all_stats, args.output_dir)


if __name__ == "__main__":
    main()
