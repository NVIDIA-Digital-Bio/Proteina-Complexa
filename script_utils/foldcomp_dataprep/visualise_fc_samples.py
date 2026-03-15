import argparse
import os
import tempfile

import ammolite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import foldcomp
import numpy as np


def convert_ss_annotations(ss_array: np.ndarray) -> np.ndarray:
    """Convert secondary structure annotations from Biotite to PyMOL format.

    Args:
        ss_array: Array of secondary structure annotations from Biotite.

    Returns:
        Array of secondary structure annotations in PyMOL format.
    """
    ss_mapping = {"c": "L", "a": "H", "b": "S"}
    return np.vectorize(ss_mapping.get)(ss_array)


def superimpose(vec1: np.ndarray, vec2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the rotation matrix and translation vector to superimpose vec2 onto vec1.

    Args:
        vec1: Reference vector of shape (n, 3).
        vec2: Vector to be superimposed of shape (n, 3).

    Returns:
        Tuple containing the rotation matrix and translation vector.
    """
    assert vec1.shape == vec2.shape
    n = vec1.shape[0]  # total points

    centroid1 = np.mean(vec1, axis=0)
    centroid2 = np.mean(vec2, axis=0)

    # Center the points
    vec1_centered = vec1 - centroid1
    vec2_centered = vec2 - centroid2

    # Singular Value Decomposition
    H = vec1_centered.T @ vec2_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid2 - centroid1 @ R
    return R, t


def apply_publication_settings() -> None:
    """Apply settings suitable for publication."""
    ammolite.cmd.set("cartoon_discrete_colors", 1)
    ammolite.cmd.set("cartoon_oval_length", 1.0)
    ammolite.cmd.set("ray_trace_mode", 1)
    ammolite.cmd.set("ambient", 0.1)


def visualise_structure(ca: struc.AtomArray, show_secondary_structure: bool = True) -> ammolite.PyMOLObject:
    """Visualize a protein structure with optional secondary structure coloring.

    Args:
        ca: C-alpha atom array of the protein structure.
        show_secondary_structure: Whether to color by secondary structure.

    Returns:
        PyMOL object representing the visualized structure.
    """
    pymol_object = ammolite.PyMOLObject.from_structure(ca)

    if show_secondary_structure:
        sse = convert_ss_annotations(struc.annotate_sse(ca, "A"))

        # Color each secondary structure element
        selections = {"L": [], "H": [], "S": []}
        for resi, ss in zip(ca.res_id, sse, strict=False):
            selections[ss].append(resi)

        colors = {"L": "white", "H": "salmon", "S": "lightblue"}
        for ss, color in colors.items():
            selection_string = f"resi {'+'.join(map(str, selections[ss]))} and model {pymol_object.name}"
            ammolite.cmd.color(color, selection_string)

        # Show secondary structure as cartoon
        for resi, ss in zip(ca.res_id, sse, strict=False):
            ammolite.cmd.alter(f"resi {resi}", f'ss="{ss}"')

    return pymol_object


def add_spheres_at_residues(pymol_object: ammolite.PyMOLObject, res_ids: np.ndarray) -> str:
    """Add spheres at specified residue positions.

    Args:
        pymol_object: PyMOL object representing the protein structure.
        res_ids: Array of residue IDs to add spheres at.

    Returns:
        Selection string for the added spheres.
    """
    selection_string = f"resi {'+'.join(map(str, res_ids))} and name CA and model {pymol_object.name}"
    ammolite.cmd.show("spheres", selection_string)
    ammolite.cmd.set("sphere_scale", 0.2, selection_string)
    return selection_string


def visualise_ca_coord_numpy(
    coords: np.ndarray,
    cond_res: np.ndarray = None,
    zoom_out: bool = True,
    save_path: str = None,
    save_for_publication: bool = False,
) -> ammolite.PyMOLObject:
    """Visualize a protein structure with optional annotations and save the image.

    Args:
        coords: Coordinates of the backbone CA atoms of shape (n, 3).
        cond_res: Indices of the conditioned residues to annotate.
        zoom_out: Whether to zoom out to show the full structure.
        save_path: Path to save the image. If None, the image is not saved.
        save_for_publication: Whether to apply publication settings to the image.

    Returns:
        PyMOL object representing the visualized structure.
    """
    ammolite.reset()
    ca = struc.from_template(struc.AtomArray(coords))
    pymol_object = visualise_structure(ca)

    if cond_res is not None:
        selection = add_spheres_at_residues(pymol_object, cond_res + 1)  # +1 for 1-indexing in Biotite
        ammolite.cmd.color("purple", selection)
        ammolite.cmd.zoom(f"resi {max(min(cond_res) - 2, 0)}-{min(max(cond_res) + 2, len(ca))}")

    if zoom_out:
        ammolite.cmd.zoom("all")

    ammolite.cmd.set("cartoon_transparency", 0.6, pymol_object.name)

    if save_path:
        if save_for_publication:
            apply_publication_settings()
            ammolite.cmd.png(str(save_path), width=3000, height=2000, dpi=300, ray=1)
        else:
            ammolite.cmd.png(str(save_path), width=500, height=330, dpi=100, ray=0)
        return save_path
    else:
        return pymol_object


def visualise_ca_coord_pdb(
    pdb_file: str,
    cond_res: np.ndarray | None = None,
    zoom_out: bool = True,
    save_path: str | None = None,
    save_for_publication: bool = False,
) -> ammolite.PyMOLObject:
    """Visualize a protein structure from a PDB file containing only C-alpha coordinates.

    Args:
        pdb_file: Path to the PDB file containing only C-alpha coordinates.
        cond_res: Indices of the conditioned residues to annotate.
        zoom_out: Whether to zoom out to show the full structure.
        save_path: Path to save the image. If None, the image is not saved.
        save_for_publication: Whether to apply publication settings to the image.

    Returns:
        PyMOL object representing the visualized structure.
    """
    ammolite.reset()

    # Load the PDB file containing only C-alpha coordinates
    pdb_file = pdb.PDBFile.read(pdb_file)
    ca = pdb_file.get_structure(model=1)

    # Ensure that the structure contains only C-alpha atoms
    ca = ca[ca.atom_name == "CA"]

    pymol_object = visualise_structure(ca)

    if cond_res is not None:
        selection = add_spheres_at_residues(pymol_object, cond_res + 1)  # +1 for 1-indexing in Biotite
        ammolite.cmd.color("purple", selection)
        ammolite.cmd.zoom(f"resi {max(min(cond_res) - 2, 0)}-{min(max(cond_res) + 2, len(ca))}")

    if zoom_out:
        ammolite.cmd.zoom("all")

    ammolite.cmd.set("cartoon_transparency", 0.6, pymol_object.name)

    if save_path:
        if save_for_publication:
            apply_publication_settings()
            ammolite.cmd.png(str(save_path), width=3000, height=2000, dpi=300, ray=1)
        else:
            ammolite.cmd.png(str(save_path), width=500, height=330, dpi=100, ray=0)
        return save_path
    else:
        return pymol_object


def process_and_write_chunk(db_path, output_dir, ids):
    db = foldcomp.open(db_path, ids=ids)

    for name, content in db:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb") as temp_file:
            temp_file.write(content)
            save_path = os.path.join(output_dir, f"{name}")
            visual_save_path = visualise_ca_coord_pdb(temp_file.name, save_path=save_path, save_for_publication=True)
    db.close()


def main(db_dir, db_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ids = [
        "AF-A0A421CQ62-F1-model_v4.cif.gz",
        "AF-A0A3B9SQX8-F1-model_v4.cif.gz",
        "AF-A0A226QHN8-F1-model_v4.cif.gz",
        "AF-A0A226QNR8-F1-model_v4.cif.gz",
        "AF-A0A1C5AD19-F1-model_v4.cif.gz",
        "AF-V6J0J8-F1-model_v4.cif.gz",
        "AF-A0A7K4RF77-F1-model_v4.cif.gz",
        "AF-A0A6L2P900-F1-model_v4.cif.gz",
        "AF-A0A7K6NC84-F1-model_v4.cif.gz",
        "AF-A0A7K6ND65-F1-model_v4.cif.gz",
        "AF-A0A7K6NRA9-F1-model_v4.cif.gz",
        "AF-A0A7K6NTL2-F1-model_v4.cif.gz",
        "AF-Q73WI4-F1-model_v4.cif.gz",
        "AF-A0A1V2PBD4-F1-model_v4.cif.gz",
        "AF-A0A1I6I3I6-F1-model_v4.cif.gz",
        "AF-A0A3M1P2M3-F1-model_v4.cif.gz",
        "AF-A0A1U7TC47-F1-model_v4.cif.gz",
        "AF-A0A0Q4UD46-F1-model_v4.cif.gz",
        "AF-A0A7C3MH68-F1-model_v4.cif.gz",
        "AF-A0A4P6ELW4-F1-model_v4.cif.gz",
        "AF-A0A4P6ENW0-F1-model_v4.cif.gz",
        "AF-A0A7L4MUH8-F1-model_v4.cif.gz",
        "AF-A0A7L4N228-F1-model_v4.cif.gz",
        "AF-A0A7L4N528-F1-model_v4.cif.gz",
        "AF-A0A7L4N5L9-F1-model_v4.cif.gz",
        "AF-A0A7L4NAL3-F1-model_v4.cif.gz",
        "AF-A0A7L4NJL7-F1-model_v4.cif.gz",
        "AF-A0A820S1N3-F1-model_v4.cif.gz",
        "AF-A0A820S2J2-F1-model_v4.cif.gz",
        "AF-A0A820SBL2-F1-model_v4.cif.gz",
        "AF-A0A820TKD6-F1-model_v4.cif.gz",
        "AF-A0A4Q1U1Z8-F1-model_v4.cif.gz",
        "AF-A0A6P4W6H2-F1-model_v4.cif.gz",
        "AF-A0A6P4WE54-F1-model_v4.cif.gz",
        "AF-A0A0K8TS14-F1-model_v4.cif.gz",
        "AF-A0A0R1QQE8-F1-model_v4.cif.gz",
        "AF-R1DJA6-F1-model_v4.cif.gz",
        "AF-R1DJB1-F1-model_v4.cif.gz",
    ]
    ids = [entry.rstrip(".cif.gz") for entry in ids]
    process_and_write_chunk(f"{db_dir}/{db_name}", output_dir, ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB files and write metrics to Parquet.")
    parser.add_argument("--db_dir", type=str, required=True, help="Path to the database directory")
    parser.add_argument("--db_name", type=str, required=True, help="Database File Name")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Parquet files",
    )

    args = parser.parse_args()
    main(args.db_dir, args.db_name, args.output_dir)
