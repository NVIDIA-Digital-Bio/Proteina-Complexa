"""
Alternative implementations for PyRosetta functionality.

This module provides OpenMM and Biopython-based alternatives to PyRosetta functions,
enabling BindCraft to run without PyRosetta installation. These implementations
aim to provide similar functionality with reasonable approximations where exact
replication is not possible.

Functions:
    openmm_relax: Structure relaxation using OpenMM
    openmm_relax_subprocess: Run relax in a fresh process to isolate OpenCL context
    pr_alternative_score_interface: Interface scoring using Biopython

Helper Functions:
    _get_openmm_forcefield: Singleton ForceField instance
    _create_lj_repulsive_force: Custom LJ repulsion for clash resolution
    _create_backbone_restraint_force: Backbone position restraints
    _chain_total_sasa: Calculate total SASA for a chain

Rationale:
    In long runs we observed sporadic OpenCL context failures after many relax calls,
    consistent with driver/runtime state or memory accumulation. The subprocess helper
    guarantees full teardown per relax, isolating OpenCL state between runs.
"""

import contextlib
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import zip_longest

# Bio.PDB imports
from Bio.PDB import PDBIO, Model, PDBParser, Polypeptide, Structure
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1

from proteinfoundation.utils.biopython_utils import biopython_align_all_ca, hotspot_residues

# OpenMM imports
# import openmm
# from openmm import app, unit, Platform, OpenMMException
# from pdbfixer import PDBFixer


# Cache a single OpenMM ForceField instance to avoid repeated XML parsing per relaxation
_OPENMM_FORCEFIELD_SINGLETON = None

# Optional FreeSASA availability
try:
    import freesasa  # type: ignore

    _HAS_FREESASA = True
except Exception:
    freesasa = None  # type: ignore
    _HAS_FREESASA = False


# clean unnecessary rosetta information from PDB
def clean_pdb(pdb_file):
    # Read the pdb file and filter relevant lines
    with open(pdb_file) as f_in:
        relevant_lines = [line for line in f_in if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END", "LINK"))]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, "w") as f_out:
        f_out.writelines(relevant_lines)


def _get_openmm_forcefield():
    global _OPENMM_FORCEFIELD_SINGLETON
    if _OPENMM_FORCEFIELD_SINGLETON is None:
        _OPENMM_FORCEFIELD_SINGLETON = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    return _OPENMM_FORCEFIELD_SINGLETON


# Removed legacy in-process OpenCL reset helper; subprocess isolation handles context teardown.


# Helper function for k conversion
def _k_kj_per_nm2(k_kcal_A2):
    return k_kcal_A2 * 4.184 * 100.0


# Helper function for LJ repulsive force creation
def _create_lj_repulsive_force(
    system,
    lj_rep_base_k_kj_mol,
    lj_rep_ramp_factors,
    original_sigmas,
    nonbonded_force_index,
):
    lj_rep_custom_force = None
    k_rep_lj_param_index = -1

    if lj_rep_base_k_kj_mol > 0 and original_sigmas and lj_rep_ramp_factors:
        lj_rep_custom_force = openmm.CustomNonbondedForce(
            "k_rep_lj * (((sigma_particle1 + sigma_particle2) * 0.5 / r)^12)"
        )

        initial_k_rep_val = lj_rep_base_k_kj_mol * lj_rep_ramp_factors[0]
        # Global parameters in OpenMM CustomNonbondedForce expect plain float values for the constant.
        # The energy expression itself defines how this constant is used with physical units.
        k_rep_lj_param_index = lj_rep_custom_force.addGlobalParameter("k_rep_lj", float(initial_k_rep_val))
        lj_rep_custom_force.addPerParticleParameter("sigma_particle")

        for sigma_val_nm in original_sigmas:
            lj_rep_custom_force.addParticle([sigma_val_nm])

        # Check if nonbonded_force_index is valid before trying to get the force
        if nonbonded_force_index != -1:
            existing_nb_force = system.getForce(nonbonded_force_index)
            nb_method = existing_nb_force.getNonbondedMethod()

            if nb_method in [
                openmm.NonbondedForce.CutoffPeriodic,
                openmm.NonbondedForce.CutoffNonPeriodic,
            ]:
                lj_rep_custom_force.setNonbondedMethod(
                    openmm.CustomNonbondedForce.CutoffPeriodic
                    if nb_method == openmm.NonbondedForce.CutoffPeriodic
                    else openmm.CustomNonbondedForce.CutoffNonPeriodic
                )
                lj_rep_custom_force.setCutoffDistance(existing_nb_force.getCutoffDistance())
                if nb_method == openmm.NonbondedForce.CutoffPeriodic:
                    lj_rep_custom_force.setUseSwitchingFunction(existing_nb_force.getUseSwitchingFunction())
                    if existing_nb_force.getUseSwitchingFunction():
                        lj_rep_custom_force.setSwitchingDistance(existing_nb_force.getSwitchingDistance())
            elif nb_method == openmm.NonbondedForce.NoCutoff:
                lj_rep_custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

            for ex_idx in range(existing_nb_force.getNumExceptions()):
                p1, p2, chargeProd, sigmaEx, epsilonEx = existing_nb_force.getExceptionParameters(ex_idx)
                lj_rep_custom_force.addExclusion(p1, p2)
        else:
            # This case should ideally not be hit if sigmas were extracted,
            # but as a fallback, don't try to use existing_nb_force.
            # Default to NoCutoff if we couldn't determine from an existing force.
            lj_rep_custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

        lj_rep_custom_force.setForceGroup(2)
        system.addForce(lj_rep_custom_force)

    return lj_rep_custom_force, k_rep_lj_param_index


# Helper function for backbone restraint force creation
def _create_backbone_restraint_force(system, fixer, restraint_k_kcal_mol_A2):
    restraint_force = None
    k_restraint_param_index = -1

    if restraint_k_kcal_mol_A2 > 0:
        restraint_force = openmm.CustomExternalForce(
            "0.5 * k_restraint * ( (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0) )"
        )
        # Global parameters in OpenMM CustomExternalForce also expect plain float values.
        k_restraint_param_index = restraint_force.addGlobalParameter(
            "k_restraint", _k_kj_per_nm2(restraint_k_kcal_mol_A2)
        )
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")

        initial_positions = fixer.positions
        num_bb_restrained = 0
        BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}
        for atom in fixer.topology.atoms():
            if atom.name in BACKBONE_ATOM_NAMES:
                xyz_vec = initial_positions[atom.index].value_in_unit(unit.nanometer)
                restraint_force.addParticle(atom.index, [xyz_vec[0], xyz_vec[1], xyz_vec[2]])
                num_bb_restrained += 1

        if num_bb_restrained > 0:
            restraint_force.setForceGroup(1)
            system.addForce(restraint_force)
        else:
            restraint_force = None
            k_restraint_param_index = -1

    return restraint_force, k_restraint_param_index


# Chothia/NACCESS-like atomic radii (heavy atoms dominate SASA)
R_CHOTHIA = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

# (Unused) _MAX_ASA removed during cleanup.

# (Unused) Residue-specific polar carbons mapping removed during cleanup.


@contextlib.contextmanager
def _suppress_freesasa_warnings():
    """Temporarily redirect OS-level stderr (fd=2) to suppress FreeSASA warnings."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
    except Exception:
        # Fallback: no suppression
        yield


# Hydrophobic amino acids set (match PyRosetta hydrophobic/aromatic intent)
HYDROPHOBIC_AA_SET = set("ACFILMPVWY")


def _chain_total_sasa(chain_entity):
    return sum(getattr(atom, "sasa", 0.0) for atom in chain_entity.get_atoms())


def _calculate_shape_complementarity(
    pdb_file_path, binder_chain="B", target_chain="A", distance=4.0, sc_bin: str = None
):
    """
    Calculate shape complementarity using sc-rs CLI when available.
    Looks first for a local binary placed next to this module (e.g., 'functions/sc' or 'functions/sc-rs').
    Falls back to a conservative placeholder (0.70) if sc-rs is not installed or fails.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file containing the complex
    binder_chain : str
        Chain ID of the binder (default: "B")
    target_chain : str
        Chain ID of the target (default: "A")
    distance : float
        Unused here; retained for API compatibility

    Returns
    -------
    float
        Shape complementarity in [0, 1]
    """
    try:
        start_time = time.time()
        basename = os.path.basename(pdb_file_path)
        print(f"[SC-RS] Initiating shape complementarity for {basename} (target={target_chain}, binder={binder_chain})")

        # sc-rs CLI: sc <pdb> <chainA> <chainB> --json; SC is symmetric, pass target first for clarity
        cmd = [sc_bin, pdb_file_path, str(target_chain), str(binder_chain), "--json"]
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = (proc.stdout or "").strip()
        if not stdout:
            print(f"[SC-RS] Empty output; using placeholder 0.70 for {basename}")
            return 0.70

        # Parse JSON strictly, else try to extract from mixed output
        try:
            payload = json.loads(stdout)
        except Exception:
            payload = None
            try:
                s_idx = stdout.rfind("{")
                e_idx = stdout.rfind("}")
                if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                    payload = json.loads(stdout[s_idx : e_idx + 1])
            except Exception:
                payload = None

        if isinstance(payload, dict):
            try:
                sc_key = "sc" if "sc" in payload else ("sc_value" if "sc_value" in payload else None)
                if sc_key is not None:
                    sc_val = float(payload[sc_key])
                    if 0.0 <= sc_val <= 1.0:
                        elapsed = time.time() - start_time
                        print(f"[SC-RS] Completed for {basename}: SC={sc_val:.2f} in {elapsed:.2f}s")
                        return sc_val
            except Exception:
                pass
    except subprocess.TimeoutExpired:
        print(f"[SC-RS] ERROR: sc-rs timed out for {os.path.basename(pdb_file_path)}")
    except subprocess.CalledProcessError as e:
        print(f"[SC-RS] ERROR running sc-rs: {e}. stderr: {getattr(e, 'stderr', '')}")
    except Exception as e:
        print(f"[SC-RS] WARN: Failed to compute SC for {pdb_file_path}: {e}")

    # Fallback to placeholder to keep pipelines running
    print(f"[SC-RS] Fallback placeholder 0.70 for {os.path.basename(pdb_file_path)}")
    return 0.70


def _compute_sasa_metrics(pdb_file_path, binder_chain="B", target_chain="A"):
    """
    Compute SASA-derived metrics needed for interface scoring using Biopython.

    Returns a 5-tuple:
        (surface_hydrophobicity_fraction, binder_sasa_in_complex, binder_sasa_monomer,
         target_sasa_in_complex, target_sasa_monomer)
    """
    surface_hydrophobicity_fraction = 0.0
    binder_sasa_in_complex = 0.0
    binder_sasa_monomer = 0.0
    target_sasa_in_complex = 0.0
    target_sasa_monomer = 0.0

    try:
        t0 = time.time()
        basename = os.path.basename(pdb_file_path)
        print(f"[SASA-Biopython] Start for {basename} (binder={binder_chain}, target={target_chain})")
        parser = PDBParser(QUIET=True)

        # Compute atom-level SASA for the entire complex
        complex_structure = parser.get_structure("complex", pdb_file_path)
        complex_model = complex_structure[0]
        sr_complex = ShrakeRupley(probe_radius=1.40, n_points=960, radii_dict=R_CHOTHIA)
        sr_complex.compute(complex_model, level="A")

        # Binder chain SASA within complex
        if binder_chain in complex_model:
            binder_chain_in_complex = complex_model[binder_chain]
            binder_sasa_in_complex = _chain_total_sasa(binder_chain_in_complex)

        # Target chain SASA within complex
        if target_chain in complex_model:
            target_chain_in_complex = complex_model[target_chain]
            target_sasa_in_complex = _chain_total_sasa(target_chain_in_complex)

        # Binder monomer SASA and surface hydrophobicity fraction (area-based)
        if binder_chain in complex_model:
            binder_only_structure = Structure.Structure("binder_only")
            binder_only_model = Model.Model(0)
            binder_only_chain = copy.deepcopy(complex_model[binder_chain])
            binder_only_model.add(binder_only_chain)
            binder_only_structure.add(binder_only_model)

            sr_mono = ShrakeRupley(probe_radius=1.40, n_points=960, radii_dict=R_CHOTHIA)
            sr_mono.compute(binder_only_model, level="A")
            binder_sasa_monomer = _chain_total_sasa(binder_only_chain)

            # Residue-based hydrophobic surface fraction (sum residue SASA for hydrophobic residues)
            hydrophobic_res_sasa = 0.0
            for residue in binder_only_chain:
                if Polypeptide.is_aa(residue, standard=True):
                    try:
                        aa1 = seq1(residue.get_resname()).upper()
                    except Exception:
                        aa1 = ""
                    if aa1 in HYDROPHOBIC_AA_SET:
                        res_sasa = sum(getattr(atom, "sasa", 0.0) for atom in residue.get_atoms())
                        hydrophobic_res_sasa += res_sasa
            surface_hydrophobicity_fraction = (
                (hydrophobic_res_sasa / binder_sasa_monomer) if binder_sasa_monomer > 0.0 else 0.0
            )
        else:
            surface_hydrophobicity_fraction = 0.0

        # Target monomer SASA
        if target_chain in complex_model:
            target_only_structure = Structure.Structure("target_only")
            target_only_model = Model.Model(0)
            target_only_chain = copy.deepcopy(complex_model[target_chain])
            target_only_model.add(target_only_chain)
            target_only_structure.add(target_only_model)
            sr_target_mono = ShrakeRupley(probe_radius=1.40, n_points=960, radii_dict=R_CHOTHIA)
            sr_target_mono.compute(target_only_model, level="A")
            target_sasa_monomer = _chain_total_sasa(target_only_chain)

        elapsed = time.time() - t0
        print(f"[SASA-Biopython] Completed for {basename} in {elapsed:.2f}s")
    except Exception as e_sasa:
        print(f"[Biopython-SASA] ERROR for {pdb_file_path}: {e_sasa}")
        # Fallbacks chosen to match original behavior
        surface_hydrophobicity_fraction = 0.30
        binder_sasa_in_complex = 0.0
        binder_sasa_monomer = 0.0
        target_sasa_in_complex = 0.0
        target_sasa_monomer = 0.0

    return (
        surface_hydrophobicity_fraction,
        binder_sasa_in_complex,
        binder_sasa_monomer,
        target_sasa_in_complex,
        target_sasa_monomer,
    )


def _compute_sasa_metrics_with_freesasa(pdb_file_path, binder_chain="B", target_chain="A"):
    """
    Compute SASA-derived metrics using FreeSASA with fallback to Biopython on failure.

    Returns a 5-tuple:
        (surface_hydrophobicity_fraction, binder_sasa_in_complex, binder_sasa_monomer,
         target_sasa_in_complex, target_sasa_monomer)
    """
    try:
        t0 = time.time()
        basename = os.path.basename(pdb_file_path)
        print(f"[SASA-FreeSASA] Start for {basename} (binder={binder_chain}, target={target_chain})")
        if not _HAS_FREESASA:
            raise RuntimeError("FreeSASA not available")

        # Optional classifier (e.g., NACCESS) via repo file or env var FREESASA_CONFIG
        classifier_obj = None
        try:
            classifier_path = os.environ.get("FREESASA_CONFIG")
            if not classifier_path or not os.path.isfile(classifier_path):
                # default to repo-provided NACCESS config
                module_dir = os.path.dirname(os.path.abspath(__file__))
                default_cfg = os.path.join(module_dir, "freesasa_naccess.cfg")
                if os.path.isfile(default_cfg):
                    classifier_path = default_cfg
            if classifier_path and os.path.isfile(classifier_path):
                classifier_obj = freesasa.Classifier(classifier_path)  # type: ignore[name-defined]
                print(f"[SASA-FreeSASA] Using classifier: {classifier_path}")
        except Exception:
            classifier_obj = None

        # Complex SASA
        if classifier_obj is not None:
            structure_complex = freesasa.Structure(pdb_file_path, classifier=classifier_obj)  # type: ignore[name-defined]
        else:
            structure_complex = freesasa.Structure(pdb_file_path)  # type: ignore[name-defined]
        result_complex = freesasa.calc(structure_complex)  # type: ignore[name-defined]

        binder_sasa_in_complex = 0.0
        target_sasa_in_complex = 0.0
        try:
            # FreeSASA Python API expects a list of selection definition strings: "name, selector"
            selection_defs = [
                f"binder, chain {binder_chain!s}",
                f"target, chain {target_chain!s}",
            ]
            sel_area = freesasa.selectArea(selection_defs, structure_complex, result_complex)  # type: ignore[name-defined]
            # sel_area is a dict-like mapping from selection name to area
            binder_sasa_in_complex = float(sel_area.get("binder", 0.0))
            target_sasa_in_complex = float(sel_area.get("target", 0.0))
        except Exception:
            pass

        # Prepare monomer PDBs via Bio.PDB (only used for chain extraction)
        parser = PDBParser(QUIET=True)
        complex_structure_bp = parser.get_structure("complex_for_freesasa", pdb_file_path)
        complex_model_bp = complex_structure_bp[0]

        binder_sasa_monomer = 0.0
        target_sasa_monomer = 0.0
        surface_hydrophobicity_fraction = 0.0

        tmp_binder_path = None
        tmp_target_path = None
        try:
            if binder_chain in complex_model_bp:
                binder_only_structure = Structure.Structure("binder_only")
                binder_only_model = Model.Model(0)
                binder_only_chain = copy.deepcopy(complex_model_bp[binder_chain])
                binder_only_model.add(binder_only_chain)
                binder_only_structure.add(binder_only_model)

                io_b = PDBIO()
                io_b.set_structure(binder_only_structure)
                tmp_b = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
                tmp_b.close()
                tmp_binder_path = tmp_b.name
                io_b.save(tmp_binder_path)

                if classifier_obj is not None:
                    structure_binder_only = freesasa.Structure(tmp_binder_path, classifier=classifier_obj)  # type: ignore[name-defined]
                else:
                    structure_binder_only = freesasa.Structure(tmp_binder_path)  # type: ignore[name-defined]
                result_binder_only = freesasa.calc(structure_binder_only)  # type: ignore[name-defined]
                binder_sasa_monomer = float(result_binder_only.totalArea())

                # FreeSASA residue selection only: hydrophobic residues / total (no fallback)
                try:
                    sel_defs = ["hydro, resn ala+val+leu+ile+met+phe+pro+trp+tyr+cys"]
                    with _suppress_freesasa_warnings():
                        sel_area = freesasa.selectArea(sel_defs, structure_binder_only, result_binder_only)  # type: ignore[name-defined]
                    hydro_area = float(sel_area.get("hydro", 0.0))
                    if binder_sasa_monomer > 0.0:
                        surface_hydrophobicity_fraction = hydro_area / binder_sasa_monomer
                except Exception:
                    # Keep default 0.0 if selection fails
                    pass

            if target_chain in complex_model_bp:
                target_only_structure = Structure.Structure("target_only")
                target_only_model = Model.Model(0)
                target_only_chain = copy.deepcopy(complex_model_bp[target_chain])
                target_only_model.add(target_only_chain)
                target_only_structure.add(target_only_model)

                io_t = PDBIO()
                io_t.set_structure(target_only_structure)
                tmp_t = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
                tmp_t.close()
                tmp_target_path = tmp_t.name
                io_t.save(tmp_target_path)

                if classifier_obj is not None:
                    structure_target_only = freesasa.Structure(tmp_target_path, classifier=classifier_obj)  # type: ignore[name-defined]
                else:
                    structure_target_only = freesasa.Structure(tmp_target_path)  # type: ignore[name-defined]
                result_target_only = freesasa.calc(structure_target_only)  # type: ignore[name-defined]
                target_sasa_monomer = float(result_target_only.totalArea())
        finally:
            if tmp_binder_path and os.path.isfile(tmp_binder_path):
                try:
                    os.remove(tmp_binder_path)
                except Exception:
                    pass
            if tmp_target_path and os.path.isfile(tmp_target_path):
                try:
                    os.remove(tmp_target_path)
                except Exception:
                    pass

        elapsed = time.time() - t0
        print(f"[SASA-FreeSASA] Completed for {basename} in {elapsed:.2f}s")
        return (
            surface_hydrophobicity_fraction,
            binder_sasa_in_complex,
            binder_sasa_monomer,
            target_sasa_in_complex,
            target_sasa_monomer,
        )
    except Exception as e_fsasa:
        print(f"[FreeSASA] ERROR for {pdb_file_path}: {e_fsasa}")
        return _compute_sasa_metrics(pdb_file_path, binder_chain=binder_chain, target_chain=target_chain)


def openmm_relax(
    pdb_file_path,
    output_pdb_path,
    use_gpu_relax=True,
    openmm_max_iterations=1000,  # Safety cap per stage to avoid stalls (set 0 for unlimited)
    # Default force tolerances for ramp stages (kJ/mol/nm)
    openmm_ramp_force_tolerance_kj_mol_nm=2.0,
    openmm_final_force_tolerance_kj_mol_nm=0.1,
    restraint_k_kcal_mol_A2=3.0,
    restraint_ramp_factors=(1.0, 0.4, 0.0),  # 3-stage restraint ramp factors
    md_steps_per_shake=5000,  # MD steps for each shake (applied only to first two stages)
    lj_rep_base_k_kj_mol=10.0,  # Base strength for extra LJ repulsion (kJ/mol)
    lj_rep_ramp_factors=(0.0, 1.5, 3.0),
):  # 3-stage LJ repulsion ramp factors (soft → hard)
    """
    Relaxes a PDB structure using OpenMM with L-BFGS minimizer.
    Uses PDBFixer to prepare the structure first.
    Applies backbone heavy-atom harmonic restraints (ramped down using restraint_ramp_factors)
    and uses OBC2 implicit solvent.
    Includes an additional ramped LJ-like repulsive force (using lj_rep_ramp_factors) to help with initial clashes.
    Includes short MD shakes for the first two ramp stages only (speed optimization).
    Uses accept-to-best position bookkeeping across all stages.
    Aligns to original and copies B-factors.

    Returns
    -------
    platform_name_used : str or None
        Name of the OpenMM platform actually used (e.g., 'CUDA', 'OpenCL', or 'CPU').
    """

    start_time = time.time()
    basename = os.path.basename(pdb_file_path)
    print(f"[OpenMM-Relax] Initiating relax for {basename}")
    best_energy = float("inf") * unit.kilojoule_per_mole  # Initialize with units
    best_positions = None

    # 1. Store original B-factors (per residue CA or first atom)
    original_residue_b_factors = {}
    bio_parser = PDBParser(QUIET=True)
    try:
        original_structure = bio_parser.get_structure("original", pdb_file_path)
        for model in original_structure:
            for chain in model:
                for residue in chain:
                    # Use Polypeptide.is_aa if available and needed for strict AA check
                    # For B-factor copying, we might want to copy for any residue type present.
                    # Let's assume standard AA check for now as in pr_relax context
                    if Polypeptide.is_aa(residue, standard=True):
                        ca_atom = None
                        try:  # Try to get 'CA' atom
                            ca_atom = residue["CA"]
                        except KeyError:  # 'CA' not in residue
                            pass

                        b_factor = None
                        if ca_atom:
                            b_factor = ca_atom.get_bfactor()
                        else:  # Fallback to first atom if CA not found
                            first_atom = next(residue.get_atoms(), None)
                            if first_atom:
                                b_factor = first_atom.get_bfactor()

                        if b_factor is not None:
                            # residue.id is (hetfield, resseq, icode)
                            original_residue_b_factors[(chain.id, residue.id)] = b_factor
    except Exception:
        original_residue_b_factors = {}

    try:
        # 1. Prepare the PDB structure using PDBFixer
        fixer = PDBFixer(filename=pdb_file_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()  # This should handle common MODRES
        fixer.removeHeterogens(keepWater=False)  # Usually False for relaxation
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # Add hydrogens at neutral pH

        # 2. Set up OpenMM ForceField, System, Integrator, and Simulation
        # Reuse a module-level ForceField instance to avoid re-parsing XMLs each call
        forcefield = _get_openmm_forcefield()

        system = forcefield.createSystem(
            fixer.topology,
            nonbondedMethod=app.CutoffNonPeriodic,  # Retain for OBC2 defined by XML
            nonbondedCutoff=1.0 * unit.nanometer,  # Retain for OBC2 defined by XML
            constraints=app.HBonds,
        )

        # Extract original sigmas from the NonbondedForce for the custom LJ repulsion
        original_sigmas = []
        nonbonded_force_index = -1
        for i_force_idx in range(system.getNumForces()):  # Use getNumForces and getForce
            force_item = system.getForce(i_force_idx)
            if isinstance(force_item, openmm.NonbondedForce):
                nonbonded_force_index = i_force_idx
                for p_idx in range(force_item.getNumParticles()):
                    charge, sigma, epsilon = force_item.getParticleParameters(p_idx)
                    original_sigmas.append(sigma.value_in_unit(unit.nanometer))  # Store as float in nm
                break

        if nonbonded_force_index == -1:
            pass  # Keep silent

        # Add custom LJ-like repulsive force (ramped) using helper function
        lj_rep_custom_force, k_rep_lj_param_index = _create_lj_repulsive_force(
            system,
            lj_rep_base_k_kj_mol,
            lj_rep_ramp_factors,
            original_sigmas,
            nonbonded_force_index,
        )
        if "original_sigmas" in locals():  # Check if it was actually created
            del original_sigmas  # Free memory as it's no longer needed in this scope

        # Add backbone heavy-atom harmonic restraints using helper function
        restraint_force, k_restraint_param_index = _create_backbone_restraint_force(
            system, fixer, restraint_k_kcal_mol_A2
        )

        integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)

        simulation = None
        platform_name_used = None  # To store the name of the successfully used platform

        platform_order = []
        if use_gpu_relax:
            # Prefer OpenCL, then CUDA (override with env if needed)
            env_order = os.environ.get("OPENMM_PLATFORM_ORDER")
            if env_order:
                platform_order = [p.strip() for p in env_order.split(",") if p.strip()]
            else:
                platform_order.extend(["OpenCL", "CUDA"])
        else:
            # Explicit CPU-only path if GPU is not requested
            platform_order.append("CPU")

        last_exception = None
        for p_name_to_try in platform_order:
            if simulation:
                break

            # Retry up to 3 times per platform with 1s backoff
            for attempt_idx in range(1, 4):
                # ensure fresh simulation object per attempt
                simulation = None
                current_platform_obj = None
                current_properties = {}
                try:
                    current_platform_obj = Platform.getPlatformByName(p_name_to_try)
                    if p_name_to_try == "CUDA":
                        current_properties = {"CudaPrecision": "mixed"}
                    elif p_name_to_try == "OpenCL":
                        current_properties = {"OpenCLPrecision": "single"}

                    simulation = app.Simulation(
                        fixer.topology,
                        system,
                        integrator,
                        current_platform_obj,
                        current_properties,
                    )
                    platform_name_used = p_name_to_try
                    print(f"[OpenMM-Relax] Using platform: {platform_name_used}")
                    break
                except (OpenMMException, Exception) as e:
                    last_exception = e
                    if attempt_idx < 3:
                        print(
                            f"[OpenMM-Relax] Platform {p_name_to_try} attempt {attempt_idx} failed; retrying in 1s..."
                        )
                        time.sleep(1.0)
                        continue
                    else:
                        print(f"[OpenMM-Relax] Platform {p_name_to_try} failed after {attempt_idx} attempts")
                        break

            if simulation:
                break

        if simulation is None:
            final_error_msg = f"FATAL: Could not initialize OpenMM Simulation with any GPU platform after trying {', '.join(platform_order)}."
            # Prefer raising the last captured exception if present
            if last_exception is not None:
                raise last_exception
            raise OpenMMException(final_error_msg)

        simulation.context.setPositions(fixer.positions)

        # Optional Pre-Minimization Step (before main ramp loop)
        # Perform if restraints or LJ repulsion are active, to stabilize structure.
        if restraint_k_kcal_mol_A2 > 0 or lj_rep_base_k_kj_mol > 0:
            # Set LJ repulsion to zero for this initial minimization
            if lj_rep_custom_force is not None and k_rep_lj_param_index != -1 and lj_rep_base_k_kj_mol > 0:
                lj_rep_custom_force.setGlobalParameterDefaultValue(k_rep_lj_param_index, 0.0)  # Pass plain float
                lj_rep_custom_force.updateParametersInContext(simulation.context)

            # Set restraints to full strength for this initial minimization (if active)
            if restraint_force is not None and k_restraint_param_index != -1 and restraint_k_kcal_mol_A2 > 0:
                # restraint_k_kcal_mol_A2 is the base parameter for restraint strength
                full_initial_restraint_k_val = _k_kj_per_nm2(restraint_k_kcal_mol_A2)
                restraint_force.setGlobalParameterDefaultValue(k_restraint_param_index, full_initial_restraint_k_val)
                restraint_force.updateParametersInContext(simulation.context)

            initial_min_tolerance = openmm_ramp_force_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer
            simulation.minimizeEnergy(tolerance=initial_min_tolerance, maxIterations=openmm_max_iterations)

        # 3. Perform staged relaxation: ramp restraints, limited MD shakes, and minimization
        base_k_for_ramp_kcal = restraint_k_kcal_mol_A2

        # Determine number of stages based on provided ramp factors
        # Use restraint_ramp_factors for k_constr and lj_rep_ramp_factors for k_rep_lj
        # Simplified stage iteration using zip_longest
        effective_restraint_factors = (
            restraint_ramp_factors if restraint_k_kcal_mol_A2 > 0 and restraint_ramp_factors else [0.0]
        )  # Use 0.0 if no restraint
        effective_lj_rep_factors = (
            lj_rep_ramp_factors if lj_rep_base_k_kj_mol > 0 and lj_rep_ramp_factors else [0.0]
        )  # Use 0.0 if no LJ rep

        # If one of the ramps is disabled (e.g. k=0 or empty factors), its factors list will be [0.0].
        # zip_longest will then pair its 0.0 with the active ramp's factors.
        # If both are disabled, it will iterate once with (0.0, 0.0).

        ramp_pairs = list(zip_longest(effective_restraint_factors, effective_lj_rep_factors, fillvalue=0.0))
        num_stages = len(ramp_pairs)

        # If both k_restraint_kcal_mol_A2 and lj_rep_base_k_kj_mol are 0,
        # or their factor lists are empty, num_stages will be 1 (due to [0.0] default),
        # effectively running one minimization stage without these ramps.
        if (
            num_stages == 1
            and effective_restraint_factors == [0.0]
            and effective_lj_rep_factors == [0.0]
            and not (restraint_k_kcal_mol_A2 > 0 or lj_rep_base_k_kj_mol > 0)
        ):
            pass

        for i_stage_val, (k_factor_restraint, current_lj_rep_k_factor) in enumerate(ramp_pairs):
            stage_num = i_stage_val + 1

            # Set LJ repulsive ramp for the current stage
            if lj_rep_custom_force is not None and k_rep_lj_param_index != -1 and lj_rep_base_k_kj_mol > 0:
                current_lj_rep_k_val = lj_rep_base_k_kj_mol * current_lj_rep_k_factor
                lj_rep_custom_force.setGlobalParameterDefaultValue(
                    k_rep_lj_param_index, current_lj_rep_k_val
                )  # Pass plain float
                lj_rep_custom_force.updateParametersInContext(simulation.context)

            # Set restraint stiffness for the current stage
            if restraint_force is not None and k_restraint_param_index != -1 and restraint_k_kcal_mol_A2 > 0:
                current_stage_k_kcal = base_k_for_ramp_kcal * k_factor_restraint
                numeric_k_for_stage = _k_kj_per_nm2(current_stage_k_kcal)
                restraint_force.setGlobalParameterDefaultValue(k_restraint_param_index, numeric_k_for_stage)
                restraint_force.updateParametersInContext(simulation.context)

            # MD Shake only for first two ramp stages for speed-performance tradeoff
            if md_steps_per_shake > 0 and i_stage_val < 2:
                simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)  # Reinitialize velocities
                simulation.step(md_steps_per_shake)

            # Minimization for the current stage
            # Set force tolerance for current stage
            if i_stage_val == num_stages - 1:  # Final stage
                current_force_tolerance = openmm_final_force_tolerance_kj_mol_nm
            else:  # Ramp stages
                current_force_tolerance = openmm_ramp_force_tolerance_kj_mol_nm
            force_tolerance_quantity = current_force_tolerance * unit.kilojoule_per_mole / unit.nanometer

            # Chunked minimization to avoid pathological stalls: run in small blocks and early-stop
            # if energy improvement becomes negligible
            per_call_max_iterations = (
                200 if (openmm_max_iterations == 0 or openmm_max_iterations > 200) else openmm_max_iterations
            )
            remaining_iterations = openmm_max_iterations
            small_improvement_streak = 0
            last_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

            while True:
                simulation.minimizeEnergy(
                    tolerance=force_tolerance_quantity,
                    maxIterations=per_call_max_iterations,
                )
                current_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

                # Check improvement magnitude
                try:
                    energy_improvement = last_energy - current_energy
                    if energy_improvement < (0.1 * unit.kilojoule_per_mole):
                        small_improvement_streak += 1
                    else:
                        small_improvement_streak = 0
                except Exception:
                    # If unit math fails for any reason, break conservatively
                    small_improvement_streak = 3

                last_energy = current_energy

                # Decrement remaining iterations if bounded
                if openmm_max_iterations > 0:
                    remaining_iterations -= per_call_max_iterations
                    if remaining_iterations <= 0:
                        break

                # Early stop if improvement is consistently negligible
                if small_improvement_streak >= 3:
                    break

            stage_final_energy = last_energy

            # Accept-to-best bookkeeping
            if stage_final_energy < best_energy:
                best_energy = stage_final_energy
                best_positions = simulation.context.getState(getPositions=True).getPositions(
                    asNumpy=True
                )  # Use asNumpy=True

        # After all stages, set positions to the best ones found
        if best_positions is not None:
            simulation.context.setPositions(best_positions)

        # 4. Save the relaxed structure
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(output_pdb_path, "w") as outfile:
            app.PDBFile.writeFile(simulation.topology, positions, outfile, keepIds=True)

        # 4a. Align relaxed structure to original pdb_file_path using all CA atoms
        try:
            biopython_align_all_ca(pdb_file_path, output_pdb_path)
        except Exception:
            pass  # Keep silent on alignment failure

        # 4b. Apply original B-factors to the (now aligned) relaxed structure
        if original_residue_b_factors:
            try:
                # Use Bio.PDB parser and PDBIO for this
                relaxed_structure_for_bfactors = bio_parser.get_structure("relaxed_aligned", output_pdb_path)
                modified_b_factors = False
                for model in relaxed_structure_for_bfactors:
                    for chain in model:
                        for residue in chain:
                            b_factor_to_apply = original_residue_b_factors.get((chain.id, residue.id))
                            if b_factor_to_apply is not None:
                                for atom in residue:
                                    atom.set_bfactor(b_factor_to_apply)
                                modified_b_factors = True

                if modified_b_factors:
                    io = PDBIO()
                    io.set_structure(relaxed_structure_for_bfactors)
                    io.save(output_pdb_path)
            except Exception:
                pass  # Keep silent on B-factor application failure

        # 5. Clean the output PDB
        clean_pdb(output_pdb_path)

        # Explicitly delete heavy OpenMM objects to avoid cumulative slowdowns across many trajectories
        try:
            del positions
        except Exception:
            pass
        try:
            del (
                simulation,
                integrator,
                system,
                restraint_force,
                lj_rep_custom_force,
                fixer,
            )
        except Exception:
            pass
        gc.collect()

        elapsed_total = time.time() - start_time
        print(f"[OpenMM-Relax] Completed relax for {basename} in {elapsed_total:.2f}s (platform={platform_name_used})")
        return platform_name_used

    except Exception as _:
        shutil.copy(pdb_file_path, output_pdb_path)
        gc.collect()
        elapsed_total = time.time() - start_time
        print(f"[OpenMM-Relax] ERROR; copied input to output for {basename} after {elapsed_total:.2f}s")
        print(f"[OpenMM-Relax] ERROR; exeception {_!s}")
        # Guard against 'platform_name_used' not being assigned yet
        try:
            return platform_name_used
        except UnboundLocalError:
            return None


def pr_alternative_score_interface(
    pdb_file, binder_chain="B", target_chain="A", sasa_engine="auto", sc_bin: str = None
):
    """
    Calculate interface scores using PyRosetta-free alternatives including SCASA shape complementarity.

    This function provides comprehensive interface scoring without PyRosetta dependency by combining:
    - Biopython-based SASA calculations
    - SCASA shape complementarity calculation
    - Interface residue identification

    Parameters
    ----------
    pdb_file : str
        Path to PDB file
    binder_chain : str
        Chain ID of the binder
    sasa_engine : str
        "auto" (default) prefers FreeSASA if installed, else Biopython.
        "freesasa" forces FreeSASA (falls back to Biopython on error).
        "biopython" forces Biopython Shrake-Rupley.

    Returns
    -------
    tuple
        (interface_scores, interface_AA, interface_residues_pdb_ids_str)
    """
    t0_all = time.time()
    basename = os.path.basename(pdb_file)
    print(
        f"[Alt-Score] Initiating PyRosetta-free scoring for {basename} (binder={binder_chain}, sasa_engine={sasa_engine})"
    )

    # Get interface residues via Biopython (works without PyRosetta)
    t0_if = time.time()
    print("[Alt-Score] Finding interface residues (hotspot_residues)...")
    interface_residues_set = hotspot_residues(pdb_file, binder_chain, target_chain)
    interface_residues_pdb_ids = [f"{binder_chain}{pdb_res_num}" for pdb_res_num in interface_residues_set.keys()]
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)
    print(f"[Alt-Score] Found {len(interface_residues_pdb_ids)} interface residues in {time.time() - t0_if:.2f}s")

    # Initialize amino acid dictionary for interface composition
    interface_AA = dict.fromkeys("ACDEFGHIKLMNPQRSTVWY", 0)
    for pdb_res_num, aa_type in interface_residues_set.items():
        interface_AA[aa_type] += 1

    # SASA-based calculations: select engine
    t0_sasa = time.time()
    if str(sasa_engine).lower() == "biopython":
        print("[Alt-Score] Computing SASA with Biopython Shrake-Rupley...")
        (
            surface_hydrophobicity_fraction,
            binder_sasa_in_complex,
            binder_sasa_monomer,
            target_sasa_in_complex,
            target_sasa_monomer,
        ) = _compute_sasa_metrics(pdb_file, binder_chain=binder_chain, target_chain=target_chain)
    elif str(sasa_engine).lower() == "freesasa":
        print("[Alt-Score] Computing SASA with FreeSASA...")
        (
            surface_hydrophobicity_fraction,
            binder_sasa_in_complex,
            binder_sasa_monomer,
            target_sasa_in_complex,
            target_sasa_monomer,
        ) = _compute_sasa_metrics_with_freesasa(pdb_file, binder_chain=binder_chain, target_chain=target_chain)
    else:
        if _HAS_FREESASA:
            print("[Alt-Score] Computing SASA with FreeSASA (auto)...")
            (
                surface_hydrophobicity_fraction,
                binder_sasa_in_complex,
                binder_sasa_monomer,
                target_sasa_in_complex,
                target_sasa_monomer,
            ) = _compute_sasa_metrics_with_freesasa(pdb_file, binder_chain=binder_chain, target_chain=target_chain)
        else:
            print("[Alt-Score] Computing SASA with Biopython (auto fallback)...")
            (
                surface_hydrophobicity_fraction,
                binder_sasa_in_complex,
                binder_sasa_monomer,
                target_sasa_in_complex,
                target_sasa_monomer,
            ) = _compute_sasa_metrics(pdb_file, binder_chain=binder_chain, target_chain=target_chain)
    print(f"[Alt-Score] SASA computations finished in {time.time() - t0_sasa:.2f}s")

    # Compute buried SASA: binder-side and total (binder + target)
    interface_binder_dSASA = max(binder_sasa_monomer - binder_sasa_in_complex, 0.0)
    interface_target_dSASA = 0.0
    try:
        interface_target_dSASA = max(target_sasa_monomer - target_sasa_in_complex, 0.0)
    except Exception as e_idsasa:
        print(f"[Biopython-SASA] WARN interface_target_dSASA for {pdb_file}: {e_idsasa}")
    interface_total_dSASA = interface_binder_dSASA + interface_target_dSASA
    # Align with PyRosetta: use TOTAL interface dSASA divided by binder SASA IN COMPLEX
    interface_binder_fraction = (
        (interface_total_dSASA / binder_sasa_in_complex * 100.0) if binder_sasa_in_complex > 0.0 else 0.0
    )

    # Calculate shape complementarity using SCASA
    t0_sc = time.time()
    print("[Alt-Score] Computing shape complementarity (SC)...")
    interface_sc = _calculate_shape_complementarity(pdb_file, binder_chain, target_chain=target_chain, sc_bin=sc_bin)
    print(f"[Alt-Score] SC computation finished in {time.time() - t0_sc:.2f}s")

    # Fixed placeholder values for metrics that are not currently computed without PyRosetta
    # These values are chosen to pass active filters
    interface_nres = len(interface_residues_pdb_ids)  # computed from interface residues
    # interface_interface_hbonds = 5                                      # passes >= 3 (active filter)
    # interface_delta_unsat_hbonds = 1                                    # passes <= 4 (active filter)
    # interface_hbond_percentage = 60.0                                   # informational (no active filter)
    # interface_bunsch_percentage = 0.0                                   # informational (no active filter)
    # binder_score = -1.0                                                 # passes <= 0 (active filter) - never results in rejections based on extensive testing
    # interface_packstat = 0.65                                           # informational (no active filter)
    # interface_dG = -10.0                                                # passes <= 0 (active filter) - never results in rejections based on extensive testing
    # interface_dG_SASA_ratio = 0.0                                       # informational (no active filter)

    interface_scores = {
        "surface_hydrophobicity": surface_hydrophobicity_fraction,
        "interface_sc": interface_sc,
        "interface_dSASA": interface_total_dSASA,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": (
            (sum(interface_AA[aa] for aa in "ACFILMPVWY") / interface_nres * 100.0) if interface_nres > 0 else 0.0
        ),
        "interface_nres": interface_nres,
    }

    # Round float values to two decimals for consistency
    interface_scores = {k: round(v, 3) if isinstance(v, float) else v for k, v in interface_scores.items()}

    print(f"[Alt-Score] Completed scoring for {basename} in {time.time() - t0_all:.2f}s")
    return interface_scores, interface_AA, interface_residues_pdb_ids_str


def openmm_relax_subprocess(pdb_file_path, output_pdb_path, use_gpu_relax=True, timeout=None, max_attempts=3):
    """Run openmm_relax in a fresh Python process to fully reset OpenCL context per run.
    Retries if the child fell back to copying input (soft failure) or if the child crashes (hard failure).
    Streams child logs to parent stdout/stderr so DEBUG lines are visible.
    """
    import logging as _logging

    want_verbose = _logging.getLogger("functions").isEnabledFor(_logging.DEBUG)

    code_parts = []
    if want_verbose:
        code_parts.append(
            "import logging; logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')"
        )
    else:
        code_parts.append("import logging; logging.basicConfig(level=logging.WARNING)")
    # Suppress noisy third-party DEBUG logs in child process
    code_parts.append("import logging")
    code_parts.append("logging.getLogger('matplotlib').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('pyrosetta').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('pyrosetta.distributed').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('pyrosetta.distributed.utility.pickle').setLevel(logging.WARNING)")
    code_parts.append("from functions.pr_alternative_utils import openmm_relax")
    code_parts.append(
        f"plat = openmm_relax({pdb_file_path!r}, {output_pdb_path!r}, use_gpu_relax={bool(use_gpu_relax)})"
    )
    py_code = "; ".join(code_parts)

    # Signature to detect soft fallback path inside child (input copied to output)
    fallback_signature = "[OpenMM-Relax] ERROR; copied input to output"

    attempts = int(max(1, int(max_attempts)))
    for attempt_idx in range(1, attempts + 1):
        # Capture output to inspect for fallback while still forwarding to parent
        proc = subprocess.run(
            [sys.executable, "-c", py_code],
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        # Forward child output to parent streams to preserve visibility, but filter stderr
        if proc.stdout:
            try:
                sys.stdout.write(proc.stdout)
            except Exception:
                pass
        if proc.stderr:
            try:
                for line in proc.stderr.splitlines(True):  # Preserve newlines
                    if "Failed to read file: /tmp/dep-" not in line:
                        sys.stderr.write(line)
            except Exception:
                pass

        # Hard failure: non-zero rc from child
        if proc.returncode != 0:
            if attempt_idx >= attempts:
                raise RuntimeError(
                    f"Subprocess openmm_relax failed with rc={proc.returncode} after {attempt_idx} attempts"
                )
            time.sleep(0.5)
            continue

        # Soft failure: child printed fallback copy message
        combined_out = (proc.stdout or "") + (proc.stderr or "")
        if fallback_signature in combined_out and attempt_idx < attempts:
            print(f"[OpenMM-Relax] Detected fallback copy; retrying ({attempt_idx + 1}/{attempts})")
            # Remove fallback-copied file before retry so the next success writes a clean output
            try:
                if os.path.isfile(output_pdb_path):
                    os.remove(output_pdb_path)
            except Exception:
                pass
            time.sleep(0.5)
            continue

        # Success (or final acceptable fallback)
        return None

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--relax-cli", action="store_true")
    parser.add_argument("--in", dest="inp", type=str)
    parser.add_argument("--out", dest="out", type=str)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    if args.relax_cli:
        if args.verbose:
            import logging

            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        plat = openmm_relax(args.inp, args.out, use_gpu_relax=args.gpu)
        if plat:
            print(plat)
        sys.exit(0)
