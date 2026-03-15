# Target Data

This directory contains PDB files used as design targets, organized by task type into subdirectories matching the `source` field in target/task config YAMLs.

The pipeline resolves target PDBs via `$DATA_PATH/target_data/<source>/<target_filename>.pdb`, so place your files accordingly or use an explicit `target_path` in the config.

## Directory Structure

```
target_data/
├── bindcraft_targets/       # Protein binder targets (PD1, PDL1, CD45, etc.)
├── alpha_proteo_targets/    # AlphaProteo benchmark binder targets
├── ligand_targets/          # Ligand-only binder targets (small molecule + de novo protein)
└── ame_input_structures/    # AME (motif + ligand) scaffolding targets
```

## Target Types

### Protein Binder Targets (`bindcraft_targets`)

Standard protein-protein binder design. Each entry specifies a target chain, hotspot residues, and binder length range.

- Config: `configs/targets/targets_dict.yaml`
- Required fields: `source`, `target_filename`, `target_input`, `hotspot_residues`, `binder_length`

### Ligand Binder Targets (`ligand_targets`)

Design a protein binder around a small-molecule ligand. The target PDB contains only the ligand; the protein is generated de novo.

- Config: `configs/targets/ligand_targets_dict.yaml`
- Additional fields: `ligand` (CCD code), `SMILES`, `ligand_only: True`, `use_bonds_from_file: True`

### AME Targets (`ame_targets`)

Motif scaffolding with ligands (Atomistic Motif Extension). The target PDB contains a protein active site plus one or more ligands; the model scaffolds a new protein around the specified motif residues and ligand atoms.

- Config: `configs/design_tasks/ame_dict_v2.yaml`
- Additional fields: `ligand`, `contig_atoms`

### Monomer Motif Targets

Motif scaffolding without ligands. Motif PDBs are referenced via `$DATA_PATH/motif_data_aa/`.

- Config: `configs/design_tasks/motif_dict.yaml`

## Preparing AME Target PDBs

> **Warning**: AME input PDBs require manual cleaning before use. Most bundled structures in `ame_input_structures/` still need preparation. `M0024_1nzy_v3.pdb` is provided as a ready-to-use example of a correctly cleaned structure. Malformed inputs will cause silent evaluation errors or incorrect RMSD values.

### Required cleaning steps

1. **Chain assignment**: Ligand(s) must be on **chain A**, motif protein residues on **chain B**. If your source PDB has different chain assignments, re-chain accordingly.

2. **Residue naming**: Set the ligand residue name to `L:0`. This prevents RF3 from reconstructing missing atoms from the CCD dictionary, which causes shape mismatches in RMSD computation.

3. **Remove metal ions (Zn, Mg, Fe, etc.)**: Strip all metal/ion atoms from the PDB. These are sometimes included by accident from the source structure and will interfere with generation and evaluation. In the task config, also remove any metal CCD codes from the `ligand` list (e.g., remove `"ZN"`, `"MG"`, `"FE"`, `"FE2"`, `"MN"`).

4. **Update `contig_atoms`**: Make sure the chain letters in `contig_atoms` match the re-chained structure (should reference chain B for motif residues).

### Example: cleaning with atomworks

```python
from atomworks.io import load_any, to_pdb_file

atom_array = load_any("M0024_1nzy.pdb")[0]

# Remove metal ions
metal_names = {"ZN"} #, "MG", "FE", "MN", "CA", "NA", "CU", "CO", "NI"}
metal_mask = [rn.strip() in metal_names for rn in atom_array.res_name]
atom_array = atom_array[~numpy.array(metal_mask)]

# Re-chain: ligand -> A, protein -> B
atom_array.chain_id[atom_array.hetero] = "A"
atom_array.chain_id[~atom_array.hetero] = "B"

# Rename ligand residues to L:0
ligand_mask = atom_array.chain_id == "A"
atom_array.res_name[ligand_mask] = "L:0"

to_pdb_file(atom_array, "M0024_1nzy_v3.pdb")
```

### Ligand naming for RF3 evaluation

RoseTTAFold3 (>=0.1.12, March 2026) reconstructs ligands from their CCD (Chemical Component Dictionary) code, adding back any atoms missing in the input PDB. If your ligand is incomplete (i.e., a substructure of the full CCD entry), RF3 will add the missing heavy atoms, causing a shape mismatch that breaks RMSD computation.

Setting `ligand: "L:0"` in the task config tells the pipeline to treat the ligand as a generic residue rather than resolving it from CCD, preserving the exact atom set from your input PDB. The `M0024_1nzy_v3` entry in `ame_dict_v2.yaml` and its corresponding `M0024_1nzy_v3.pdb` are provided as a fully cleaned, ready-to-use reference.
