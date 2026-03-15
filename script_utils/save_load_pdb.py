import os

import biotite.structure.io as strucio
from atomworks.io.utils.io_utils import load_any

# Load the PDB
pdb = load_any("2lgv.pdb")

# Create directory
os.makedirs("2lgv", exist_ok=True)

# Iterate and save
for idx, structure in enumerate(pdb):
    # Filter out residues with res_id > 108
    filtered = structure[structure.res_id <= 108]

    # Save to PDB file
    strucio.save_structure(f"2lgv/2lgv_{idx}.pdb", filtered)
    print(f"Saved 2lgv/2lgv_{idx}.pdb ({len(filtered)} atoms)")
