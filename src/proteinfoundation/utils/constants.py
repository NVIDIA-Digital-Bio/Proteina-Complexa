import torch
from atomworks.constants import AA_LIKE_CHEM_TYPES, DNA_LIKE_CHEM_TYPES, RNA_LIKE_CHEM_TYPES
from atomworks.ml.encoding_definitions import TokenEncoding
from graphein.protein.resi_atoms import ATOM_NUMBERING
from openfold.np.residue_constants import atom_types

# PDB and OpenFold have different atom ordering, these utils convert between the two
# PDB ordering: https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# OpenFold ordering: https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L556
# more background: https://kdidi.netlify.app/blog/proteins/2024-02-03-protein-representations/
PDB_TO_OPENFOLD_INDEX_TENSOR = torch.tensor([ATOM_NUMBERING[atom] for atom in atom_types])
OPENFOLD_TO_PDB_INDEX_TENSOR = torch.tensor([atom_types.index(atom) for atom in ATOM_NUMBERING])

AA_CHARACTER_PROTORP = {
    "ALA": "A",
    "CYS": "P",
    "GLU": "C",
    "ASP": "C",
    "GLY": "A",
    "PHE": "A",
    "ILE": "A",
    "HIS": "P",
    "LYS": "C",
    "MET": "A",
    "LEU": "A",
    "ASN": "P",
    "GLN": "P",
    "PRO": "A",
    "SER": "P",
    "ARG": "C",
    "THR": "P",
    "TRP": "P",
    "VAL": "A",
    "TYR": "P",
}

SIDECHAIN_TIP_ATOMS = {
    "ALA": ["CA", "CB"],
    "ARG": ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "CYS": ["CA", "CB", "SG"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "GLN": ["CG", "CD", "NE2", "OE1"],
    "GLY": [],
    "HIS": ["CB", "CG", "CD2", "CE1", "ND1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CE", "NZ"],
    "MET": ["CG", "CE", "SD"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CA", "CB", "CG", "CD", "N"],
    "SER": ["CA", "CB", "OG"],
    "THR": ["CA", "CB", "CG2", "OG1"],
    "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}


DEBUG_ATOMS = {  #! M0024_1nzy this task + tip atoms for residues that do not exist there
    "PHE": ["C", "O"],
    "ALA": ["C", "CA", "CB", "N"],
    "HIS": ["CD2", "CE1", "CG", "ND1", "NE2"],
    "GLY": ["CA", "N"],
    "TRP": ["CD1", "CD2", "CE2", "CG", "CZ2", "NE1"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ARG": ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "CYS": ["CA", "CB", "SG"],
    "GLU": ["CD", "CG", "OE1", "OE2"],
    "HIS": ["CD2", "CE1", "CG", "ND1", "NE2"],
    "GLN": ["CD", "CG", "NE2", "OE1"],
    "ILE": ["CB", "CD1", "CG1", "CG2"],
    "LEU": ["CB", "CD1", "CD2", "CG"],
    "LYS": ["CE", "NZ"],
    "MET": ["CE", "CG", "SD"],
    "PRO": ["CA", "CB", "CD", "CG", "N"],
    "SER": ["CA", "CB", "OG"],
    "THR": ["CA", "CB", "CG2", "OG1"],
    "TYR": ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}

AME_ATOMS = {
    "PHE": [["C", "O"], ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ"]],
    "ALA": [
        ["C", "CA", "CB", "N"],
        ["CA", "N"],
        ["CA", "CB"],
        ["C", "CA", "O"],
        ["C", "O"],
    ],
    "HIS": [
        ["CD2", "CE1", "CG", "ND1", "NE2"],
        ["CB", "CD2", "CE1", "CG", "ND1", "NE2"],
        ["CD2", "CE1", "NE2"],
        ["CE1", "CG", "ND1"],
        ["CE1", "ND1", "NE2"],
    ],
    "GLY": [["CA", "N"], ["C", "CA", "N"]],
    "TRP": [
        ["CD1", "CD2", "CE2", "CG", "CZ2", "NE1"],
        ["CB", "CD1", "CD2", "CE2", "CE3", "CG", "CH2", "CZ2", "CZ3", "NE1"],
    ],
    "ASP": [
        ["CB", "CG", "OD1", "OD2"],
        ["CG", "OD2"],
        ["CG", "OD1"],
        ["C", "CA", "CB", "N"],
        ["C", "CA", "CB", "CG", "N", "O"],
    ],
    "ARG": [
        ["CZ", "NE", "NH1", "NH2"],
        ["C", "CA", "CB", "N"],
        ["CZ", "NH1"],
        ["CD", "CZ", "NE"],
        ["CD", "CG", "CZ", "NE", "NH1", "NH2"],
        ["CB", "CD", "CG", "CZ", "NE"],
        ["CZ", "NH2"],
        ["C", "O"],
        ["CD", "CZ", "NE", "NH1", "NH2"],
    ],
    "LYS": [["CD", "CE", "NZ"], ["CE", "NZ"]],
    "THR": [["CA", "CB", "CG2", "OG1"], ["C", "CA", "CB", "N"], ["CB", "OG1"]],
    "GLU": [
        ["C", "CA", "O"],
        ["CA", "N"],
        ["CD", "OE2"],
        ["CD", "OE1"],
        ["CD", "CG", "OE1", "OE2"],
    ],
    "ILE": [["C", "CA", "O"], ["C", "CA", "CB", "N"], ["CB", "CD1", "CG1", "CG2"]],
    "SER": [
        ["CB", "OG"],
        ["C", "CA", "O"],
        ["CA", "CB", "OG"],
        ["C", "O"],
        ["CA", "N"],
        ["C", "CA", "CB", "N"],
    ],
    "ASN": [["CB", "CG", "ND2", "OD1"], ["CA", "N"], ["CG", "ND2"]],
    "CYS": [["CA", "CB", "SG"], ["CB", "SG"]],
    "GLN": [["CA", "N"], ["CD", "OE1"], ["CD", "CG", "NE2", "OE1"]],
    "TYR": [
        ["CE1", "CE2", "CZ", "OH"],
        ["CZ", "OH"],
        ["CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "OH"],
    ],
    "PRO": [["CD", "CG", "N"], ["CA", "CB", "CD", "CG", "N"]],
    "LEU": [["CB", "CD1", "CD2", "CG"]],
    "MET": [["CE", "CG", "SD"]],
    "VAL": [["CB", "CG1", "CG2"]],
}

# fmt: off
UNIFIED_ATOM37_ENCODING = TokenEncoding(
    token_atoms={

        # Standard amino acids (classes 0-19)
        #        0       1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22      23      24      25      26      27      28      29      30      31      32      33      34      35      36
        'ALA': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'ARG': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'NE ', '   ', '   ', '   ', '   ', '   ', 'NH1', 'NH2', '   ', 'CZ ', '   ', '   ', '   ', 'OXT'],
        'ASN': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'ND2', 'OD1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'ASP': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OD1', 'OD2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'CYS': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', '   ', '   ', '   ', 'SG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'GLN': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'NE2', 'OE1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'GLU': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OE1', 'OE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'GLY': ['N  ', 'CA ', 'C  ', '   ', 'O  ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'HIS': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD2', 'ND1', '   ', '   ', '   ', '   ', '   ', 'CE1', '   ', '   ', '   ', '   ', 'NE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'ILE': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', 'CG1', 'CG2', '   ', '   ', '   ', '   ', 'CD1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'LEU': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'LYS': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CE ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'NZ ', 'OXT'],
        'MET': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'SD ', 'CE ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'PHE': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', 'CE1', 'CE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CZ ', '   ', '   ', '   ', 'OXT'],
        'PRO': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', 'CD ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'SER': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', '   ', 'OG ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'THR': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', '   ', 'CG2', '   ', 'OG1', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],
        'TRP': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'CE2', 'CE3', '   ', 'NE1', '   ', '   ', '   ', 'CH2', '   ', '   ', '   ', '   ', 'CZ2', 'CZ3', '   ', 'OXT'],
        'TYR': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', 'CG ', '   ', '   ', '   ', '   ', '   ', '   ', 'CD1', 'CD2', '   ', '   ', '   ', '   ', '   ', '   ', 'CE1', 'CE2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OH ', 'CZ ', '   ', '   ', '   ', 'OXT'],
        'VAL': ['N  ', 'CA ', 'C  ', 'CB ', 'O  ', '   ', 'CG1', 'CG2', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', 'OXT'],

        # Mask token (class 20)
        '<M>': ['   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   '],
        # Unknown amino acid (class 21)
        'UNK': ['   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   '],

        # RNA nucleotides (classes 22-25): A, C, G, U
        #       0     1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17     18     19     20     21     22     23     24     25     26     27     28     29     30     31     32     33     34     35     36
        'A':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  'N6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'C':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  'N4',  '',    '',    '',    '',    ''],
        'G':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  '',    'N2',  'O6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'U':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  '',    'O4',  '',    '',    '',    ''],

        # Unknown RNA (class 26)
        'N':  ['P',   "C1'",  "C2'", "O2'",  "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],

        # DNA nucleotides (classes 27-30): DA, DC, DG, DT
        'DA': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  'N6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'DC': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  'N4',  '',    '',    '',    '',    ''],
        'DG': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', 'N9',  'C8',  'N7',  'C5',  'C4',  'N3',  'C2',  'N1',  'C6',  '',    'N2',  'O6',  '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
        'DT': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    'N1',  'C2',  'O2',  'N3',  'C4',  'C5',  'C6',  '',    'O4',  'C7',  '',    '',    ''],

        # Unknown DNA (class 31)
        'DN': ['P',   "C1'",  "C2'", '',     "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'OP1', 'OP2', '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],

        # Atomised token (class 32) - placeholder for atomised small molecules, always put atom in the second position
        '<A>': ['   ', 'X', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   '],

        # Gap token (class 33) - represents alignment gaps in MSAs
        '<G>': ['   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   '],
    },
    chemcomp_type_to_unknown=(
        dict.fromkeys(AA_LIKE_CHEM_TYPES, "UNK")
        | dict.fromkeys(DNA_LIKE_CHEM_TYPES, "DN")
        | dict.fromkeys(RNA_LIKE_CHEM_TYPES, "N")
    ),
)
"""Unified atom37 encoding for all token types in ConditionalResidueTypeSeqFeat.

Provides a comprehensive 37-slot encoding that encompasses:
- Class 0: MASK token (special masking token)
- Classes 1-20: Standard amino acids (ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE,
                LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL)
- Class 21: UNK (unknown amino acid)
- Classes 22-25: RNA nucleotides (A, C, G, U)
- Class 26: N (unknown RNA)
- Classes 27-30: DNA nucleotides (DA, DC, DG, DT)
- Class 31: DN (unknown DNA)
- Class 32: ATOMIZED (atomized small molecule token)
- Class 33: GAP (alignment gap in MSAs)

This encoding is compatible with the conditional residue type feature used in protein
foundation models, enabling unified handling of proteins, RNA, DNA, and small molecules
in a single representation space.

Usage:
    UNIFIED_ATOM37_ENCODING serves as the single source of truth for:
    - Atom37 layout operations (coordinate processing):
        * atom_array_to_encoding() / atom_array_from_encoding()
        * Converting between AtomArray and atom37 coordinate tensors
    - Sequence encoding operations (residue type indices):
        * Use UNIFIED_ATOM37_ENCODING.token_to_idx to encode residue names
        * Use UNIFIED_ATOM37_ENCODING.idx_to_token to decode indices
"""
# fmt: on

# Derived automatically from dict order - no duplication!
RESNAME_TO_IDX = {token: i for i, token in enumerate(UNIFIED_ATOM37_ENCODING.token_atoms.keys())}
