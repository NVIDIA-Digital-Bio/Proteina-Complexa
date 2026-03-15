import logging
from typing import Any

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from atomworks.constants import CCD_MIRROR_PATH, ELEMENT_NAME_TO_ATOMIC_NUMBER, UNKNOWN_LIGAND

# pip install openbabel-wheel==3.1.1.22
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.ccd import get_available_ccd_codes
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from atomworks.ml.transforms.atom_array import get_within_entity_idx
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.openbabel_utils import atom_array_from_openbabel, atom_array_to_openbabel
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from rdkit import Chem
from scipy.linalg import sqrtm
from scipy.sparse.linalg import eigs

logger = logging.getLogger("atomworks.ml")

# UNL is a special CCD code for unknown ligands; we do not consider it "known" as it has no structure
KNOWN_CCD_CODES = get_available_ccd_codes(CCD_MIRROR_PATH) - {UNKNOWN_LIGAND}


def _encode_atom_names_like_af3(atom_names: np.ndarray) -> np.ndarray:
    """Encodes atom names like AF3.

    This generates the `ref_atom_name_chars` feature used in AF3.
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) - 32, and names are padded to
        length 4.

    Reference:
        - https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
    # Ensure uppercase
    atom_names = np.char.upper(atom_names)
    # Turn into 4 character ASCII string (this truncates longer atom names)
    atom_names = atom_names.astype("|S4")
    # Pad to 4 char string with " " (ord(" ") = 32)
    atom_names = np.char.ljust(atom_names, width=4, fillchar=" ")
    # Interpret ASCII bytes to uint8
    atom_names = atom_names.view(np.uint8)
    # Reshape to (N, 4) and subtract 32 to get back to range [0, 64]
    return atom_names.reshape(-1, 4) - 32


def _map_reference_conformer_to_residue(
    res_name: str, atom_names: np.ndarray, conformer: AtomArray
) -> tuple[np.ndarray, np.ndarray]:
    """Maps the coordinate information from a reference conformer to a
    given residue, dropping all atoms that are not in the residue.

    Args:
        - res_name (str): The name of the residue to map to.
        - atom_names (np.ndarray): Array of atom names in the residue to map to.
        - conformer (AtomArray): The reference conformer.

    Returns:
        - ref_pos (np.ndarray): Reference positions for atoms in the residue.
        - ref_mask (np.ndarray): Mask indicating valid reference positions.
    """

    # ... mark the atoms that are in the residue (keep) and where they are in the residue (to_within_res_idx)
    keep = np.zeros(len(conformer), dtype=bool)  # [n_atoms_in_conformer]
    # Mapping from conformer atom indices to residue atom indices
    to_within_res_idx = -np.ones(len(conformer), dtype=int)  # [n_atoms_in_conformer]

    for i, atom_name in enumerate(atom_names):
        matching_atom_idx = np.where(conformer.atom_name == atom_name)[0]
        if len(matching_atom_idx) == 0:
            logger.warning(f"Atom {atom_name} not found in conformer for residue {res_name} with {atom_names=}.")
            continue
        matching_atom_idx = matching_atom_idx[0]
        keep[matching_atom_idx] = True
        to_within_res_idx[matching_atom_idx] = i

    # ... fill the reference positions
    # (We must handle the case where to_within_res_idx[keep] contains indices out of bounds for the filtered conformer)
    kept_atoms = np.where(keep)[0]
    ordering = np.array([to_within_res_idx[idx] for idx in kept_atoms])
    coord = conformer.coord[kept_atoms][np.argsort(ordering)]  # [n_atoms_in_res, 3]

    ref_pos = coord
    ref_mask = np.isfinite(coord).all(axis=-1)  # [n_atoms_in_res]

    return ref_pos, ref_mask  # [n_atoms_in_res, 3], [n_atoms_in_res]


def get_af3_raw_molecule_features(
    atom_array: AtomArray,
    use_element_for_atom_names_of_atomized_tokens: bool = False,
    residue_conformer_indices: dict[int, np.ndarray] | None = None,
    **generate_conformers_kwargs,
) -> tuple[dict[str, Any], dict[str, Chem.Mol]]:
    """Get AF3 reference features for each residue in the atom array.

    Args:
        atom_array: The input atom array.
        conformer_generation_timeout: Maximum time allowed for conformer generation per residue.
            Defaults to (3.0, 0.15), which gives a timeout of 3.0 + 0.15 * (n_conformers - 1) seconds.
            If None, no timeout is applied and the timeout strategy is ignored (no subprocesses will be spawned).
        apply_random_rotation_and_translation: Whether to apply a random rotation and translation to each conformer (AF-3-style)
        timeout_strategy: The strategy to use for the timeout.
            Defaults to "subprocess" (which is the most reliable choice).
        max_conformers_per_residue: Maximum number of conformers to generate per residue type.
            If None, generates conformers equal to residue count. If set, generates min(count, max_conformers_per_residue)
            and randomly samples from those conformers for each residue instance.
        cached_residue_level_data: Optional cached conformer data by residue name. If provided,
            cached conformers will be preferred over generated ones.
        residue_conformer_indices: Optional mapping of global residue IDs to specific conformer indices.
            If provided, these specific conformers will be used for the corresponding residues.
        **generate_conformers_kwargs: Additional keyword arguments to pass to the generate_conformers function.

    Returns:
        ref_conformer: A dictionary containing the generated reference features.
        ref_mols: A dictionary containing all generated RDKit molecules, including those with unknown CCD codes.

    This function generates the following reference features, following AF3:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
            translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms, 128] One-hot encoding of the element atomic number for each atom in the
            reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
            Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
            this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.

    (Optionally) The following custom features, helpful for extra conditioning:
        - ref_pos_is_ground_truth (optional): [N_atoms] Whether the reference conformer is the ground-truth conformer.
            Determined by the `ground_truth_conformer_policy` annotation.
        - ref_pos_ground_truth (optional): [N_atoms, 3] The ground-truth conformer positions.
            Determined by the `ground_truth_conformer_policy` annotation.
        - is_atomized_atom_level: [N_atoms] Whether the atom is atomized (atom-level version of "is_ligand")

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
    "ground_truth_conformer_policy" in atom_array.get_annotation_categories()
    "res_id_global" in atom_array.get_annotation_categories()

    # Generate reference conformers for each residue (if cropped, each residue that has tokens in the crop)
    # ... get residue-level stochiometry
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
    _res_names = atom_array.res_name[_res_starts]
    res_stochiometry = dict(zip(*np.unique(_res_names, return_counts=True), strict=False))

    atom_array.coord
    ref_mask = np.ones(len(atom_array), dtype=bool)

    # Generate remaining reference features
    # ... element
    ref_element = (
        atom_array.atomic_number
        if "atomic_number" in atom_array.get_annotation_categories()
        else np.vectorize(ELEMENT_NAME_TO_ATOMIC_NUMBER.get)(atom_array.element)
    )
    # ... charge
    ref_charge = atom_array.charge

    # ... atom name
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)

    if use_element_for_atom_names_of_atomized_tokens:
        assert "atomize" in atom_array.get_annotation_categories(), (
            "Atomize annotation is required when using element for atom names of atomized tokens."
        )
        ref_atom_name_chars[atom_array.atomize] = _encode_atom_names_like_af3(atom_array.element[atom_array.atomize])

    is_atomized_atom_level = atom_array.atomize if "atomize" in atom_array.get_annotation_categories() else None
    ref_conformer = {
        "atom_element": ref_element,  # F.one_hot(torch.from_numpy(ref_element).long(), num_classes=128),  # (n_atoms)
        "atom_charge": ref_charge,  # (n_atoms)
        "atom_name_chars": ref_atom_name_chars,  # F.one_hot(torch.from_numpy(ref_atom_name_chars).long(), num_classes=64),  # (n_atoms, 4)
        "is_atomized_atom_level": is_atomized_atom_level,  # (n_atoms)
    }

    return ref_conformer


class GetAF3MoleculeFeatures(Transform):
    """Generate AF3 reference molecule features for each residue in the atom array.

    This transform adds the following features to the data dictionary under the 'feats' key, following AF3:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
          translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms] One-hot encoding of the element atomic number for each atom in the
          reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
          Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
          this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.

    And the following custom features, helpful for extra conditioning/downstream use:
        - ref_pos_is_ground_truth: [N_atoms] Whether the reference conformer is the ground-truth conformer.
          Determined by the `ground_truth_conformer_policy` annotation.
        - ref_pos_ground_truth: [N_atoms, 3] The ground-truth conformer positions.
          Determined by the `ground_truth_conformer_policy` annotation.
        - is_atomized_atom_level: [N_atoms] Whether the atom is atomized (atom-level version of "is_ligand")

    Note: This transform should be applied after cropping.

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """

    def __init__(
        self,
        use_element_for_atom_names_of_atomized_tokens: bool = False,
        **generate_conformers_kwargs,
    ):
        self.generate_conformers_kwargs = generate_conformers_kwargs
        self.use_element_for_atom_names_of_atomized_tokens = use_element_for_atom_names_of_atomized_tokens

        if self.use_element_for_atom_names_of_atomized_tokens:
            logger.warning("Using element type for atom names of atomized tokens.")

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

        if "charge" not in data["atom_array"].get_annotation_categories():
            data["atom_array"].set_annotation(
                "charge", np.zeros(len(data["atom_array"]), dtype=np.int32)
            )  #! SAIR does not have charges
        check_atom_array_annotation(data, ["res_name", "element", "charge", "atom_name"])

        if self.use_element_for_atom_names_of_atomized_tokens:
            check_atom_array_annotation(data, ["atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        # Generate reference features
        reference_features = get_af3_raw_molecule_features(
            atom_array,
            use_element_for_atom_names_of_atomized_tokens=self.use_element_for_atom_names_of_atomized_tokens,
            residue_conformer_indices=None,
            **self.generate_conformers_kwargs,
        )

        # Add reference features to the 'feats' dictionary
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(reference_features)

        return data


class ProteinaLigandTransform(Transform):
    def __init__(self, use_raw_file=False, use_rdkit_from_smiles=False, use_openbabel=False):
        self.use_raw_file = use_raw_file
        self.use_rdkit_from_smiles = use_rdkit_from_smiles
        self.use_openbabel = use_openbabel
        assert not (use_rdkit_from_smiles and use_openbabel), "Cannot use both RDKit and OpenBabel"

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        # check_atom_array_annotation(data, ["is_ligand"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        is_ligand = data["feats"]["is_ligand"]  #!token level
        if not is_ligand.any():
            return data
        # assert is_ligand.any(), "No ligands found"

        atom_mask = data["ground_truth"]["mask_atom_lvl"]
        ligand_atom_mask = is_ligand[data["feats"]["atom_to_token_map"]] * atom_mask
        #! used as SAIR does not have bonds
        ligand_atom_array = atom_array[ligand_atom_mask]

        use_openbabel = self.use_openbabel
        use_rdkit_from_smiles = self.use_rdkit_from_smiles
        use_raw_file = self.use_raw_file  #! this is False for everything but PLINDER
        if use_openbabel:
            obmol = atom_array_to_openbabel(
                ligand_atom_array,
                infer_hydrogens=False,
                infer_aromaticity=False,
                annotations_to_keep=[
                    "chain_id",
                    "res_id",
                    "res_name",
                    "atom_name",
                    "atom_id",
                ],
            )
            obmol.ConnectTheDots()
            obmol.PerceiveBondOrders()
            new_lig = atom_array_from_openbabel(obmol)
            #! has many issues with infering bonds
        elif use_rdkit_from_smiles:
            smi = data["extra_info"]["SMILES"]
            mol = Chem.MolFromSmiles(smi)
            if mol.GetNumAtoms() != ligand_atom_array.shape[0]:
                mol = Chem.RemoveHs(mol)
            if mol.GetNumAtoms() != ligand_atom_array.shape[0]:
                print("\n\n ERROR: RDKit mol and atom array do not match")

            new_lig = atom_array_from_rdkit(mol)
            new_lig.element = np.char.upper(new_lig.element)

            if use_raw_file:
                if new_lig.bonds.as_array().shape[0] != ligand_atom_array.bonds.as_array().shape[0]:
                    print(
                        f"\n\n WARNING: atom array {ligand_atom_array.bonds.as_array().shape[0]} and RDKit mol bonds {new_lig.bonds.as_array().shape[0]} do not match"
                    )
                new_lig = ligand_atom_array
            else:
                try:
                    if not (new_lig.element == ligand_atom_array.element).all():
                        print("\n\n ERROR: atom array and RDKit mol elements do not match")
                except Exception as e:
                    print(e)
                    assert e
            bonds = new_lig.bonds.as_array()
            # bond_mask = np.ones_like(bonds)
            # bond_mask[:, -1] = 0  #! (i, j, bond type where 0 is no bond)
            atom_id_2_index = dict(
                zip(atom_array.atom_id, range(len(atom_array)), strict=False)
            )  # {atom.atom_id: i for i, atom in enumerate(atom_array)}

            for bond in ligand_atom_array.bonds.as_array():  #! clear any bad ones
                atom_id_1, atom_id_2, bond_type = bond
                ligand_atom_array.bonds.remove_bond(atom_id_1, atom_id_2)

            for bond in bonds:  #! changed this logic in ame to fix array out of bounds issue post cropping
                atom_id_1, atom_id_2, bond_type = bond
                atom_1_index = atom_id_2_index[ligand_atom_array.atom_id[atom_id_1]]
                atom_2_index = atom_id_2_index[ligand_atom_array.atom_id[atom_id_2]]
                atom_array.bonds.remove_bond(atom_1_index, atom_2_index)
                atom_array.bonds.add_bond(atom_1_index, atom_2_index, bond_type)
        elif use_raw_file:
            new_lig = ligand_atom_array
        else:
            raise ValueError(
                f"Invalid use_raw_file: {use_raw_file}, use_rdkit_from_smiles: {use_rdkit_from_smiles}, use_openbabel: {use_openbabel}"
            )

        adj = atom_array.bonds.adjacency_matrix()
        adj_ligand = adj[ligand_atom_mask][:, ligand_atom_mask]
        # ligand_atom_mask,sum() = 10 is_ligand.sum() = 12
        pe = get_laplacian_pe(adj_ligand.astype(np.float32))  # adj_ligand.numpy()
        pe_all = torch.zeros(is_ligand.shape[0], 32)
        pe_all[is_ligand] = (
            pe  # RuntimeError: shape mismatch: value tensor of shape [10, 32] cannot be broadcast to indexing result of shape [12, 32]
        )

        data["feats"]["ligand_laplacian_pe"] = pe
        data["feats"]["ligand_laplacian_pe_all"] = pe_all
        return data


def one_hot_encoding(x, num_classes):
    """
    One-hot encoding of a categorical variable.
    """
    return np.eye(num_classes)[x]


def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def build_adj_matrix(
    num_nodes: int,
    s: np.ndarray,
    r: np.ndarray,
    weights: np.ndarray,
):
    adj = np.zeros((num_nodes, num_nodes))
    adj[s, r] = weights
    return adj


def add_self_loops(
    senders: np.ndarray,
    receivers: np.ndarray,
    weights: np.ndarray,
    num_nodes: int,
    fill_value: float = 1.0,
):
    appendix = []
    for i in range(num_nodes):
        appendix.append(i)
    senders = np.concatenate([senders, appendix])
    receivers = np.concatenate([receivers, appendix])
    weights = np.concatenate([weights, fill_value * np.ones(num_nodes)])
    return senders, receivers, weights


def fix_PE_shape(pe, k=32):
    if pe.shape[1] < k:
        pe = np.hstack([pe, np.zeros((pe.shape[0], k - pe.shape[1]))])
    elif pe.shape[1] > k:
        pe = pe[:, :k]
    return pe


def get_laplacian_pe(
    adj_matrix: np.ndarray,
    # num_nodes: int = None,
    # s: np.ndarray = None,
    # r: np.ndarray = None,
    k: int = 32,  # final laplacian_pe dimension
):
    """
    Get the Laplacian positional encodings of a molecule.
    We use the symmetric normalization of laplacian.

    Part of the code is borrowed from torch_geometric.transforms.AddLaplacianEigenvectorPE

    Args:
        num_nodes: number of nodes in the molecule
        s: sender nodes
        r: receiver nodes
        k: final laplacian_pe dimension

    Returns:
        eig_vals: eigenvalues of the laplacian
        pe: positional encodings
    """
    SPARSE_THRESHOLD = 100
    num_nodes = adj_matrix.shape[0]

    if num_nodes <= 3:
        logger.warning(f"num_nodes={num_nodes} <= 3, returning zero positional encodings")
        return torch.zeros(num_nodes, k).float()
    n_lap = min(num_nodes - 1, k)  # for small molecules, we only use the first few eigenvectors
    # edge_weight = np.ones_like(s, dtype=np.float32)
    # A = build_adj_matrix(num_nodes, s, r, edge_weight) # adjacency matrix
    # A = np.asarray(A, dtype=np.float32)
    A = adj_matrix
    D = np.diag(np.sum(A, axis=1)) + 1e-8 * np.eye(num_nodes)  # degree matrix

    # Symmetric normalization laplacian
    # L_sym = I - (D*)^(1/2) @ A @ (D*)^(1/2), D* is pinv(D)
    D_inv_sqrt = np.linalg.pinv(sqrtm(D))

    L = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    L = np.asmatrix(L, dtype=np.float32)
    L = scipy.sparse.coo_matrix(L)

    if num_nodes < SPARSE_THRESHOLD:
        L = L.todense()
        eig_vals, eig_vecs = np.linalg.eig(L)
    else:
        eig_vals, eig_vecs = eigs(L, k=n_lap + 1, which="SR", return_eigenvectors=True)
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = eig_vecs[:, 1 : n_lap + 1]
    pe = np.asarray(pe)
    sign = -1 + 2 * np.random.randint(0, 2, (n_lap,))
    pe *= sign

    # pad or truncate to the final dimension
    pe = fix_PE_shape(pe, k)

    return torch.from_numpy(pe).float()


class EncodeAF3TokenLevelFeatures(Transform):
    """
    A transform that encodes token-level features like AF3. The token-level features are returned as:

    - feats:
        # (Standard AF3 token-level features)
        - `residue_index`: Residue number in the token's original input chain (pre-crop)
        - `token_index`: Token number. Increases monotonically; does not restart at 1 for new
            chains. (Runs from 0 to N_tokens)
        - `asym_id`: Unique integer for each distinct chain (pn_unit_iid)
            NOTE: We use `pn_unit_iid` rather than `chain_iid` to be more consistent
            with handling of multi-residue/multi-chain ligands (especially sugars)
        - `entity_id`: Unique integer for each distinct sequence (pn_unit entity)
        - `sym_id`: Unique integer within chains of this sequence. E.g. if pn_units A, B and C
            share a sequence but D does not, their `sym_id`s would be [0, 1, 2, 0].
        - `restype`: Integer encoding of the sequence. 32 possible values: 20 AA + unknown,
            4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap. Ligands are
            represented as unknown amino acid (`UNK`)
        - `is_protein`: whether a token is of protein type
        - `is_rna`: whether a token is of RNA type
        - `is_dna`: whether a token is of DNA type
        - `is_ligand`: whether a token is a ligand residue

        # (Custom token-level features)
        - `is_atomized`: whether a token is an atomized token

    - feat_metadata:
        - `asym_name`: The asymmetric unit name for each id in `asym_id`. Acts as a legend.
        - `entity_name`: The entity name for each id in `entity_id`. Acts as a legend.
        - `sym_name`: The symmetric unit name for each id in `sym_id`. Acts as a legend.

    Reference:
        - Section 2.8 of the AF3 supplementary (Table 5)
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """

    def __init__(self, sequence_encoding: AF3SequenceEncoding):
        self.sequence_encoding = sequence_encoding

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data,
            [
                "atomize",
                "pn_unit_iid",
                "chain_entity",
                "res_name",
                "within_chain_res_idx",
            ],
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        # ... get token-level array
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]

        # ... identifier tokens
        # ... (residue)
        residue_index = token_level_array.within_chain_res_idx
        # ... (token)
        token_index = np.arange(len(token_starts))
        # ... (chain instance)
        asym_name, asym_id = np.unique(token_level_array.pn_unit_iid, return_inverse=True)
        # ... (chain entity)
        entity_name, entity_id = np.unique(token_level_array.pn_unit_entity, return_inverse=True)
        # ... (within chain entity)
        sym_name, sym_id = get_within_entity_idx(token_level_array, level="pn_unit")

        # ... sequence tokens
        restype = self.sequence_encoding.encode(token_level_array.res_name)

        # HACK: MSA transformations rely on the encoded query sequence being stored in "encoded/seq"
        # We could consider finding a consistent place to store the encoded query sequence across RF2AA and AF3 (e.g., "encoded" vs. "feats/restype")
        data["encoded"] = {"seq": restype}

        # ...one-hot encode the restype (NOTE: We one-hot encode here, since we have access to the sequence encoding object)
        restype = F.one_hot(torch.tensor(restype), num_classes=self.sequence_encoding.n_tokens).numpy()

        # ... molecule type
        _aa_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_aa_like]
        is_protein = np.isin(token_level_array.res_name, _aa_like_res_names)

        _rna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_rna_like]
        is_rna = np.isin(token_level_array.res_name, _rna_like_res_names)

        _dna_like_res_names = self.sequence_encoding.all_res_names[self.sequence_encoding.is_dna_like]
        is_dna = np.isin(token_level_array.res_name, _dna_like_res_names)

        is_ligand = ~(is_protein | is_rna | is_dna)
        if ~is_ligand.any():
            is_protein = is_protein & token_level_array.is_polymer
            is_ligand = ~(is_protein | is_rna | is_dna)
            print("\n\n WARNING: No ligands found due to residue name of ligand so looking at 'is_polymer'")

        # ... add to data dict
        if "feats" not in data:
            data["feats"] = {}
        if "feat_metadata" not in data:
            data["feat_metadata"] = {}

        # ... add to data dict
        data["feats"] |= {
            "residue_index": residue_index,  # (N_tokens) (int)
            "token_index": token_index,  # (N_tokens) (int)
            "asym_id": asym_id,  # (N_tokens) (int)
            "entity_id": entity_id,  # (N_tokens) (int)
            "sym_id": sym_id,  # (N_tokens) (int)
            "restype": restype,  # (N_tokens, 32) (float, one-hot)
            "is_protein": is_protein,  # (N_tokens) (bool)
            "is_rna": is_rna,  # (N_tokens) (bool)
            "is_dna": is_dna,  # (N_tokens) (bool)
            "is_ligand": is_ligand,  # (N_tokens) (bool)
            "is_atomized": token_level_array.atomize,  # (N_tokens) (bool)
        }
        data["feat_metadata"] |= {
            "asym_name": asym_name,  # (N_asyms)
            "entity_name": entity_name,  # (N_entities)
            "sym_name": sym_name,  # (N_entities)
        }

        return data
