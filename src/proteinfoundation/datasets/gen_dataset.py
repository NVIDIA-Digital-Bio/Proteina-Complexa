import os
from collections.abc import Callable, Iterable
from typing import Literal

import atomworks
import biotite
import numpy as np
import torch
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from loguru import logger
from rdkit import Chem
from torch.nn import functional as F
from torch.utils.data import Dataset

from proteinfoundation.datasets.atomworks_ligand_transforms import get_af3_raw_molecule_features, get_laplacian_pe
from proteinfoundation.nn.feature_factory.feature_utils import BOND_ORDER_MAP
from proteinfoundation.utils.motif_utils import parse_motif, save_motif_csv
from proteinfoundation.utils.pdb_utils import load_target_from_pdb


def UniformInt(low: int, high: int | None, nsamples: int = 1, endpoint: bool = False) -> np.ndarray:
    """
    Generates an array of integers sampled uniformly from [low, high) if endpoint is False,
    or [low, high] if endpoint is True. If high is None or equal to low, returns fixed array.

    Args:
        low (int): The lowest integer to be drawn from the distribution (inclusive).
        high (int, optional): The highest integer (exclusive unless endpoint=True).
            If None, uses low for fixed length.
        nsamples (int): Number of samples to draw.
        endpoint (bool, optional): If True, 'high' is inclusive. Defaults to False.

    Returns:
        np.ndarray: Array of shape (nsamples,) containing integers in the specified range.
    """
    if high is None or high == low:
        return np.full(nsamples, low)
    return np.random.randint(low, high + endpoint, nsamples)


class ConditionalFeature:
    """
    Base class for conditional features used in GenDataset.

    This class provides an interface for defining conditional features that can be
    attached to each sample in the dataset. Subclasses should implement the `setup`
    and `__call__` methods to provide custom behavior.

    Methods:
        setup(nres: List[int], nrepeat_per_sample: int):
            Optional setup method called before dataset iteration. Can be used to
            precompute or cache any data needed for the feature.

        __call__(result: Dict, sample_idx: int) -> Dict:
            Method to augment or modify the sample dictionary for a given index.
            Should return the updated result dictionary.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}({self.args}, {self.kwargs})"

    def setup(self, nres: list[int]):
        pass

    def __call__(self, result: dict, sample_idx: int):
        pass


class GenDataset(Dataset):
    """
    PyTorch Dataset for generating protein design samples, supporting optional conditional features.

    Args:
        nres (Optional[Iterable[int]], optional):
            List of residue lengths.
        nrepeat_per_sample (int, optional):
            Number of times to repeat each sample (default: 1).
        conditional_features (Optional[List[ConditionalFeature]], optional):
            List of conditional feature objects to add conditional features.
            Conditional features will be called in the order they are added.
        transforms: Optional[Callable, Iterable[Callable]], optional):
            Transforms to apply to the sample. Can be a single transform or an iterable of transforms.
        task_name: Optional[str], optional):
            Name of the task.
    """

    def __init__(
        self,
        nres: Iterable[int] | None = None,
        nrepeat_per_sample: int = 1,
        conditional_features: list[ConditionalFeature] | None = None,
        transforms: Callable | Iterable[Callable] | None = None,
        task_name: str | None = None,
    ):
        super().__init__()
        self.nres = list(int(x) for x in nres) if nres is not None else []
        self.nrepeat_per_sample = nrepeat_per_sample
        self.conditional_features = conditional_features
        # Simple compose: chain transforms sequentially
        if transforms is not None:
            if callable(transforms) and not isinstance(transforms, (list, tuple)):
                self.transforms = transforms
            else:
                transforms_list = list(transforms)

                def compose_fn(data):
                    for t in transforms_list:
                        data = t(data)
                    return data

                self.transforms = compose_fn
        else:
            self.transforms = None

        # set up conditional features
        if self.conditional_features is not None:
            for conditional_feature_func in self.conditional_features:
                conditional_feature_func.setup(self.nres)

        assert len(self.nres) > 0, (
            "nres must be provided either in the constructor or extended by the conditional features"
        )
        self.task_name = task_name

    def __repr__(self):
        return f"GenDataset(nres={self.nres}, nrepeat_per_sample={self.nrepeat_per_sample}, conditional_features={self.conditional_features})"

    def __len__(self):
        return len(self.nres) * self.nrepeat_per_sample

    def __getitem__(self, index: int):
        sample_idx = index // self.nrepeat_per_sample
        result = {
            "nres": self.nres[sample_idx],
            "mask": torch.ones(self.nres[sample_idx], dtype=torch.bool),
        }
        if self.conditional_features is not None:
            for conditional_feature_func in self.conditional_features:
                result = conditional_feature_func(result, sample_idx)

        if self.transforms is not None:
            result = self.transforms(result)
        return result


class CathCodes(ConditionalFeature):
    """
    ConditionalFeature subclass for assigning CATH codes to dataset samples.

    Args:
        cath_codes (List[str]): List of CATH codes to assign.

    Methods:
        setup(nres: List[int], nsamples: int):
            Validates that the number of CATH codes matches the number of lengths.

        __call__(result: Dict, sample_idx: int) -> Dict:
            Adds the appropriate CATH code to the result dictionary for the given sample index.
    """

    def __init__(self, cath_codes: list[str]):
        super().__init__()
        self.cath_codes = cath_codes

    def __repr__(self):
        return f"CathCodes(cath_codes={self.cath_codes})"

    def setup(self, nres: list[int]):
        assert len(nres) == len(self.cath_codes), (
            f"Number of nres ({len(nres)}) must match number of CATH codes ({len(self.cath_codes)})"
        )

    def __call__(self, result: dict, sample_idx: int):
        assert "cath_code" not in result, "cath_code already exists"
        result["cath_code"] = self.cath_codes[sample_idx]
        return result


class MotifFeatures(ConditionalFeature):
    """
    ConditionalFeature for adding motif-related features to dataset samples.

    Two padding modes (indexed vs unindexed) control how the motif is
    returned, and two specification modes (contig string vs atom spec) control
    how motif atoms are selected from the PDB.

    **Indexed** (``padding=True``):
        The motif is embedded in a scaffold whose total length varies per
        sample.  ``MotifFeatures`` takes ownership of ``nres`` (which must
        be empty) and populates it with ``nsamples`` random scaffold
        placements.  Output tensors include scaffold zero-padding so that
        motif positions are indexed within a longer sequence.

        Requires ``contig_string``, ``nsamples``, ``min_length``,
        ``max_length``, ``segment_order``, and ``atom_selection_mode``.
        This is the standard mode for all-atom / tip-atom motif scaffolding
        (see ``motif_dict.yaml``).

    **Unindexed** (``padding=False``, default):
        Output tensors contain only motif atoms — no scaffold padding.
        Binder lengths come from an external ``UniformInt`` via ``nres``.
        Two specification sub-modes are available:

        - *Contig string*: uses the same ``contig_string`` / ``nsamples`` /
          ``min_length`` / ``max_length`` / ``segment_order`` /
          ``atom_selection_mode`` inputs as indexed mode, but the scaffold
          padding is stripped after generation, returning only the motif
          residues.

        - *Atom spec*: directly specifies which atoms to extract per
          residue (e.g. ``"B64: [O, C]; B86: [CB, CA, N, C]"``).  Used by
          AME tasks where arbitrary atoms per residue must be specified.
          Only ``task_name``, ``pdb_path``, and ``motif_atom_spec`` are
          needed.  Produces a single deterministic sample that is
          replicated to match ``len(nres)``.

    Args:
        task_name: Name of the motif task.
        pdb_path: Path to the motif PDB file.
        atom_selection_mode: Which atoms to keep per residue
            (``"ca"``, ``"bb3o"``, ``"all_atom"``, ``"tip_atoms"``).
            Used in contig-string mode; ignored for atom-spec.
        nsamples: How many random scaffold placements to generate (contig
            mode).  For indexed mode this determines the number of samples;
            for unindexed contig mode this controls how many variants are
            produced before padding is stripped.  Ignored for atom-spec.
        contig_string: Contig string specifying motif segments and scaffold
            ranges (e.g. ``"10-40/A163-181/10-40"``).
        motif_only: If ``True``, use only the motif chains from the PDB
            (contig mode).
        motif_atom_spec: Atom-level specification string for AME tasks
            (e.g. ``"B64: [O, C]; B86: [CB, CA, N, C]"``).  Mutually
            exclusive with ``contig_string``.
        min_length: Minimum total length for scaffold placement (contig mode).
        max_length: Maximum total length for scaffold placement (contig mode).
        segment_order: Order of motif segments (contig mode).
        motif_csv_path: Optional path to save motif placement CSV (contig mode).
        padding: ``True`` for indexed scaffolding, ``False`` (default) for
            unindexed output.
    """

    def __init__(
        self,
        task_name: str,
        pdb_path: str,
        atom_selection_mode: Literal["ca", "bb3o", "all_atom", "tip_atoms"] = "ca",
        nsamples: int = 1,
        contig_string: str | None = None,
        motif_only: bool = False,
        motif_atom_spec: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        segment_order: str | None = None,
        motif_csv_path: str | None = None,
        padding: bool = False,
    ):
        super().__init__()
        if contig_string is None and motif_atom_spec is None:
            raise ValueError("Exactly one of contig_string or motif_atom_spec must be provided")
        if contig_string is not None and motif_atom_spec is not None:
            raise ValueError("contig_string and motif_atom_spec are mutually exclusive")

        self.task_name = task_name
        self.pdb_path = pdb_path
        self.atom_selection_mode = atom_selection_mode
        self.nsamples = nsamples
        self.contig_string = contig_string
        self.motif_only = motif_only
        self.motif_atom_spec = motif_atom_spec
        self.min_length = min_length
        self.max_length = max_length
        self.segment_order = segment_order
        self.motif_csv_path = motif_csv_path
        self.padding = padding

    def __repr__(self):
        if self.motif_atom_spec is not None:
            spec = f"motif_atom_spec={self.motif_atom_spec!r}"
        else:
            spec = (
                f"contig_string={self.contig_string!r}, "
                f"nsamples={self.nsamples!r}, "
                f"atom_selection_mode={self.atom_selection_mode!r}"
            )
        return (
            f"MotifFeatures(task_name={self.task_name!r}, pdb_path={self.pdb_path!r}, padding={self.padding!r}, {spec})"
        )

    def setup(self, nres: list[int]):
        (
            self.motif_masks,
            self.x_motifs,
            self.residue_types,
        ) = self._generate_motif_info()

        if self.padding:
            # Indexed: MotifFeatures owns nres — populate from scaffold lengths
            assert len(nres) == 0, (
                "nres must be empty for indexed mode (padding=True) "
                "because MotifFeatures populates it from scaffold placements"
            )
            for x_motif in self.x_motifs:
                nres.append(int(x_motif.shape[0]))
        elif self.contig_string is not None:
            # Unindexed + contig: strip scaffold padding, keep only motif residues
            for idx, motif_mask in enumerate(self.motif_masks):
                res_mask = motif_mask.sum(dim=-1).bool()
                self.x_motifs[idx] = self.x_motifs[idx][res_mask]
                self.residue_types[idx] = self.residue_types[idx][res_mask]
                self.motif_masks[idx] = motif_mask[res_mask]
            n_needed = len(nres)
            if len(self.x_motifs) < n_needed:
                self.x_motifs = (self.x_motifs * n_needed)[:n_needed]
                self.residue_types = (self.residue_types * n_needed)[:n_needed]
                self.motif_masks = (self.motif_masks * n_needed)[:n_needed]
        else:
            # Unindexed + atom-spec: already bare motif, replicate to match nres
            n_needed = len(nres)
            if len(self.x_motifs) == 1 and n_needed > 1:
                self.x_motifs = self.x_motifs * n_needed
                self.residue_types = self.residue_types * n_needed
                self.motif_masks = self.motif_masks * n_needed

    def _generate_motif_info(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate motif information for the current motif configuration.

        This method extracts motif masks, coordinates, and residue types from the specified motif PDB file
        and contig string or atom specification. It sorts motifs by length, centers them to the origin,
        and optionally saves motif information to a CSV file.

        Returns:
            motif_masks (List[torch.Tensor]): List of motif atom masks (n_res, 37).
            x_motifs (List[torch.Tensor]): List of motif coordinates (n_res, 37, 3).
            residue_types (List[torch.Tensor]): List of motif residue types (n_res).
        """
        lengths, motif_masks, x_motifs, residue_types, outstrs = parse_motif(
            motif_pdb_path=self.pdb_path,
            contig_string=self.contig_string,
            nsamples=self.nsamples,
            motif_only=self.motif_only,
            motif_min_length=self.min_length,
            motif_max_length=self.max_length,
            segment_order=self.segment_order,
            motif_atom_spec=self.motif_atom_spec,
            atom_selection_mode=self.atom_selection_mode,
        )
        idx = np.argsort(lengths)
        motif_masks = [motif_masks[i] for i in idx]
        x_motifs = [x_motifs[i] for i in idx]
        residue_types = [residue_types[i] for i in idx]

        # Only save CSV for contig_string (residue/range) case
        if self.motif_atom_spec is None and self.motif_csv_path is not None:
            outstrs = [outstrs[i] for i in idx]
            save_motif_csv(
                self.pdb_path,
                self.task_name,
                outstrs,
                outpath=self.motif_csv_path,
                segment_order=self.segment_order,
            )
        return motif_masks, x_motifs, residue_types

    def __call__(self, result: dict, sample_idx: int):
        for key in ["motif_mask", "x_motif", "seq_motif_mask", "seq_motif"]:
            assert key not in result, f"{key} already exists"
        result["motif_mask"] = self.motif_masks[sample_idx].bool().clone()
        # we use the motif mask for uidx and idx mode
        result["seq_motif_mask"] = self.motif_masks[sample_idx].sum(dim=-1).bool()
        result["x_motif"] = self.x_motifs[sample_idx].clone()
        result["seq_motif"] = self.residue_types[sample_idx].clone()
        return result


class TargetFeatures(ConditionalFeature):
    """
    ConditionalFeature for generating target structural features for binder design.

    Args:
        task_name (str): Name of the target task.
        pdb_path (str): Path to the target PDB file.
        input_spec (str): Input string specifying the target protein regions.
        target_hotspots (Optional[List[int]], optional): List of hotspot residue indices. Default: None.
        binder_center (Optional[List[float]], optional): Coordinates [x, y, z] for binder center in nanometers. Default: None.
        pdb_id (Optional[str], optional): PDB ID of the target protein. Default: None.

    Methods:
        setup(nres: List[int]):
            Generates target structural features for binder design.

        _generate_target_info(nres: List[int]):
            Internal method to generate target structural features for binder design.

    Notes:
        - All tensors are in nanometers.
        - total_len = target_length if binder_gen_only else target_length + binder_length.
        - For each binder length in nres, one set of tensors is generated.
    """

    def __init__(
        self,
        task_name: str,
        binder_gen_only: bool,
        pdb_path: str,
        input_spec: str,
        target_hotspots: list[int] | None = None,
        binder_center: list[float] | None = None,
        pdb_id: str | None = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.binder_gen_only = binder_gen_only
        self.input_spec = input_spec

        # Validate pdb_path exists with helpful error message
        if not os.path.exists(pdb_path):
            data_path = os.environ.get("DATA_PATH", "<not set>")
            # Use opt(raw=True) to preserve newlines in multi-line output
            logger.opt(raw=True).error(f"""
{"=" * 70}
ERROR: Target PDB file not found!
{"=" * 70}
  Task name:    {task_name}
  PDB path:     {pdb_path}
  DATA_PATH:    {data_path}
{"=" * 70}
Possible causes:
  1. DATA_PATH environment variable is not set correctly
     - Current value: {data_path}
     - Set it with: export DATA_PATH=/path/to/data
  2. The target PDB file does not exist at the expected location
  3. The target config has incorrect 'source' or 'target_filename'

To fix:
  - Check your .env file or export DATA_PATH
  - Verify the target exists: ls -la {pdb_path}
  - Run 'complexa download' to download target data
{"=" * 70}
""")
            raise FileNotFoundError(f"Target PDB not found: {pdb_path}")

        self.pdb_path = pdb_path
        self.target_hotspots = target_hotspots
        self.binder_center = torch.tensor(binder_center).reshape(1, 3) if binder_center is not None else None
        self.pdb_id = pdb_id

    def __repr__(self):
        return (
            f"TargetFeatures("
            f"task_name={self.task_name!r}, "
            f"binder_gen_only={self.binder_gen_only!r}, "
            f"pdb_path={self.pdb_path!r}, "
            f"input_spec={self.input_spec!r}, "
            f"target_hotspots={self.target_hotspots!r}, "
            f"binder_center={self.binder_center!r}, "
            f"pdb_id={self.pdb_id!r})"
        )

    def setup(self, nres: list[int]):
        (
            self.target_structures,
            self.target_masks,
            self.target_chains,
            self.target_residue_types,
            self.target_hotspots_masks,
            self.binder_centers,
        ) = self._generate_target_info(nres)

    def _generate_target_info(
        self, nres: list[int]
    ) -> tuple[
        list[torch.Tensor],  # target_structures
        list[torch.Tensor],  # target_masks
        list[torch.Tensor],  # target_chains
        list[torch.Tensor],  # target_residue_types
        list[torch.Tensor],  # target_hotspots_masks
        list[torch.Tensor],  # binder_centers
    ]:
        """
        Generate target structural features for binder design.

        This method extracts and processes the target protein structure and generates
        tensors required for binder design, including structure coordinates, masks,
        residue types, and (optionally) hotspot and binder center information for each
        requested binder length.

        Args:
            nres: List of integers specifying binder lengths to generate.

        Returns:
            Tuple containing:
                - target_structures: List of [total_len, 37, 3] tensors (target only if binder_gen_only else target + binder).
                - target_masks: List of [total_len, 37] boolean tensors.
                - target_chains: List of [total_len] integer tensors.
                - target_residue_types: List of [total_len] integer tensors.
                - target_hotspots_masks: List of [total_len] boolean tensors, or None.
                - binder_centers: List of [1, 3] tensors, or None.
        """
        # Extract, center and convert target structure
        (
            target_mask,
            target_structure,
            target_residue_type,
            target_hotspots_mask,
            target_chain,
        ) = load_target_from_pdb(
            self.input_spec,
            self.pdb_path,
            self.target_hotspots,
        )

        target_len = target_structure.shape[0]
        assert target_hotspots_mask.shape[0] == target_len, (
            "target_hotspots_mask and target_structure have different lengths"
        )

        # Initialize output lists
        target_masks = []
        target_structures = []
        target_residue_types = []
        target_hotspots_masks = [] if self.target_hotspots is not None else None
        target_chains = []
        binder_centers = [] if self.binder_center is not None else None

        # Process each binder length and sample count
        for binder_len in nres:
            if self.binder_gen_only:
                total_len = target_len
            else:
                total_len = target_len + binder_len

            target_structure_extended = torch.zeros((total_len, 37, 3), dtype=torch.float32)
            target_structure_extended[:target_len] = target_structure

            target_mask_extended = torch.zeros((total_len, 37), dtype=torch.bool)
            target_mask_extended[:target_len] = target_mask

            target_residue_type_extended = torch.zeros((total_len,), dtype=torch.long)
            target_residue_type_extended[:target_len] = target_residue_type

            n_chains = target_chain.max() + 1
            target_chain_extended = torch.ones((total_len,), dtype=torch.long) * n_chains
            target_chain_extended[:target_len] = target_chain

            target_structures.append(target_structure_extended)
            target_masks.append(target_mask_extended)
            target_residue_types.append(target_residue_type_extended)
            target_chains.append(target_chain_extended)

            if self.target_hotspots is not None:
                target_hotspots_masks.append(target_hotspots_mask)
            if self.binder_center is not None:
                binder_centers.append(self.binder_center)

        return (
            target_structures,
            target_masks,
            target_chains,
            target_residue_types,
            target_hotspots_masks,
            binder_centers,
        )

    def __call__(self, result: dict, sample_idx: int):
        nres = result["nres"]
        for key in [
            "target_mask",
            "x_target",
            "seq_target_mask",
            "seq_target",
            "target_chains",
            "chains",
            "target_hotspot_mask",
            "binder_center",
            "prepend_target",
            "atomistic_target",
        ]:
            assert key not in result, f"{key} already exists"
        result["target_mask"] = self.target_masks[sample_idx].bool().clone()  # [num_res, 37]
        result["x_target"] = self.target_structures[sample_idx].clone()  # [num_res, 37, 3]
        result["seq_target_mask"] = self.target_masks[sample_idx].sum(dim=-1).bool()  # [num_res]
        result["seq_target"] = self.target_residue_types[sample_idx].clone()  # [num_res]

        result["target_chains"] = self.target_chains[sample_idx].clone()  # [num_res]
        result["chains"] = (self.target_chains[sample_idx].max(dim=-1, keepdim=True).values + 1).repeat(nres)
        if self.target_hotspots is not None and self.target_hotspots_masks is not None:
            result["target_hotspot_mask"] = self.target_hotspots_masks[sample_idx].clone()
        if self.binder_center is not None and self.binder_centers is not None:
            result["binder_center"] = self.binder_centers[sample_idx].clone()
        result["prepend_target"] = self.binder_gen_only  # give atomistic if the target  has this then set to True
        result["atomistic_target"] = (
            False  # hasattr(self, "target_laplacian_pes") and self.target_laplacian_pes[index] is not None # whether to prepend target to the generated sequence
        )
        return result


class LigandFeatures(ConditionalFeature):
    """
    ConditionalFeature for generating ligand features for the target structure.

    Supports both single-ligand (standard ligand binder) and multi-ligand (AME)
    configurations. When multiple ligands are specified via ``ligand`` as a
    list, features are extracted per-ligand and merged with block-diagonal bond
    matrices so that intra-ligand bonds are preserved while inter-ligand bonds
    remain zero.

    Args:
        task_name (str): Name of the ligand task.
        pdb_path (str): Path to the target PDB file containing the ligand(s).
        ligand: Residue name(s) of the ligand(s). A single string for one
            ligand, or a list of strings for multiple ligands in the same PDB
            (e.g. ``["DHZ", "ZN"]`` for AME). Ignored when ``ligand_only=True``.
        ligand_only (bool): Whether to use the entire file as the ligand
            (True) or extract specific residues by name (False).
        SMILES: SMILES string for bond regeneration via RDKit.  Required when
            ``use_bonds_from_file=False`` and there is a single ligand.
            Not supported for multi-ligand (must use bonds from file).
        use_bonds_from_file (bool): Whether to use bonds from the file.
        pdb_id: PDB ID of the target protein (optional, for logging only).

    Notes:
        - If use_bonds_from_file is True, the bonds are extracted from the file.
        - If use_bonds_from_file is False, the bonds are regenerated from the
          SMILES string using RDKit (single-ligand only).
        - The ligand can be extracted by residue name or kept as the full complex
          depending on 'ligand_only'.
    """

    def __init__(
        self,
        task_name: str,
        pdb_path: str,
        ligand: str | list[str] | None,
        ligand_only: bool = False,
        SMILES: str | None = None,
        use_bonds_from_file: bool = True,
        pdb_id: str | None = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.pdb_id = pdb_id
        self.ligand_res_names = ligand
        self.ligand_only = ligand_only
        self.SMILES = SMILES
        self.use_bonds_from_file = use_bonds_from_file

        # Normalize ligand to a list for uniform handling
        if isinstance(ligand, str):
            self._res_names = [ligand]
        elif isinstance(ligand, (list, tuple)):
            self._res_names = list(ligand)
        else:
            self._res_names = []

        if not self._res_names and not ligand_only:
            raise ValueError(
                f"Task '{task_name}': ligand is null/empty but ligand_only=False. "
                "When no residue names are provided, ligand_only must be True "
                "so the entire file is used as the ligand."
            )

        if len(self._res_names) > 1 and not use_bonds_from_file:
            raise ValueError(
                "Multi-ligand mode requires use_bonds_from_file=True. "
                "SMILES-based bond regeneration is only supported for a single ligand."
            )

        # Validate pdb_path exists with helpful error message
        if not os.path.exists(pdb_path):
            data_path = os.environ.get("DATA_PATH", "<not set>")
            logger.opt(raw=True).error(f"""
{"=" * 70}
ERROR: Ligand target PDB/CIF file not found!
{"=" * 70}
  Task name:    {task_name}
  PDB path:     {pdb_path}
  PDB ID:       {pdb_id}
  Ligand:       {ligand}
  DATA_PATH:    {data_path}
{"=" * 70}
Possible causes:
  1. DATA_PATH environment variable is not set correctly
     - Current value: {data_path}
     - Set it with: export DATA_PATH=/path/to/data
  2. The ligand target file does not exist at the expected location
  3. The target config has incorrect 'source' or 'target_filename'

To fix:
  - Check your .env file or export DATA_PATH
  - Verify the target exists: ls -la {pdb_path}
  - Run 'complexa download' to download target data
{"=" * 70}
""")
            raise FileNotFoundError(f"Ligand target not found: {pdb_path}")

        self.pdb_path = pdb_path

    def setup(self, nres: list[int]):
        (
            self.target_structures,
            self.target_masks,
            self.target_residue_types,
            self.target_charges,
            self.target_bond_orders,
            self.target_laplacian_pes,
            self.target_bond_masks,
            self.target_atom_names,
            self.ligand,
        ) = self._generate_ligand_info(nres)

    def _extract_ligand_atoms(self, pl_complex: biotite.structure.AtomArray) -> list[biotite.structure.AtomArray]:
        """Extract one or more ligand atom groups from a loaded complex.

        Returns a list of AtomArrays — one per ligand residue name, or
        a single-element list containing the whole complex when ligand_only=True.
        """
        if self.ligand_only:
            return [pl_complex]

        groups = []
        for rn in self._res_names:
            lig = pl_complex[pl_complex.res_name == rn]
            if len(set(lig.chain_id)) > 1:
                lig = lig[lig.chain_id == lig.chain_id[0]]
            groups.append(lig)
        return groups

    def _compute_features_for_ligand(self, ligand: biotite.structure.AtomArray) -> dict[str, torch.Tensor]:
        """Compute structural features for a single ligand AtomArray.

        Coordinates are returned in nanometers. Bond information is read from the
        file or regenerated from SMILES depending on ``use_bonds_from_file``.

        Returns dict with keys: structure, mask, seq, charge, bond_order,
        laplacian_pe, bond_mask, atom_name.
        """
        structure = torch.from_numpy(ligand.coord) / 10.0

        if "charge" not in ligand.get_annotation_categories():
            ligand.set_annotation("charge", np.zeros(len(ligand), dtype=np.int32))

        if self.use_bonds_from_file:
            bonds = ligand.bonds
            adj = bonds.adjacency_matrix()
        else:
            mol = Chem.MolFromSmiles(self.SMILES)
            if mol.GetNumAtoms() != ligand.shape[0]:
                mol = Chem.RemoveHs(mol)
            if mol.GetNumAtoms() != ligand.shape[0]:
                logger.warning("RDKit mol and atom array do not match")
            new_lig = atom_array_from_rdkit(mol)
            new_lig.element = np.char.upper(new_lig.element)
            try:
                if not (new_lig.element == ligand.element).all():
                    logger.warning("atom array and RDKit mol elements do not match")
            except Exception as e:
                logger.warning(e)
            bonds = new_lig.bonds
            adj = bonds.adjacency_matrix()

        ligand_feats = get_af3_raw_molecule_features(ligand)
        pe = get_laplacian_pe(adj.astype(np.float32))
        bm = bonds.bond_type_matrix()
        bm = np.vectorize(BOND_ORDER_MAP.get)(bm)

        return {
            "structure": structure,
            "mask": torch.ones(len(ligand), dtype=torch.bool),
            "seq": F.one_hot(torch.from_numpy(ligand_feats["atom_element"]).long(), num_classes=128),
            "charge": torch.from_numpy(ligand.charge).float(),
            "bond_order": torch.from_numpy(bm).float(),
            "laplacian_pe": pe,
            "bond_mask": torch.from_numpy(adj),
            "atom_name": F.one_hot(torch.from_numpy(ligand_feats["atom_name_chars"]).long(), num_classes=64).reshape(
                len(ligand), 64 * 4
            ),
        }

    @staticmethod
    def _build_block_diagonal(matrices: list[torch.Tensor]) -> torch.Tensor:
        """Combine square matrices into a single block-diagonal matrix.

        Used to merge bond-order / bond-mask matrices from multiple ligands so
        that intra-ligand bonds are preserved and inter-ligand entries are zero.
        """
        total = sum(m.shape[0] for m in matrices)
        result = torch.zeros(total, total, dtype=matrices[0].dtype)
        offset = 0
        for m in matrices:
            n = m.shape[0]
            result[offset : offset + n, offset : offset + n] = m
            offset += n
        return result

    def _generate_ligand_info(
        self, nres: list[int]
    ) -> tuple[
        list[torch.Tensor],  # target_structures
        list[torch.Tensor],  # target_masks
        list[torch.Tensor],  # target_residue_types
        list[torch.Tensor],  # target_charges
        list[torch.Tensor],  # target_bond_orders
        list[torch.Tensor],  # target_laplacian_pes
        list[torch.Tensor],  # target_bond_masks
        list[torch.Tensor],  # target_atom_names
        biotite.structure.AtomArray,  # ligand
    ]:
        """Generate ligand features for the target structure.

        Handles both single-ligand and multi-ligand (AME) cases:
        - Single ligand: features computed directly.
        - Multiple ligands: features computed per-ligand, then merged with
          block-diagonal bond matrices and concatenated atom-level tensors.

        Args:
            nres: List of binder lengths (one set of features is replicated per length).

        Returns:
            Tuple of per-sample lists (one entry per binder length) plus the
            merged biotite AtomArray for the ligand(s).
        """
        pl_complex = atomworks.io.utils.io_utils.load_any(self.pdb_path)[0]

        # --- Extract and compute per-ligand features ---
        ligand_groups = self._extract_ligand_atoms(pl_complex)
        per_ligand = [self._compute_features_for_ligand(lig) for lig in ligand_groups]

        # --- Merge if multiple ligands (AME) ---
        if len(per_ligand) == 1:
            feats = per_ligand[0]
            merged_ligand = ligand_groups[0]
        else:
            concat_keys = [
                "structure",
                "mask",
                "seq",
                "charge",
                "laplacian_pe",
                "atom_name",
            ]
            feats = {k: torch.cat([f[k] for f in per_ligand], dim=0) for k in concat_keys}
            feats["bond_order"] = self._build_block_diagonal([f["bond_order"] for f in per_ligand])
            feats["bond_mask"] = self._build_block_diagonal([f["bond_mask"] for f in per_ligand]).bool()
            merged_ligand = biotite.structure.concatenate(ligand_groups)

        # Save centered PDB for single-ligand (backwards compatibility)
        # if len(ligand_groups) == 1 and "centered" not in self.pdb_path:
        #     ligand_path = self.pdb_path.replace(".pdb", "_ligand_centered.pdb")
        #     if not os.path.exists(ligand_path):
        #         biotite.structure.io.save_structure(ligand_path, ligand_groups[0])
        #     logger.info(f"Saved ligand {ligand_groups[0].shape} to {ligand_path}")

        # --- Replicate for each binder length ---
        target_structures = []
        target_masks = []
        target_residue_types = []
        target_charges = []
        target_bond_orders = []
        target_laplacian_pes = []
        target_bond_masks = []
        target_atom_names = []

        for _binder_len in nres:
            target_structures.append(feats["structure"].clone())
            target_masks.append(feats["mask"].clone())
            target_residue_types.append(feats["seq"].clone())
            target_charges.append(feats["charge"].clone())
            target_bond_orders.append(feats["bond_order"].clone())
            target_laplacian_pes.append(feats["laplacian_pe"].clone())
            target_bond_masks.append(feats["bond_mask"].clone())
            target_atom_names.append(feats["atom_name"].clone())

        return (
            target_structures,
            target_masks,
            target_residue_types,
            target_charges,
            target_bond_orders,
            target_laplacian_pes,
            target_bond_masks,
            target_atom_names,
            merged_ligand,
        )

    def __call__(self, result: dict, sample_idx: int):
        for key in [
            "target_mask",
            "x_target",
            "seq_target_mask",
            "seq_target",
            "target_charge",
            "target_bond_order",
            "target_laplacian_pe",
            "target_bond_mask",
            "target_atom_name",
            "prepend_target",
            "atomistic_target",
        ]:
            assert key not in result, f"{key} already exists"
        result["target_mask"] = self.target_masks[sample_idx].bool().clone()
        result["x_target"] = self.target_structures[sample_idx].clone()
        result["seq_target_mask"] = self.target_masks[sample_idx].sum(dim=-1).bool()
        result["seq_target"] = self.target_residue_types[sample_idx].clone()
        result["target_charge"] = self.target_charges[sample_idx].clone()
        result["target_bond_order"] = self.target_bond_orders[sample_idx].clone()
        result["target_laplacian_pe"] = self.target_laplacian_pes[sample_idx].clone()
        result["target_bond_mask"] = self.target_bond_masks[sample_idx].clone()
        result["target_atom_name"] = self.target_atom_names[sample_idx].clone()
        result["prepend_target"] = self.ligand_only  # give atomistic if the target  has this then set to True
        result["atomistic_target"] = (
            True  # hasattr(self, "target_laplacian_pes") and self.target_laplacian_pes[index] is not None # whether to prepend target to the generated sequence
        )
        return result


class MultimerFeatures(ConditionalFeature):
    def __init__(self, chains: list[list[int]]):
        super().__init__()
        self.chains = chains

    def setup(self, nres: list[int]):
        pass


def collate_fn(batch: list[dict], padding_values: dict | None = None):
    """
    Collate function to assemble a batch of protein design samples.

    Args:
        batch (List[Dict]): List of sample dictionaries. Each dictionary contains entries for
            structural and sequence tensors such as 'nres', 'x_target', 'seq_target', etc.
        padding_values (Optional[Dict], optional): A dictionary specifying the padding value
            to use for each tensor key during batching. If None, defaults to zero for all keys.

    Returns:
        Dict: A batch dictionary where each key contains a padded tensor or value aggregated
            across the batch.
            'nres' contains the maximal residue length for the batch,
            'nsamples' contains the number of samples in the batch,
            'mask' is a boolean mask indicating valid positions,
            remaining keys are padded tensors for each respective field in the input samples.
    """
    padding_values = padding_values or {}
    res = {
        "nsamples": len(batch),
    }
    for key in batch[0].keys():
        value_list = [sample[key] for sample in batch]
        if key == "nres":
            res["nres"] = max([sample["nres"] for sample in batch])
        elif isinstance(value_list[0], torch.Tensor):
            if value_list[0].dim() > 0:
                res[key] = torch.nn.utils.rnn.pad_sequence(
                    value_list,
                    batch_first=True,
                    padding_value=padding_values.get(key, 0),
                )
            else:
                res[key] = torch.stack(value_list)
        else:
            res[key] = value_list
    return res
