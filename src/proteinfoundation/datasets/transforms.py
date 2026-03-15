import gzip
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import wget
from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_from_encoding

# atomworks
from biotite.structure.bonds import connect_via_residue_names
from loguru import logger
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.np.residue_constants import atom_order, atom_types
from openfold.utils import rigid_utils
from torch.nn import functional as F

from proteinfoundation.datasets.atomworks_ligand_transforms import get_af3_raw_molecule_features, get_laplacian_pe
from proteinfoundation.nn.feature_factory.feature_utils import BOND_ORDER_MAP
from proteinfoundation.utils.align_utils import mean_w_mask
from proteinfoundation.utils.constants import AA_CHARACTER_PROTORP, AME_ATOMS, SIDECHAIN_TIP_ATOMS
from proteinfoundation.utils.coors_utils import ang_to_nm, sample_uniform_rotation


# Simple data container that supports both attribute and dict-style access
# Replaces torch_geometric.data.Data
class Data:
    """Simple data container with attribute and dict-style access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        for key in self.keys():
            yield key, getattr(self, key)

    def keys(self) -> list:
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def register_token_group(self, group: str, fields: list[str]) -> None:
        """Register fields belonging to a non-binder token group for collation.

        Extraction transforms call this in compact mode so that the collate
        function knows which fields share a padding dimension that differs
        from the binder length.

        Args:
            group: Group name, e.g. ``"target"`` or ``"motif"``.
            fields: Field names on this Data that belong to the group.
        """
        if not hasattr(self, "_token_groups"):
            self._token_groups = {}
        self._token_groups[group] = fields


class BaseTransform:
    """Base transform class - replaces torch_geometric.transforms.BaseTransform."""

    def __call__(self, data: Data) -> Data:
        raise NotImplementedError

    def forward(self, data: Data) -> Data:
        return self.__call__(data)


class CopyCoordinatesTransform(BaseTransform):
    """Copies coords to coords_unmodified. Useful if other transforms like noising or rotations/translations are applied later on."""

    def __call__(self, graph: Data) -> Data:
        graph.coords_unmodified = graph.coords.clone()


class ChainBreakCountingTransform(BaseTransform):
    """Counting the number of chain breaks in the protein coordinates and saving it as an attribute."""

    def __init__(
        self,
        chain_break_cutoff: float = 4.0,
    ):
        self.chain_break_cutoff = chain_break_cutoff

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        graph.chain_breaks = (ca_dists > self.chain_break_cutoff).sum().item()
        return graph


class ChainBreakPerResidueTransform(BaseTransform):
    """Creates a binary mask indicating whether residue has chain break or not."""

    def __init__(
        self,
        chain_break_cutoff: float = 4.0,
    ):
        self.chain_break_cutoff = chain_break_cutoff

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        chain_breaks_per_residue = ca_dists > self.chain_break_cutoff
        graph.chain_breaks_per_residue = torch.cat(
            (
                chain_breaks_per_residue,
                torch.tensor([False], dtype=torch.bool, device=chain_breaks_per_residue.device),
            )
        )
        return graph


class PaddingTransform(BaseTransform):
    def __init__(self, max_size=256, fill_value=0):
        self.max_size = max_size
        self.fill_value = fill_value

    def __call__(self, graph: Data) -> Data:
        for key, value in graph:
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1:  # Only pad tensors with 2 or more dimensions
                    pad_dim = 0
                    graph[key] = self.pad_tensor(value, self.max_size, pad_dim, self.fill_value)
                    # logger.info(f"Padded {key} to {graph[key].shape}")
        return graph

    def pad_tensor(self, tensor, max_size, dim, fill_value=0):
        if tensor.size(dim) >= max_size:
            return tensor

        pad_size = max_size - tensor.size(dim)
        padding = [0] * (2 * tensor.dim())
        padding[2 * (tensor.dim() - 1 - dim) + 1] = pad_size
        return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_size={self.max_size}, fill_value={self.fill_value})"


class GlobalRotationTransform(BaseTransform):
    """Modifies the global rotation of the atom37 representation randomly.

    Should be used as the first transform in the pipeline that modifies coordinates in order to keep
    e.g. frame construction or other things consistent down the pipeline."""

    def __init__(self, rotation_strategy: Literal["uniform"] = "uniform"):
        self.rotation_strategy = rotation_strategy

    def __call__(self, graph: Data) -> Data:
        if self.rotation_strategy == "uniform":
            rot = sample_uniform_rotation(dtype=graph.coords_nm.dtype, device=graph.coords_nm.device)
        else:
            raise ValueError(f"Rotation strategy {self.rotation_strategy} not supported")
        graph.coords_nm = torch.matmul(graph.coords_nm, rot)  # [n, 37, 3] * [3, 3] -> [n, 37, 3]
        # masked coords will still be 0
        return graph


class StructureNoiseTransform(BaseTransform):
    """Adds noise to the coordinates of a protein structure.

    Sets the following attributes on the protein data object:
        coords_uncorrupted (torch.Tensor): The original coordinates of the protein.
        noise (torch.Tensor): The noise added to the coordinates.
        coords (torch.Tensor): The original coordinates with added noise.

    Args:
        corruption_rate (float): Magnitude of corruption to apply to the coordinates.
        corruption_strategy (str): Noise strategy to use for corruption. Must be
            either "uniform" or "gaussian".
        gaussian_mean (float, optional): Mean of the Gaussian distribution.
            Defaults to 0.0.
        gaussian_std (float, optional): Standard deviation of the Gaussian
            distribution. Defaults to 1.0.
        uniform_min (float, optional): Minimum value of the uniform distribution.
            Defaults to -1.0.
        uniform_max (float, optional): Maximum value of the uniform distribution.
            Defaults to 1.0.

    Raises:
        ValueError: If the corruption strategy is not supported.
    """

    def __init__(
        self,
        corruption_strategy: Literal["uniform", "gaussian"] = "gaussian",
        gaussian_mean: float = 0.0,
        gaussian_std: float = 1.0,
        uniform_min: float = -1.0,
        uniform_max: float = 1.0,
    ):
        self.corruption_strategy = corruption_strategy
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max

    def __call__(self, graph: Data) -> Data:
        """Adds noise to the coordinates of a protein structure.

        Args:
            graph (Data): Protein data object.

        Returns:
            Data: Protein data object with corrupted coordinates.
        """

        if self.corruption_strategy == "uniform":
            noise = torch.empty_like(graph.coords_nm).uniform_(self.uniform_min, self.uniform_max)
        elif self.corruption_strategy == "gaussian":
            noise = torch.normal(
                mean=self.gaussian_mean,
                std=self.gaussian_std,
                size=graph.coords_nm.size(),
            )
        else:
            raise ValueError(f"Corruption strategy '{self.corruption_strategy}' not supported.")

        graph.noise = noise
        graph.noise[graph.coord_mask == 0] = 0
        graph.coords_nm += noise
        return graph

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(corruption_strategy={self.corruption_strategy}, "
            f"gaussian_mean={self.gaussian_mean}, "
            f"gaussian_std={self.gaussian_std}, uniform_min={self.uniform_min}, "
            f"uniform_max={self.uniform_max})"
        )


class CenterStructureTransform(BaseTransform):
    """Centers the structure based on CA coordinates."""

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords_nm[:, 1, :]  # [n, 3]
        mask = torch.ones(ca_coords.shape[0], dtype=torch.bool, device=ca_coords.device)
        com = mean_w_mask(ca_coords, mask, keepdim=True)  # [1, 3]
        graph.coords_nm = graph.coords_nm - com[None, ...]  # [n, 37, 3] - [1, 3]
        return graph


class GlobalTranslationTransform(BaseTransform):
    """Applies a global translation to the coordinates of a protein structure."""

    def __init__(
        self,
        translation_strategy: Literal["uniform", "normal"] = "uniform",
        uniform_min: float = -1.0,
        uniform_max: float = 1.0,
        normal_mean: float = 0.0,
        normal_std: float = 1.0,
    ):
        self.translation_strategy = translation_strategy
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max
        self.normal_mean = normal_mean
        self.normal_std = normal_std

    def __call__(self, graph: Data) -> Data:

        if self.translation_strategy == "uniform":
            translation = torch.empty(3, dtype=graph.coords_nm.dtype, device=graph.coords_nm.device).uniform_(
                self.uniform_min, self.uniform_max
            )
        elif self.translation_strategy == "normal":
            translation = torch.normal(
                mean=self.normal_mean,
                std=self.normal_std,
                size=(3,),
                dtype=graph.coords_nm.dtype,
                device=graph.coords_nm.device,
            )
        else:
            raise ValueError(f"Translation strategy '{self.translation_strategy}' not supported.")

        graph.translation = translation
        graph.coords_nm += translation
        return graph

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(translation_strategy={self.translation_strategy}, "
            f"gaussian_mean={self.normal_mean}, gaussian_std={self.normal_std}, "
            f"uniform_min={self.uniform_min}, uniform_max={self.uniform_max})"
        )


class CoordsToNanometers(BaseTransform):
    """Gets cordinates in nanometers."""

    def __call__(self, graph: Data) -> Data:
        graph.coords_nm = ang_to_nm(graph.coords)
        return graph


class OpenFoldFrameTransform(BaseTransform):
    """OpenFold frame transform."""

    def __call__(self, graph: Data) -> Data:
        aatype = torch.zeros_like(graph.residue_type).long()
        coords = graph.coords.double()
        atom_mask = graph.coord_mask.double()
        # Run through OpenFold data transforms.
        chain_feats = {
            "aatype": aatype,
            "all_atom_positions": coords,
            "all_atom_mask": atom_mask,
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_gt_frames"])[:, 0]
        rotations_gt = rigids_1.get_rots().get_rot_mats()
        translations_gt = rigids_1.get_trans()

        graph.translations_gt = translations_gt
        graph.rotations_gt = rotations_gt

        return graph


class CATHLabelTransform(BaseTransform):
    """Adds CATH labels if available to the protein."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.pdb_chain_cath_uniprot_url = (
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_cath_uniprot.tsv.gz"
        )
        self.cath_id_cath_code_url = (
            "http://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-all.gz"
        )
        self.cath_id_cath_code_filename = Path(self.cath_id_cath_code_url).name
        self.pdb_chain_cath_uniprot_filename = Path(self.pdb_chain_cath_uniprot_url).name

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        if not os.path.exists(self.root_dir / self.pdb_chain_cath_uniprot_filename):
            logger.info("Downloading Uniprot/PDB CATH map...")
            wget.download(self.pdb_chain_cath_uniprot_url, out=str(self.root_dir))

        if not os.path.exists(self.root_dir / self.cath_id_cath_code_filename):
            logger.info("Downloading CATH ID to CATH code map...")
            wget.download(self.cath_id_cath_code_url, out=str(self.root_dir))

        logger.info("Processing Uniprot/PDB CATH map...")
        self.pdbchain_to_cathid_mapping = self._parse_cath_id()
        logger.info("Processing CATH ID to CATH code map...")
        self.cathid_to_cathcode_mapping, self.cathid_to_segment_mapping = self._parse_cath_code()

    def __call__(self, graph: Data) -> Data:
        """Map each PDB chain to its CATH ID and CATH code."""
        cath_ids = self.pdbchain_to_cathid_mapping.get(graph.id, None)
        if cath_ids:
            cath_code = [self.cathid_to_cathcode_mapping.get(cath_id, None) for cath_id in cath_ids]
        else:
            cath_code = None
        if cath_code:  # check for list of Nones in cath code list
            graph.cath_code = cath_code
        else:
            graph.cath_code = []
        return graph

    def _parse_cath_id(self) -> dict[str, str]:
        """Parse the CATH ID for all PDB chains.

        :return: Dictionary of PDB chain ID with their
            corresponding CATH ID.
        :rtype: Dict[str, str]
        """
        pdbchain_to_cathid_mapping = defaultdict(list)
        with gzip.open(self.root_dir / self.pdb_chain_cath_uniprot_filename, "rt") as f:
            next(f)  # Skip header line
            for line in f:
                try:
                    pdb, chain, uniprot_id, cath_id = line.strip().split("\t")
                    key = f"{pdb}_{chain}"
                    pdbchain_to_cathid_mapping[key].append(cath_id)
                except ValueError as e:
                    logger.warning(e)
                    continue
        return pdbchain_to_cathid_mapping

    def _parse_cath_code(self) -> dict[str, str]:
        """Parse the CATH code for all CATH IDs.

        :return: Dictionary of CATH ID with their
            corresponding CATH code.
        :rtype: Dict[str, str]
        """
        cathid_to_cathcode_mapping = {}
        cathid_to_segment_mapping = {}
        with gzip.open(self.root_dir / self.cath_id_cath_code_filename, "rt") as f:
            for line in f:
                try:
                    cath_id, cath_version, cath_code, cath_segment_and_chain = line.strip().split()
                    # check if cath segment is one or multiple by presence of comma
                    if "," in cath_segment_and_chain:
                        # multiple segments, process each one separately by putting them into a list
                        cath_segments_and_chains = [cath_segment for cath_segment in cath_segment_and_chain.split(",")]
                    else:
                        # single segment, make it into a list as well for uniform processing
                        cath_segments_and_chains = [cath_segment_and_chain]
                    # separate segments and chains by colon
                    cath_segments, cath_chains = zip(
                        *[cath_segment_and_chain.split(":") for cath_segment_and_chain in cath_segments_and_chains],
                        strict=False,
                    )
                    # separate start and end positions of each segment by hyphen
                    cath_segments_start, cath_segments_end = zip(
                        *[self.split_segment(cath_segment) for cath_segment in cath_segments], strict=False
                    )
                    cathid_to_cathcode_mapping[cath_id] = cath_code
                    cathid_to_segment_mapping[cath_id] = [
                        (cath_chain, cath_segment_start, cath_segment_end)
                        for (cath_chain, cath_segment_start, cath_segment_end) in zip(
                            cath_chains, cath_segments_start, cath_segments_end, strict=False
                        )
                    ]
                except ValueError as e:
                    logger.warning(e)
                    continue
        return cathid_to_cathcode_mapping, cathid_to_segment_mapping

    def split_segment(self, segment: str) -> tuple[str, str]:
        """Split a segment into start position and end position. Handles cases where start or end position are negative numbers.

        Args:
            segment (str): segment description, for example `1-48` or `-2-36` or `1T-14M`.

        Returns:
            Tuple[str, str]: tuple containing start and end position, for example `(1, 48)` or `(-2, 36)` or `(1T, 14M)`.
        """
        # This regex pattern matches (potentially negative) numbers with potentially letters after them and separates segments by hyphen
        pattern = r"(-?\d+[A-Za-z]*)-(-?\d+[A-Za-z]*)"
        match = re.match(pattern, segment)
        if match:
            return match.groups()
        raise ValueError(f"Segment {segment} is not in the correct format")


class CroppingTransform2(BaseTransform):
    """
    Contiguous cropping on the binder chain, and both contiguous and spatial cropping on the target chains.

    1. Based on the interface threshold, select interface residues.
    2. Randomly select one residue from the interface residues and use its corresponding chain as the binder chain, and all the other chains as the target chains.
    3. Do contiguous cropping on the binder chain, with its length being randomly sampled from Uniform(binder_min_length, binder_max_length).
        Minimal number of residues (=binder_padding_length) on the both sides of the binder seed residue on the binder chain will be included.
    4. For the target chains, will do both contiguous cropping and spatial cropping.
        a): First do spatial cropping on the target chains, select target residues with distance to the whole binder chain less than target_spatial_crop_threshold.
            If there are more than (crop_size - binder_length) residues selected, take the top (crop_size - binder_length) residues.
        b): Then do contiguous cropping on the target chains for each segments of the target chains
    """

    def __init__(
        self,
        crop_size: int = 384,
        interface_threshold: float = 5.0,
        dist_mode: Literal["bb_ca", "all-atom"] = "bb_ca",
        data_mode: Literal["bb_ca", "all-atom"] = "all-atom",
        binder_min_length: int = 50,
        binder_max_length: int = 200,
        binder_padding_length: int = 10,
        target_spatial_crop_threshold: float = 15.0,
        target_min_length: int = 50,
        enforce_target_min_length: bool = False,
        max_num_target_chains: int = -1,  # -1 for no limit
    ):
        """
        Args:
            crop_size: the total number of residues to keep after cropping. Default is 384.
            interface_threshold: the threshold (in Angstrom) for interface residues. Default is 5.0.
            dist_mode: the distance mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom for computing the distance between residues. Default is "bb_ca".
            data_mode: the data mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom. Default is "all-atom".
            binder_min_length: the minimum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 50.
            binder_max_length: the maximum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 200.
            binder_padding_length: the minimal number of residues to include on each side of the binder seed residue when cropping the binder chain. Default is 10.
            target_spatial_crop_threshold: the threshold (in Angstrom) for target spatial cropping, which is the cutoff distanceof target residues to the whole binder chain. Default is 15.0.
            enforce_target_min_length: whether to enforce the target chain to have at least target_min_length residues after cropping.
                If True, target might have more chains than max_num_target_chains. Default is True.
            max_num_target_chains: the maximum number of target chains to include in the cropping, -1 for no limit. Default is -1.
        """
        self.crop_size = crop_size
        self.interface_threshold = interface_threshold
        self.dist_mode = dist_mode
        self.data_mode = data_mode
        self.binder_min_length = binder_min_length
        self.binder_max_length = binder_max_length
        self.binder_padding_length = binder_padding_length
        self.target_spatial_crop_threshold = target_spatial_crop_threshold
        self.target_min_length = target_min_length
        self.enforce_target_min_length = enforce_target_min_length
        self.max_num_target_chains = max_num_target_chains
        assert self.binder_min_length >= 2 * self.binder_padding_length + 1, (
            "binder_min_length must be >= 2 * binder_padding_length + 1"
        )

    def __call__(self, graph: Data) -> Data:
        asym_id = graph.chains
        n_res = len(graph.chains)
        residues_idxs = torch.arange(n_res)
        chain_lens = torch.bincount(asym_id)
        selected_idxs = torch.zeros(n_res, dtype=torch.bool)
        remaining_idxs = torch.ones(n_res, dtype=torch.bool)
        chain_residue_ranges = {}  # {chain_id: [start_idx, end_idx]}
        for chain_id in torch.unique(asym_id).sort().values.tolist():
            chain_mask = asym_id == chain_id
            chain_indices = residues_idxs[chain_mask]
            chain_residue_ranges[chain_id] = [
                int(chain_indices.min()),
                int(chain_indices.max()),
            ]

        # get interface residues, and residue-wise distances: (n_res, n_res)
        interface_residues, min_dist_per_res = self._get_interface_residues(graph)
        # filter out interface residues that are from too short chains
        chain_of_interface_residues = asym_id[interface_residues]
        binder_candidate_seed_residues = interface_residues[
            chain_lens[chain_of_interface_residues] >= self.binder_min_length
        ]

        if len(binder_candidate_seed_residues) == 0:
            # Select the top k residues in min_dist_per_res and generate the chain_of_interface_residues
            k = min(10, min_dist_per_res.shape[0])  # or another value for k as appropriate
            # Get the indices of the k smallest min_dist_per_res values (i.e., closest interface residues)
            topk_vals, topk_indices = torch.topk(min_dist_per_res, k, largest=False)
            interface_residues = topk_indices
            chain_of_interface_residues = asym_id[interface_residues]
            binder_candidate_seed_residues = interface_residues[
                chain_lens[chain_of_interface_residues] >= self.binder_min_length
            ]
            # raise ValueError(f"No binder candidate seed residues found for pdb: {graph.id}")
            logger.warning(
                f"No binder candidate seed residues found for pdb: {graph.id}, using top {k} interface residues as binder candidate seed residues, with min interface distance: {topk_vals.min().item():.2f}A"
            )

        # randomly select a binder seed residue from interface residues
        try:
            binder_seed_residue = random.choice(binder_candidate_seed_residues)
        except Exception:
            raise ValueError(f"No binder candidate seed residues found for pdb: {graph.pdb_id}")
        binder_chain_id = int(asym_id[binder_seed_residue])

        if (n_res - int(chain_lens[binder_chain_id])) < self.target_min_length and self.enforce_target_min_length:
            raise ValueError(f"Target chain too short for pdb: {graph.id}")

        if hasattr(graph, "cat1") and hasattr(graph, "cat2"):
            if binder_chain_id == 0:
                cath_code = [graph.cat1]
            else:
                cath_code = [graph.cat2]
            graph.cath_code = []
            for code in cath_code:
                if len(code.split(".")) == 3:
                    graph.cath_code.append(code + ".x")
                elif len(code.split(".")) == 4:
                    graph.cath_code.append(code)
                else:
                    raise ValueError(f"Invalid cath code: {code}")

        target_mask = asym_id != binder_chain_id
        # in the case of controlling the number of target chains, binder_mask not necessarily equal to ~target_mask
        # binder_mask = asym_id == binder_chain_id
        if self.data_mode == "all-atom":
            graph.target_mask = target_mask[:, None] * graph.coord_mask.bool()
        else:
            graph.target_mask = target_mask

        if len(graph.chains) < self.crop_size:
            return graph

        # Contiguous cropping on the binder chain
        binder_length = int(torch.randint(self.binder_min_length, self.binder_max_length + 1, (1,))[0])
        binder_chain_start = chain_residue_ranges[binder_chain_id][0]
        binder_chain_end = chain_residue_ranges[binder_chain_id][1]
        binder_left_length = int(
            torch.randint(
                self.binder_padding_length,
                binder_length - self.binder_padding_length - 1 + 1,
                (1,),
            )[0]
        )
        binder_chain_crop_start = max(binder_chain_start, binder_seed_residue - binder_left_length)
        binder_chain_crop_end = min(binder_chain_end + 1, binder_chain_crop_start + binder_length)
        if binder_chain_crop_end == binder_chain_end + 1:
            binder_chain_crop_start = max(binder_chain_start, binder_chain_crop_end - binder_length)
        binder_chain_crop_idxs = torch.arange(binder_chain_crop_start, binder_chain_crop_end)
        selected_idxs[binder_chain_crop_idxs] = True

        # Spatial cropping on the target chains first to get the target seed residues
        to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
        target_seed_residues = residues_idxs[target_mask][to_binder_dist < self.target_spatial_crop_threshold]
        target_seed_residue_chains = asym_id[target_seed_residues]
        target_chains = target_seed_residue_chains.unique()
        target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        if self.max_num_target_chains > 0 and len(target_chains) > self.max_num_target_chains:
            # can't use random.shuffle to shuffle torch tensors.
            target_chains = (
                target_chains[torch.randperm(len(target_chains))][: self.max_num_target_chains].sort().values
            )
            target_seed_residues = target_seed_residues[
                (target_seed_residue_chains[:, None] == target_chains[None, :]).any(dim=-1)
            ]
            target_seed_residue_chains = asym_id[target_seed_residues]
            target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        # put chains not included in the target_chains to be False in remaining_idxs
        remaining_idxs[(asym_id[:, None] != target_chains[None, :]).all(dim=-1)] = False
        num_budget = self.crop_size - int(selected_idxs.sum())
        num_remaining = int(remaining_idxs.sum())

        if num_remaining <= num_budget:
            selected_idxs[residues_idxs[target_mask]] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        if len(target_seed_residues) >= num_budget:
            to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
            target_seed_residues = residues_idxs[target_mask][torch.argsort(to_binder_dist)[:num_budget]]
            selected_idxs[target_seed_residues] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        # Get the segments of the target chains
        segments = []
        for chain_id in target_chains:
            for segment in self._convert_to_segments(
                target_seed_residues[target_seed_residue_chains == chain_id].tolist()
            ):
                segments.append(segment)

        # Contiguous cropping on the target chains from the target seed residues/segments
        remaining_idxs[target_seed_residues] = False
        selected_idxs[target_seed_residues] = True
        segment_idxs = list(range(len(segments)))
        segment_used_mask = torch.zeros((len(segments), 2), dtype=torch.bool)  # [left_crop, right_crop]
        random.shuffle(segment_idxs)

        for segment_idx in segment_idxs:
            segment_start, segment_end = segments[segment_idx]
            chain_id = int(asym_id[segment_start])
            chain_start = chain_residue_ranges[chain_id][0]
            chain_end = chain_residue_ranges[chain_id][1]

            orders = [0, 1]
            random.shuffle(orders)
            for order in orders:
                if order == 0:
                    # left crop
                    segment_used_mask[segment_idx, 0] = True
                    if segment_idx == 0:
                        remaining_idxs[chain_start:segment_start] = False
                    else:
                        if segment_used_mask[segment_idx - 1, 1]:  # and segment_used_mask[segment_idx, 0]
                            remaining_idxs[segments[segment_idx - 1][1] : segment_start] = False
                        elif segments[segment_idx - 1][1] < chain_start:
                            remaining_idxs[chain_start:segment_start] = False
                    num_budget = self.crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx >= 1:
                        segment_crop_size_max = min(
                            num_budget,
                            segment_start - chain_start,
                            int((~selected_idxs[(segments[segment_idx - 1][1] + 1) : segment_start]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, segment_start - chain_start)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_start - segment_crop_size) : (segment_start)] = True
                    remaining_idxs[(segment_start - segment_crop_size) : (segment_start)] = False

                else:
                    # right crop
                    segment_used_mask[segment_idx, 1] = True
                    if segment_idx == len(segment_idxs) - 1:
                        remaining_idxs[segment_end : chain_end + 1] = False
                    else:
                        if segment_used_mask[segment_idx + 1, 0]:  # and segment_used_mask[segment_idx, 1]
                            remaining_idxs[segment_end : segments[segment_idx + 1][0]] = False
                        elif segments[segment_idx + 1][0] > chain_end:
                            remaining_idxs[segment_end : chain_end + 1] = False
                    num_budget = self.crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx < len(segment_idxs) - 1:
                        segment_crop_size_max = min(
                            num_budget,
                            chain_end - segment_end,
                            int((~selected_idxs[(segment_end + 1) : segments[segment_idx + 1][0]]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, chain_end - segment_end)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = True
                    remaining_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = False

        graph = self._crop_graph(graph, residues_idxs[selected_idxs])

        return graph

    @staticmethod
    def _convert_to_segments(sorted_integers):
        """
        Convert a list of sorted integers to a combination of ranges.

        Args:
            sorted_integers: List of sorted integers

        Returns:
            List of tuples, each tuple is a range of integers
        """
        segments = []
        start = sorted_integers[0]
        end = sorted_integers[0]

        for i in range(1, len(sorted_integers)):
            if sorted_integers[i] == end + 1:
                end = sorted_integers[i]
            else:
                segments.append((start, end))
                start = sorted_integers[i]
                end = sorted_integers[i]

        # Add the last range
        segments.append((start, end))

        return segments

    def _get_interface_residues(self, graph: Data) -> torch.Tensor:
        asym_id = graph.chains
        diff_chain_mask = asym_id[..., None, :] != asym_id[..., :, None]
        if self.dist_mode == "bb_ca":
            ca_idx = atom_order["CA"]
            ca_positions = graph.coords[..., ca_idx, :]
            ca_pairwise_dists = torch.cdist(ca_positions, ca_positions)
            min_dist_per_res = torch.where(diff_chain_mask, ca_pairwise_dists, torch.inf)  # .min(dim=-1)
        elif self.dist_mode == "all-atom":
            positions = graph.coords
            n_res, n_atom, _ = positions.shape
            atom_mask = graph.coord_mask
            pairwise_dists = (
                torch.cdist(
                    positions.view(n_res * n_atom, -1),
                    positions.view(n_res * n_atom, -1),
                )
                .view(n_res, n_atom, n_res, n_atom)
                .permute(0, 2, 1, 3)
            )
            pair_mask = atom_mask[None, :, None, :] * atom_mask[:, None, :, None]
            mask = diff_chain_mask[:, :, None, None] * pair_mask
            min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1).values.min(dim=-1).values
        else:
            raise ValueError(f"Invalid dist mode: {self.dist_mode}")

        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
        return interface_residues_idxs, min_dist_per_res

    def _crop_graph(self, graph: Data, crop_idxs: torch.Tensor) -> Data:
        num_residues = graph.coords.size(0)
        for key, value in graph:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_residues:
                graph[key] = value[crop_idxs]
            elif isinstance(value, list) and len(value) == num_residues:
                graph[key] = [value[i] for i in crop_idxs]
        if hasattr(graph, "chain_names"):
            graph["chain_names"] = [graph["chain_names"][i] for i in torch.unique_consecutive(graph.chains)]

        return graph


class CroppingTransform2Ligand(BaseTransform):
    """
    Contiguous cropping on the binder chain, and both contiguous and spatial cropping on the target chains.

    1. Based on the interface threshold, select interface residues.
    2. Randomly select one residue from the interface residues and use its corresponding chain as the binder chain, and all the other chains as the target chains.
    3. Do contiguous cropping on the binder chain, with its length being randomly sampled from Uniform(binder_min_length, binder_max_length).
        Minimal number of residues (=binder_padding_length) on the both sides of the binder seed residue on the binder chain will be included.
    4. For the target chains, will do both contiguous cropping and spatial cropping.
        a): First do spatial cropping on the target chains, select target residues with distance to the whole binder chain less than target_spatial_crop_threshold.
            If there are more than (crop_size - binder_length) residues selected, take the top (crop_size - binder_length) residues.
        b): Then do contiguous cropping on the target chains for each segments of the target chains
    """

    def __init__(
        self,
        crop_size: int = 384,
        interface_threshold: float = 5.0,
        dist_mode: Literal["bb_ca", "all-atom"] = "bb_ca",
        data_mode: Literal["bb_ca", "all-atom"] = "all-atom",
        binder_min_length: int = 50,
        binder_max_length: int = 200,
        binder_padding_length: int = 10,
        target_spatial_crop_threshold: float = 15.0,
        target_min_length: int = 1,
        target_max_length: int = 5,
        enforce_target_min_length: bool = True,
        max_num_target_chains: int = -1,  # -1 for no limit
    ):
        """
        Args:
            crop_size: the total number of residues to keep after cropping. Default is 384.
            interface_threshold: the threshold (in Angstrom) for interface residues. Default is 5.0.
            dist_mode: the distance mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom for computing the distance between residues. Default is "bb_ca".
            data_mode: the data mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom. Default is "all-atom".
            binder_min_length: the minimum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 50.
            binder_max_length: the maximum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 200.
            binder_padding_length: the minimal number of residues to include on each side of the binder seed residue when cropping the binder chain. Default is 10.
            target_spatial_crop_threshold: the threshold (in Angstrom) for target spatial cropping, which is the cutoff distanceof target residues to the whole binder chain. Default is 15.0.
            enforce_target_min_length: whether to enforce the target chain to have at least target_min_length residues after cropping.
                If True, target might have more chains than max_num_target_chains. Default is True.
            max_num_target_chains: the maximum number of target chains to include in the cropping, -1 for no limit. Default is -1.
        """
        self.crop_size = crop_size
        self.interface_threshold = interface_threshold
        self.dist_mode = dist_mode
        self.data_mode = data_mode
        self.binder_min_length = binder_min_length
        self.binder_max_length = binder_max_length
        self.binder_padding_length = binder_padding_length
        self.target_spatial_crop_threshold = target_spatial_crop_threshold
        self.target_min_length = target_min_length
        self.target_max_length = target_max_length
        self.enforce_target_min_length = enforce_target_min_length
        self.max_num_target_chains = max_num_target_chains
        assert self.binder_min_length >= 2 * self.binder_padding_length + 1, (
            "binder_min_length must be >= 2 * binder_padding_length + 1"
        )

    def __call__(self, graph: Data) -> Data:
        asym_id = graph.chains
        n_res = len(graph.chains)
        residues_idxs = torch.arange(n_res)
        chain_lens = torch.bincount(asym_id)
        selected_idxs = torch.zeros(n_res, dtype=torch.bool)
        remaining_idxs = torch.ones(n_res, dtype=torch.bool)
        chain_residue_ranges = {}  # {chain_id: [start_idx, end_idx]}
        for chain_id in torch.unique(asym_id).sort().values.tolist():
            chain_mask = asym_id == chain_id
            chain_indices = residues_idxs[chain_mask]
            chain_residue_ranges[chain_id] = [
                int(chain_indices.min()),
                int(chain_indices.max()),
            ]

        # get interface residues, and residue-wise distances: (n_res, n_res)
        interface_residues, min_dist_per_res = self._get_interface_residues(graph)
        # filter out interface residues that are from too short chains
        chain_of_interface_residues = asym_id[interface_residues]
        binder_candidate_seed_residues = interface_residues[
            chain_lens[chain_of_interface_residues] >= self.binder_min_length
        ]

        if len(binder_candidate_seed_residues) == 0:
            # Select the top k residues in min_dist_per_res and generate the chain_of_interface_residues
            k = min(10, min_dist_per_res.shape[0])  # or another value for k as appropriate
            # Get the indices of the k smallest min_dist_per_res values (i.e., closest interface residues)
            topk_vals, topk_indices = torch.topk(min_dist_per_res, k, largest=False)
            interface_residues = topk_indices
            chain_of_interface_residues = asym_id[interface_residues]
            binder_candidate_seed_residues = interface_residues[
                chain_lens[chain_of_interface_residues] >= self.binder_min_length
            ]
            # raise ValueError(f"No binder candidate seed residues found for pdb: {graph.id}")
            logger.warning(
                f"No binder candidate seed residues found for pdb: {graph.id}, using top {k} interface residues as binder candidate seed residues, with min interface distance: {topk_vals.min().item():.2f}A"
            )

        # randomly select a binder seed residue from interface residues
        try:
            binder_seed_residue = random.choice(binder_candidate_seed_residues)
        except Exception as e:
            print(e)
            print(graph)
            raise e
        binder_chain_id = int(asym_id[binder_seed_residue])
        1 - binder_chain_id
        target_seed_residues = min_dist_per_res[binder_seed_residue].argmin()
        # target_seed_residues = random.choice(interface_residues[chain_of_interface_residues == target_chain_id])

        if (n_res - int(chain_lens[binder_chain_id])) < self.target_min_length and self.enforce_target_min_length:
            raise ValueError(f"Target chain too short for pdb: {graph.id}")

        # Initialize target mask and mark seed residue
        target_mask = (asym_id != binder_chain_id) * False
        target_mask[target_seed_residues] = True
        target_length = int(torch.randint(self.target_min_length, self.target_max_length + 1, (1,))[0])
        # Get all residues on the target chain in order
        target_chain_residues = torch.where(asym_id != binder_chain_id)[0]
        # Find position of seed residue in the ordered chain
        seed_pos = torch.where(target_chain_residues == target_seed_residues)[0].item()

        # Keep track of current selected residues
        current_selected = {target_seed_residues.item()}
        remaining_to_select = target_length - 1  # -1 because we already have seed

        while remaining_to_select > 0:
            # Find valid neighbors (left and right of current selection)
            valid_left = None
            valid_right = None
            min_selected = min(current_selected)
            max_selected = max(current_selected)

            # Check left neighbor
            left_idx = torch.where(target_chain_residues == min_selected)[0].item()
            if left_idx > 0:
                valid_left = target_chain_residues[left_idx - 1].item()

            # Check right neighbor
            right_idx = torch.where(target_chain_residues == max_selected)[0].item()
            if right_idx < len(target_chain_residues) - 1:
                valid_right = target_chain_residues[right_idx + 1].item()

            # If no valid neighbors, break
            if valid_left is None and valid_right is None:
                break

            # Choose direction based on available options
            if valid_left is not None and valid_right is not None:
                # 50/50 chance to go left or right
                if random.random() < 0.5:
                    current_selected.add(valid_left)
                    target_mask[valid_left] = True
                else:
                    current_selected.add(valid_right)
                    target_mask[valid_right] = True
            elif valid_left is not None:
                current_selected.add(valid_left)
                target_mask[valid_left] = True
            else:  # valid_right is not None
                current_selected.add(valid_right)
                target_mask[valid_right] = True

            remaining_to_select -= 1

        for selected_target_res_id in current_selected:
            target_mask[selected_target_res_id] = True
        selected_idxs[target_mask] = True

        # in the case of controlling the number of target chains, binder_mask not necessarily equal to ~target_mask
        # binder_mask = asym_id == binder_chain_id
        if self.data_mode == "all-atom":
            graph.target_mask = target_mask[:, None] * graph.coord_mask.bool()
        else:
            graph.target_mask = target_mask

        if len(graph.chains) < self.crop_size:
            #! we can fit the entire binder so we do that
            selected_idxs[asym_id == binder_chain_id] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            # from proteinfoundation.utils.pdb_utils import write_prot_to_pdb
            # write_prot_to_pdb(
            #     prot_pos=graph.coords.float().detach().cpu().numpy(),
            #     aatype=graph.residue_type.detach().cpu().numpy(),
            #     file_path="ted_crop.pdb",
            #     chain_index=graph.chains.numpy(),
            #     overwrite=True,
            #     no_indexing=True,
            # )

            return graph

        # Contiguous cropping on the binder chain
        binder_length = int(torch.randint(self.binder_min_length, self.binder_max_length + 1, (1,))[0])
        binder_chain_start = chain_residue_ranges[binder_chain_id][0]
        binder_chain_end = chain_residue_ranges[binder_chain_id][1]
        binder_left_length = int(
            torch.randint(
                self.binder_padding_length,
                binder_length - self.binder_padding_length - 1 + 1,
                (1,),
            )[0]
        )
        binder_chain_crop_start = max(binder_chain_start, binder_seed_residue - binder_left_length)
        binder_chain_crop_end = min(binder_chain_end + 1, binder_chain_crop_start + binder_length)
        if binder_chain_crop_end == binder_chain_end + 1:
            binder_chain_crop_start = max(binder_chain_start, binder_chain_crop_end - binder_length)
        binder_chain_crop_idxs = torch.arange(binder_chain_crop_start, binder_chain_crop_end)
        selected_idxs[binder_chain_crop_idxs] = True

        # # Set the target fake ligand indices into the crop
        # selected_idxs[target_mask] = True

        # # Spatial cropping on the target chains first to get the target seed residues
        # to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
        # target_seed_residues = residues_idxs[target_mask][to_binder_dist < self.target_spatial_crop_threshold]
        # target_seed_residue_chains = asym_id[target_seed_residues]
        # target_chains = target_seed_residue_chains.unique()
        # target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)
        # # assert False
        # if self.max_num_target_chains > 0 and len(target_chains) > self.max_num_target_chains:
        #     # can't use random.shuffle to shuffle torch tensors.
        #     target_chains = target_chains[torch.randperm(len(target_chains))][:self.max_num_target_chains].sort().values
        #     target_seed_residues = target_seed_residues[(target_seed_residue_chains[:, None] == target_chains[None,:]).any(dim=-1)]
        #     target_seed_residue_chains = asym_id[target_seed_residues]
        #     target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        # # put chains not included in the target_chains to be False in remaining_idxs
        # remaining_idxs[(asym_id[:,None] != target_chains[None,:]).all(dim=-1)] = False
        # num_budget = self.crop_size - int(selected_idxs.sum())
        # num_remaining = int(remaining_idxs.sum())
        # if num_remaining <= num_budget:
        #     selected_idxs[residues_idxs[target_mask]] = True
        #     graph = self._crop_graph(graph, residues_idxs[selected_idxs])
        #     return graph

        # if len(target_seed_residues) >= num_budget:
        #     to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
        #     target_seed_residues = residues_idxs[target_mask][torch.argsort(to_binder_dist)[:num_budget]]
        #     selected_idxs[target_seed_residues] = True
        #     graph = self._crop_graph(graph, residues_idxs[selected_idxs])
        #     return graph

        # # Get the segments of the target chains
        # segments = []
        # for chain_id in target_chains:
        #     for segment in self._convert_to_segments(target_seed_residues[target_seed_residue_chains == chain_id].tolist()):
        #         segments.append(segment)

        # # Contiguous cropping on the target chains from the target seed residues/segments
        # remaining_idxs[target_seed_residues] = False
        # selected_idxs[target_seed_residues] = True
        # segment_idxs = list(range(len(segments)))
        # segment_used_mask = torch.zeros((len(segments), 2), dtype=torch.bool) # [left_crop, right_crop]
        # random.shuffle(segment_idxs)

        # for segment_idx in segment_idxs:
        #     segment_start, segment_end = segments[segment_idx]
        #     chain_id = int(asym_id[segment_start])
        #     chain_start = chain_residue_ranges[chain_id][0]
        #     chain_end = chain_residue_ranges[chain_id][1]

        #     orders = [0, 1]
        #     random.shuffle(orders)
        #     for order in orders:
        #         if order == 0:
        #             # left crop
        #             segment_used_mask[segment_idx, 0] = True
        #             if segment_idx == 0:
        #                 remaining_idxs[chain_start:segment_start] = False
        #             else:
        #                 if segment_used_mask[segment_idx - 1, 1]: # and segment_used_mask[segment_idx, 0]
        #                     remaining_idxs[segments[segment_idx - 1][1]:segment_start] = False
        #                 elif segments[segment_idx - 1][1] < chain_start:
        #                     remaining_idxs[chain_start:segment_start] = False
        #             num_budget = self.crop_size - int(selected_idxs.sum())
        #             num_remaining = int(remaining_idxs.sum())
        #             if segment_idx >= 1:
        #                 segment_crop_size_max = min(
        #                     num_budget,
        #                     segment_start - chain_start,
        #                     int((~selected_idxs[(segments[segment_idx - 1][1] + 1):segment_start]).sum()),
        #                 )
        #             else:
        #                 segment_crop_size_max = min(num_budget, segment_start - chain_start)
        #             segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
        #             segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
        #             selected_idxs[(segment_start - segment_crop_size):(segment_start)] = True
        #             remaining_idxs[(segment_start - segment_crop_size):(segment_start)] = False

        #         else:
        #             # right crop
        #             segment_used_mask[segment_idx, 1] = True
        #             if segment_idx == len(segment_idxs) - 1:
        #                 remaining_idxs[segment_end:chain_end + 1] = False
        #             else:
        #                 if segment_used_mask[segment_idx + 1, 0]: # and segment_used_mask[segment_idx, 1]
        #                     remaining_idxs[segment_end:segments[segment_idx + 1][0]] = False
        #                 elif segments[segment_idx + 1][0] > chain_end:
        #                     remaining_idxs[segment_end:chain_end + 1] = False
        #             num_budget = self.crop_size - int(selected_idxs.sum())
        #             num_remaining = int(remaining_idxs.sum())
        #             if segment_idx < len(segment_idxs) - 1:
        #                 segment_crop_size_max = min(
        #                     num_budget,
        #                     chain_end - segment_end,
        #                     int((~selected_idxs[(segment_end + 1):segments[segment_idx + 1][0]]).sum()),
        #                 )
        #             else:
        #                 segment_crop_size_max = min(num_budget, chain_end - segment_end)
        #             segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
        #             segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
        #             selected_idxs[(segment_end + 1):(segment_end + 1 + segment_crop_size)] = True
        #             remaining_idxs[(segment_end + 1):(segment_end + 1 + segment_crop_size)] = False

        graph = self._crop_graph(graph, residues_idxs[selected_idxs])

        return graph

    @staticmethod
    def _convert_to_segments(sorted_integers):
        """
        Convert a list of sorted integers to a combination of ranges.

        Args:
            sorted_integers: List of sorted integers

        Returns:
            List of tuples, each tuple is a range of integers
        """
        segments = []
        start = sorted_integers[0]
        end = sorted_integers[0]

        for i in range(1, len(sorted_integers)):
            if sorted_integers[i] == end + 1:
                end = sorted_integers[i]
            else:
                segments.append((start, end))
                start = sorted_integers[i]
                end = sorted_integers[i]

        # Add the last range
        segments.append((start, end))

        return segments

    def _get_interface_residues(self, graph: Data) -> torch.Tensor:
        asym_id = graph.chains
        diff_chain_mask = asym_id[..., None, :] != asym_id[..., :, None]
        if self.dist_mode == "bb_ca":
            ca_idx = atom_order["CA"]
            ca_positions = graph.coords[..., ca_idx, :]
            ca_pairwise_dists = torch.cdist(ca_positions, ca_positions)
            min_dist_per_res = torch.where(diff_chain_mask, ca_pairwise_dists, torch.inf)  # .min(dim=-1)
        elif self.dist_mode == "all-atom":
            positions = graph.coords
            n_res, n_atom, _ = positions.shape
            atom_mask = graph.coord_mask
            pairwise_dists = (
                torch.cdist(
                    positions.view(n_res * n_atom, -1),
                    positions.view(n_res * n_atom, -1),
                )
                .view(n_res, n_atom, n_res, n_atom)
                .permute(0, 2, 1, 3)
            )
            pair_mask = atom_mask[None, :, None, :] * atom_mask[:, None, :, None]
            mask = diff_chain_mask[:, :, None, None] * pair_mask
            min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1).values.min(dim=-1).values
        else:
            raise ValueError(f"Invalid dist mode: {self.dist_mode}")

        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
        return interface_residues_idxs, min_dist_per_res

    def _crop_graph(self, graph: Data, crop_idxs: torch.Tensor) -> Data:
        num_residues = graph.coords.size(0)
        for key, value in graph:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_residues:
                graph[key] = value[crop_idxs]
            elif isinstance(value, list) and len(value) == num_residues:
                graph[key] = [value[i] for i in crop_idxs]
        if hasattr(graph, "chain_names"):
            graph["chain_names"] = [graph["chain_names"][i] for i in torch.unique_consecutive(graph.chains)]

        return graph


class TargetSelectionTransform(BaseTransform):
    def __init__(
        self,
        interface_threshold: float = 5.0,
        dist_mode: Literal["bb_ca", "all-atom"] = "bb_ca",
        data_mode: Literal["bb_ca", "all-atom"] = "all-atom",
        binder_min_length: int = 50,
    ):
        """
        Args:
            interface_threshold: the threshold (in Angstrom) for interface residues. Default is 5.0.
            dist_mode: the distance mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom for computing the distance between residues. Default is "bb_ca".
            data_mode: the data mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom. Default is "all-atom".
            binder_min_length: the minimum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 50.
        """
        self.interface_threshold = interface_threshold
        self.dist_mode = dist_mode
        self.data_mode = data_mode
        self.binder_min_length = binder_min_length

    def __call__(self, graph: Data) -> Data:
        asym_id = graph.chains
        n_res = len(graph.chains)
        residues_idxs = torch.arange(n_res)
        chain_lens = torch.bincount(asym_id)
        chain_residue_ranges = {}  # {chain_id: [start_idx, end_idx]}
        for chain_id in torch.unique(asym_id).sort().values.tolist():
            chain_mask = asym_id == chain_id
            chain_indices = residues_idxs[chain_mask]
            chain_residue_ranges[chain_id] = [
                int(chain_indices.min()),
                int(chain_indices.max()),
            ]

        if hasattr(graph, "binder_chain_id"):
            binder_chain_id = graph.binder_chain_id
            binder_chain_id = ord(binder_chain_id) - ord("A")
        else:
            # get interface residues, and residue-wise distances: (n_res, n_res)
            interface_residues, min_dist_per_res = self._get_interface_residues(graph)
            # filter out interface residues that are from too short chains
            chain_of_interface_residues = asym_id[interface_residues]
            binder_candidate_seed_residues = interface_residues[
                chain_lens[chain_of_interface_residues] >= self.binder_min_length
            ]

            if len(binder_candidate_seed_residues) == 0:
                # Select the top k residues in min_dist_per_res and generate the chain_of_interface_residues
                k = min(10, min_dist_per_res.shape[0])  # or another value for k as appropriate
                # Get the indices of the k smallest min_dist_per_res values (i.e., closest interface residues)
                topk_vals, topk_indices = torch.topk(min_dist_per_res, k, largest=False)
                interface_residues = topk_indices
                chain_of_interface_residues = asym_id[interface_residues]
                binder_candidate_seed_residues = interface_residues[
                    chain_lens[chain_of_interface_residues] >= self.binder_min_length
                ]
                # raise ValueError(f"No binder candidate seed residues found for pdb: {graph.id}")
                logger.warning(
                    f"No binder candidate seed residues found for pdb: {graph.id}, using top {k} interface residues as binder candidate seed residues, with min interface distance: {topk_vals.min().item():.2f}A"
                )

            # randomly select a binder seed residue from interface residues
            binder_seed_residue = random.choice(binder_candidate_seed_residues)
            binder_chain_id = int(asym_id[binder_seed_residue])

        target_mask = asym_id != binder_chain_id
        # in the case of controlling the number of target chains, binder_mask not necessarily equal to ~target_mask
        # binder_mask = asym_id == binder_chain_id
        if self.data_mode == "all-atom":
            graph.target_mask = target_mask[:, None] * graph.coord_mask.bool()
        else:
            graph.target_mask = target_mask
        return graph

    def _get_interface_residues(self, graph: Data) -> torch.Tensor:
        asym_id = graph.chains
        diff_chain_mask = asym_id[..., None, :] != asym_id[..., :, None]
        if self.dist_mode == "bb_ca":
            ca_idx = atom_order["CA"]
            ca_positions = graph.coords[..., ca_idx, :]
            ca_pairwise_dists = torch.cdist(ca_positions, ca_positions)
            min_dist_per_res = torch.where(diff_chain_mask, ca_pairwise_dists, torch.inf)  # .min(dim=-1)
        elif self.dist_mode == "all-atom":
            positions = graph.coords
            n_res, n_atom, _ = positions.shape
            atom_mask = graph.coord_mask
            pairwise_dists = (
                torch.cdist(
                    positions.view(n_res * n_atom, -1),
                    positions.view(n_res * n_atom, -1),
                )
                .view(n_res, n_atom, n_res, n_atom)
                .permute(0, 2, 1, 3)
            )
            pair_mask = atom_mask[None, :, None, :] * atom_mask[:, None, :, None]
            mask = diff_chain_mask[:, :, None, None] * pair_mask
            min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1).values.min(dim=-1).values
        else:
            raise ValueError(f"Invalid dist mode: {self.dist_mode}")

        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
        return interface_residues_idxs, min_dist_per_res


class CroppingTransform(BaseTransform):
    """
    Randomly applies either spatial or contiguous cropping.

    Refactored from openfold multimer random_crop_to_size():
    https://github.com/aqlaboratory/openfold/blob/a1192c8d3a0f3004b1284aaf6437681e6b558c10/openfold/data/data_transforms_multimer.py#L419
    """

    def __init__(
        self,
        crop_size: int = 256,
        spatial_crop_prob: float = 0.5,
        interface_threshold: float = 8.0,
        generator: torch.Generator = None,
    ):
        self.crop_size = crop_size
        self.spatial_crop_prob = spatial_crop_prob
        self.interface_threshold = interface_threshold
        self.generator = generator if generator is not None else torch.default_generator

    def __call__(self, graph: Data) -> Data:
        use_spatial_crop = torch.rand((1,), generator=self.generator) < self.spatial_crop_prob
        if graph.chains.max() == 0:
            use_spatial_crop = False
        num_res = graph.coords.size(0)
        if num_res <= self.crop_size:
            return graph
        elif use_spatial_crop:
            crop_idxs = self._get_spatial_crop_idx(graph)
        else:
            crop_idxs = self._get_contiguous_crop_idx(graph)
        cropped_graph = self._crop_graph(graph, crop_idxs)
        return cropped_graph

    def _get_spatial_crop_idx(self, graph: Data) -> torch.Tensor:

        interface_residues = self._get_interface_residues(graph)
        if not torch.any(interface_residues):
            return self._get_contiguous_crop_idx(graph)

        target_res_idx = self._randint(
            lower=0,
            upper=interface_residues.shape[-1] - 1,
            generator=self.generator,
        )

        target_res = interface_residues[target_res_idx]

        # Get CA positions and distances
        ca_idx = atom_order["CA"]
        ca_positions = graph.coords[..., ca_idx, :]
        ca_mask = graph.coord_mask[..., ca_idx].bool()

        coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
        ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff**2, dim=-1))

        to_target_distances = ca_pairwise_dists[target_res]
        break_tie = (
            torch.arange(
                0,
                to_target_distances.shape[-1],
            ).float()
            * 1e-3
        )
        to_target_distances = torch.where(ca_mask, to_target_distances, torch.inf) + break_tie

        ret = torch.argsort(to_target_distances)[: self.crop_size]
        return ret.sort().values

    def _get_contiguous_crop_idx(self, graph: Data) -> torch.Tensor:
        unique_asym_ids, chain_idxs, chain_lens = graph["chains"].unique(
            dim=-1, return_inverse=True, return_counts=True
        )
        shuffle_idx = torch.randperm(chain_lens.shape[-1], generator=self.generator)

        _, idx_sorted = torch.sort(chain_idxs, stable=True)
        cum_sum = chain_lens.cumsum(dim=0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]), dim=0)
        asym_offsets = idx_sorted[cum_sum]

        num_budget = self.crop_size
        num_remaining = graph.seq_pos.size(0)

        crop_idxs = []
        for idx in shuffle_idx:
            chain_len = int(chain_lens[idx])
            num_remaining -= chain_len

            crop_size_max = min(num_budget, chain_len)
            crop_size_min = min(chain_len, max(0, num_budget - num_remaining))
            chain_crop_size = self._randint(
                lower=crop_size_min,
                upper=crop_size_max,
                generator=self.generator,
            )

            num_budget -= chain_crop_size

            chain_start = self._randint(
                lower=0,
                upper=chain_len - chain_crop_size,
                generator=self.generator,
            )

            asym_offset = asym_offsets[idx]
            crop_idxs.append(
                torch.arange(
                    asym_offset + chain_start,
                    asym_offset + chain_start + chain_crop_size,
                )
            )

        return torch.concat(crop_idxs).sort().values

    def _crop_graph(self, graph: Data, crop_idxs: torch.Tensor) -> Data:
        num_residues = graph.coords.size(0)
        for key, value in graph:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_residues:
                graph[key] = value[crop_idxs]
            elif isinstance(value, list) and len(value) == num_residues:
                graph[key] = [value[i] for i in crop_idxs]
        if hasattr(graph, "chain_names"):
            graph["chain_names"] = [graph["chain_names"][i] for i in torch.unique_consecutive(graph.chains)]

        return graph

    def _get_interface_residues(self, graph: Data) -> torch.Tensor:
        positions = graph.coords
        asym_id = graph.chains
        atom_mask = graph.coord_mask
        # coord_diff = positions[..., None, :, :] - positions[..., None, :, :, :]
        # pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
        pairwise_dists = torch.cdist(positions.transpose(0, 1), positions.transpose(0, 1)).permute(1, 2, 0)

        diff_target_mask = (asym_id[..., None, :] != asym_id[..., :, None]).float()
        pair_mask = atom_mask[..., None, :] * atom_mask[..., None, :, :]
        mask = (diff_target_mask[..., None] * pair_mask).bool()

        min_dist_per_res, _ = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)
        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]

        return interface_residues_idxs

    @staticmethod
    def _randint(lower: int, upper: int, generator: torch.Generator):
        return int(
            torch.randint(
                lower,
                upper + 1,
                (1,),
                generator=generator,
            )[0]
        )


class ContactMaskTransform(BaseTransform):
    """ """

    def __init__(
        self,
        cutoff: float = 8.0,
        mode: Literal["all_atom", "ca"] = "all_atom",
    ):
        self.cutoff = cutoff
        self.mode = mode

    def __call__(self, graph: Data) -> Data:
        if graph.chains.max() == 0:
            graph.contact_mask = torch.zeros(len(graph.residues), dtype=torch.bool)
            return graph
        target_chain = graph.target_mask.any(dim=-1)
        binder_chain = ~target_chain
        if self.mode == "all_atom":
            target_coord_mask = graph["coord_mask"][target_chain]
            binder_coord_mask = graph["coord_mask"][binder_chain]
            pair_dists = torch.norm(
                graph["coords"][target_chain, None, :, None, :] - graph["coords"][None, binder_chain, None, :, :],
                dim=-1,
            )
            contact_mask = (
                (pair_dists < self.cutoff) & target_coord_mask[:, None, :, None] & binder_coord_mask[None, :, None, :]
            ).any(dim=[2, 3])
        elif self.mode == "ca":
            ca_index = atom_types.index("CA")
            contact_mask = (
                torch.norm(
                    graph["coords"][target_chain, None, ca_index, :] - graph["coords"][None, binder_chain, ca_index, :],
                    dim=-1,
                )
                < self.cutoff
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        binder_contact_mask = contact_mask.any(dim=0)
        target_contact_mask = contact_mask.any(dim=1)
        graph.contact_mask = torch.zeros(len(graph.residues), dtype=torch.bool)
        graph.contact_mask[binder_chain] = binder_contact_mask
        graph.contact_mask[target_chain] = target_contact_mask
        return graph


class HotspotTransform(BaseTransform):
    """ """

    def __init__(self, min_perc: float = 0, max_perc: float = 0.2):
        """
        min_perc: minimum percentage of contacts to be considered a hotspot
        max_perc: maximum percentage of contacts to be considered a hotspot
        """
        self.min_perc = min_perc
        self.max_perc = max_perc

    def __call__(self, graph: Data) -> Data:
        if graph.chains.max() == 0:
            graph.hotspot_mask = torch.zeros(len(graph.residues), dtype=torch.bool)
            return graph
        target_chain = graph.target_mask.any(dim=-1)
        target_contact_mask = graph.contact_mask[target_chain]

        unique_target_hotspots = torch.arange(len(graph.residues))[target_chain][target_contact_mask]
        hotspot_perc = random.uniform(self.min_perc, self.max_perc)
        num_to_select = int(target_contact_mask.sum().item() * hotspot_perc)
        perm = torch.randperm(len(unique_target_hotspots))
        sampled_target_hotspots = unique_target_hotspots[perm[:num_to_select]]
        graph.hotspot_mask = torch.zeros(len(graph.chains), dtype=torch.bool)
        graph.hotspot_mask[sampled_target_hotspots] = True

        return graph


class ContactTransform(BaseTransform):
    """Computes contact-based features for protein binder design.

    This transform analyzes inter-chain contacts and generates features for protein binder design tasks.
    It identifies contact residues between chains, analyzes their amino acid composition, and selects
    target hotspots for design optimization.

    The transform adds the following attributes to the graph:
    - contact_mask: Binary mask indicating which residues are in contact across chains
    - contact: Contact composition features [total_contacts, apolar_frac, polar_frac, charged_frac]
    - target_hotspots: Binary mask indicating selected target hotspot residues

    Args:
        cutoff (float): Distance cutoff in Ångströms for defining contacts. Defaults to 5.0.
        min_perc (float): Minimum percentage of contact residues to select as hotspots. Defaults to 0.0.
        max_perc (float): Maximum percentage of contact residues to select as hotspots. Defaults to 0.2.
        contact_prob (float): Probability of including contact composition features. Defaults to 0.5.
        contact_dropout_prob (float): Probability of zeroing out contact features during training
            to enable inference without contact features. Defaults to 0.5.
        hotspot_dropout_prob (float): Probability of zeroing out hotspot features during training
            to enable inference without hotspot features. Defaults to 0.5.

    Note:
        - Requires multi-chain structures (graph.chains.max() > 0)
        - Uses target_mask (collapsed to residue level) to distinguish between target and binder chains
        - Contact composition is based on amino acid polarity classification (apolar, polar, charged)
        - Hotspots are selected from interface contact residues when available, otherwise random from target chain
        - Features are always computed normally, then dropout is applied at the end:
          * Contact dropout zeroes out contact_mask and contact features
          * Hotspot dropout zeroes out target_hotspots features
          * Dropouts are applied independently at the end of computation
    """

    def __init__(
        self,
        cutoff: float = 8.0,
        min_perc: float = 0,
        max_perc: float = 0.2,
        contact_prob: float = 0.5,
        contact_dropout_prob: float = 0.5,
        hotspot_dropout_prob: float = 0.5,
    ) -> None:
        self.cutoff = cutoff
        self.min_perc = min_perc
        self.max_perc = max_perc
        self.contact_prob = contact_prob
        self.contact_dropout_prob = contact_dropout_prob
        self.hotspot_dropout_prob = hotspot_dropout_prob

    def __call__(self, graph: Data) -> Data:
        if graph.chains.max() == 0:
            graph.contact_mask = torch.zeros(len(graph.residues))
            graph.contact = torch.zeros(len(graph.residues), 4)
            graph.hotspot_mask = torch.zeros(len(graph.residues))
            return graph

        # Always compute contact and hotspot features normally
        target_chain = graph.target_mask.any(dim=-1)  # [n] - collapse atom dim to get residue-level target mask
        binder_chain = ~target_chain
        target_coords = graph.coords[target_chain]  # [n_target, 37, 3]
        binder_coords = graph.coords[binder_chain]  # [n_binder, 37, 3]
        target_coord_mask = graph.coord_mask[target_chain].bool()  # [n_target, 37]
        binder_coord_mask = graph.coord_mask[binder_chain].bool()  # [n_binder, 37]

        # Compute pairwise distances between target and binder CA atoms
        # target_coords: [n_target, 37, 3] -> [n_target, 1, 3]
        # binder_coords: [n_binder, 37, 3] -> [1, n_binder, 1, 3]
        pair_dists = torch.norm(
            target_coords[:, None, 1, :] - binder_coords[None, :, 1, :],
            dim=-1,
        )  # [n_target, n_binder]

        contact_mask = (
            (pair_dists < self.cutoff) & target_coord_mask[:, None, 1] & binder_coord_mask[None, :, 1]
        )  # [n_target, n_binder]

        binder_contact_mask = contact_mask.any(dim=0)  # [n_binder]
        target_contact_mask = contact_mask.any(dim=1)  # [n_target]

        # Compute contact features
        graph.contact_mask = torch.zeros(len(graph.residues), dtype=torch.bool)
        graph.contact_mask[binder_chain] = binder_contact_mask
        graph.contact_mask[target_chain] = target_contact_mask

        contact_residue_types = [
            AA_CHARACTER_PROTORP.get(graph.residues[idx])
            for idx in torch.arange(len(graph.residues))[binder_chain][binder_contact_mask]
        ]
        counts = Counter([x for x in contact_residue_types if x is not None])
        total_contacts = sum(counts.values())
        apolar_contacts = counts.get("A", 0)
        polar_contacts = counts.get("P", 0)
        charged_contacts = counts.get("C", 0)
        graph.contact = torch.zeros(len(graph.residues), 4)
        if total_contacts > 0 and torch.rand(1) <= self.contact_prob:
            graph.contact[binder_chain] = torch.tensor(
                [
                    total_contacts,
                    apolar_contacts / total_contacts,
                    polar_contacts / total_contacts,
                    charged_contacts / total_contacts,
                ]
            )

        # Compute hotspot features
        if target_contact_mask.sum() > 0:
            # Use contact-based hotspot selection
            unique_target_hotspots = torch.arange(len(graph.residues))[target_chain][target_contact_mask]
            hotspot_perc = random.uniform(self.min_perc, self.max_perc)
            num_to_select = int(target_contact_mask.sum().item() * hotspot_perc)
            perm = torch.randperm(len(unique_target_hotspots))
            sampled_target_hotspots = unique_target_hotspots[perm[:num_to_select]]
        else:
            # Fallback to random hotspot selection from target chain
            target_residues = torch.arange(len(graph.residues))[target_chain]
            hotspot_perc = random.uniform(self.min_perc, self.max_perc)
            num_to_select = int(len(target_residues) * hotspot_perc)
            perm = torch.randperm(len(target_residues))
            sampled_target_hotspots = target_residues[perm[:num_to_select]]

        graph.hotspot_mask = torch.zeros(len(graph.chains))
        graph.hotspot_mask[sampled_target_hotspots] = 1

        # Apply dropout at the end - zero out features based on dropout probabilities
        if torch.rand(1) < self.contact_dropout_prob:
            graph.contact_mask = torch.zeros(len(graph.residues), dtype=torch.bool)
            graph.contact = torch.zeros(len(graph.residues), 4)

        if torch.rand(1) < self.hotspot_dropout_prob:
            graph.hotspot_mask = torch.zeros(len(graph.chains))

        return graph


class CenteringTransform(BaseTransform):
    """Centers protein structures based on one of their chains or a provided mask."""

    def __init__(
        self,
        center_mode: str = "full",
        data_mode: str = "bb_ca",
        variance_perturbation: float = 0.01,
    ) -> None:
        """Initializes the transform with the chain ID to center on.
        Args:
            center_mode (str): type of centering to perform. Options:
                - "full": Center on all atoms
                - "random_chain": Center on a random chain
                - "random_unique_chain": Center on a random unique chain
                - "motif": Center on motif mask if available
                - "target": Center on target mask if available
                - "stochastic_centering": center whole protein, then add stochastic translation
            data_mode (str): The data to center on. Options:
                - "bb_ca": Center on CA atoms only
                - "ligand_atom37": Center on ligand atom37 atoms only
                - "all-atom": Center on all atoms
            variance_perturbation (float): Variance of the stochastic translation if enabled. Defaults to 0.01.
        """
        self.center_mode = center_mode
        self.data_mode = data_mode
        self.variance_perturbation = variance_perturbation

    def __call__(self, graph: Data) -> Data:
        """Centers the graph based on the center mode and data mode.
        Args:
            graph (Data): The graph to center

        Returns:
            Data: The centered graph
        """
        # set the correct mask for centering depending on task
        if self.center_mode == "full" or self.center_mode == "stochastic_centering":
            if self.data_mode == "bb_ca":
                centering_mask = graph.coord_mask[:, 1]
            else:
                centering_mask = graph.coord_mask.flatten(0, 1)
        elif self.center_mode == "random_chain":
            # TODO: the way to randomly choose chains is wrong
            # TODO: there seems a shape mismatch in centering_mask, to be checked
            # choose a random chain
            random_chain = np.random.choice(graph.chains[-1].item() + 1)
            centering_mask = graph.coord_mask.flatten(0, 1)
            # set the masks to 0 for all chains except the random chain
            centering_mask[graph.chains != random_chain] = False
        elif self.center_mode == "random_unique_chain":
            # TODO: there seems a shape mismatch in centering_mask, to be checked
            # choose a random unique chain
            names, counts = np.unique(graph.chain_names, return_counts=True)
            unique_names = names[counts == 1]
            if len(unique_names) == 0:
                raise ValueError(f"No unique chain found in pdb: {graph.id}")
            random_name = np.random.choice(unique_names)
            random_chain = graph.chain_names.index(random_name)
            centering_mask = graph.coord_mask.flatten(0, 1)
            # set the masks to 0 for all chains except the random chain
            centering_mask[graph.chains != random_chain] = False
        elif self.center_mode == "motif":
            if not hasattr(graph, "motif_mask"):
                raise ValueError("Motif mask not found in graph. Apply MotifMaskTransform first.")
            centering_mask = graph.motif_mask.flatten(0, 1)
        elif self.center_mode == "target":
            if not hasattr(graph, "target_mask") and not hasattr(graph, "target_residue_mask"):
                raise ValueError("Target mask not found in graph. Apply TargetMaskTransform first.")
            if self.data_mode == "ligand_atom37":
                target_mask = graph.target_residue_mask
            else:
                target_mask = graph.target_mask
            # Normalize: target_mask can be (N,) from ligand pipeline or (N, 37) from protein pipeline
            # if target_mask.dim() == 1:
            #     # 1D mask (N,) - expand to (N, 37) using coord_mask
            #     target_mask = target_mask[:, None] * graph.coord_mask.bool()
            if target_mask.dim() > 1:
                centering_mask = target_mask.flatten(0, 1)
            else:
                centering_mask = target_mask
        else:
            raise ValueError(f"Invalid center mode {self.center_mode}")

        # Store original shape for reshaping later
        original_coords_shape = graph.coords_nm.shape

        # set the correct data mode for centering
        if self.data_mode == "bb_ca" or self.data_mode == "ligand_atom37":
            coords = graph.coords_nm[:, 1, :]
        elif self.data_mode == "all-atom":
            coords = graph.coords_nm.flatten(0, 1)
        else:
            raise ValueError(f"Invalid data mode {self.data_mode}")
        # get the mean of the selected chain
        masked_mean = mean_w_mask(coords, centering_mask.bool(), keepdim=True)
        # If stochastic centering, add random translation to masked mean that is used for centering
        if self.center_mode == "stochastic_centering":
            translation = torch.normal(
                mean=0.0,
                std=self.variance_perturbation**0.5,
                size=(3,),
                dtype=graph.coords.dtype,
                device=graph.coords.device,
            )
            masked_mean += translation
            graph.stochastic_translation = translation
        # substract the mean of that chain from all coordinates
        if self.data_mode == "bb_ca":
            graph["coords_nm"] -= masked_mean
            graph["coords_nm"] = graph["coords_nm"] * graph["coord_mask"][..., None]
        else:  # all-atom
            graph["coords_nm"] = graph["coords_nm"].flatten(0, 1) - masked_mean
            graph["coords_nm"] = graph["coords_nm"].view(original_coords_shape)
            graph["coords_nm"] = graph["coords_nm"] * graph["coord_mask"][..., None]
        return graph


class MotifMaskTransform(BaseTransform):
    """
    Creates a motif mask for a protein structure, supporting multiple residue and atom selection strategies.

    Args:
        atom_selection_mode (str): How to select atoms within each motif residue. Options:
            - "random": Randomly select between 1 and all available atoms per residue.
            - "backbone": Select only backbone atoms (N, CA, C, O).
            - "sidechain": Select only sidechain atoms (all non-backbone atoms).
            - "all": Select all available atoms.
            - "ca_only": Select only CA atom if available.
            - "tip_atoms": Select only the tip atoms of sidechains (e.g., OH for Ser, NH2 for Arg).
            - "bond_graph": Use a chemically-aware expansion from a seed atom using the residue's bond graph.
        residue_selection_mode (str): How to select which residues are in the motif. Options:
            - "relative_fraction": Select a fraction of residues (see motif_min_pct_res, motif_max_pct_res, and segment logic applies).
            - "absolute_number": Select a fixed number of residues (see motif_min_n_res, motif_max_n_res, each residue is its own segment).
        motif_prob (float, optional): Probability of creating a motif. Defaults to 1.0.
        # Used if residue_selection_mode == "relative_fraction":
        motif_min_pct_res (float, optional): Minimum percentage of residues in motif. Defaults to 0.05.
        motif_max_pct_res (float, optional): Maximum percentage of residues in motif. Defaults to 0.5.
        motif_min_n_seg (int, optional): Minimum number of segments in motif. Defaults to 1.
        motif_max_n_seg (int, optional): Maximum number of segments in motif. Defaults to 4.
        # Used if residue_selection_mode == "absolute_number":
        motif_min_n_res (int, optional): Minimum number of residues in motif. Defaults to 1.
        motif_max_n_res (int, optional): Maximum number of residues in motif. Defaults to 8.

    Returns:
        The input graph with the following attributes added:
            - motif_mask: Binary tensor of shape (num_res, 37) indicating which atoms are in the motif.


    Notes:
        - For 'bond_graph' atom selection, the motif is expanded from a seed atom using the residue's bond graph (from AlphaFold's stereo_chemical_props).
        - In 'absolute_number' mode, each selected residue is its own segment and residues are chosen randomly (not necessarily contiguous).
        - In 'relative_fraction' mode, the selected residues are grouped into contiguous segments.
    """

    def __init__(
        self,
        atom_selection_mode: Literal[
            "random",
            "backbone",
            "sidechain",
            "all",
            "ca_only",
            "tip_atoms",
            "bond_graph",
        ] = "ca_only",
        residue_selection_mode: Literal["relative_fraction", "absolute_number"] = "relative_fraction",
        motif_prob: float = 1.0,
        motif_min_pct_res: float = 0.05,
        motif_max_pct_res: float = 0.5,
        motif_min_n_seg: int = 1,
        motif_max_n_seg: int = 4,
        motif_min_n_res: int = 1,
        motif_max_n_res: int = 8,
    ):
        self.atom_selection_mode = atom_selection_mode
        self.residue_selection_mode = residue_selection_mode
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg
        self.motif_min_n_res = motif_min_n_res
        self.motif_max_n_res = motif_max_n_res

        # Define backbone atom indices based on atom_types from residue_constants
        self.backbone_atoms = [
            residue_constants.atom_types.index("N"),
            residue_constants.atom_types.index("CA"),
            residue_constants.atom_types.index("C"),
            residue_constants.atom_types.index("O"),
        ]
        self.ca_index = residue_constants.atom_types.index("CA")

    def _select_atoms(self, available_atoms: torch.Tensor, residue_idx: int = None, graph: Data = None) -> list[int]:
        """Select atoms for a residue based on the specified mode.

        Args:
            available_atoms (torch.Tensor): Tensor of available atom indices.
            residue_idx (int, optional): Residue index in the graph (needed for tip_atoms mode).
            graph (Data, optional): The full graph (needed for tip_atoms mode).

        Returns:
            List[int]: List of selected atom indices.
        """
        if self.atom_selection_mode == "random":
            n_atoms = random.randint(1, len(available_atoms))
            return random.sample(available_atoms.tolist(), n_atoms)

        elif self.atom_selection_mode == "backbone":
            # Select only backbone atoms that are available
            return [i for i in self.backbone_atoms if i in available_atoms]

        elif self.atom_selection_mode == "sidechain":
            # Select only sidechain atoms (all non-backbone atoms)
            sidechain_atoms = [i for i in available_atoms if i not in self.backbone_atoms]
            if len(sidechain_atoms) > 0:
                n_atoms = random.randint(1, len(sidechain_atoms))
                return random.sample(sidechain_atoms, n_atoms)
            return []

        elif self.atom_selection_mode == "all":
            # Select all available atoms
            return available_atoms.tolist()

        elif self.atom_selection_mode == "ca_only":
            # Select only CA atom if available
            return [self.ca_index] if self.ca_index in available_atoms else []

        elif self.atom_selection_mode == "tip_atoms":
            # Select only tip atoms of sidechains based on residue type
            if graph is None or residue_idx is None:
                raise ValueError("graph and residue_idx must be provided for tip_atoms mode")

            # Get residue type and convert to residue name
            res_type_idx = graph.residue_type[residue_idx].item()
            resname = residue_constants.restype_1to3.get(residue_constants.restypes[res_type_idx], "UNK")

            # Get tip atoms for this residue type
            tip_atom_names = SIDECHAIN_TIP_ATOMS.get(resname, [])

            # Map atom names to indices
            tip_atom_indices = []
            for atom_name in tip_atom_names:
                if atom_name in atom_types:
                    tip_atom_indices.append(atom_types.index(atom_name))

            # Filter for only available atoms
            selected_atoms = [i for i in tip_atom_indices if i in available_atoms]
            return selected_atoms
        elif self.atom_selection_mode == "ame":
            # Select only tip atoms of sidechains based on residue type
            if graph is None or residue_idx is None:
                raise ValueError("graph and residue_idx must be provided for tip_atoms mode")

            # Get residue type and convert to residue name
            res_type_idx = graph.residue_type[residue_idx].item()
            resname = residue_constants.restype_1to3.get(residue_constants.restypes[res_type_idx], "UNK")

            # Get tip atoms for this residue type
            tip_atom_names = AME_ATOMS.get(resname, [])
            tip_atom_names = random.choice(tip_atom_names)  # randomly select one of the possible atom configurations
            # Map atom names to indices
            tip_atom_indices = []
            for atom_name in tip_atom_names:
                if atom_name in atom_types:
                    tip_atom_indices.append(atom_types.index(atom_name))

            # Filter for only available atoms
            selected_atoms = [i for i in tip_atom_indices if i in available_atoms]
            return selected_atoms

        elif self.atom_selection_mode == "bond_graph":
            if graph is None or residue_idx is None:
                raise ValueError("graph and residue_idx must be provided for bond_graph mode")
            # Always use backbone O atom index for this residue
            ref_atom_idx = residue_constants.atom_order["O"]
            if ref_atom_idx not in available_atoms:
                ref_atom_idx = residue_constants.atom_order["CA"]
            ref_atom_coord = graph.coords_nm[residue_idx, ref_atom_idx, :]
            atom_coords = graph.coords_nm[residue_idx, available_atoms, :]
            dists = torch.norm(atom_coords - ref_atom_coord, dim=-1)
            # 80%: farthest from ref_atom, 20%: random
            if random.random() < 0.8:
                seed_atom_idx = torch.argmax(dists).item()
            else:
                seed_atom_idx = random.randint(0, len(available_atoms) - 1)
            seed_atom = available_atoms[seed_atom_idx].item()
            # Build bond graph using residue_constants
            res_type_idx = graph.residue_type[residue_idx].item()
            resname = residue_constants.restype_1to3.get(residue_constants.restypes[res_type_idx], "UNK")
            residue_bonds, _, _ = residue_constants.load_stereo_chemical_props()
            bonds = residue_bonds.get(resname, [])
            # Map atom names to local indices in available_atoms
            atom_name_to_local_idx = {
                residue_constants.atom_types[atom_idx]: i
                for i, atom_idx in enumerate(available_atoms.tolist())
                if atom_idx < len(residue_constants.atom_types)
            }
            n_atoms = len(available_atoms)
            adj = torch.zeros((n_atoms, n_atoms), dtype=torch.bool)
            for bond in bonds:
                a1 = bond.atom1_name
                a2 = bond.atom2_name
                if a1 in atom_name_to_local_idx and a2 in atom_name_to_local_idx:
                    i = atom_name_to_local_idx[a1]
                    j = atom_name_to_local_idx[a2]
                    adj[i, j] = True
                    adj[j, i] = True
            # If no bonds found, fallback to fully connected
            if adj.sum() == 0:
                adj = torch.ones((n_atoms, n_atoms), dtype=torch.bool)
                torch.diagonal(adj).fill_(0)
            # Map atom indices to local indices
            atom_idx_map = {atom.item(): i for i, atom in enumerate(available_atoms)}
            # BFS expansion from seed_atom
            n_expand = np.random.geometric(p=0.5)
            visited = set([seed_atom])
            queue = [seed_atom]
            while len(visited) < min(n_expand, len(available_atoms)) and queue:
                current = queue.pop(0)
                current_local = atom_idx_map[current]
                neighbors = [available_atoms[i].item() for i in range(n_atoms) if adj[current_local, i]]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    if len(visited) >= min(n_expand, len(available_atoms)):
                        break
            return list(visited)

        else:
            raise ValueError(f"Unknown atom selection mode: {self.atom_selection_mode}")

    def __call__(self, graph: Data) -> Data:
        if random.random() > self.motif_prob:
            motif_mask = torch.zeros_like(graph.coord_mask)
            graph.motif_mask = motif_mask.bool()
            return graph

        num_res = graph.coords_nm.shape[0]
        if self.residue_selection_mode == "relative_fraction":
            motif_n_res = int(
                random.random() * (num_res * self.motif_max_pct_res - num_res * self.motif_min_pct_res)
                + num_res * self.motif_min_pct_res
            )
            motif_n_res = min(motif_n_res, num_res)
            # Segment logic: contiguous segments within the selected motif region
            motif_n_seg = int(
                random.random() * (min(motif_n_res, self.motif_max_n_seg) - self.motif_min_n_seg + 1)
                + self.motif_min_n_seg
            )
            indices = np.sort(random.sample(range(1, motif_n_res), motif_n_seg - 1)) if motif_n_seg > 1 else []
            indices = (
                np.concatenate([[0], indices, [motif_n_res]]).astype(int)
                if motif_n_seg > 1
                else np.array([0, motif_n_res])
            )
            segments = []
            for i in range(len(indices) - 1):
                start, end = indices[i], indices[i + 1]
                segment_length = end - start
                segments.append("".join(["1"] * segment_length))
            segments.extend(["0"] * (num_res - motif_n_res))
            random.shuffle(segments)
            motif_sequence_mask = torch.tensor([int(elt) for elt in "".join(segments)]).bool()
        elif self.residue_selection_mode == "absolute_number":
            motif_n_res = random.randint(self.motif_min_n_res, self.motif_max_n_res)
            motif_n_res = min(motif_n_res, num_res)
            # Randomly pick motif_n_res unique residue indices
            motif_indices = random.sample(range(num_res), motif_n_res)
            motif_sequence_mask = torch.zeros(num_res, dtype=torch.bool)
            motif_sequence_mask[motif_indices] = True
        else:
            raise ValueError(f"Unknown residue_selection_mode: {self.residue_selection_mode}")
        motif_mask = torch.zeros_like(graph.coord_mask)
        for res_idx in torch.where(motif_sequence_mask)[0]:
            available_atoms = torch.where(graph.coord_mask[res_idx])[0]
            if len(available_atoms) == 0:
                continue
            if self.atom_selection_mode in ["tip_atoms", "bond_graph", "ame"]:
                selected_atoms = self._select_atoms(available_atoms, residue_idx=res_idx.item(), graph=graph)
            else:
                selected_atoms = self._select_atoms(available_atoms)
            motif_mask[res_idx, selected_atoms] = True
        motif_mask = motif_mask.bool()
        graph.motif_mask = motif_mask.bool()  # [n, 37]
        return graph


class TargetMaskTransform(BaseTransform):
    """Creates a target mask for a protein structure.

    The target mask is a binary tensor of shape (num_res, 37) indicating which atoms
    are part of the selected target. The mask is created by:
    1. Selecting target based on the selection_mode
    2. For each selected residue, selecting a subset of its available atoms
       based on the atom_selection_mode and coord_mask
    3. Create a binary mask indicating the selected atoms

    The transform can be used to create conditional target masks for protein structure prediction
    or design tasks. The target selection process follows these steps:
    1. With probability target_prob, create a target mask (otherwise return empty mask)
    2. Select target based on the selection_mode
    3. For each selected residue, use the atom_selection_mode to choose which atoms to include
    4. Create a binary mask indicating the selected atoms

    The transform adds the following attributes to the graph:
    - target_mask: Binary tensor of shape (num_res, 37) indicating which atoms are in the selected target
    """

    def __init__(
        self,
        chain_prob: float = 0.8,
        selection_mode: Literal["random", "sequential", "interface", "all"] = "random",
        min_chains: int = 1,
        max_chains: int = None,
        interface_threshold: float = 8.0,
        atom_selection_mode: Literal[
            "random", "backbone", "sidechain", "all", "ca_only", "tip_atoms", "ame"
        ] = "ca_only",
    ):
        """
        Initialize ChainMaskTransform with parameters.

        Args:
            chain_prob (float, optional): Probability of creating a chain mask. Defaults to 0.8.
            selection_mode (str, optional): Mode for selecting chains. Options:
                - "random": Randomly select between min_chains and max_chains chains
                - "sequential": Select chains in sequential order
                - "interface": Select chains that form interfaces with other chains
                - "all": Select all chains
            min_chains (int, optional): Minimum number of chains to select. Defaults to 1.
            max_chains (int, optional): Maximum number of chains to select. If None, uses all available chains. Defaults to None.
            interface_threshold (float, optional): Distance threshold for interface detection. Defaults to 8.0.
            atom_selection_mode (str, optional): Mode for selecting atoms in each residue. Options:
                - "random": Randomly select between 1 and all available atoms
                - "backbone": Select only backbone atoms (N, CA, C, O)
                - "sidechain": Select only sidechain atoms
                - "all": Select all available atoms
                - "ca_only": Select only CA atoms
                - "tip_atoms": Select only the tip atoms of sidechains (e.g., OH for Ser, NH2 for Arg)
                - "ame": Select only the AME atoms of sidechains (e.g., CZ for Phe, CE1 for Val)
        """
        self.chain_prob = chain_prob
        self.selection_mode = selection_mode
        self.min_chains = min_chains
        self.max_chains = max_chains
        self.interface_threshold = interface_threshold
        self.atom_selection_mode = atom_selection_mode

        # Define backbone atom indices based on atom_types from residue_constants
        self.backbone_atoms = [
            atom_types.index("N"),
            atom_types.index("CA"),
            atom_types.index("C"),
            atom_types.index("O"),
        ]
        self.ca_index = atom_types.index("CA")

    def _select_atoms(self, available_atoms: torch.Tensor, residue_idx: int = None, graph: Data = None) -> list[int]:
        """Select atoms for a residue based on the specified mode.

        Args:
            available_atoms (torch.Tensor): Tensor of available atom indices.
            residue_idx (int, optional): Residue index in the graph (needed for tip_atoms mode).
            graph (Data, optional): The full graph (needed for tip_atoms mode).

        Returns:
            List[int]: List of selected atom indices.
        """
        if self.atom_selection_mode == "random":
            n_atoms = random.randint(1, len(available_atoms))
            return random.sample(available_atoms.tolist(), n_atoms)

        elif self.atom_selection_mode == "backbone":
            # Select only backbone atoms that are available
            return [i for i in self.backbone_atoms if i in available_atoms]

        elif self.atom_selection_mode == "sidechain":
            # Select only sidechain atoms (all non-backbone atoms)
            sidechain_atoms = [i for i in available_atoms if i not in self.backbone_atoms]
            if len(sidechain_atoms) > 0:
                n_atoms = random.randint(1, len(sidechain_atoms))
                return random.sample(sidechain_atoms, n_atoms)
            return []

        elif self.atom_selection_mode == "all":
            # Select all available atoms
            return available_atoms.tolist()

        elif self.atom_selection_mode == "ca_only":
            # Select only CA atom if available
            return [self.ca_index] if self.ca_index in available_atoms else []

        elif self.atom_selection_mode == "tip_atoms":
            # Select only tip atoms of sidechains based on residue type
            if graph is None or residue_idx is None:
                raise ValueError("graph and residue_idx must be provided for tip_atoms mode")

            # Get residue type and convert to residue name
            res_type_idx = graph.residue_type[residue_idx].item()
            resname = residue_constants.restype_1to3.get(residue_constants.restypes[res_type_idx], "UNK")

            # Get tip atoms for this residue type
            tip_atom_names = SIDECHAIN_TIP_ATOMS.get(resname, [])

            # Map atom names to indices
            tip_atom_indices = []
            for atom_name in tip_atom_names:
                if atom_name in atom_types:
                    tip_atom_indices.append(atom_types.index(atom_name))

            # Filter for only available atoms
            selected_atoms = [i for i in tip_atom_indices if i in available_atoms]
            return selected_atoms

        else:
            raise ValueError(f"Unknown atom selection mode: {self.atom_selection_mode}")

    def _get_interface_chains(self, graph: Data) -> torch.Tensor:
        """Get chains that form interfaces with other chains."""
        positions = graph.coords
        asym_id = graph.chains
        atom_mask = graph.coord_mask

        # Calculate pairwise distances between CA atoms
        ca_positions = positions[:, 1, :]  # [n, 3]
        pairwise_dists = torch.cdist(ca_positions, ca_positions)

        # Create mask for different chains
        diff_target_mask = asym_id[..., None] != asym_id[..., None, :]

        # Create mask for valid atoms
        pair_mask = atom_mask[:, 1] * atom_mask[:, 1].unsqueeze(-1)

        # Combine masks
        mask = (diff_target_mask * pair_mask).bool()

        # Find minimum distance per chain pair
        min_dist_per_chain = torch.where(mask, pairwise_dists, torch.inf)
        min_dist_per_chain = min_dist_per_chain.min(dim=0)[0]

        # Get chains that form interfaces
        interface_chains = torch.unique(asym_id[min_dist_per_chain < self.interface_threshold])
        return interface_chains

    def _select_chains(self, graph: Data) -> torch.Tensor:
        """Select chains based on the specified mode."""
        unique_chains = torch.unique(graph.chains)
        num_chains = len(unique_chains)

        if num_chains <= 1:  # if only one chain, we use it as binder
            return torch.tensor([], dtype=torch.long)

        if self.max_chains is None:
            self.max_chains = num_chains - 1

        n_chains = min(max(self.min_chains, 1), min(self.max_chains, num_chains - 1))

        if self.selection_mode == "random":
            # Select one chain randomly and return all OTHER chains as targets
            excluded_chain = unique_chains[np.random.choice(num_chains)]
            # Return all chains except the excluded one
            return unique_chains[unique_chains != excluded_chain]

        elif self.selection_mode == "sequential":
            # Select first n_chains
            return unique_chains[:n_chains]

        elif self.selection_mode == "interface":
            # Get interface chains
            interface_chains = self._get_interface_chains(graph)
            if len(interface_chains) == 0:
                # Fallback to random if no interfaces found
                excluded_chain = unique_chains[np.random.choice(num_chains)]
                return unique_chains[unique_chains != excluded_chain]
            # Select up to n_chains interface chains
            return interface_chains[:n_chains]

        elif self.selection_mode == "all":
            return unique_chains

        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

    def __call__(self, graph: Data) -> Data:
        """Creates a chain mask for the protein structure.

        Args:
            graph (Data): PyG Data object containing the protein structure.

        Returns:
            Data: The same graph with added target_mask, x_target, seq_target_mask, seq_target attributes.
        """
        num_residues = graph.chains.shape[0]

        # Create empty mask if probability check fails or no chains
        if graph.chains.max() == 0 or torch.rand(1) > self.chain_prob:
            graph.target_mask = torch.zeros((num_residues, 37), dtype=torch.bool)
            return graph

        # Select chains based on mode
        selected_chains = self._select_chains(graph)

        # Create initial chain mask for residues
        residue_mask = torch.isin(graph.chains, selected_chains)

        # Create full chain mask for all atoms
        target_mask = torch.zeros((num_residues, 37), dtype=torch.bool)

        # For each residue in selected chains, select atoms based on the specified mode
        for res_idx in torch.where(residue_mask)[0]:
            # Get available atoms for this residue
            available_atoms = torch.where(graph.coord_mask[res_idx])[0]
            if len(available_atoms) == 0:
                continue
            # Select atoms based on the specified mode
            if self.atom_selection_mode == "tip_atoms":
                selected_atoms = self._select_atoms(available_atoms, residue_idx=res_idx.item(), graph=graph)
            else:
                selected_atoms = self._select_atoms(available_atoms)
            # Update chain mask
            target_mask[res_idx, selected_atoms] = True

        graph.target_mask = target_mask.bool()
        return graph


class ExtractMotifCoordinatesTransform(BaseTransform):
    """
    Extracts motif coordinates and sequence information from a graph using the motif_mask.
    Adds x_motif, seq_motif_mask, and seq_motif attributes.

    Args:
        compact_mode (bool): If True, only keeps residues where motif_mask is True,
                           collapsing the sequence dimension. If False, keeps original
                           sequence length with zeros for non-motif residues.
    """

    def __init__(self, compact_mode: bool = False):
        self.compact_mode = compact_mode

    def __call__(self, graph: Data) -> Data:
        if not hasattr(graph, "motif_mask") or graph.motif_mask is None:
            raise ValueError("motif_mask not found in graph. Apply MotifMaskTransform first.")

        # Compute per-residue motif mask
        motif_residue_mask = graph.motif_mask.sum(dim=-1).bool()  # [n]

        if self.compact_mode:
            # Compact mode: only keep residues where motif_mask is True
            if motif_residue_mask.any():
                # Extract only motif residues
                graph.x_motif = graph.coords_nm[motif_residue_mask]  # [n_motif, 37, 3]
                graph.motif_mask = graph.motif_mask[motif_residue_mask]  # [n_motif, 37]
                graph.seq_motif = graph.residue_type[motif_residue_mask]  # [n_motif]
                graph.seq_motif_mask = torch.ones(
                    motif_residue_mask.sum(),
                    dtype=torch.bool,
                    device=graph.residue_type.device,
                )  # [n_motif] all True
                graph.register_token_group(
                    "motif",
                    [
                        "x_motif",
                        "motif_mask",
                        "seq_motif",
                        "seq_motif_mask",
                    ],
                )
            else:
                # No motif residues - create empty tensors
                device = graph.coords_nm.device
                graph.x_motif = torch.zeros(0, 37, 3, device=device)  # [0, 37, 3]
                graph.motif_mask = torch.zeros(0, 37, dtype=torch.bool, device=device)  # [0, 37]
                graph.seq_motif = torch.zeros(0, dtype=graph.residue_type.dtype, device=device)  # [0]
                graph.seq_motif_mask = torch.zeros(0, dtype=torch.bool, device=device)  # [0]
        else:
            # Original mode: keep full sequence length, zero out non-motif residues
            graph.x_motif = graph.coords_nm * graph.motif_mask[..., None]  # [n, 37, 3]
            graph.seq_motif_mask = motif_residue_mask  # [n]
            graph.seq_motif = graph.residue_type * graph.seq_motif_mask  # [n]

        # Always add the per-residue mask for compatibility
        graph.motif_residue_mask = motif_residue_mask  # [n] (original sequence length)
        return graph


class ExtractTargetCoordinatesTransform(BaseTransform):
    """
    Extracts target coordinates and sequence information from a graph using the target_mask.
    Adds x_target, seq_target_mask, and seq_target attributes.

    Args:
        compact_mode (bool): If True, only keeps residues where target_mask is True,
                           collapsing the sequence dimension. If False, keeps original
                           sequence length with zeros for non-target residues.
    """

    def __init__(self, compact_mode: bool = False):
        self.compact_mode = compact_mode

    def __call__(self, graph: Data) -> Data:
        if not hasattr(graph, "target_mask") or graph.target_mask is None:
            raise ValueError("target_mask not found in graph. Apply TargetMaskTransform first.")

        # Compute per-residue target mask
        target_residue_mask = graph.target_mask.sum(dim=-1).bool()  # [n]

        if self.compact_mode:
            # Compact mode: only keep residues where target_mask is True
            if target_residue_mask.any():
                # Extract only target residues
                graph.x_target = graph.coords_nm[target_residue_mask]  # [n_target, 37, 3]
                graph.target_mask = graph.target_mask[target_residue_mask]  # [n_target, 37]
                graph.seq_target = graph.residue_type[target_residue_mask]  # [n_target]
                graph.seq_target_mask = torch.ones(
                    target_residue_mask.sum(),
                    dtype=torch.bool,
                    device=graph.residue_type.device,
                )  # [n_target] all True
                if hasattr(graph, "hotspot_mask"):
                    graph.target_hotspot_mask = graph.hotspot_mask[target_residue_mask]
                else:
                    graph.target_hotspot_mask = torch.zeros(0, dtype=torch.bool, device=graph.coords_nm.device)  # [0]
                if hasattr(graph, "residue_pdb_idx"):
                    graph.target_pdb_idx = graph.residue_pdb_idx[target_residue_mask]
                if hasattr(graph, "chains"):
                    graph.target_chains = graph.chains[target_residue_mask]
                # Now do the same for potential ligand features
                if hasattr(graph, "target_charge"):
                    graph.target_charge = graph.target_charge[target_residue_mask]
                if hasattr(graph, "target_atom_name"):
                    graph.target_atom_name = graph.target_atom_name[target_residue_mask]
                if hasattr(graph, "target_bond_order"):
                    graph.target_bond_order = graph.target_bond_order[target_residue_mask, :][:, target_residue_mask]
                if hasattr(graph, "target_bond_mask"):
                    graph.target_bond_mask = graph.target_bond_mask[target_residue_mask, :][:, target_residue_mask]
                if hasattr(graph, "target_laplacian_pe"):
                    graph.target_laplacian_pe = graph.target_laplacian_pe[target_residue_mask]
                if hasattr(graph, "target_residue_mask"):
                    graph.target_residue_mask = graph.target_residue_mask[target_residue_mask]
                target_fields = [
                    "x_target",
                    "target_mask",
                    "seq_target",
                    "seq_target_mask",
                    "target_hotspot_mask",
                    "target_pdb_idx",
                    "target_chains",
                ]
                for f in (
                    "target_charge",
                    "target_atom_name",
                    "target_bond_order",
                    "target_bond_mask",
                    "target_laplacian_pe",
                    "target_residue_mask",
                ):
                    if hasattr(graph, f):
                        target_fields.append(f)
                graph.register_token_group("target", target_fields)
            else:
                # No target residues - create empty tensors
                device = graph.coords_nm.device
                graph.x_target = torch.zeros(0, 37, 3, device=device)  # [0, 37, 3]
                graph.target_mask = torch.zeros(0, 37, dtype=torch.bool, device=device)  # [0, 37]
                graph.seq_target = torch.zeros(0, dtype=graph.residue_type.dtype, device=device)  # [0]
                graph.seq_target_mask = torch.zeros(0, dtype=torch.bool, device=device)  # [0]
                graph.target_hotspot_mask = torch.zeros(0, dtype=torch.bool, device=device)  # [0]
                graph.target_pdb_idx = torch.zeros(0, dtype=torch.long, device=device)  # [0]
                graph.target_chains = torch.zeros(0, dtype=graph.chains.dtype, device=device)  # [0]
                # add fake ligand feautures make sure squares are squared are correct

                graph.target_charge = torch.zeros(0, dtype=torch.float32, device=device)
                graph.target_atom_name = torch.zeros(0, 64 * 4, dtype=torch.float32, device=device)
                graph.target_bond_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
                graph.target_bond_order = torch.zeros(0, 0, dtype=torch.float32, device=device)
                graph.target_laplacian_pe = torch.zeros(0, 32, dtype=torch.float32, device=device)
        else:
            # Original mode: keep full sequence length, zero out non-target residues
            graph.x_target = graph.coords_nm * graph.target_mask[..., None]  # [n, 37, 3]
            graph.seq_target_mask = target_residue_mask  # [n]
            graph.seq_target = graph.residue_type * graph.seq_target_mask  # [n]
            graph.target_hotspot_mask = graph.hotspot_mask * graph.target_mask  # [n]
            graph.target_pdb_idx = graph.residue_pdb_idx  # [n]
            graph.target_chains = graph.chains  # [n]

        # Always add the per-residue mask for compatibility
        graph.target_residue_mask = target_residue_mask  # [n] (original sequence length)
        return graph


# not used
class ExtractTargetCoordinatesLigandTransform(BaseTransform):
    """
    Extracts target coordinates and sequence information from a graph using the target_mask.
    Adds x_target, seq_target_mask, and seq_target attributes.

    Args:
        compact_mode (bool): If True, only keeps residues where target_mask is True,
                           collapsing the sequence dimension. If False, keeps original
                           sequence length with zeros for non-target residues.
    """

    def __init__(self, compact_mode: bool = False):
        self.compact_mode = compact_mode

    def __call__(self, graph: Data) -> Data:
        if not hasattr(graph, "target_mask") or graph.target_mask is None:
            raise ValueError("target_mask not found in graph. Apply TargetMaskTransform first.")

        # Compute per-residue target mask
        target_residue_mask = graph.target_mask.sum(dim=-1).bool()  # [n]
        # from atomworks.ml.transforms.encoding import atom_array_from_encoding
        # from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
        atom_array = atom_array_from_encoding(
            encoded_coord=graph.coords_nm[target_residue_mask],
            encoded_mask=graph.coord_mask[target_residue_mask],
            encoded_seq=graph.residue_type[target_residue_mask],
            encoding=AF2_ATOM37_ENCODING,
        )

        atom_array = atom_array[atom_array.occupancy > 0]
        if "charge" not in atom_array.get_annotation_categories():
            atom_array.set_annotation("charge", np.zeros(len(atom_array), dtype=np.int32))
        bonds = connect_via_residue_names(atom_array)
        adj = bonds.adjacency_matrix()
        ligand_feats = get_af3_raw_molecule_features(atom_array)
        pe = get_laplacian_pe(adj.astype(np.float32))
        bm = bonds.bond_type_matrix()
        bm = np.vectorize(BOND_ORDER_MAP.get)(bm)
        bm = torch.from_numpy(bm).float()

        if self.compact_mode:
            # Compact mode: only keep residues where target_mask is True
            if target_residue_mask.any():
                # Extract only target residues
                # x_target = graph.coords_nm[
                #     target_residue_mask
                # ]
                # graph.x_target = graph.coords_nm[
                #     target_residue_mask
                # ]  # [n_target, 37, 3]
                graph.x_target = torch.from_numpy(atom_array.coord).float()
                graph.target_charge = torch.from_numpy(atom_array.charge).float()
                graph.target_bond_order = bm
                graph.target_laplacian_pe = pe  # torch.from_numpy(pe).float()
                graph.target_bond_mask = torch.from_numpy(adj)  # .float()
                graph.target_atom_name = F.one_hot(
                    torch.from_numpy(ligand_feats["atom_name_chars"]).long(),
                    num_classes=64,
                ).reshape(len(atom_array), 64 * 4)
                graph.seq_target = F.one_hot(
                    torch.from_numpy(ligand_feats["atom_element"]).long(),
                    num_classes=128,
                )
                graph.target_mask = torch.ones(len(atom_array), dtype=torch.bool)
                # graph.seq_target = graph.residue_type[target_residue_mask]  # [n_target]
                # graph.seq_target_mask = torch.ones(
                #     target_residue_mask.sum(),
                #     dtype=torch.bool,
                #     device=graph.residue_type.device,
                # )  # [n_target] all True
                # graph.target_hotspot_mask = graph.hotspot_mask[target_residue_mask]
                # graph.target_pdb_idx = graph.residue_pdb_idx[target_residue_mask]
                # graph.target_chains = graph.chains[target_residue_mask]
            else:
                # No target residues - create empty tensors
                device = graph.coords_nm.device
                graph.x_target = torch.zeros(0, 37, 3, device=device)  # [0, 37, 3]
                graph.target_mask = torch.zeros(0, 37, dtype=torch.bool, device=device)  # [0, 37]
                graph.seq_target = torch.zeros(0, dtype=graph.residue_type.dtype, device=device)  # [0]
                graph.seq_target_mask = torch.zeros(0, dtype=torch.bool, device=device)  # [0]
                graph.target_hotspot_mask = torch.zeros(0, dtype=torch.bool, device=device)  # [0]
                graph.target_pdb_idx = torch.zeros(0, dtype=torch.long, device=device)  # [0]
                graph.target_chains = torch.zeros(0, dtype=graph.chains.dtype, device=device)  # [0]
        else:
            # Original mode: keep full sequence length, zero out non-target residues
            raise ValueError("Non-compact mode not supported for LigandExtractTargetCoordinatesTransform")
            graph.x_target = graph.coords_nm * graph.target_mask[..., None]  # [n, 37, 3]
            graph.seq_target_mask = target_residue_mask  # [n]
            graph.seq_target = graph.residue_type * graph.seq_target_mask  # [n]
            graph.target_hotspot_mask = graph.hotspot_mask * graph.target_mask  # [n]
            graph.target_pdb_idx = graph.residue_pdb_idx  # [n]
            graph.target_chains = graph.chains  # [n]

        # Always add the per-residue mask for compatibility
        graph.target_residue_mask = target_residue_mask  # [n] (original sequence length)
        return graph


class FilterTargetResiduesTransform(BaseTransform):
    """
    Filters target residues from main generation features while keeping target features intact.

    This transform is useful for binder design where:
    - Target features (x_target, target_mask, seq_target, etc.) provide conditioning context
    - Main features (coords_nm, residue_type, mask, etc.) should only include binder residues for generation

    The transform modifies the main protein features to exclude target residues:
    - coords_nm, coords, residue_type, ... -> filtered to binder residues only
    - Target features remain unchanged for use as conditioning in concat features.

    Requirements:
    - Must be applied after TargetMaskTransform and ExtractTargetCoordinatesTransform
    - The graph must have target_mask attribute
    """

    def __call__(self, graph: Data) -> Data:
        """
        Filters target residues from main generation features.

        Args:
            graph (Data): PyG Data object containing the protein structure.

        Returns:
            Data: The same graph with main features filtered to binder residues only.
        """
        # Check if target_mask exists
        if not hasattr(graph, "target_mask") or graph.target_mask is None:
            raise ValueError(
                "target_mask not found in graph. Apply TargetMaskTransform and ExtractTargetCoordinatesTransform first."
            )

        target_residue_mask = graph.target_residue_mask.clone()  # [n] - True for target residues
        binder_residue_mask = ~target_residue_mask  # [n] - True for binder residues
        # Check if there are any binder residues
        if not binder_residue_mask.any():
            logger.warning("No binder residues found after filtering targets. All residues are marked as targets.")
        num_original_residues = target_residue_mask.shape[0]
        # Iterate through all graph attributes and filter per-residue features
        for attr_name in list(graph.keys()):
            # Skip target-related features (they should remain at full length for conditioning)
            if "target" in attr_name.lower():
                continue

            attr_value = getattr(graph, attr_name)
            # Handle tensor attributes
            if (
                isinstance(attr_value, torch.Tensor)
                and attr_value.dim() > 0  # Check tensor has at least one dimension
                and attr_value.shape[0] == num_original_residues
            ):
                setattr(graph, attr_name, attr_value[binder_residue_mask])

            # Handle list attributes
            elif isinstance(attr_value, list) and len(attr_value) == num_original_residues:
                setattr(
                    graph,
                    attr_name,
                    [attr_value[i] for i in torch.where(binder_residue_mask)[0]],
                )

        # Update mask_dict if it exists
        if hasattr(graph, "mask_dict"):
            for key, mask_tensor in graph.mask_dict.items():
                if (
                    isinstance(mask_tensor, torch.Tensor)
                    and mask_tensor.dim() > 0  # Check tensor has at least one dimension
                    and mask_tensor.shape[0] == num_original_residues
                ):
                    graph.mask_dict[key] = mask_tensor[binder_residue_mask]

        # Store information about the filtering for potential later use
        graph.original_residue_count = num_original_residues
        graph.binder_residue_count = binder_residue_mask.sum().item()
        graph.target_residue_count = target_residue_mask.sum().item()

        # Remove full-complex masks - they served their purpose and would cause
        # shape mismatches in collate (neither binder-sized nor target-sized)
        if hasattr(graph, "target_residue_mask"):
            delattr(graph, "target_residue_mask")
        if hasattr(graph, "binder_residue_mask"):
            delattr(graph, "binder_residue_mask")

        return graph


class SyntheticCroppingTransform(BaseTransform):
    """
    Contiguous cropping on the binder chain, and both contiguous and spatial cropping on the target chains.

    1. Based on the interface threshold, select interface residues.
    2. Randomly select one residue from the interface residues and use its corresponding chain as the binder chain, and all the other chains as the target chains.
    3. Do contiguous cropping on the binder chain, with its length being randomly sampled from Uniform(binder_min_length, binder_max_length).
        Minimal number of residues (=binder_padding_length) on the both sides of the binder seed residue on the binder chain will be included.
    4. For the target chains, will do both contiguous cropping and spatial cropping.
        a): First do spatial cropping on the target chains, select target residues with distance to the whole binder chain less than target_spatial_crop_threshold.
            If there are more than (crop_size - binder_length) residues selected, take the top (crop_size - binder_length) residues.
        b): Then do contiguous cropping on the target chains for each segments of the target chains
    """

    def __init__(
        self,
        crop_size: int = 384,
        interface_threshold: float = 5.0,
        dist_mode: Literal["bb_ca", "all-atom"] = "bb_ca",
        data_mode: Literal["bb_ca", "all-atom"] = "all-atom",
        binder_min_length: int = 50,
        binder_max_length: int = 200,
        binder_padding_length: int = 10,
        target_spatial_crop_threshold: float = 15.0,
        target_min_length: int = 50,
        enforce_target_min_length: bool = False,
        max_num_target_chains: int = -1,  # -1 for no limit
        min_num_interface_binder_res: int = 1,  # the minimum number of interface residues on the binder chain
    ):
        """
        Args:
            crop_size: the total number of residues to keep after cropping. Default is 384.
            interface_threshold: the threshold (in Angstrom) for interface residues. Default is 5.0.
            dist_mode: the distance mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom for computing the distance between residues. Default is "bb_ca".
            data_mode: the data mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom. Default is "all-atom".
            binder_min_length: the minimum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 50.
            binder_max_length: the maximum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 200.
            binder_padding_length: the minimal number of residues to include on each side of the binder seed residue when cropping the binder chain. Default is 10.
            target_spatial_crop_threshold: the threshold (in Angstrom) for target spatial cropping, which is the cutoff distanceof target residues to the whole binder chain. Default is 15.0.
            enforce_target_min_length: whether to enforce the target chain to have at least target_min_length residues after cropping.
                If True, target might have more chains than max_num_target_chains. Default is True.
            max_num_target_chains: the maximum number of target chains to include in the cropping, -1 for no limit. Default is -1.
        """
        self.crop_size = crop_size
        self.interface_threshold = interface_threshold
        self.dist_mode = dist_mode
        self.data_mode = data_mode
        self.binder_min_length = binder_min_length
        self.binder_max_length = binder_max_length
        self.binder_padding_length = binder_padding_length
        self.target_spatial_crop_threshold = target_spatial_crop_threshold
        self.target_min_length = target_min_length
        self.enforce_target_min_length = enforce_target_min_length
        self.max_num_target_chains = max_num_target_chains
        self.min_num_interface_binder_res = min_num_interface_binder_res
        assert self.binder_min_length >= 2 * self.binder_padding_length + 1, (
            "binder_min_length must be >= 2 * binder_padding_length + 1"
        )

    def __call__(self, graph: Data) -> Data:
        # In processed graph, binder_chain_id is single character from A to Z. All complexes with >26 chains are removed.
        # We need to convert binder_chain_id to integer for the following operations.
        binder_chain_id = graph.binder_chain_id
        binder_chain_id = ord(binder_chain_id) - ord("A")

        asym_id = graph.chains  # [n_res], each element is a integer representing the chain_id
        n_res = len(graph.chains)  # number of residues
        residues_idxs = torch.arange(n_res)  # [n_res]
        chain_lens = torch.bincount(asym_id)  # [n_chain] number of residues in each chain
        selected_idxs = torch.zeros(n_res, dtype=torch.bool)
        remaining_idxs = torch.ones(n_res, dtype=torch.bool)
        chain_residue_ranges = {}  # {chain_id: [start_idx, end_idx]}
        for chain_id in torch.unique(asym_id).sort().values.tolist():
            chain_mask = asym_id == chain_id
            chain_indices = residues_idxs[chain_mask]
            chain_residue_ranges[chain_id] = [
                int(chain_indices.min()),
                int(chain_indices.max()),
            ]

        # get interface residues, and residue-wise distances: (n_res, n_res)
        interface_residues, min_dist_per_res = self._get_interface_residues(graph)
        # get the re-designed residue indices
        redesigned_res_idx = torch.where(asym_id == binder_chain_id)[0]
        # binder candidate seed residues are the interface residues that are from the re-designed chain
        intersection_mask = (interface_residues.unsqueeze(1) == redesigned_res_idx).any(dim=1)
        binder_candidate_seed_residues = interface_residues[intersection_mask]
        # binder_candidate_seed_residues = asym_id[binder_interface_indices]
        # graph.binder_interface_residues = binder_candidate_seed_residues

        if len(binder_candidate_seed_residues) < self.min_num_interface_binder_res:
            raise ValueError(
                f"Interface residues on the binder chain are too few for pdb: {graph.id}, {len(binder_candidate_seed_residues)} < {self.min_num_interface_binder_res}"
            )

        # randomly select a binder seed residue from interface residues
        # For synthetic data, we selected the re-designed chain as the binder chain.
        binder_seed_residue = random.choice(binder_candidate_seed_residues)

        if (n_res - int(chain_lens[binder_chain_id])) < self.target_min_length and self.enforce_target_min_length:
            raise ValueError(f"Target chain too short for pdb: {graph.id}")

        target_mask = asym_id != binder_chain_id
        # in the case of controlling the number of target chains, binder_mask not necessarily equal to ~target_mask
        # binder_mask = asym_id == binder_chain_id
        if self.data_mode == "all-atom":
            graph.target_mask = target_mask[:, None] * graph.coord_mask.bool()
        else:
            graph.target_mask = target_mask

        if len(graph.chains) < self.crop_size:
            return graph

        # Contiguous cropping on the binder chain
        redesigned_binder_len = torch.sum(asym_id == binder_chain_id)
        # if the re-designed binder chain is too short, use the entire re-designed chain, else use the contiguous cropping
        if redesigned_binder_len > self.binder_min_length:
            binder_length = int(torch.randint(self.binder_min_length, self.binder_max_length + 1, (1,))[0])
            binder_chain_start = chain_residue_ranges[binder_chain_id][0]
            binder_chain_end = chain_residue_ranges[binder_chain_id][1]
            binder_left_length = int(
                torch.randint(
                    self.binder_padding_length,
                    binder_length - self.binder_padding_length - 1 + 1,
                    (1,),
                )[0]
            )
            binder_chain_crop_start = max(binder_chain_start, binder_seed_residue - binder_left_length)
            binder_chain_crop_end = min(binder_chain_end + 1, binder_chain_crop_start + binder_length)
            if binder_chain_crop_end == binder_chain_end + 1:
                binder_chain_crop_start = max(binder_chain_start, binder_chain_crop_end - binder_length)
        else:
            binder_chain_crop_start = redesigned_res_idx[0]
            binder_chain_crop_end = redesigned_res_idx[-1] + 1
        binder_chain_crop_idxs = torch.arange(binder_chain_crop_start, binder_chain_crop_end)
        selected_idxs[binder_chain_crop_idxs] = True

        # Spatial cropping on the target chains first to get the target seed residues
        to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
        target_seed_residues = residues_idxs[target_mask][to_binder_dist < self.target_spatial_crop_threshold]
        target_seed_residue_chains = asym_id[target_seed_residues]
        target_chains = target_seed_residue_chains.unique()
        target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        if self.max_num_target_chains > 0 and len(target_chains) > self.max_num_target_chains:
            # can't use random.shuffle to shuffle torch tensors.
            target_chains = (
                target_chains[torch.randperm(len(target_chains))][: self.max_num_target_chains].sort().values
            )
            target_seed_residues = target_seed_residues[
                (target_seed_residue_chains[:, None] == target_chains[None, :]).any(dim=-1)
            ]
            target_seed_residue_chains = asym_id[target_seed_residues]
            target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        # put chains not included in the target_chains to be False in remaining_idxs
        remaining_idxs[(asym_id[:, None] != target_chains[None, :]).all(dim=-1)] = False
        num_budget = self.crop_size - int(selected_idxs.sum())
        num_remaining = int(remaining_idxs.sum())

        if num_remaining <= num_budget:
            selected_idxs[residues_idxs[target_mask]] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        if len(target_seed_residues) >= num_budget:
            to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
            target_seed_residues = residues_idxs[target_mask][torch.argsort(to_binder_dist)[:num_budget]]
            selected_idxs[target_seed_residues] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        # Get the segments of the target chains
        segments = []
        for chain_id in target_chains:
            for segment in self._convert_to_segments(
                target_seed_residues[target_seed_residue_chains == chain_id].tolist()
            ):
                segments.append(segment)

        # Contiguous cropping on the target chains from the target seed residues/segments
        remaining_idxs[target_seed_residues] = False
        selected_idxs[target_seed_residues] = True
        segment_idxs = list(range(len(segments)))
        segment_used_mask = torch.zeros((len(segments), 2), dtype=torch.bool)  # [left_crop, right_crop]
        random.shuffle(segment_idxs)

        for segment_idx in segment_idxs:
            segment_start, segment_end = segments[segment_idx]
            chain_id = int(asym_id[segment_start])
            chain_start = chain_residue_ranges[chain_id][0]
            chain_end = chain_residue_ranges[chain_id][1]

            orders = [0, 1]
            random.shuffle(orders)
            for order in orders:
                if order == 0:
                    # left crop
                    segment_used_mask[segment_idx, 0] = True
                    if segment_idx == 0:
                        remaining_idxs[chain_start:segment_start] = False
                    else:
                        if segment_used_mask[segment_idx - 1, 1]:  # and segment_used_mask[segment_idx, 0]
                            remaining_idxs[segments[segment_idx - 1][1] : segment_start] = False
                        elif segments[segment_idx - 1][1] < chain_start:
                            remaining_idxs[chain_start:segment_start] = False
                    num_budget = self.crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx >= 1:
                        segment_crop_size_max = min(
                            num_budget,
                            segment_start - chain_start,
                            int((~selected_idxs[(segments[segment_idx - 1][1] + 1) : segment_start]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, segment_start - chain_start)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_start - segment_crop_size) : (segment_start)] = True
                    remaining_idxs[(segment_start - segment_crop_size) : (segment_start)] = False

                else:
                    # right crop
                    segment_used_mask[segment_idx, 1] = True
                    if segment_idx == len(segment_idxs) - 1:
                        remaining_idxs[segment_end : chain_end + 1] = False
                    else:
                        if segment_used_mask[segment_idx + 1, 0]:  # and segment_used_mask[segment_idx, 1]
                            remaining_idxs[segment_end : segments[segment_idx + 1][0]] = False
                        elif segments[segment_idx + 1][0] > chain_end:
                            remaining_idxs[segment_end : chain_end + 1] = False
                    num_budget = self.crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx < len(segment_idxs) - 1:
                        segment_crop_size_max = min(
                            num_budget,
                            chain_end - segment_end,
                            int((~selected_idxs[(segment_end + 1) : segments[segment_idx + 1][0]]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, chain_end - segment_end)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = True
                    remaining_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = False

        graph = self._crop_graph(graph, residues_idxs[selected_idxs])

        return graph

    @staticmethod
    def _convert_to_segments(sorted_integers):
        """
        Convert a list of sorted integers to a combination of ranges.

        Args:
            sorted_integers: List of sorted integers

        Returns:
            List of tuples, each tuple is a range of integers
        """
        segments = []
        start = sorted_integers[0]
        end = sorted_integers[0]

        for i in range(1, len(sorted_integers)):
            if sorted_integers[i] == end + 1:
                end = sorted_integers[i]
            else:
                segments.append((start, end))
                start = sorted_integers[i]
                end = sorted_integers[i]

        # Add the last range
        segments.append((start, end))

        return segments

    def _get_interface_residues(self, graph: Data) -> torch.Tensor:
        asym_id = graph.chains
        diff_chain_mask = asym_id[..., None, :] != asym_id[..., :, None]
        if self.dist_mode == "bb_ca":
            ca_idx = atom_order["CA"]
            ca_positions = graph.coords[..., ca_idx, :]
            ca_pairwise_dists = torch.cdist(ca_positions, ca_positions)
            min_dist_per_res = torch.where(diff_chain_mask, ca_pairwise_dists, torch.inf)  # .min(dim=-1)
        elif self.dist_mode == "all-atom":
            positions = graph.coords
            n_res, n_atom, _ = positions.shape
            atom_mask = graph.coord_mask
            pairwise_dists = (
                torch.cdist(
                    positions.view(n_res * n_atom, -1),
                    positions.view(n_res * n_atom, -1),
                )
                .view(n_res, n_atom, n_res, n_atom)
                .permute(0, 2, 1, 3)
            )
            pair_mask = atom_mask[None, :, None, :] * atom_mask[:, None, :, None]
            mask = diff_chain_mask[:, :, None, None] * pair_mask
            min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1).values.min(dim=-1).values
        else:
            raise ValueError(f"Invalid dist mode: {self.dist_mode}")

        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
        return interface_residues_idxs, min_dist_per_res

    def _crop_graph(self, graph: Data, crop_idxs: torch.Tensor) -> Data:
        num_residues = graph.coords.size(0)
        for key, value in graph:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_residues:
                graph[key] = value[crop_idxs]
            elif isinstance(value, list) and len(value) == num_residues:
                graph[key] = [value[i] for i in crop_idxs]
        if hasattr(graph, "chain_names"):
            graph["chain_names"] = [graph["chain_names"][i] for i in torch.unique_consecutive(graph.chains)]

        return graph


class RefoldedCroppingTransform(BaseTransform):
    """
    Contiguous cropping on the binder chain, and both contiguous and spatial cropping on the target chains.

    1. Based on the interface threshold, select interface residues.
    2. Randomly select one residue from the interface residues and use its corresponding chain as the binder chain, and all the other chains as the target chains.
    3. Do contiguous cropping on the binder chain, with its length being randomly sampled from Uniform(binder_min_length, binder_max_length).
        Minimal number of residues (=binder_padding_length) on the both sides of the binder seed residue on the binder chain will be included.
    4. For the target chains, will do both contiguous cropping and spatial cropping.
        a): First do spatial cropping on the target chains, select target residues with distance to the whole binder chain less than target_spatial_crop_threshold.
            If there are more than (crop_size - binder_length) residues selected, take the top (crop_size - binder_length) residues.
        b): Then do contiguous cropping on the target chains for each segments of the target chains
    """

    def __init__(
        self,
        crop_size: int = 384,
        interface_threshold: float = 5.0,
        dist_mode: Literal["bb_ca", "all-atom"] = "bb_ca",
        data_mode: Literal["bb_ca", "all-atom"] = "all-atom",
        binder_min_length: int = 50,
        binder_max_length: int = 200,
        binder_padding_length: int = 10,
        target_spatial_crop_threshold: float = 15.0,
        target_min_length: int = 50,
        enforce_target_min_length: bool = False,
        max_num_target_chains: int = -1,  # -1 for no limit
        min_num_interface_binder_res: int = 1,  # the minimum number of interface residues on the binder chain
    ):
        """
        Args:
            crop_size: the total number of residues to keep after cropping. Default is 384.
            interface_threshold: the threshold (in Angstrom) for interface residues. Default is 5.0.
            dist_mode: the distance mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom for computing the distance between residues. Default is "bb_ca".
            data_mode: the data mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom. Default is "all-atom".
            binder_min_length: the minimum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 50.
            binder_max_length: the maximum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 200.
            binder_padding_length: the minimal number of residues to include on each side of the binder seed residue when cropping the binder chain. Default is 10.
            target_spatial_crop_threshold: the threshold (in Angstrom) for target spatial cropping, which is the cutoff distanceof target residues to the whole binder chain. Default is 15.0.
            enforce_target_min_length: whether to enforce the target chain to have at least target_min_length residues after cropping.
                If True, target might have more chains than max_num_target_chains. Default is True.
            max_num_target_chains: the maximum number of target chains to include in the cropping, -1 for no limit. Default is -1.
        """
        self.crop_size = crop_size
        self.interface_threshold = interface_threshold
        self.dist_mode = dist_mode
        self.data_mode = data_mode
        self.binder_min_length = binder_min_length
        self.binder_max_length = binder_max_length
        self.binder_padding_length = binder_padding_length
        self.target_spatial_crop_threshold = target_spatial_crop_threshold
        self.target_min_length = target_min_length
        self.enforce_target_min_length = enforce_target_min_length
        self.max_num_target_chains = max_num_target_chains
        self.min_num_interface_binder_res = min_num_interface_binder_res
        assert self.binder_min_length >= 2 * self.binder_padding_length + 1, (
            "binder_min_length must be >= 2 * binder_padding_length + 1"
        )

    def __call__(self, graph: Data) -> Data:
        # In processed graph, binder_chain_id is single character from A to Z. All complexes with >26 chains are removed.
        # We need to convert binder_chain_id to integer for the following operations.
        binder_chain_id = graph.binder_chain_id
        binder_chain_id = ord(binder_chain_id) - ord("A")

        asym_id = graph.chains  # [n_res], each element is a integer representing the chain_id
        n_res = len(graph.chains)  # number of residues
        residues_idxs = torch.arange(n_res)  # [n_res]
        chain_lens = torch.bincount(asym_id)  # [n_chain] number of residues in each chain
        selected_idxs = torch.zeros(n_res, dtype=torch.bool)
        remaining_idxs = torch.ones(n_res, dtype=torch.bool)
        chain_residue_ranges = {}  # {chain_id: [start_idx, end_idx]}
        for chain_id in torch.unique(asym_id).sort().values.tolist():
            chain_mask = asym_id == chain_id
            chain_indices = residues_idxs[chain_mask]
            chain_residue_ranges[chain_id] = [
                int(chain_indices.min()),
                int(chain_indices.max()),
            ]

        # get interface residues, and residue-wise distances: (n_res, n_res)
        interface_residues, min_dist_per_res = self._get_interface_residues(graph)
        # get the re-designed residue indices
        redesigned_res_idx = torch.where(asym_id == binder_chain_id)[0]
        # binder candidate seed residues are the interface residues that are from the re-designed chain
        intersection_mask = (interface_residues.unsqueeze(1) == redesigned_res_idx).any(dim=1)
        binder_candidate_seed_residues = interface_residues[intersection_mask]
        # binder_candidate_seed_residues = asym_id[binder_interface_indices]
        # graph.binder_interface_residues = binder_candidate_seed_residues

        if len(binder_candidate_seed_residues) < self.min_num_interface_binder_res:
            raise ValueError(
                f"Interface residues on the binder chain are too few for pdb: {graph.id}, {len(binder_candidate_seed_residues)} < {self.min_num_interface_binder_res}"
            )

        # randomly select a binder seed residue from interface residues
        # For synthetic data, we selected the re-designed chain as the binder chain.
        binder_seed_residue = random.choice(binder_candidate_seed_residues)

        if (n_res - int(chain_lens[binder_chain_id])) < self.target_min_length and self.enforce_target_min_length:
            raise ValueError(f"Target chain too short for pdb: {graph.id}")

        target_mask = asym_id != binder_chain_id
        # in the case of controlling the number of target chains, binder_mask not necessarily equal to ~target_mask
        # binder_mask = asym_id == binder_chain_id
        if self.data_mode == "all-atom":
            graph.target_mask = target_mask[:, None] * graph.coord_mask.bool()
        else:
            graph.target_mask = target_mask

        if len(graph.chains) < self.crop_size:
            return graph

        # Contiguous cropping on the binder chain
        redesigned_binder_len = torch.sum(asym_id == binder_chain_id)
        # if the re-designed binder chain is too short, use the entire re-designed chain, else use the contiguous cropping
        if redesigned_binder_len > self.binder_min_length:
            binder_length = int(torch.randint(self.binder_min_length, self.binder_max_length + 1, (1,))[0])
            binder_chain_start = chain_residue_ranges[binder_chain_id][0]
            binder_chain_end = chain_residue_ranges[binder_chain_id][1]
            binder_left_length = int(
                torch.randint(
                    self.binder_padding_length,
                    binder_length - self.binder_padding_length - 1 + 1,
                    (1,),
                )[0]
            )
            binder_chain_crop_start = max(binder_chain_start, binder_seed_residue - binder_left_length)
            binder_chain_crop_end = min(binder_chain_end + 1, binder_chain_crop_start + binder_length)
            if binder_chain_crop_end == binder_chain_end + 1:
                binder_chain_crop_start = max(binder_chain_start, binder_chain_crop_end - binder_length)
        else:
            binder_chain_crop_start = redesigned_res_idx[0]
            binder_chain_crop_end = redesigned_res_idx[-1] + 1
        binder_chain_crop_idxs = torch.arange(binder_chain_crop_start, binder_chain_crop_end)
        selected_idxs[binder_chain_crop_idxs] = True

        # Spatial cropping on the target chains first to get the target seed residues
        to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
        target_seed_residues = residues_idxs[target_mask][to_binder_dist < self.target_spatial_crop_threshold]
        target_seed_residue_chains = asym_id[target_seed_residues]
        target_chains = target_seed_residue_chains.unique()
        target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        if self.max_num_target_chains > 0 and len(target_chains) > self.max_num_target_chains:
            # can't use random.shuffle to shuffle torch tensors.
            target_chains = (
                target_chains[torch.randperm(len(target_chains))][: self.max_num_target_chains].sort().values
            )
            target_seed_residues = target_seed_residues[
                (target_seed_residue_chains[:, None] == target_chains[None, :]).any(dim=-1)
            ]
            target_seed_residue_chains = asym_id[target_seed_residues]
            target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        # put chains not included in the target_chains to be False in remaining_idxs
        remaining_idxs[(asym_id[:, None] != target_chains[None, :]).all(dim=-1)] = False
        num_budget = self.crop_size - int(selected_idxs.sum())
        num_remaining = int(remaining_idxs.sum())

        if num_remaining <= num_budget:
            selected_idxs[residues_idxs[target_mask]] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        if len(target_seed_residues) >= num_budget:
            to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
            target_seed_residues = residues_idxs[target_mask][torch.argsort(to_binder_dist)[:num_budget]]
            selected_idxs[target_seed_residues] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        # Get the segments of the target chains
        segments = []
        for chain_id in target_chains:
            for segment in self._convert_to_segments(
                target_seed_residues[target_seed_residue_chains == chain_id].tolist()
            ):
                segments.append(segment)

        # Contiguous cropping on the target chains from the target seed residues/segments
        remaining_idxs[target_seed_residues] = False
        selected_idxs[target_seed_residues] = True
        segment_idxs = list(range(len(segments)))
        segment_used_mask = torch.zeros((len(segments), 2), dtype=torch.bool)  # [left_crop, right_crop]
        random.shuffle(segment_idxs)

        for segment_idx in segment_idxs:
            segment_start, segment_end = segments[segment_idx]
            chain_id = int(asym_id[segment_start])
            chain_start = chain_residue_ranges[chain_id][0]
            chain_end = chain_residue_ranges[chain_id][1]

            orders = [0, 1]
            random.shuffle(orders)
            for order in orders:
                if order == 0:
                    # left crop
                    segment_used_mask[segment_idx, 0] = True
                    if segment_idx == 0:
                        remaining_idxs[chain_start:segment_start] = False
                    else:
                        if segment_used_mask[segment_idx - 1, 1]:  # and segment_used_mask[segment_idx, 0]
                            remaining_idxs[segments[segment_idx - 1][1] : segment_start] = False
                        elif segments[segment_idx - 1][1] < chain_start:
                            remaining_idxs[chain_start:segment_start] = False
                    num_budget = self.crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx >= 1:
                        segment_crop_size_max = min(
                            num_budget,
                            segment_start - chain_start,
                            int((~selected_idxs[(segments[segment_idx - 1][1] + 1) : segment_start]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, segment_start - chain_start)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_start - segment_crop_size) : (segment_start)] = True
                    remaining_idxs[(segment_start - segment_crop_size) : (segment_start)] = False

                else:
                    # right crop
                    segment_used_mask[segment_idx, 1] = True
                    if segment_idx == len(segment_idxs) - 1:
                        remaining_idxs[segment_end : chain_end + 1] = False
                    else:
                        if segment_used_mask[segment_idx + 1, 0]:  # and segment_used_mask[segment_idx, 1]
                            remaining_idxs[segment_end : segments[segment_idx + 1][0]] = False
                        elif segments[segment_idx + 1][0] > chain_end:
                            remaining_idxs[segment_end : chain_end + 1] = False
                    num_budget = self.crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx < len(segment_idxs) - 1:
                        segment_crop_size_max = min(
                            num_budget,
                            chain_end - segment_end,
                            int((~selected_idxs[(segment_end + 1) : segments[segment_idx + 1][0]]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, chain_end - segment_end)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = True
                    remaining_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = False

        graph = self._crop_graph(graph, residues_idxs[selected_idxs])

        return graph

    @staticmethod
    def _convert_to_segments(sorted_integers):
        """
        Convert a list of sorted integers to a combination of ranges.

        Args:
            sorted_integers: List of sorted integers

        Returns:
            List of tuples, each tuple is a range of integers
        """
        segments = []
        start = sorted_integers[0]
        end = sorted_integers[0]

        for i in range(1, len(sorted_integers)):
            if sorted_integers[i] == end + 1:
                end = sorted_integers[i]
            else:
                segments.append((start, end))
                start = sorted_integers[i]
                end = sorted_integers[i]

        # Add the last range
        segments.append((start, end))

        return segments

    def _get_interface_residues(self, graph: Data) -> torch.Tensor:
        asym_id = graph.chains
        diff_chain_mask = asym_id[..., None, :] != asym_id[..., :, None]
        if self.dist_mode == "bb_ca":
            ca_idx = atom_order["CA"]
            ca_positions = graph.coords[..., ca_idx, :]
            ca_pairwise_dists = torch.cdist(ca_positions, ca_positions)
            min_dist_per_res = torch.where(diff_chain_mask, ca_pairwise_dists, torch.inf)  # .min(dim=-1)
        elif self.dist_mode == "all-atom":
            positions = graph.coords
            n_res, n_atom, _ = positions.shape
            atom_mask = graph.coord_mask
            pairwise_dists = (
                torch.cdist(
                    positions.view(n_res * n_atom, -1),
                    positions.view(n_res * n_atom, -1),
                )
                .view(n_res, n_atom, n_res, n_atom)
                .permute(0, 2, 1, 3)
            )
            pair_mask = atom_mask[None, :, None, :] * atom_mask[:, None, :, None]
            mask = diff_chain_mask[:, :, None, None] * pair_mask
            min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1).values.min(dim=-1).values
        else:
            raise ValueError(f"Invalid dist mode: {self.dist_mode}")

        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
        return interface_residues_idxs, min_dist_per_res

    def _crop_graph(self, graph: Data, crop_idxs: torch.Tensor) -> Data:
        num_residues = graph.coords.size(0)
        for key, value in graph:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_residues:
                graph[key] = value[crop_idxs]
            elif isinstance(value, list) and len(value) == num_residues:
                graph[key] = [value[i] for i in crop_idxs]
        if hasattr(graph, "chain_names"):
            graph["chain_names"] = [graph["chain_names"][i] for i in torch.unique_consecutive(graph.chains)]

        return graph


class CoordsTensorCenteringTransform(BaseTransform):
    """Centers a primary coordinates tensor and optionally shifts others by the same center.

    Computes the masked center of mass from the primary tensor, subtracts
    it, then applies the same translation to any ``additional_tensors``.
    This lets multiple coordinate tensors (e.g. motif + ligand) share the
    same reference frame.

    The shift is shape-agnostic for additional tensors: flat ``(n, 3)`` and
    atom37 ``(n, 37, 3)`` tensors are both handled automatically via
    broadcasting from the ``(1, 3)`` center.

    Args:
        tensor_name: Key for the primary coordinates tensor.
        mask_name: Key for the primary mask tensor.  When ``None`` or absent
            from the data, valid positions are inferred from non-zero
            coordinates.
        data_mode: How to compute the center from the primary tensor:

            - ``"bb_ca"``: atom37 ``(n_res, 37, 3)`` — center on CA (index 1)
            - ``"all-atom"``: atom37 — center on all masked atoms
            - ``"ligand_only"``: flat ``(n_atoms, 3)``
        additional_tensors: Optional list of dicts with ``tensor_name`` and
            optional ``mask_name``.  Each is shifted by the same center.
    """

    def __init__(
        self,
        tensor_name: str = "coords_nm",
        mask_name: str | None = "coord_mask",
        data_mode: Literal["bb_ca", "all-atom", "ligand_only"] = "all-atom",
        additional_tensors: list[dict[str, str]] | None = None,
    ):
        self.tensor_name = tensor_name
        self.mask_name = mask_name
        self.data_mode = data_mode
        self.additional_tensors = additional_tensors or []

    @staticmethod
    def _resolve_mask(coords: torch.Tensor, mask_name: str | None, graph) -> torch.Tensor:
        """Return an explicit mask or infer one from non-zero coordinates.

        The returned mask has the same leading dimensions as *coords* minus
        the last (xyz) dimension, i.e. ``(n,)`` for flat or ``(n, 37)`` for
        atom37.
        """
        if mask_name and mask_name in graph:
            return graph[mask_name].bool()
        return coords.abs().sum(-1) > 1e-8

    @staticmethod
    def _compute_center(
        coords: torch.Tensor,
        mask: torch.Tensor,
        data_mode: str,
    ) -> torch.Tensor:
        """Compute the masked mean of *coords*, always returning ``(1, 3)``.

        *data_mode* controls which atoms contribute:
        ``"bb_ca"`` uses CA only (index 1 of atom37), ``"all-atom"`` uses
        every masked atom in atom37, ``"ligand_only"`` treats coords as
        flat ``(n, 3)``.
        """
        if data_mode == "bb_ca":
            return mean_w_mask(coords[:, 1, :], mask[:, 1].bool(), keepdim=True)
        if data_mode == "all-atom":
            return mean_w_mask(coords.flatten(0, 1), mask.flatten(0, 1).bool(), keepdim=True)
        if data_mode == "ligand_only":
            return mean_w_mask(coords, mask.bool(), keepdim=True)
        logger.warning(f"Invalid data_mode: {data_mode} - Using Nx3 point cloud instead")
        return mean_w_mask(coords, mask.bool(), keepdim=True)

    @staticmethod
    def _apply_shift(
        coords: torch.Tensor,
        mask: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        """Subtract *center* ``(1, 3)`` and zero out masked positions.

        Shape-agnostic: works for flat ``(n, 3)`` with ``(n,)`` mask,
        atom37 ``(n, 37, 3)`` with ``(n, 37)`` mask, or atom37 with
        residue-level ``(n,)`` mask (zeros entire residues).
        """
        return (coords - center) * mask.unsqueeze(-1)

    def __call__(self, graph: Data) -> Data:
        coords = graph[self.tensor_name]
        mask = self._resolve_mask(coords, self.mask_name, graph)
        center = self._compute_center(coords, mask, self.data_mode)
        graph[self.tensor_name] = self._apply_shift(graph[self.tensor_name], mask, center)

        for spec in self.additional_tensors:
            tname = spec["tensor_name"]
            mname = spec.get("mask_name")
            t_coords = graph[tname]
            t_mask = self._resolve_mask(t_coords, mname, graph)
            graph[tname] = self._apply_shift(t_coords, t_mask, center)

        return graph


class AtomworksLigandFeaturesTransform(BaseTransform):
    """Transform that extracts ligand-specific features from atomworks pipeline output.

    This transform processes the raw atomworks data stored in `_atomworks_data` attribute
    and creates token-level features for ligand modeling.

    Features extracted:
        - token_charge: Token-level charges from representative atoms
        - token_element: Token-level element types from representative atoms
        - token_name_chars: Token-level atom name characters
        - token_bond_order: Token-token bond order matrix
        - token_bond_mask: Token-token bond connectivity mask
        - token_laplacian_pe: Laplacian positional encodings for tokens
        - binder_mask: Mask for protein/binder tokens
        - target_mask: Mask for ligand/target tokens

    Args:
        remove_atomworks_data: If True, removes the raw `_atomworks_data` after processing
            to save memory. Default: True.
    """

    def __init__(self, remove_atomworks_data: bool = True):
        self.remove_atomworks_data = remove_atomworks_data

    def __call__(self, data: Data) -> Data:
        # Get the raw atomworks data stored during atom37 conversion
        atomworks_data = getattr(data, "_atomworks_data", None)
        if atomworks_data is None:
            # No atomworks data - nothing to do
            return data

        atom_array = atomworks_data.get("atom_array", None)
        data.residue_type = (data.residue_type < 20) * data.residue_type  # zero out non-amino acids
        # Extract masks
        data.binder_residue_mask = atomworks_data.get("protein_token_mask", None)
        data.target_residue_mask = atomworks_data.get("ligand_token_mask", None)
        data.target_mask = data.target_residue_mask[:, None] * data.coord_mask.bool()
        data.target_bond_mask = atomworks_data.get("token_bond_mask", None)
        data.target_laplacian_pe = atomworks_data.get("token_laplacian_pe", None)

        # Get atom-level features
        target_atom_bond_order = atomworks_data.get("atom_bond_order", None)
        atomworks_data.get("atom_element", None)
        target_atom_name_chars = atomworks_data.get("atom_name_chars", None)
        target_atom_charge = atomworks_data.get("atom_charge", None)

        # Get representative atom indices for creating token-level features
        from atomworks.ml.utils.token import get_af3_token_representative_idxs

        representative_atom_idx = atomworks_data.get("ground_truth", {}).get(
            "rep_atom_idxs",
            (get_af3_token_representative_idxs(atom_array) if atom_array is not None else None),
        )

        if representative_atom_idx is not None:
            # Create token-level features from atom-level features
            # 1D: [n_atoms] -> [n_tokens]
            if target_atom_charge is not None:
                data.target_charge = target_atom_charge[representative_atom_idx]
            # if target_atom_element is not None:
            #     data.target_residue_element = target_atom_element[representative_atom_idx]
            # 2D: [n_atoms, char_dim] -> [n_tokens, char_dim]
            if target_atom_name_chars is not None:
                t = target_atom_name_chars[representative_atom_idx]
                data.target_atom_name = t.reshape(t.shape[0], -1)  # flatten [n, 4, 64] -> [n, 256]
            # 2D: [n_atoms, n_atoms] -> [n_tokens, n_tokens]
            if target_atom_bond_order is not None:
                data.target_bond_order = target_atom_bond_order[representative_atom_idx][:, representative_atom_idx]

        # Clean up raw data to save memory
        if self.remove_atomworks_data:
            if hasattr(data, "_atomworks_data"):
                delattr(data, "_atomworks_data")

        return data


class LigandAtom37SqueezeTransform(BaseTransform):
    """Unsqueezes the ligand atom37 features to the original shape."""

    def __init__(self):
        pass

    def __call__(self, data: Data) -> Data:
        # for k in data.keys(): v=getattr(data,k); print(f"{k}: {v.shape}" if hasattr(v,'shape') else f"{k}: {type(v).__name__}")
        data.x_target = data.x_target[:, 1, :]
        return data
