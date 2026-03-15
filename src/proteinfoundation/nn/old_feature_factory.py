import gzip
import math
import os
import random
from typing import Literal

import einops
import torch
from jaxtyping import Float
from loguru import logger
from openfold.data import data_transforms
from openfold.np.residue_constants import atom_types
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean

from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle
from proteinfoundation.utils.fold_utils import extract_cath_code_by_level
from proteinfoundation.utils.tensor_utils import concat_padded_tensor

BOND_ORDER_MAP = {
    -1: 0,
    0: 1,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 2,
    7: 3,
    8: 5,
    9: 6,
}
NUM_BOND_ORDERS = 7

# import esm


################################
# # Some auxiliary functions # #
################################
# From genie2 code
def sinusoidal_encoding(v, N, D):
    # v: [*]

    # [D]
    k = torch.arange(1, D + 1).to(v.device)

    # [*, D]
    sin_div_term = N ** (2 * k / D)
    sin_div_term = sin_div_term.view(*((1,) * len(v.shape) + (len(sin_div_term),)))
    sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

    # [*, D]
    cos_div_term = N ** (2 * (k - 1) / D)
    cos_div_term = cos_div_term.view(*((1,) * len(v.shape) + (len(cos_div_term),)))
    cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

    # [*, D]
    enc = torch.zeros_like(sin_enc).to(v.device)
    enc[..., 0::2] = cos_enc[..., 0::2]
    enc[..., 1::2] = sin_enc[..., 1::2]

    return enc


# From frameflow code
def get_index_embedding(indices, edim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of type integer, shape either [n] or [b, n].
        edim: dimension of the embeddings to create.
        max_len: maximum length.

    Returns:
        positional embedding of shape either [n, edim] or [b, n, edim]
    """
    # indices [n] of [b, n]
    K = torch.arange(edim // 2, device=indices.device)  # [edim / 2]

    if len(indices.shape) == 1:  # [n]
        K = K[None, ...]
    elif len(indices.shape) == 2:  # [b, n]
        K = K[None, None, ...]

    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K / edim))).to(indices.device)
    # [n, 1] / [1, edim/2] -> [n, edim/2] or [b, n, 1] / [1, 1, edim/2] -> [b, n, edim/2]
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K / edim))).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)  # [n, edim]
    return pos_embedding


def get_time_embedding(t: Float[torch.Tensor, "b"], edim: int, max_positions: int = 2000) -> torch.Tensor:
    """
    Code from Frameflow, which got it from
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Creates embedding for a given vector of times t.

    Args:
        t: vector of times (float) of shape [b].
        edim: dimension of the embeddings.
        max_positions: ...

    Returns:
        Embedding for the vector t of shape [b, edim]
    """
    assert len(t.shape) == 1
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if edim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (t.shape[0], edim)
    return emb


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(min_dist, max_dist, dim - 1, device=x.device)  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: dict):
        pass  # Implemented by each class

    def extract_bs_and_n(self, batch: dict):
        """
        Extracts batch size and n from input batch.
        Once we add more modalities just try modaliites until you hit one.
        Other option is to pass arguments to nn which modalities are used.
        """
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "coords_nm" in batch:
            v = batch["coords_nm"]  # [b, n, 37,3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise OSError("Don't know how to extract batch size and n from batch...")
        bs, n = v.shape[0], v.shape[1]
        return bs, n

    def extract_device(self, batch: dict):
        """Extracts device from input batch."""
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "coords_nm" in batch:
            v = batch["coords_nm"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise OSError("Don't know how to extract device from batch...")
        return v.device

    def assert_defaults_allowed(self, batch: dict, ftype: str):
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if "strict_feats" in batch:
            if batch["strict_feats"]:
                raise OSError(
                    f"{ftype} feature requested but no appropriate feature provided. "
                    "Make sure to include the relevant transform in the data config."
                )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(
        self,
        dim_feats_out=128,
        mode: Literal["seq", "pair"] = "seq",
        name=None,
        **kwargs,
    ):
        super().__init__(dim=128)
        self.mode = mode

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=device)
        elif self.mode == "pair":
            torch.zeros((b, n, n, self.dim_feats_out), device=device)
        else:
            raise OSError(f"Mode {self.mode} wrong for zero feature")


class CroppedFlagSeqFeat(Feature):
    """Computes feature of shape [b, n, 1] indicating if protein is cropped
    (1s if it is, 0s if it isn't)."""

    def __init__(self):
        super().__init__(dim=1)

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if "cropped" in batch:
            ones = torch.ones((b, n, self.dim), device=device)
            cropped = batch["cropped"]  # boolean [b]
            return ones * cropped[..., None, None]  # [b, n, dim(=1)]
        else:
            return torch.zeros((b, n, self.dim), device=device)


class FoldEmbeddingSeqFeat(Feature):
    """Computes fold class embedding and returns as sequence feature of shape [b, n, fold_emb_dim * 3]."""

    def __init__(
        self,
        fold_emb_dim,
        cath_code_dir,
        multilabel_mode="sample",
        fold_nhead=4,
        fold_nlayer=2,
        **kwargs,
    ):
        """
        multilabel_mode (["sample", "average", "transformer"]): Schemes to handle multiple fold labels
            "sample": randomly sample one label
            "average": average fold embeddings over all labels
            "transformer": pad labels together and feed into a transformer, take the average over the output
        """
        super().__init__(dim=fold_emb_dim * 3)
        self.create_mapping(cath_code_dir)
        self.embedding_C = torch.nn.Embedding(
            self.num_classes_C + 1, fold_emb_dim
        )  # The last class is left as null embedding
        self.embedding_A = torch.nn.Embedding(self.num_classes_A + 1, fold_emb_dim)
        self.embedding_T = torch.nn.Embedding(self.num_classes_T + 1, fold_emb_dim)
        self.register_buffer("_device_param", torch.tensor(0), persistent=False)
        assert multilabel_mode in ["sample", "average", "transformer"]
        self.multilabel_mode = multilabel_mode
        if multilabel_mode == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                fold_emb_dim * 3,
                nhead=fold_nhead,
                dim_feedforward=fold_emb_dim * 3,
                batch_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, fold_nlayer)

    @property
    def device(self):
        return next(self.buffers()).device

    def create_mapping(self, cath_code_dir):
        """Create cath label vocabulary for C, A, T levels."""
        mapping_file = os.path.join(cath_code_dir, "cath_label_mapping.pt")
        if os.path.exists(mapping_file):
            class_mapping = torch.load(mapping_file)
        else:
            cath_code_file = os.path.join(cath_code_dir, "cath-b-newest-all.gz")
            cath_code_set = {"C": set(), "A": set(), "T": set()}
            with gzip.open(cath_code_file, "rt") as f:
                for line in f:
                    cath_id, cath_version, cath_code, cath_segment_and_chain = line.strip().split()
                    cath_code_set["C"].add(extract_cath_code_by_level(cath_code, "C"))
                    cath_code_set["A"].add(extract_cath_code_by_level(cath_code, "A"))
                    cath_code_set["T"].add(extract_cath_code_by_level(cath_code, "T"))
            class_mapping = {
                "C": {k: i for i, k in enumerate(sorted(list(cath_code_set["C"])))},
                "A": {k: i for i, k in enumerate(sorted(list(cath_code_set["A"])))},
                "T": {k: i for i, k in enumerate(sorted(list(cath_code_set["T"])))},
            }
            torch.save(class_mapping, mapping_file)

        self.class_mapping_C = class_mapping["C"]
        self.class_mapping_A = class_mapping["A"]
        self.class_mapping_T = class_mapping["T"]
        self.num_classes_C = len(self.class_mapping_C)
        self.num_classes_A = len(self.class_mapping_A)
        self.num_classes_T = len(self.class_mapping_T)

    def parse_label(self, cath_code_list):
        """Parse cath_code into corresponding indices at C, A, T levels

        Args:
            cath_code_list (List[List[str]]): List of cath codes for each protein. Each protein can have no, one or multiple labels.

        Return:
            results: for each label of each protein, return its C, A, T label indices
        """
        results = []
        for cath_codes in cath_code_list:
            result = []
            for cath_code in cath_codes:
                result.append(
                    [
                        self.class_mapping_C.get(
                            extract_cath_code_by_level(cath_code, "C"),
                            self.num_classes_C,
                        ),  # If unknown or masked, set as null
                        self.class_mapping_A.get(
                            extract_cath_code_by_level(cath_code, "A"),
                            self.num_classes_A,
                        ),
                        self.class_mapping_T.get(
                            extract_cath_code_by_level(cath_code, "T"),
                            self.num_classes_T,
                        ),
                    ]
                )
            if len(cath_codes) == 0:
                result = [
                    [self.num_classes_C, self.num_classes_A, self.num_classes_T]
                ]  # If no cath code is provided, return null
            results.append(result)
        return results  # [b, num_label, 3]

    def sample(self, cath_code_list):
        """Randomly sample one cath code"""
        results = []
        for cath_codes in cath_code_list:
            idx = random.randint(0, len(cath_codes) - 1)
            results.append(cath_codes[idx])
        return results

    def flatten(self, cath_code_list):
        """Flatten variable lengths of cath codes into a long cath code tensor"""
        results = []
        batch_id = []
        for i, cath_codes in enumerate(cath_code_list):
            results += cath_codes
            batch_id += [i] * len(cath_codes)
        results = torch.as_tensor(results, device=self.device)
        batch_id = torch.as_tensor(batch_id, device=self.device)
        return results, batch_id

    def pad(self, cath_code_list):
        """Pad variable lengths of cath codes into a batched cath code tensor"""
        results = []
        max_num_label = 0
        for cath_codes in cath_code_list:
            results.append(cath_codes)
            max_num_label = max(max_num_label, len(cath_codes))
        mask = []
        for i in range(len(results)):
            mask_i = [False] * len(results[i])
            if len(results[i]) < max_num_label:
                mask_i += [True] * (max_num_label - len(results[i]))
                results[i] += [[self.num_classes_C, self.num_classes_A, self.num_classes_T]] * (
                    max_num_label - len(results[i])
                )
            mask.append(mask_i)
        results = torch.as_tensor(results, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        return results, mask

    def forward(self, batch):
        bs, n = self.extract_bs_and_n(batch)
        if "cath_code" not in batch:
            cath_code = [["x.x.x.x"]] * bs  # If no cath code provided, return null embeddings
        else:
            cath_code = batch["cath_code"]

        cath_code_list = self.parse_label(cath_code)
        if self.multilabel_mode == "sample":
            cath_code_list = self.sample(cath_code_list)  # Random sample one label for each protein
            cath_code = torch.as_tensor(cath_code_list, device=self.device)  # [b, 3]
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, 0]),
                    self.embedding_A(cath_code[:, 1]),
                    self.embedding_T(cath_code[:, 2]),
                ],
                dim=-1,
            )  # [b, fold_emb_dim * 3]
        elif self.multilabel_mode == "average":
            cath_code, batch_id = self.flatten(cath_code_list)
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, 0]),
                    self.embedding_A(cath_code[:, 1]),
                    self.embedding_T(cath_code[:, 2]),
                ],
                dim=-1,
            )  # [num_code, fold_emb_dim * 3]
            fold_emb = scatter_mean(fold_emb, batch_id, dim=0, dim_size=bs)
        elif self.multilabel_mode == "transformer":
            cath_code, mask = self.pad(cath_code_list)
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, :, 0]),
                    self.embedding_A(cath_code[:, :, 1]),
                    self.embedding_T(cath_code[:, :, 2]),
                ],
                dim=-1,
            )  # [b, max_num_label, fold_emb_dim * 3]
            fold_emb = self.transformer(fold_emb, src_key_padding_mask=mask)  # [b, max_num_label, fold_emb_dim * 3]
            fold_emb = (fold_emb * (~mask[:, :, None]).float()).sum(dim=1) / (
                (~mask[:, :, None]).float().sum(dim=1) + 1e-10
            )  # [b, fold_emb_dim * 3]
        fold_emb = fold_emb[:, None, :]  # [b, 1, fold_emb_dim * 3]
        return fold_emb.expand((fold_emb.shape[0], n, fold_emb.shape[2]))  # [b, n, fold_emb_dim * 3]


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingSeqFeatGenie2(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, n_timestep, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use
        self.n_timestep = n_timestep

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = sinusoidal_encoding(t * self.n_timestep, self.n_timestep, self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        # If it has the actual residue indices
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
        return get_index_embedding(inds, edim=self.dim)  # [b, n, idx_embed_dim]


class IdxEmbeddingSeqFeatGenie2(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, c_pos_emb, max_n_res, **kwargs):
        super().__init__(dim=c_pos_emb)
        self.max_n_res = max_n_res

    def forward(self, batch):
        # If it has the actual residue indices
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
        return sinusoidal_encoding(inds, self.max_n_res, self.dim)


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        # If it has the actual chain breaks
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"] * 1.0  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Chain break sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            chain_breaks = torch.zeros((b, n), device=device) * 1.0  # [b, n]
        return chain_breaks[..., None]  # [b, n, 1]


class XscBBCASeqFeat(Feature):
    """Computes feature from backbone CA self conditining coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, mode_key="x_sc", **kwargs):
        super().__init__(dim=3)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert "bb_ca" in data_modes_avail, (
                f"`bb_ca` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            )
            return batch[self.mode_key]["bb_ca"]  # [b, n, 3]
        else:
            # If we do not provide self-conditioning as input to the nn
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} in batch, returning zeros for XscBBCASeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, 3, device=device)


class XscLocalLatentsSeqFeat(Feature):
    """Computes feature from local latents self conditining, seq feature of shape [b, n, dim]."""

    def __init__(self, latent_dim, mode_key="x_sc", **kwargs):
        super().__init__(dim=latent_dim)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert "local_latents" in data_modes_avail, (
                f"`local_latents` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            )
            return batch[self.mode_key]["local_latents"]  # [b, n, latent_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} in batch, returning zeros for XscLocalLatentsSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class XtBBCASeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert "bb_ca" in data_modes_avail, (
            f"`bb_ca` seq feat feature requested but key not available in data modes {data_modes_avail}"
        )
        return batch["x_t"]["bb_ca"]  # [b, n, 3]


class XtLocalLatentsSeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, latent_dim, **kwargs):
        super().__init__(dim=latent_dim)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert "local_latents" in data_modes_avail, (
            f"`local_latents` seq feat feature requested but key not available in data modes {data_modes_avail}"
        )
        return batch["x_t"]["local_latents"]  # [b, n, latent_dim]


class CaCoorsNanometersSeqFeat(Feature):
    """Computes feature from ca coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        assert "ca_coors_nm" in batch or "coords_nm" in batch, (
            "`ca_coors_nm` nor `coords_nm` in batch, cannot compute CaCoorsNanometersSeqFeat"
        )
        if "ca_coors_nm" in batch:
            return batch["ca_coors_nm"]  # [b, n, 3]
        else:
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]


class TryCaCoorsNanometersSeqFeat(CaCoorsNanometersSeqFeat):
    """
    If `ca_coors_nm` in batch, returns sequence feature with CA coordinates (in nm) of shape [b, n, 3].

    If `ca_coors_nm` not in batch return zero feature.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if "ca_coors_nm" in batch or "coords_nm" in batch:
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No ca_coors_nm or coords_nm in batch, returning zeros for TryCaCoorsNanometersSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class OptionalCaCoorsNanometersSeqFeat(CaCoorsNanometersSeqFeat):
    """
    If `use_ca_coors_nm_feature` in batch and true, returns sequence feature with CA coordinates (in nm) of shape [b, n, 3].

    If `use_ca_coors_nm_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if batch.get("use_ca_coors_nm_feature", False):  # defaults to False
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "use_ca_coors_nm_feature disabled or not in batch, returning zeros for OptionalCaCoorsNanometersSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class TargetChargeSeqFeat(Feature):
    """
    Computes feature from ligand charge, feature of shape [b, n, 1].
    """

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        assert "target_charge" in batch, "`target_charge` not in batch, cannot compute LigandChargeSeqFeat"
        return batch["target_charge"][..., None]  # [b, n, 1]


class TargetAtomNameSeqFeat(Feature):
    """
    Computes feature from ligand charge, feature of shape [b, n, 1].
    """

    def __init__(self, **kwargs):
        super().__init__(dim=4 * 64)

    def forward(self, batch):
        assert "target_atom_name" in batch, "`target_atom_name` not in batch, cannot compute LigandChargeSeqFeat"
        return batch["target_atom_name"]


class TargetGraphPESeqFeat(Feature):
    """
    Computes feature from ligand charge, feature of shape [b, n, 1].
    """

    def __init__(self, **kwargs):
        super().__init__(dim=32)

    def forward(self, batch):
        assert "target_laplacian_pe" in batch, "`target_laplacian_pe` not in batch, cannot compute LigandChargeSeqFeat"
        return batch["target_laplacian_pe"]


class AtomTypeSeqFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 128].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, input_is_onehot=True, **kwargs):
        super().__init__(dim=128)
        self.input_is_onehot = input_is_onehot

    def forward(self, batch):
        assert "residue_type" in batch, "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        rpadmask = batch["mask_dict"]["residue_type"]  # [b, n] binary
        # [b, n], the -1 padding becomes 0
        if self.input_is_onehot:
            rtype = rtype * rpadmask[..., None]
            rtype_onehot = rtype
        else:
            rtype_onehot = F.one_hot(rtype, num_classes=self.dim)  # [b, n, dim]
            rtype_onehot = rtype_onehot * rpadmask[..., None]  # zero out padding rows just in case
        return rtype_onehot * 1.0


class BondMaskPairFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 128].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        return batch["target_bond_mask"][..., None]


class BondOrderPairFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 128].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=NUM_BOND_ORDERS)

    def forward(self, batch):
        return F.one_hot(batch["target_bond_order"].long(), num_classes=self.dim)


class ResidueTypeSeqFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 20].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=20)

    def forward(self, batch):
        assert "residue_type" in batch, "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        if "mask_dict" in batch:
            rpadmask = batch["mask_dict"]["residue_type"]  # [b, n] binary
        else:
            rpadmask = batch["mask"]  # [b, n] binary
        rtype = rtype * rpadmask  # [b, n], the -1 padding becomes 0
        rtype_onehot = F.one_hot(rtype, num_classes=20)  # [b, n, 20]
        rtype_onehot = rtype_onehot * rpadmask[..., None]  # zero out padding rows just in case
        return rtype_onehot * 1.0


class OptionalResidueTypeSeqFeat(ResidueTypeSeqFeat):
    """
    If `use_residue_type_feature` in batch and true, adds residue type feature of shape [b, n, 20].

    If `use_residue_type_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if batch.get("use_residue_type_feature", False):  # defaults to False
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "use_residue_type_feature disabled or not in batch, returning zeros for OptionalResidueTypeSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, 20, device=device)


class AtomisticCoorsSeqFeat(Feature):
    """
    Computes feature from the atom representation (in Å), feature of shape [b, n, 1 * 4].
    """

    def __init__(self, **kwargs):
        super().__init__(dim=4)

    def forward(self, batch):
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot compute AtomisticCoorsSeqFeat"
        assert "coord_mask" in batch, "`coord_mask` not in batch, cannot compute AtomisticCoorsSeqFeat"
        coors = batch["coords_nm"]  # [b, n, 3]
        coors_mask = batch["coord_mask"]  # [b, n] (different for different residue types)
        coors = coors * coors_mask[..., None]  # Zero-out non-atoms (padding and no side chain atoms)

        feat = torch.cat([coors, coors_mask[..., None]], dim=-1)  # [b, n, 4]
        return feat


class Atom37NanometersCoorsSeqFeat(Feature):
    """
    Computes feature from the atom37 representation (in Å), feature of shape [b, n, 37 * 4].

    Atom37 has shape [b, n, 37, 3], and the appropriate mask (for the residue type) has shape
    [b, n, 37]. This feature concatenates the flattened mask (shape [b, n, 37]) with the flattened coordinates (of shape
    [b, n, 37 * 3])

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, rel=False, **kwargs):
        super().__init__(dim=(37 * 4))
        # 37 * 4, 37 * 3 for the coordinates, + 37 for the mask
        # 37 * 4 = 148
        self.rel = rel
        # Whether to get features relative to CA or absolute

    def forward(self, batch):
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        assert "coord_mask" in batch, "`coord_mask` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        coors = batch["coords_nm"]  # [b, n, 37, 3]
        coors_mask = batch["coord_mask"]  # [b, n, 37] (different for different residue types)
        coors = coors * coors_mask[..., None]  # Zero-out non-atoms (padding and no side chain atoms)

        if self.rel:
            # If relative remove CA coordinates
            ca_coors = coors[:, :, 1, :]  # [b, n, 3]
            coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
            coors = coors * coors_mask[..., None]

        # coors[:, :, 3:, :] = coors[:, :, 3:, :] * 0.0  # If I don't want to pass sidechain info

        coors_flat = einops.rearrange(coors, "b n a t -> b n (a t)")  # [b, n, 37, 3] -> [b, n, 37 * 3]
        feat = torch.cat([coors_flat, coors_mask], dim=-1)  # [b, n, 37 * 4]
        return feat


class BackboneTorsionAnglesSeqFeat(Feature):
    """
    Computes torsion angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=(3 * 21))

    def forward(self, batch):
        # # # # # # # # # # # # # # # # # # # #
        bb_torsion = self._get_bb_torsion_angles(batch)  # [b, n, 3]
        bb_torsion_feats = bin_and_one_hot(
            bb_torsion,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_torsion.device),
        )  # [b, n, 3, nbins], nbins in 20+1
        bb_torsion_feats = einops.rearrange(bb_torsion_feats, "b n t d -> b n (t d)")  # [b, n, 3 * nbins]
        return bb_torsion_feats

    def _get_bb_torsion_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
            # no need to force 1 since taking difference
            idx = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]

        psi = signed_dihedral_angle(N[:, :-1, :], CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])  # [b, n-1]
        omega = signed_dihedral_angle(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # [b, n-1]
        phi = signed_dihedral_angle(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :], C[:, 1:, :])  # [b, n-1]
        bb_angles = torch.stack([psi, omega, phi], dim=-1)  # [b, n-1, 3]

        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]
        bb_angles = bb_angles * good_pair[..., None]  # [b, n-1, 3]

        zero_pad = torch.zeros((a37.shape[0], 1, 3), device=bb_angles.device)
        bb_angles = torch.cat([bb_angles, zero_pad], dim=1)  # [b, n, 3]
        return bb_angles


class BackboneBondAnglesSeqFeat(Feature):
    """
    Computes bond angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=(3 * 21))

    def forward(self, batch):
        # TODO: Pass arguments here, just the 20
        # # # # # # # # # # # # # # # # # # # #
        bb_bond_angle = self._get_bb_bond_angles(batch)  # [b, n, 3]
        bb_bond_angle_feats = bin_and_one_hot(
            bb_bond_angle,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_bond_angle.device),
        )  # [b, n, 3, nbins]
        # I think this is always between 0 and pi
        bb_bond_angle_feats = einops.rearrange(bb_bond_angle_feats, "b n t d -> b n (t d)")  # [b, n, 3 * nbins]
        return bb_bond_angle_feats

    def _get_bb_bond_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        if "mask_dict" in batch:
            mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n]
        else:
            mask = batch["mask"]  # [b, n]

        if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
            # no need to force 1 since taking difference
            idx = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
        b = a37.shape[0]

        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]
        theta_1 = bond_angles(N[:, :, :], CA[:, :, :], C[:, :, :])  # [b, n]
        theta_2 = bond_angles(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])  # [b, n-1]
        theta_3 = bond_angles(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # [b, n-1]

        # Account for chain breaks in theta_2 and theta_3
        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]
        theta_2 = theta_2 * good_pair  # [b, n-1]
        theta_3 = theta_3 * good_pair  # [b, n-1]

        # Add a zero at the end of theta_2 and theta_3 to get shape [b, n]
        zero_pad = torch.zeros((b, 1), device=theta_2.device)  # [b, 1]
        theta_2 = torch.cat([theta_2, zero_pad], dim=-1)  # [b, n]
        theta_3 = torch.cat([theta_3, zero_pad], dim=-1)  # [b, n]

        bb_angles = torch.stack([theta_1, theta_2, theta_3], dim=-1)  # [b, n, 3]
        return bb_angles


class OpenfoldSideChainAnglesSeqFeat(Feature):
    """Computes sequence features from side chain angles."""

    def __init__(self, **kwargs):
        super().__init__(dim=(4 * 21 + 4))  # 88

    def forward(self, batch):
        # TODO: Pass arguments here, just the 20
        # # # # # # # # # # # # # # # # # # # #
        _, angles, torsion_angles_mask = self._get_sidechain_angles(batch)
        # _, [b, n, 4] and [b, n, 4]
        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, 4, nbins]
        angles_feat = angles_feat * torsion_angles_mask[..., None]
        angles_feat = einops.rearrange(angles_feat, "b n s d -> b n (s d)")  # [b, n, 4 * nbins]
        feat = torch.cat([angles_feat, torsion_angles_mask], dim=-1)  # [b, n, 4 * nbins + 4]
        return feat

    def _get_sidechain_angles(self, batch):
        orig_dtype = batch["coords"].dtype
        aatype = batch["residue_type"]  # [b, n]
        coords = batch["coords"].double()  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"].double()  # [b, n, 37]
        p = {
            "aatype": aatype,
            "all_atom_positions": coords,
            "all_atom_mask": atom_mask,
        }
        # Next function defined with curry1 decorator
        p = data_transforms.atom37_to_torsion_angles(prefix="")(p)
        torsion_angles_sin_cos = p["torsion_angles_sin_cos"]  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = p["alt_torsion_angles_sin_cos"]  # [b, n, 7, 2]
        # For cases with symmetry
        # Normalize, all these vectors should have norm 1
        torsion_angles_sin_cos = torsion_angles_sin_cos / (
            torch.linalg.norm(torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos / (
            torch.linalg.norm(alt_torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        torsion_angles_mask = p["torsion_angles_mask"]  # [b, n, 7]
        # This symmetry is important if predicting these angles, as you need to take the min
        # when computing the loss, since both predictions are correct
        # Not important when used as features
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[..., None]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[..., None]
        angles = torch.atan2(torsion_angles_sin_cos[..., 0], torsion_angles_sin_cos[..., 1])  # [b, n, 7]
        angles = angles * torsion_angles_mask
        # Keep only sidechain
        torsion_angles_sin_cos = torsion_angles_sin_cos[..., -4:, :]  # [b, n, 4, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos[..., -4:, :]  # [b, n, 4, 2]
        angles = angles[..., -4:]  # [b, n, 4]
        torsion_angles_mask = torsion_angles_mask[..., -4:]  # [b, n, 4]
        return (
            torsion_angles_sin_cos.to(dtype=orig_dtype),
            angles.to(dtype=orig_dtype),
            torsion_angles_mask.bool(),
        )  # [b, n, 4, 2], [b, n, 4] and [b, n, 4]


class LatentVariableSeqFeat(Feature):
    """Returns sequence feature from latent variable."""

    def __init__(self, latent_z_dim, **kwargs):
        print([k for k in kwargs])
        super().__init__(dim=latent_z_dim)

    def forward(self, batch):
        assert "z_latent" in batch, "`z_latent` not in batch, cannot compute LatentVariableSeqFeat"
        return batch["z_latent"]  # [b, n, latent_dim]


class MotifAbsoluteCoordsSeqFeat(Feature):
    """Computes absolute coordinates feature from motif coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for absolute coords
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch:
            batch_coors = {
                "coords_nm": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return Atom37NanometersCoorsSeqFeat(rel=False)(batch_coors)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No x_motif or motif_mask in batch, returning zeros for MotifAbsoluteCoordsSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifRelativeCoordsSeqFeat(Feature):
    """Computes relative coordinates feature from motif coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for relative coords
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif_mask" in batch:
            required_atoms = torch.tensor([atom_types.index("CA")], device=batch["motif_mask"].device)  # CA
            has_required_atoms = torch.all(batch["motif_mask"][:, :, required_atoms], dim=-1)  # [batch, seq_len]
            relevant_has_required_atoms = torch.where(
                batch["seq_motif_mask"],
                has_required_atoms,
                torch.ones_like(has_required_atoms, dtype=torch.bool),
            )
            if not torch.all(relevant_has_required_atoms):
                if not self._has_logged:
                    logger.warning(
                        "Missing required CA atoms in motif region, returning zeros for MotifRelativeCoordsSeqFeat"
                    )
                    self._has_logged = True
                b, n = self.extract_bs_and_n(batch)
                device = self.extract_device(batch)
                return torch.zeros(b, n, self.dim, device=device)
            batch_coors = {
                "coords_nm": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return Atom37NanometersCoorsSeqFeat(rel=True)(batch_coors)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No x_motif or motif_mask in batch, returning zeros for MotifRelativeCoordsSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifSequenceSeqFeat(Feature):
    """Computes sequence feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=20)  # 20 for one-hot encoded residues
        self._has_logged = False

    def forward(self, batch):
        if "seq_motif" in batch and "seq_motif_mask" in batch:
            batch_seq = {
                "residue_type": batch["seq_motif"],
                "mask_dict": {
                    "residue_type": batch["seq_motif_mask"],
                },
            }
            return ResidueTypeSeqFeat()(batch_seq)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No seq_motif or seq_motif_mask in batch, returning zeros for MotifSequenceSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifSideChainAnglesSeqFeat(Feature):
    """Computes side chain angles feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=88)  # 4 * 21 + 4 for side chain angles
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif" in batch:
            batch_sc_angles = {
                "residue_type": batch["seq_motif"],
                "coords": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return OpenfoldSideChainAnglesSeqFeat()(batch_sc_angles)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("Missing required motif data in batch, returning zeros for MotifSideChainAnglesSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifTorsionAnglesSeqFeat(Feature):
    """Computes torsion angles feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=63)  # 3 * 21 for torsion angles
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif_mask" in batch:
            backbone_atoms = torch.tensor(
                [
                    atom_types.index("N"),
                    atom_types.index("CA"),
                    atom_types.index("C"),
                    atom_types.index("O"),
                ],
                device=batch["motif_mask"].device,
            )
            motif_mask_per_residue_backbone = torch.any(
                batch["motif_mask"][:, :, backbone_atoms], dim=-1
            )  # [batch, seq_len]
            relevant_motif_mask = torch.where(
                batch["seq_motif_mask"],
                motif_mask_per_residue_backbone,
                torch.ones_like(motif_mask_per_residue_backbone, dtype=torch.bool),
            )
            if not torch.all(relevant_motif_mask):
                if not self._has_logged:
                    logger.warning("Missing backbone atoms in motif region, returning zeros")
                    self._has_logged = True
                b, n = self.extract_bs_and_n(batch)
                device = self.extract_device(batch)
                return torch.zeros(b, n, self.dim, device=device)

            batch_torsion_angles = {
                "coords": batch["x_motif"],
                "residue_pdb_idx": batch.get("residue_pdb_idx", None),
            }
            return BackboneTorsionAnglesSeqFeat()(batch_torsion_angles)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No x_motif or motif_mask in batch, returning zeros for MotifTorsionAnglesSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class MotifMaskSeqFeat(Feature):
    """Computes motif mask feature."""

    def __init__(self, **kwargs):
        super().__init__(dim=37)  # 37 for atom mask
        self._has_logged = False

    def forward(self, batch):
        if "motif_mask" in batch:
            return batch["motif_mask"] * 1.0  # [b, n, 37]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No motif_mask in batch, returning zeros for MotifMaskSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


# Similar classes for target features
class TargetAbsoluteCoordsSeqFeat(Feature):
    """Computes absolute coordinates feature from target coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for absolute coords
        self._has_logged = False

    def forward(self, batch):
        if "x_target" in batch and "target_mask" in batch:
            required_atoms = torch.tensor([atom_types.index("CA")], device=batch["target_mask"].device)  # CA
            has_required_atoms = torch.all(batch["target_mask"][:, :, required_atoms], dim=-1)  # [batch, seq_len]
            # Only check positions that are part of the target
            target_positions = batch["seq_target_mask"].bool()  # [batch, seq_len]
            # For positions not in target, set to True (so they don't affect the all())
            relevant_has_required_atoms = torch.where(
                target_positions,
                has_required_atoms,
                torch.ones_like(has_required_atoms, dtype=torch.bool),
            )
            if not torch.all(relevant_has_required_atoms):
                if not self._has_logged:
                    logger.warning(
                        "Missing required CA atoms in target region, returning zeros for TargetAbsoluteCoordsSeqFeat"
                    )
                    self._has_logged = True
                b, _ = self.extract_bs_and_n(batch)
                n = batch["x_target"].shape[1]
                device = self.extract_device(batch)
                return torch.zeros(b, n, self.dim, device=device)
            batch_coors = {
                "coords_nm": batch["x_target"],
                "coord_mask": batch["target_mask"],
            }
            return Atom37NanometersCoorsSeqFeat(rel=False)(batch_coors)
        else:
            b, _ = self.extract_bs_and_n(batch)
            n = 0
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No x_target or target_mask in batch, returning zeros for TargetAbsoluteCoordsSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class TargetRelativeCoordsSeqFeat(Feature):
    """Computes relative coordinates feature from target coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for relative coords
        self._has_logged = False

    def forward(self, batch):
        if "x_target" in batch and "target_mask" in batch:
            required_atoms = torch.tensor([atom_types.index("CA")], device=batch["target_mask"].device)  # CA
            has_required_atoms = torch.all(batch["target_mask"][:, :, required_atoms], dim=-1)  # [batch, seq_len]
            relevant_has_required_atoms = torch.where(
                batch["seq_target_mask"],
                has_required_atoms,
                torch.ones_like(has_required_atoms, dtype=torch.bool),
            )
            if not torch.all(relevant_has_required_atoms):
                if not self._has_logged:
                    logger.warning(
                        "Missing required CA atoms in target region, returning zeros for TargetRelativeCoordsSeqFeat"
                    )
                    self._has_logged = True
                b, _ = self.extract_bs_and_n(batch)
                n = batch["x_target"].shape[1]
                device = self.extract_device(batch)
                return torch.zeros(b, n, self.dim, device=device)
            batch_coors = {
                "coords_nm": batch["x_target"],
                "coord_mask": batch["target_mask"],
            }
            return Atom37NanometersCoorsSeqFeat(rel=True)(batch_coors)
        else:
            b, _ = self.extract_bs_and_n(batch)
            n = 0
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No x_target or target_mask in batch, returning zeros for TargetRelativeCoordsSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class TargetSequenceSeqFeat(Feature):
    """Computes sequence feature from target."""

    def __init__(self, **kwargs):
        super().__init__(dim=20)  # 20 for one-hot encoded residues
        self._has_logged = False

    def forward(self, batch):
        if "seq_target" in batch and "seq_target_mask" in batch:
            batch_seq = {
                "residue_type": batch["seq_target"],
                "mask_dict": {
                    "residue_type": batch["seq_target_mask"],
                },
            }
            return ResidueTypeSeqFeat()(batch_seq)
        else:
            b, _ = self.extract_bs_and_n(batch)
            n = 0
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No seq_target or seq_target_mask in batch, returning zeros for TargetSequenceSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class TargetSideChainAnglesSeqFeat(Feature):
    """Computes side chain angles feature from target."""

    def __init__(self, **kwargs):
        super().__init__(dim=88)  # 4 * 21 + 4 for side chain angles
        self._has_logged = False

    def forward(self, batch):
        if "x_target" in batch and "target_mask" in batch and "seq_target" in batch and batch["x_target"].shape[1] > 0:
            batch_sc_angles = {
                "residue_type": batch["seq_target"],
                "coords": batch["x_target"],
                "coord_mask": batch["target_mask"],
            }
            return OpenfoldSideChainAnglesSeqFeat()(batch_sc_angles)
        else:
            b, _ = self.extract_bs_and_n(batch)
            n = 0
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "Missing required target data in batch, returning zeros for TargetSideChainAnglesSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class TargetTorsionAnglesSeqFeat(Feature):
    """Computes torsion angles feature from target."""

    def __init__(self, **kwargs):
        super().__init__(dim=63)  # 3 * 21 for torsion angles
        self._has_logged = False

    def forward(self, batch):
        if "x_target" in batch and "target_mask" in batch and "seq_target" in batch and batch["x_target"].shape[1] > 0:
            # Check that backbone atoms are present in target_mask for all target residues
            backbone_atoms = torch.tensor(
                [
                    atom_types.index("N"),
                    atom_types.index("CA"),
                    atom_types.index("C"),
                    atom_types.index("O"),
                ],
                device=batch["target_mask"].device,
            )
            target_mask_per_residue_backbone = torch.any(
                batch["target_mask"][:, :, backbone_atoms], dim=-1
            )  # [batch, seq_len]
            # For positions not in target, set to True (so they don't affect the all())
            relevant_target_mask = torch.where(
                batch["seq_target_mask"],
                target_mask_per_residue_backbone,
                torch.ones_like(target_mask_per_residue_backbone, dtype=torch.bool),
            )
            if not torch.all(relevant_target_mask):
                if not self._has_logged:
                    logger.warning(
                        "Missing backbone atoms in target region, returning zeros for TargetTorsionAnglesSeqFeat"
                    )
                    self._has_logged = True
                b, _ = self.extract_bs_and_n(batch)
                n = batch["target_mask"].shape[1]
                device = self.extract_device(batch)
                return torch.zeros(b, n, self.dim, device=device)
            batch_bb_angles = {
                "residue_type": batch["seq_target"],
                "coords": batch["x_target"],
                "coord_mask": batch["target_mask"],
            }
            return BackboneTorsionAnglesSeqFeat()(batch_bb_angles)
        else:
            b, _ = self.extract_bs_and_n(batch)
            n = 0
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("Missing required target data in batch, returning zeros for TargetTorsionAnglesSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class TargetMaskSeqFeat(Feature):
    """Computes target mask feature."""

    def __init__(self, **kwargs):
        super().__init__(dim=37)  # 37 for atom mask
        self._has_logged = False

    def forward(self, batch):
        if "target_mask" in batch:
            return batch["target_mask"] * 1.0  # [b, n, 37]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No target_mask in batch, returning zeros for TargetMaskSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class TargetMaskPairFeat(Feature):
    """Computes target mask feature for pairs."""

    def __init__(self, **kwargs):
        super().__init__(dim=74)  # 37 * 2 for concatenated atom masks
        self._has_logged = False

    def forward(self, batch):
        if "target_mask" in batch:
            target_mask = batch["target_mask"]  # [b, n, 37]
            # Create pairwise target mask features by concatenating masks from both residues
            target_i = target_mask[:, :, None, :].expand(-1, -1, target_mask.size(1), -1)  # [b, n, n, 37]
            target_j = target_mask[:, None, :, :].expand(-1, target_mask.size(1), -1, -1)  # [b, n, n, 37]
            pair_target = torch.cat([target_i, target_j], dim=-1) * 1.0  # [b, n, n, 74]
            return pair_target
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No target_mask in batch, returning zeros for TargetMaskPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class ChainIdxSeqFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch):
        if "chains" in batch:
            mask = batch["chains"].unsqueeze(-1)  # [b, n, 1]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No chains in batch, returning zeros for ChainIdxSeqFeat")
                self._has_logged = True
            mask = torch.zeros((b, n, 1), device=device)
        return mask


class HotspotMaskSeqFeat(Feature):
    """Gets target hotspot feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch):
        if "hotspot_mask" in batch:
            mask = batch["hotspot_mask"].float().unsqueeze(-1)  # [b, n, 1]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No hotspot_mask in batch, returning zeros for HotspotMaskSeqFeat")
                self._has_logged = True
            mask = torch.zeros((b, n, 1), device=device)
        return mask


class BinderCenterFeat(Feature):
    """Gets binder center and returns feature of shape [b, n, 3] by broadcasting over residue dimension."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)
        self._has_logged = False

    def forward(self, batch):
        if "binder_center" in batch:
            b, n = self.extract_bs_and_n(batch)
            mask = torch.tile(batch["binder_center"], (1, n, 1))  # [b, n, 3]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No binder_center in batch, returning zeros for BinderCenterFeat")
                self._has_logged = True
            mask = torch.zeros((b, n, 3), device=device)
        return mask


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim, **kwargs):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert self.dim % 2 == 1, "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.dim - 1, device=inds.device)

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


class XtBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert "bb_ca" in data_modes_avail, (
            f"`bb_ca` pair dist feature requested but key not available in data modes {data_modes_avail}"
        )
        return bin_pairwise_distances(
            x=batch["x_t"]["bb_ca"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class CaCoorsNanometersPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=30)
        self.min_dist = 0.1
        self.max_dist = 3.0

    def forward(self, batch):
        assert "ca_coors_nm" in batch or "coords_nm" in batch, (
            "`ca_coors_nm` pair dist feature requested but key `ca_coors_nm` nor `coords_nm` not available"
        )
        if "ca_coors_nm" in batch:
            ca_coors = batch["ca_coors_nm"]
        else:
            ca_coors = batch["coords_nm"][:, :, 1, :]
        return bin_pairwise_distances(
            x=ca_coors,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class OptionalCaCoorsNanometersPairwiseDistancesPairFeat(CaCoorsNanometersPairwiseDistancesPairFeat):
    """
    If `use_ca_coors_nm_feature` in batch and true, returns pair feature with CA pairwise distances binned, shape [b, n, n, nbins].

    If `use_ca_coors_nm_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if batch.get("use_ca_coors_nm_feature", False):  # defaults to False
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    "use_ca_coors_nm_feature disabled or not in batch, returning zeros for OptionalCaCoorsNanometersPairwiseDistancesPairFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class XscBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self,
        x_sc_pair_dist_dim,
        x_sc_pair_dist_min,
        x_sc_pair_dist_max,
        mode_key="x_sc",
        **kwargs,
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert "bb_ca" in data_modes_avail, (
                f"`bb_ca` sc/recycle pair dist feature requested but key not available in data modes {data_modes_avail}"
            )
            return bin_pairwise_distances(
                x=batch[self.mode_key]["bb_ca"],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} in batch, returning zeros for XscBBCAPairwiseDistancesPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class RelativeResidueOrientationPairFeat(Feature):
    """Computes pair feature with pairwise residue orientations.

    See paper "Improved protein structure prediction using
    predicted inter-residue orientations".

    TODO: Impute beta carbon for Glycine
    TODO: 20 as argument
    """

    def __init__(self, **kwargs):
        super().__init__(dim=(5 * 21))  # 105

    def forward(self, batch):
        aatype = batch["residue_type"]  # [b, n]
        coords = batch["coords"]  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding
        # residue has a beta carbon or not (equivalent, if residue type its glycine)
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n] boolean
        beta_carbon_pair_mask = has_cb[:, :, None] * has_cb[:, :, None]  # [b, n, n] boolean
        pair_mask = pair_mask * beta_carbon_pair_mask  # [b, n, n]

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        N_p1, CA_p1, CB_p1 = map(lambda v: v[:, :, None, :], (N, CA, CB))  # Each [b, n, 1, 3]
        N_p2, CA_p2, CB_p2 = map(lambda v: v[:, None, :, :], (N, CA, CB))  # Each [b, 1, n, 3]

        theta_12 = signed_dihedral_angle(N_p1, CA_p1, CB_p1, CB_p2)  # [b, n, n]
        theta_21 = signed_dihedral_angle(N_p2, CA_p2, CB_p2, CB_p1)  # [b, n, n]
        phi_12 = bond_angles(CA_p1, CB_p1, CB_p2)  # [b, n, n]
        phi_21 = bond_angles(CA_p2, CB_p2, CB_p1)  # [b, n, n]
        w = signed_dihedral_angle(CA_p1, CB_p1, CB_p2, CA_p2)  # [b, n, n]
        angles = torch.stack([theta_12, theta_21, phi_12, phi_21, w], dim=-1)  # [b, n, n, 5]

        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, n, 5, nbins]
        angles_feat = einops.rearrange(angles_feat, "b n m f d -> b n m (f d)")  # [b, n, n, 5 * nbins]
        angles_feat = angles_feat * pair_mask[..., None]  # Mask padding and GLY
        return angles_feat


class BackbonePairDistancesNanometerPairFeat(Feature):
    """
    Computes pairwise distances between backbone atoms.

    Position (i, j) encodes the distance between CA_i and
    {N_j, CA_j, C_j, CB_j}.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=(4 * 21))  # 84

    def forward(self, batch):
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot comptue BackbonePairDistancesNanometerPairFeat"
        coords = batch["coords_nm"]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding
        # residue has a beta carbon or not (equivalent, if residue type its glycine)

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        C = coords[:, :, 2, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        CA_i = CA[:, :, None, :]  # [b, n, 1, 3]
        N_j, CA_j, C_j, CB_j = map(lambda v: v[:, None, :, :], (N, CA, C, CB))  # Each [b, 1, n, 3]

        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: torch.linalg.norm(v[0] - v[1], dim=-1),
            ((CA_i, N_j), (CA_i, CA_j), (CA_i, C_j), (CA_i, CB_j)),
        )  # Each shape [b, n, n]
        # CA_X[..., i, j] has distance (nm) between CA[..., i] and X[..., j]

        # Accomodate residues without CB
        # CA_CB has shape [b, n, n], CA_CB[..., i, j] has distance between
        # CA[i] and CB[j]. If residue j has no CB, then CA_CB[..., i, j]
        # has to be zero for all i
        CA_CB = CA_CB * has_cb[:, None, :]  # [b, n, n]

        # Fix for mask
        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: v * pair_mask,
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each shape [b, n, n]

        bin_limits = torch.linspace(0.1, 2, 20, device=coords.device)
        CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat = map(
            lambda v: bin_and_one_hot(v, bin_limits=bin_limits),
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each [b, n, n, 21]

        feat = torch.cat([CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat], dim=-1)  # [b, n, n, 4 * 21]
        feat = feat * pair_mask[..., None]
        return feat


class XmotifPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone motif atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.const = BackbonePairDistancesNanometerPairFeat()
        self.dim = self.const.dim  # Fix dim, cannot put init here
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch:
            # print("Calling motif pair feat")
            batch_bbpd = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat = self.const(batch_bbpd)  # [b, n, n, some #]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No x_motif in batch, returning zeros for XmotifPairwiseDistancesPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class XtargetPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.const = BackbonePairDistancesNanometerPairFeat()
        self.dim = self.const.dim  # Fix dim, cannot put init here
        self._has_logged = False

    def forward(self, batch):
        if "x_target" in batch:
            batch_bbpd = {
                "coords_nm": batch["x_target"],  # [b, n, 37, 3]
                "coord_mask": batch["target_mask"],  # [b, n, 37]
            }
            feat = self.const(batch_bbpd)  # [b, n, n, some #]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No x_target in batch, returning zeros for XtargetPairwiseDistancesPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class CrossSequenceBackbonePairDistancesPairFeat(Feature):
    """
    Computes pairwise distances between backbone atoms of two sequences.

    Position (i, j) encodes the distance between CA_i (from sequence 1) and
    {N_j, CA_j, C_j, CB_j} (from sequence 2).

    Returns a rectangular matrix [b, n1, n2, 4*21] instead of square.
    """

    def __init__(
        self,
        coords1_key="coords_nm",
        mask1_key="coord_mask",
        coords2_key="coords_nm_2",
        mask2_key="coord_mask_2",
        **kwargs,
    ):
        super().__init__(dim=(4 * 21))  # 84
        self.coords1_key = coords1_key
        self.mask1_key = mask1_key
        self.coords2_key = coords2_key
        self.mask2_key = mask2_key

    def forward(self, batch):
        # Sequence 1 (rows of output matrix)
        assert self.coords1_key in batch, (
            f"`{self.coords1_key}` not in batch, cannot compute CrossSequenceBackbonePairDistancesPairFeat"
        )
        assert self.mask1_key in batch, (
            f"`{self.mask1_key}` not in batch, cannot compute CrossSequenceBackbonePairDistancesPairFeat"
        )

        coords1 = batch[self.coords1_key]  # [b, n1, 37, 3]
        atom_mask1 = batch[self.mask1_key]  # [b, n1, 37]
        mask1 = atom_mask1[:, :, 1]  # [b, n1] - CA mask for seq1
        has_cb1 = atom_mask1[:, :, 3]  # [b, n1] - CB mask for seq1

        # Sequence 2 (columns of output matrix)
        assert self.coords2_key in batch, (
            f"`{self.coords2_key}` not in batch, cannot compute CrossSequenceBackbonePairDistancesPairFeat"
        )
        assert self.mask2_key in batch, (
            f"`{self.mask2_key}` not in batch, cannot compute CrossSequenceBackbonePairDistancesPairFeat"
        )

        coords2 = batch[self.coords2_key]  # [b, n2, 37, 3]
        atom_mask2 = batch[self.mask2_key]  # [b, n2, 37]
        mask2 = atom_mask2[:, :, 1]  # [b, n2] - CA mask for seq2
        has_cb2 = atom_mask2[:, :, 3]  # [b, n2] - CB mask for seq2

        # Cross-sequence pair mask [b, n1, n2]
        cross_pair_mask = mask1[:, :, None] * mask2[:, None, :]  # [b, n1, n2]

        # Extract backbone atoms from sequence 1
        N1 = coords1[:, :, 0, :]  # [b, n1, 3]
        CA1 = coords1[:, :, 1, :]  # [b, n1, 3]
        C1 = coords1[:, :, 2, :]  # [b, n1, 3]
        CB1 = coords1[:, :, 3, :]  # [b, n1, 3]

        # Extract backbone atoms from sequence 2
        N2 = coords2[:, :, 0, :]  # [b, n2, 3]
        CA2 = coords2[:, :, 1, :]  # [b, n2, 3]
        C2 = coords2[:, :, 2, :]  # [b, n2, 3]
        CB2 = coords2[:, :, 3, :]  # [b, n2, 3]

        # Prepare for distance calculation: CA from seq1 to all atoms in seq2
        CA1_expanded = CA1[:, :, None, :]  # [b, n1, 1, 3]
        N2_expanded, CA2_expanded, C2_expanded, CB2_expanded = map(
            lambda v: v[:, None, :, :], (N2, CA2, C2, CB2)
        )  # Each [b, 1, n2, 3]

        # Compute distances from CA_i (seq1) to {N_j, CA_j, C_j, CB_j} (seq2)
        CA1_N2, CA1_CA2, CA1_C2, CA1_CB2 = map(
            lambda v: torch.linalg.norm(v[0] - v[1], dim=-1),
            (
                (CA1_expanded, N2_expanded),
                (CA1_expanded, CA2_expanded),
                (CA1_expanded, C2_expanded),
                (CA1_expanded, CB2_expanded),
            ),
        )  # Each shape [b, n1, n2]

        # Handle residues without CB in sequence 2
        # CA1_CB2[..., i, j] has distance between CA1[i] and CB2[j]
        # If residue j in seq2 has no CB, then CA1_CB2[..., i, j] should be zero for all i
        CA1_CB2 = CA1_CB2 * has_cb2[:, None, :]  # [b, n1, n2]

        # Apply cross-sequence mask
        CA1_N2, CA1_CA2, CA1_C2, CA1_CB2 = map(
            lambda v: v * cross_pair_mask,
            (CA1_N2, CA1_CA2, CA1_C2, CA1_CB2),
        )  # Each shape [b, n1, n2]

        # Bin distances
        bin_limits = torch.linspace(0.1, 2, 20, device=coords1.device)
        CA1_N2_feat, CA1_CA2_feat, CA1_C2_feat, CA1_CB2_feat = map(
            lambda v: bin_and_one_hot(v, bin_limits=bin_limits),
            (CA1_N2, CA1_CA2, CA1_C2, CA1_CB2),
        )  # Each [b, n1, n2, 21]

        feat = torch.cat([CA1_N2_feat, CA1_CA2_feat, CA1_C2_feat, CA1_CB2_feat], dim=-1)  # [b, n1, n2, 4 * 21]
        feat = feat * cross_pair_mask[..., None]
        return feat


class CrossSequenceBackboneAtomPairDistancesPairFeat(Feature):
    """
    Computes pairwise distances between backbone atoms of two sequences.

    Position (i, j) encodes the distance between CA_i (from sequence 1) and
    {N_j, CA_j, C_j, CB_j} (from sequence 2).

    Returns a rectangular matrix [b, n1, n2, 4*21] instead of square.
    """

    def __init__(
        self,
        coords1_key="coords_nm",
        mask1_key="coord_mask",
        coords2_key="coords_nm_2",
        mask2_key="coord_mask_2",
        **kwargs,
    ):
        super().__init__(dim=(4 * 21))  # 84
        self.coords1_key = coords1_key
        self.mask1_key = mask1_key
        self.coords2_key = coords2_key
        self.mask2_key = mask2_key

    def forward(self, batch):
        # Sequence 1 (rows of output matrix)
        assert self.coords1_key in batch, (
            f"`{self.coords1_key}` not in batch, cannot compute CrossSequenceBackboneAtomPairDistancesPairFeat"
        )
        assert self.mask1_key in batch, (
            f"`{self.mask1_key}` not in batch, cannot compute CrossSequenceBackboneAtomPairDistancesPairFeat"
        )

        coords1 = batch[self.coords1_key]  # [b, n1, 37, 3]
        atom_mask1 = batch[self.mask1_key]  # [b, n1, 37]
        mask1 = atom_mask1[:, :, 1]  # [b, n1] - CA mask for seq1
        has_cb1 = atom_mask1[:, :, 3]  # [b, n1] - CB mask for seq1

        # Sequence 2 (columns of output matrix)
        assert self.coords2_key in batch, (
            f"`{self.coords2_key}` not in batch, cannot compute CrossSequenceBackboneAtomPairDistancesPairFeat"
        )
        assert self.mask2_key in batch, (
            f"`{self.mask2_key}` not in batch, cannot compute CrossSequenceBackboneAtomPairDistancesPairFeat"
        )

        coords2 = batch[self.coords2_key]  # [b, n2, 3]
        atom_mask2 = batch[self.mask2_key]  # [b, n2]
        mask2 = atom_mask2.clone()  # [b, n2] - CA mask for seq2

        # Cross-sequence pair mask [b, n1, n2]
        cross_pair_mask = mask1[:, :, None] * mask2[:, None, :]  # [b, n1, n2]

        # Extract backbone atoms from sequence 1
        N1 = coords1[:, :, 0, :]  # [b, n1, 3]
        CA1 = coords1[:, :, 1, :]  # [b, n1, 3]
        C1 = coords1[:, :, 2, :]  # [b, n1, 3]
        CB1 = coords1[:, :, 3, :]  # [b, n1, 3]

        # Extract backbone atoms from sequence 2
        # N2 = coords2[:, :, 0, :]  # [b, n2, 3]
        # CA2 = coords2[:, :, 1, :]  # [b, n2, 3]
        # C2 = coords2[:, :, 2, :]  # [b, n2, 3]
        # CB2 = coords2[:, :, 3, :]  # [b, n2, 3]
        CA2 = coords2

        # Prepare for distance calculation: CA from seq1 to all atoms in seq2
        CA2_expanded = CA2[:, None, :, :]  # [b, n1, 1, 3]
        # N1_expanded, CA1_expanded, C1_expanded, CB1_expanded = map(
        #     lambda v: v[:, None, :, :], (N1, CA1, C1, CB1)
        # )  # Each [b, 1, n2, 3]
        N1_expanded, CA1_expanded, C1_expanded, CB1_expanded = map(lambda v: v[:, :, None, :], (N1, CA1, C1, CB1))

        # Compute distances from CA_i (seq1) to {N_j, CA_j, C_j, CB_j} (seq2)
        N1_CA2, CA1_CA2, C1_CA2, CB1_CA2 = map(
            lambda v: torch.linalg.norm(v[0] - v[1], dim=-1),
            (
                (N1_expanded, CA2_expanded),
                (CA1_expanded, CA2_expanded),
                (C1_expanded, CA2_expanded),
                (CB1_expanded, CA2_expanded),
            ),
        )  # Each shape [b, n1, n2]

        # Handle residues without CB in sequence 2
        # CA1_CB2[..., i, j] has distance between CA1[i] and CB2[j]
        # If residue j in seq2 has no CB, then CA1_CB2[..., i, j] should be zero for all i
        CB1_CA2 = CB1_CA2 * has_cb1[:, :, None]  # [b, n1, n2]

        # Apply cross-sequence mask
        N1_CA2, CA1_CA2, C1_CA2, CB1_CA2 = map(
            lambda v: v * cross_pair_mask,
            (N1_CA2, CA1_CA2, C1_CA2, CB1_CA2),
        )  # Each shape [b, n1, n2]

        # Bin distances
        bin_limits = torch.linspace(0.1, 2, 20, device=coords1.device)
        N1_CA2_feat, CA1_CA2_feat, C1_CA2_feat, CB1_CA2_feat = map(
            lambda v: bin_and_one_hot(v, bin_limits=bin_limits),
            (N1_CA2, CA1_CA2, C1_CA2, CB1_CA2),
        )  # Each [b, n1, n2, 21]

        feat = torch.cat([N1_CA2_feat, CA1_CA2_feat, C1_CA2_feat, CB1_CA2_feat], dim=-1)  # [b, n1, n2, 4 * 21]
        feat = feat * cross_pair_mask[..., None]
        return feat


class TargetAtomPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for target atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self,
        x_target_pair_dist_dim=30,
        x_target_pair_dist_min=0.1,
        x_target_pair_dist_max=3,
        mode_key="x_target",
        **kwargs,
    ):
        super().__init__(dim=x_target_pair_dist_dim)
        self.min_dist = x_target_pair_dist_min
        self.max_dist = x_target_pair_dist_max
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = batch.keys()
            assert "x_target" in data_modes_avail, (
                f"`x_target` target pair dist feature requested but key not available in data modes {data_modes_avail}"
            )
            return bin_pairwise_distances(
                x=batch[self.mode_key],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} in batch, returning zeros for TargetAtomPairwiseDistancesPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.dim, device=device)


class TargetToSamplePairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances between target and sample backbone atoms and returns rectangular feature of shape [b, n_target, n_sample, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.cross_seq_feat = CrossSequenceBackbonePairDistancesPairFeat(
            coords1_key="x_target",
            mask1_key="target_mask",
            coords2_key="coords_nm",
            mask2_key="coord_mask",
            **kwargs,
        )
        self.dim = self.cross_seq_feat.dim  # 4 * 21 = 84
        self._has_logged = False

    def forward(self, batch):
        if "x_target" in batch and "target_mask" in batch and "coords_nm" in batch and "coord_mask" in batch:
            return self.cross_seq_feat(batch)  # [b, n_target, n_sample, 4*21]
        else:
            if not self._has_logged:
                logger.warning(
                    "Missing required data for TargetToSamplePairwiseDistancesPairFeat: need x_target, target_mask, coords_nm, coord_mask"
                )
                self._has_logged = True

            # Determine dimensions for fallback
            if "x_target" in batch and "coords_nm" in batch:
                b = batch["x_target"].shape[0]
                n_target = batch["x_target"].shape[1]
                n_sample = batch["coords_nm"].shape[1]
                device = batch["x_target"].device
            elif "coords_nm" in batch:
                b, n_sample = batch["coords_nm"].shape[:2]
                n_target = n_sample  # fallback assumption
                device = batch["coords_nm"].device
            else:
                # Last resort fallback
                b, n_target = 1, 1
                n_sample = 1
                device = torch.device("cpu")

            return torch.zeros(b, n_target, n_sample, self.dim, device=device)


class ChainIdxPairFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch):
        if "chains" in batch:
            seq_mask = batch["chains"]  # [b, n]
            # mask = torch.einsum("bi,bj->bij", seq_mask, seq_mask).unsqueeze(
            #     -1
            # )  # [b, n, n, 1]
            seq_mask = (seq_mask[:, :, None] != seq_mask[:, None, :]).float()
            mask = seq_mask.unsqueeze(-1)  # [b, n, n, 1]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No chains in batch, returning zeros for ChainIdxPairFeat")
                self._has_logged = True
            mask = torch.zeros((b, n, n, 1), device=device)
        return mask


class HotspotMaskPairFeat(Feature):
    """Gets target hotspot feature for pairs and returns feature of shape [b, n, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch):
        if "hotspot_mask" in batch:
            hotspots = batch["hotspot_mask"]  # [b, n]
            # Create pairwise hotspot feature: 1 if either residue is a hotspot
            pair_hotspots = (hotspots[:, :, None] + hotspots[:, None, :]).clamp(0, 1)  # [b, n, n]
            mask = pair_hotspots.unsqueeze(-1)  # [b, n, n, 1]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No hotspot_mask in batch, returning zeros for HotspotMaskPairFeat")
                self._has_logged = True
            mask = torch.zeros((b, n, n, 1), device=device)
        return mask


class StochasticTranslationSeqFeat(Feature):
    """Gets stochastic translation from centering transform and returns feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)
        self._has_logged = False

    def forward(self, batch):
        if "stochastic_translation" in batch:
            b, n = self.extract_bs_and_n(batch)
            translation = batch["stochastic_translation"]  # [b, 3]
            # Broadcast translation to all residues
            mask = translation[:, None, :].expand(b, n, -1)  # [b, n, 3]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No stochastic_translation in batch, returning zeros for StochasticTranslationSeqFeat")
                self._has_logged = True
            mask = torch.zeros((b, n, 3), device=device)
        return mask


class ContactTypeSeqFeat(Feature):
    """Embeds contact composition features and returns feature of shape [b, n, 4]."""

    def __init__(self, **kwargs):
        super().__init__(dim=4)
        self._has_logged = False

    def forward(self, batch):
        if "contact" in batch:
            return batch["contact"]  # [b, n, 4] - [total_contacts, apolar_frac, polar_frac, charged_frac]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No contact in batch, returning zeros for ContactTypeSeqFeat")
                self._has_logged = True
            return torch.zeros((b, n, 4), device=device)


class ContactTypePairFeat(Feature):
    """Embeds contact composition features for pairs and returns feature of shape [b, n, n, 8]."""

    def __init__(self, **kwargs):
        super().__init__(dim=8)
        self._has_logged = False

    def forward(self, batch):
        if "contact" in batch:
            contact_feats = batch["contact"]  # [b, n, 4]
            # Create pairwise contact features by concatenating features from both residues
            contact_i = contact_feats[:, :, None, :].expand(-1, -1, contact_feats.size(1), -1)  # [b, n, n, 4]
            contact_j = contact_feats[:, None, :, :].expand(-1, contact_feats.size(1), -1, -1)  # [b, n, n, 4]
            pair_contact = torch.cat([contact_i, contact_j], dim=-1)  # [b, n, n, 8]
            return pair_contact
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning("No contact in batch, returning zeros for ContactTypePairFeat")
                self._has_logged = True
            return torch.zeros((b, n, n, 8), device=device)


class MotifToSamplePairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances between motif and sample backbone atoms and returns rectangular feature of shape [b, n_motif, n_sample, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.cross_seq_feat = CrossSequenceBackbonePairDistancesPairFeat(
            coords1_key="x_motif",
            mask1_key="motif_mask",
            coords2_key="coords_nm",
            mask2_key="coord_mask",
            **kwargs,
        )
        self.dim = self.cross_seq_feat.dim  # 4 * 21 = 84
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "coords_nm" in batch and "coord_mask" in batch:
            return self.cross_seq_feat(batch)  # [b, n_motif, n_sample, 4*21]
        else:
            if not self._has_logged:
                logger.warning(
                    "Missing required data for MotifToSamplePairwiseDistancesPairFeat: need x_motif, motif_mask, coords_nm, coord_mask"
                )
                self._has_logged = True

            # Determine dimensions for fallback
            if "x_motif" in batch and "coords_nm" in batch:
                b = batch["x_motif"].shape[0]
                n_motif = batch["x_motif"].shape[1]
                n_sample = batch["coords_nm"].shape[1]
                device = batch["x_motif"].device
            elif "coords_nm" in batch:
                b, n_sample = batch["coords_nm"].shape[:2]
                n_motif = n_sample  # fallback assumption
                device = batch["coords_nm"].device
            else:
                # Last resort fallback
                b, n_motif = 1, 1
                n_sample = 1
                device = torch.device("cpu")

            return torch.zeros(b, n_motif, n_sample, self.dim, device=device)


class MotifConcatSeqFeat(Feature):
    """Computes concat motif features combining coordinates, sequence, and mask."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.coords_feat = Atom37NanometersCoorsSeqFeat(rel=False)
        self.seq_feat = ResidueTypeSeqFeat()
        self.dim = self.coords_feat.dim + self.seq_feat.dim + 37  # 148 + 20 + 37 = 205
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" not in batch or "motif_mask" not in batch or "seq_motif" not in batch:
            if not self._has_logged:
                logger.warning("Missing required motif data for MotifConcatSeqFeat")
                self._has_logged = True
            b = batch.get("batch_size", 1) if "batch_size" in batch else 1
            device = self.extract_device(batch)
            return torch.zeros(b, 0, self.dim, device=device), torch.zeros(b, 0, dtype=torch.bool, device=device)

        # Check if data is already in compact mode (motif coordinates are already extracted)
        # In compact mode, x_motif will have shape [b, n_motif, 37, 3] where n_motif <= n_orig
        # In non-compact mode, x_motif will have shape [b, n_orig, 37, 3] with zeros for non-motif residues

        # Detect compact mode by checking if motif_mask has all True values along the sequence dimension
        motif_residue_mask = batch["motif_mask"].sum(dim=-1).bool()  # [b, n]
        is_compact_mode = motif_residue_mask.all(dim=1).any()  # Check if any batch has all True (compact mode)

        if is_compact_mode:
            # Compact mode: data is already extracted, use directly
            coords_feats = self.coords_feat(
                {
                    "coords_nm": batch["x_motif"],  # [b, n_motif, 37, 3]
                    "coord_mask": batch["motif_mask"],  # [b, n_motif, 37]
                }
            )  # [b, n_motif, 148]

            # Sequence features
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n_motif]
                "mask_dict": {"residue_type": batch["seq_motif_mask"]},  # [b, n_motif]
            }
            seq_feats = self.seq_feat(batch_seq)  # [b, n_motif, 20]

            # Motif mask features
            mask_feats = batch["motif_mask"] * 1.0  # [b, n_motif, 37]

            # Concatenate all features
            combined_feats = torch.cat([coords_feats, seq_feats, mask_feats], dim=-1)  # [b, n_motif, 205]
            combined_feats = combined_feats * batch["seq_motif_mask"][..., None]  # Apply mask

            # Return as-is since it's already compact
            return combined_feats, batch["seq_motif_mask"]

        else:
            # Non-compact mode: extract motif residues from full sequence
            coords_feats = self.coords_feat(
                {
                    "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                    "coord_mask": batch["motif_mask"],  # [b, n, 37]
                }
            )  # [b, n, 148]

            # Sequence features
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {"residue_type": motif_residue_mask},
            }
            seq_feats = self.seq_feat(batch_seq)  # [b, n, 20]

            # Motif mask features
            mask_feats = batch["motif_mask"] * 1.0  # [b, n, 37]

            # Concatenate all features
            combined_feats = torch.cat([coords_feats, seq_feats, mask_feats], dim=-1)  # [b, n, 205]
            combined_feats = combined_feats * motif_residue_mask[..., None]  # Apply mask

            # Extract only residues that have motif atoms
            batch_size = combined_feats.shape[0]
            concat_feats = []
            concat_masks = []

            for b in range(batch_size):
                residue_mask = motif_residue_mask[b]  # [n]
                if residue_mask.any():
                    selected_feats = combined_feats[b][residue_mask]  # [n_motif, 205]
                    selected_mask = torch.ones(
                        selected_feats.shape[0],
                        dtype=torch.bool,
                        device=selected_feats.device,
                    )
                else:
                    selected_feats = torch.zeros(0, self.dim, device=combined_feats.device)
                    selected_mask = torch.zeros(0, dtype=torch.bool, device=combined_feats.device)

                concat_feats.append(selected_feats)
                concat_masks.append(selected_mask)

            # Pad to same length
            padded_feats = pad_sequence(concat_feats, batch_first=True, padding_value=0.0)
            padded_masks = pad_sequence(concat_masks, batch_first=True, padding_value=False)

            return padded_feats, padded_masks


class TargetConcatSeqFeat(Feature):
    """Computes concat target features combining coordinates, sequence, and mask."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.coords_feat = Atom37NanometersCoorsSeqFeat(rel=False)
        self.seq_feat = ResidueTypeSeqFeat()
        self.hotspot_feat = HotspotMaskSeqFeat()
        self.rel_coords_feat = Atom37NanometersCoorsSeqFeat(rel=True)
        self.side_chain_feat = TargetSideChainAnglesSeqFeat()
        self.torsion_feat = TargetTorsionAnglesSeqFeat()
        self.dim = (
            self.coords_feat.dim * 2
            + self.seq_feat.dim
            + 37
            + self.hotspot_feat.dim
            + self.side_chain_feat.dim
            + self.torsion_feat.dim
        )  # 148 * 2 + 20 + 37 + 1 + 102 + 102 = 558
        self._has_logged = False

    def forward(self, batch):
        if "x_target" not in batch or "target_mask" not in batch or "seq_target" not in batch:
            if not self._has_logged:
                logger.warning("Missing required target data for TargetConcatSeqFeat")
                self._has_logged = True
            # b = batch.get("batch_size", 1) if "batch_size" in batch else 1
            b, _ = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, 0, self.dim, device=device), torch.zeros(b, 0, dtype=torch.bool, device=device)

        # Check if data is already in compact mode (target coordinates are already extracted)
        # In compact mode, x_target will have shape [b, n_target, 37, 3] where n_target <= n_orig
        # In non-compact mode, x_target will have shape [b, n_orig, 37, 3] with zeros for non-target residues

        # Detect compact mode by checking if target_mask has all True values along the sequence dimension
        target_residue_mask = batch["target_mask"].sum(dim=-1).bool()  # [b, n]
        is_compact_mode = target_residue_mask.all(dim=1).any()  # Check if any batch has all True (compact mode)

        if is_compact_mode:
            # Compact mode: data is already extracted, use directly
            coords_feats = self.coords_feat(
                {
                    "coords_nm": batch["x_target"],  # [b, n_target, 37, 3]
                    "coord_mask": batch["target_mask"],  # [b, n_target, 37]
                }
            )  # [b, n_target, 148]

            rel_coords_feats = self.rel_coords_feat(
                {
                    "coords_nm": batch["x_target"],  # [b, n_target, 37, 3]
                    "coord_mask": batch["target_mask"],  # [b, n_target, 37]
                }
            )  # [b, n_target, 148]

            # Sequence features
            batch_seq = {
                "residue_type": batch["seq_target"],  # [b, n_target]
                "mask_dict": {"residue_type": batch["seq_target_mask"]},  # [b, n_target]
            }
            seq_feats = self.seq_feat(batch_seq)  # [b, n_target, 20]

            # Target mask features
            mask_feats = batch["target_mask"] * 1.0  # [b, n_target, 37]

            hotspot_feats = self.hotspot_feat(
                {
                    "hotspot_mask": batch["target_hotspot_mask"],
                }
            )  # [b, n_target, 1]

            side_chain_feats = self.side_chain_feat(batch)  # [b, n_target, 102]
            torsion_feats = self.torsion_feat(batch)  # [b, n_target, 102]

            # Concatenate all features
            combined_feats = torch.cat(
                [
                    coords_feats,
                    seq_feats,
                    mask_feats,
                    hotspot_feats,
                    rel_coords_feats,
                    side_chain_feats,
                    torsion_feats,
                ],
                dim=-1,
            )  # [b, n_target, 558]
            combined_feats = combined_feats * batch["seq_target_mask"][..., None]  # Apply mask

            # Return as-is since it's already compact
            return combined_feats, batch["seq_target_mask"]

        else:
            # Non-compact mode: extract target residues from full sequence
            coords_feats = self.coords_feat(
                {
                    "coords_nm": batch["x_target"],  # [b, n, 37, 3]
                    "coord_mask": batch["target_mask"],  # [b, n, 37]
                }
            )  # [b, n, 148]

            rel_coords_feats = self.rel_coords_feat(
                {
                    "coords_nm": batch["x_target"],  # [b, n_target, 37, 3]
                    "coord_mask": batch["target_mask"],  # [b, n_target, 37]
                }
            )  # [b, n, 148]

            # Sequence features
            batch_seq = {
                "residue_type": batch["seq_target"],  # [b, n]
                "mask_dict": {"residue_type": target_residue_mask},
            }
            seq_feats = self.seq_feat(batch_seq)  # [b, n, 20]

            # Target mask features
            mask_feats = batch["target_mask"] * 1.0  # [b, n, 37]

            # Hotspot features
            hotspot_feats = self.hotspot_feat(
                {
                    "hotspot_mask": batch["target_hotspot_mask"],
                }
            )  # [b, n, 1]

            side_chain_feats = self.side_chain_feat(batch)  # [b, n, 102]
            torsion_feats = self.torsion_feat(batch)  # [b, n, 102]

            # Concatenate all features
            combined_feats = torch.cat(
                [
                    coords_feats,
                    seq_feats,
                    mask_feats,
                    hotspot_feats,
                    rel_coords_feats,
                    side_chain_feats,
                    torsion_feats,
                ],
                dim=-1,
            )  # [b, n, 558]
            combined_feats = combined_feats * target_residue_mask[..., None]  # Apply mask

            # Extract only residues that have target atoms
            batch_size = combined_feats.shape[0]
            concat_feats = []
            concat_masks = []

            for b in range(batch_size):
                residue_mask = target_residue_mask[b]  # [n]
                if residue_mask.any():
                    selected_feats = combined_feats[b][residue_mask]  # [n_target, 205]
                    selected_mask = torch.ones(
                        selected_feats.shape[0],
                        dtype=torch.bool,
                        device=selected_feats.device,
                    )
                else:
                    selected_feats = torch.zeros(0, self.dim, device=combined_feats.device)
                    selected_mask = torch.zeros(0, dtype=torch.bool, device=combined_feats.device)

                concat_feats.append(selected_feats)
                concat_masks.append(selected_mask)

            # Pad to same length
            padded_feats = pad_sequence(concat_feats, batch_first=True, padding_value=0.0)
            padded_masks = pad_sequence(concat_masks, batch_first=True, padding_value=False)

            return padded_feats, padded_masks


class LigandConcatSeqFeat(Feature):
    """Computes concat ligand features combining coordinates, sequence, and mask."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.coords_feat = AtomisticCoorsSeqFeat()
        self.seq_feat = AtomTypeSeqFeat(input_is_onehot=True)
        self.charge_feat = TargetChargeSeqFeat()
        self.atom_name_feat = TargetAtomNameSeqFeat()
        self.graph_pe_feat = TargetGraphPESeqFeat()

        self.dim = (
            self.coords_feat.dim
            + self.seq_feat.dim
            + self.charge_feat.dim
            + self.atom_name_feat.dim
            + self.graph_pe_feat.dim
            + 1
        )  # 1 for the mask
        self._has_logged = False

    def forward(self, batch):
        if "x_target" not in batch or "target_mask" not in batch or "seq_target" not in batch:
            if not self._has_logged:
                logger.warning("Missing required target data for TargetConcatSeqFeat")
                self._has_logged = True
            # b = batch.get("batch_size", 1) if "batch_size" in batch else 1
            b, _ = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, 0, self.dim, device=device), torch.zeros(b, 0, dtype=torch.bool, device=device)

        # Check if data is already in compact mode (target coordinates are already extracted)
        # In compact mode, x_target will have shape [b, n_target, 37, 3] where n_target <= n_orig
        # In non-compact mode, x_target will have shape [b, n_orig, 37, 3] with zeros for non-target residues

        # Detect compact mode by checking if target_mask has all True values along the sequence dimension
        target_residue_mask = batch["target_mask"]  # .sum(dim=-1).bool()  # [b, n]
        is_compact_mode = target_residue_mask.all(dim=1).any()  # Check if any batch has all True (compact mode)
        if is_compact_mode:
            # Compact mode: data is already extracted, use directly
            coords_feats = self.coords_feat(
                {
                    "coords_nm": batch["x_target"],  # [b, n_target, 3]
                    "coord_mask": batch["target_mask"],  # [..., None],  # [b, n_target]
                }
            )  # [b, n_target, 148]

            # Sequence features
            batch_seq = {
                "residue_type": batch["seq_target"],  # [b, n_target, 128] #! already one hotted
                "mask_dict": {"residue_type": batch["target_mask"]},  # [b, n_target]
            }
            seq_feats = self.seq_feat(batch_seq)  # [b, n_target, 128]

            # Target mask features
            mask_feats = batch["target_mask"][..., None] * 1.0  # [b, n_target, 1]

            charge_feats = self.charge_feat(batch)  # [b, n_target, 1]
            atom_name_feats = self.atom_name_feat(batch)  # [b, n_target, 4*64]
            graph_pe_feats = self.graph_pe_feat(batch)  # [b, n_target, 32]

            # Concatenate all features
            combined_feats = torch.cat(
                [
                    coords_feats,
                    seq_feats,
                    mask_feats,
                    charge_feats,
                    atom_name_feats,
                    graph_pe_feats,
                ],
                dim=-1,
            )  # [b, n_target, 558]
            combined_feats = combined_feats * batch["target_mask"][..., None]  # Apply mask

            # Return as-is since it's already compact
            return combined_feats, batch["target_mask"].bool()
        else:
            raise ValueError("LigandConcatSeqFeat only supports compact mode")

        # else:
        #     # Non-compact mode: extract target residues from full sequence
        #     coords_feats = self.coords_feat(
        #         {
        #             "coords_nm": batch["x_target"],  # [b, n, 37, 3]
        #             "coord_mask": batch["target_mask"],  # [b, n, 37]
        #         }
        #     )  # [b, n, 148]

        #     rel_coords_feats = self.rel_coords_feat(
        #         {
        #             "coords_nm": batch["x_target"],  # [b, n_target, 37, 3]
        #             "coord_mask": batch["target_mask"],  # [b, n_target, 37]
        #         }
        #     )  # [b, n, 148]

        #     # Sequence features
        #     batch_seq = {
        #         "residue_type": batch["seq_target"],  # [b, n]
        #         "mask_dict": {"residue_type": target_residue_mask},
        #     }
        #     seq_feats = self.seq_feat(batch_seq)  # [b, n, 20]

        #     # Target mask features
        #     mask_feats = batch["target_mask"] * 1.0  # [b, n, 37]

        #     # Hotspot features
        #     hotspot_feats = self.hotspot_feat({
        #         "hotspot_mask": batch["target_hotspot_mask"],
        #     })  # [b, n, 1]

        #     side_chain_feats = self.side_chain_feat(batch)  # [b, n, 102]
        #     torsion_feats = self.torsion_feat(batch)  # [b, n, 102]

        #     # Concatenate all features
        #     combined_feats = torch.cat(
        #         [coords_feats, seq_feats, mask_feats, hotspot_feats, rel_coords_feats, side_chain_feats, torsion_feats], dim=-1
        #     )  # [b, n, 558]
        #     combined_feats = (
        #         combined_feats * target_residue_mask[..., None]
        #     )  # Apply mask

        #     # Extract only residues that have target atoms
        #     batch_size = combined_feats.shape[0]
        #     concat_feats = []
        #     concat_masks = []

        #     for b in range(batch_size):
        #         residue_mask = target_residue_mask[b]  # [n]
        #         if residue_mask.any():
        #             selected_feats = combined_feats[b][residue_mask]  # [n_target, 205]
        #             selected_mask = torch.ones(
        #                 selected_feats.shape[0],
        #                 dtype=torch.bool,
        #                 device=selected_feats.device,
        #             )
        #         else:
        #             selected_feats = torch.zeros(
        #                 0, self.dim, device=combined_feats.device
        #             )
        #             selected_mask = torch.zeros(
        #                 0, dtype=torch.bool, device=combined_feats.device
        #             )

        #         concat_feats.append(selected_feats)
        #         concat_masks.append(selected_mask)

        #     # Pad to same length
        #     padded_feats = pad_sequence(
        #         concat_feats, batch_first=True, padding_value=0.0
        #     )
        #     padded_masks = pad_sequence(
        #         concat_masks, batch_first=True, padding_value=False
        #     )

        #     return padded_feats, padded_masks


####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: list[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair", "target"],
        **kwargs,
    ):
        """
        Feature factory for creating sequence and pair features.

        Sequence features include:
            Time embeddings:
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents

            Position and structure:
                - "res_seq_pdb_idx": Residue sequence position (requires ResidueSequencePositionPdbTransform)
                - "chain_break_per_res": Chain break per residue (requires ChainBreakPerResidueTransform)
                - "chain_idx_seq": Chain index as sequence feature
                - "fold_emb": Fold embedding

            Coordinates and angles:
                - "x_sc_bb_ca": Self-conditioning backbone CA coordinates
                - "x_recycle_bb_ca": Recycled backbone CA coordinates
                - "x_sc_local_latents": Self-conditioning local latents
                - "x_recycle_local_latents": Recycled local latents
                - "xt_bb_ca": Target backbone CA coordinates
                - "xt_local_latents": Target local latents
                - "x_target": Target coordinates with atom selection
                - "ca_coors_nm": CA coordinates in nanometers
                - "ca_coors_nm_try": Try CA coordinates in nanometers
                - "optional_ca_coors_nm_seq_feat": Optional CA coordinates in nanometers
                - "x1_bb_angles": Backbone torsion angles
                - "x1_bond_angles": Backbone bond angles
                - "x1_sidechain_angles": Sidechain angles

            Residue information:
                - "x1_aatype": Residue type
                - "optional_res_type_seq_feat": Optional residue type
                - "x1_a37coors_nm": Atom37 coordinates in nanometers
                - "x1_a37coors_nm_rel": Relative atom37 coordinates in nanometers

            Motif and target:
                - "x_motif": Motif coordinates and features
                - "z_latent_seq": Latent variable sequence

            Contact and binder features:
                - "hotspot_idx_seq": Target hotspot indices
                - "binder_center": Binder center coordinates
                - "stochastic_translation": Stochastic translation from centering
                - "contact_type_seq": Contact composition features

        Pair features include:
            Distance features:
                - "xt_bb_ca_pair_dists": Target backbone CA pairwise distances
                - "x_sc_bb_ca_pair_dists": Self-conditioning backbone CA pairwise distances
                - "x_recycle_bb_ca_pair_dists": Recycled backbone CA pairwise distances
                - "x_target_pair_dists": Target pairwise distances
                - "ca_coors_nm_pair_dists": CA coordinates pairwise distances in nanometers
                - "x1_bb_pair_dists_nm": Backbone pairwise distances in nanometers
                - "optional_ca_pair_dist": Optional CA pairwise distances
                - "cross_seq_bb_pair_dists": Cross-sequence backbone pairwise distances (rectangular matrix)
                - "x_motif_pair_dists": Motif pairwise distances
                - "x_target_pair_dists": Target pairwise distances
                - "target_to_sample_pair_dists": Target-to-sample cross-sequence pairwise distances (rectangular matrix)
                - "motif_to_sample_pair_dists": Motif-to-sample cross-sequence pairwise distances (rectangular matrix)
                - "sample_to_target_pair_dists": Sample-to-target cross-sequence pairwise distances (rectangular matrix)
                - "target_to_sample_xsc_pair_dists": Target-to-sample cross-sequence x_sc pairwise distances (rectangular matrix)
                - "sample_to_target_xsc_pair_dists": Sample-to-target cross-sequence x_sc pairwise distances (rectangular matrix)
                - "target_to_target_xsc_pair_dists": Target-to-target cross-sequence x_sc pairwise distances (square matrix)
                - "target_to_sample_optional_ca_dists": Target-to-sample cross-sequence optional CA distances (rectangular matrix)
                - "sample_to_target_optional_ca_dists": Sample-to-target cross-sequence optional CA distances (rectangular matrix)
                - "target_to_target_optional_ca_dists": Target-to-target cross-sequence optional CA distances (square matrix)
                - "cross_seq_xsc_pair_dists": Generic cross-sequence x_sc pairwise distances
                - "cross_seq_optional_ca_dists": Generic cross-sequence optional CA distances

            Sequence and time:
                - "rel_seq_sep": Relative sequence separation
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents
                - "target_to_sample_seq_sep": Target-to-sample cross-sequence relative sequence separation (rectangular matrix)
                - "target_to_target_seq_sep": Target-to-target relative sequence separation (square matrix)
                - "cross_seq_rel_sep": Generic cross-sequence relative sequence separation

            Structure and orientation:
                - "x1_bb_pair_orientation": Relative residue orientation
                - "chain_idx_pair": Chain index pairwise feature
                - "target_to_sample_chain_idx": Target-to-sample cross-sequence chain index features (rectangular matrix)
                - "target_to_target_chain_idx": Target-to-target chain index features (square matrix)
                - "cross_seq_chain_idx": Generic cross-sequence chain index features

            Contact and hotspot features:
                - "hotspot_idx_pair": Target hotspot pairwise features
                - "contact_type_pair": Contact composition pairwise features
                - "target_mask_pair": Target mask pairwise features
        """
        super().__init__()
        self.mode = mode

        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList([self.get_creator(f, **kwargs) for f in feats])
        self.ln_out = torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        self.linear_out = torch.nn.Linear(sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False)

    def get_creator(self, f, **kwargs):
        """Returns the right class for the requested feature f (a string)."""

        if self.mode in ["seq", "target"]:
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingSeqFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingSeqFeat(data_mode_use="local_latents", **kwargs)

            # Position and indexing
            elif f == "res_seq_pdb_idx":
                return IdxEmbeddingSeqFeat(**kwargs)
            elif f == "chain_break_per_res":
                return ChainBreakPerResidueSeqFeat(**kwargs)
            elif f == "chain_idx_seq":
                return ChainIdxSeqFeat(**kwargs)
            elif f == "fold_emb":
                return FoldEmbeddingSeqFeat(**kwargs)
            elif f == "cropped_flag_seq":
                return CroppedFlagSeqFeat()

            # Genie2 embeddings
            elif f == "time_emb_bb_ca_genie2":
                return TimeEmbeddingSeqFeatGenie2(data_mode_use="bb_ca", **kwargs)
            elif f == "res_seq_pdb_idx_genie2":
                return IdxEmbeddingSeqFeatGenie2(**kwargs)

            # Basic residue information
            elif f == "x1_aatype":
                return ResidueTypeSeqFeat(**kwargs)
            elif f == "optional_res_type_seq_feat":
                return OptionalResidueTypeSeqFeat(**kwargs)

            # Raw coordinate features
            elif f == "ca_coors_nm":
                return CaCoorsNanometersSeqFeat(**kwargs)
            elif f == "ca_coors_nm_try":
                return TryCaCoorsNanometersSeqFeat(**kwargs)
            elif f == "optional_ca_coors_nm_seq_feat":
                return OptionalCaCoorsNanometersSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm":
                return Atom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm_rel":
                return Atom37NanometersCoorsSeqFeat(rel=True, **kwargs)

            # Diffusion/sampling coordinates
            elif f == "xt_bb_ca":
                return XtBBCASeqFeat(**kwargs)
            elif f == "xt_local_latents":
                return XtLocalLatentsSeqFeat(**kwargs)
            elif f == "x_sc_bb_ca":
                return XscBBCASeqFeat(**kwargs)
            elif f == "x_recycle_bb_ca":
                return XscBBCASeqFeat(mode_key="x_recycle", **kwargs)
            elif f == "x_sc_local_latents":
                return XscLocalLatentsSeqFeat(**kwargs)
            elif f == "x_recycle_local_latents":
                return XscLocalLatentsSeqFeat(mode_key="x_recycle", **kwargs)

            # Structural features (angles)
            elif f == "x1_bb_angles":
                return BackboneTorsionAnglesSeqFeat(**kwargs)
            elif f == "x1_bond_angles":
                return BackboneBondAnglesSeqFeat(**kwargs)
            elif f == "x1_sidechain_angles":
                return OpenfoldSideChainAnglesSeqFeat(**kwargs)

            # Latent variables
            elif f == "z_latent_seq":
                return LatentVariableSeqFeat(**kwargs)

            # Motif features
            elif f == "motif_abs_coords":
                return MotifAbsoluteCoordsSeqFeat(**kwargs)
            elif f == "motif_rel_coords":
                return MotifRelativeCoordsSeqFeat(**kwargs)
            elif f == "motif_seq":
                return MotifSequenceSeqFeat(**kwargs)
            elif f == "motif_sc_angles":
                return MotifSideChainAnglesSeqFeat(**kwargs)
            elif f == "motif_torsion_angles":
                return MotifTorsionAnglesSeqFeat(**kwargs)
            elif f == "motif_mask":
                return MotifMaskSeqFeat(**kwargs)

            # Target features
            elif f == "target_abs_coords":
                return TargetAbsoluteCoordsSeqFeat(**kwargs)
            elif f == "target_rel_coords":
                return TargetRelativeCoordsSeqFeat(**kwargs)
            elif f == "target_seq":
                return TargetSequenceSeqFeat(**kwargs)
            elif f == "target_sc_angles":
                return TargetSideChainAnglesSeqFeat(**kwargs)
            elif f == "target_torsion_angles":
                return TargetTorsionAnglesSeqFeat(**kwargs)
            elif f == "target_mask_seq":
                return TargetMaskSeqFeat(**kwargs)

            # Design and binder features
            elif f == "hotspot_mask_seq":
                return HotspotMaskSeqFeat(**kwargs)
            elif f == "binder_center":
                return BinderCenterFeat(**kwargs)
            elif f == "stochastic_translation":
                return StochasticTranslationSeqFeat(**kwargs)
            elif f == "contact_type_seq":
                return ContactTypeSeqFeat(**kwargs)

            # Special/utility features
            elif f == "zero_feat_seq":
                return ZeroFeat(**kwargs)
            else:
                raise OSError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingPairFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingPairFeat(data_mode_use="local_latents", **kwargs)

            # Sequence separation
            elif f == "rel_seq_sep":
                return SequenceSeparationPairFeat(**kwargs)

            # Distance features
            elif f == "xt_bb_ca_pair_dists":
                return XtBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_sc_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_recycle_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(mode_key="x_recycle", **kwargs)
            elif f == "ca_coors_nm_pair_dists":
                return CaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "optional_ca_pair_dist":
                return OptionalCaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "x1_bb_pair_dists_nm":
                return BackbonePairDistancesNanometerPairFeat(**kwargs)
            elif f == "cross_seq_bb_pair_dists":
                return CrossSequenceBackbonePairDistancesPairFeat(**kwargs)
            elif f == "x_motif_pair_dists":
                return XmotifPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_target_pair_dists":
                return XtargetPairwiseDistancesPairFeat(**kwargs)
            elif f == "target_to_sample_pair_dists":
                return TargetToSamplePairwiseDistancesPairFeat(**kwargs)
            elif f == "motif_to_sample_pair_dists":
                return MotifToSamplePairwiseDistancesPairFeat(**kwargs)
            elif f == "sample_to_target_pair_dists":
                return SampleToTargetPairwiseDistancesPairFeat(**kwargs)
            elif f == "target_to_sample_xsc_pair_dists":
                return TargetToSampleXscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "sample_to_target_xsc_pair_dists":
                return SampleToTargetXscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "target_to_target_xsc_pair_dists":
                return TargetToTargetXscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "target_to_sample_optional_ca_dists":
                return TargetToSampleOptionalCaPairDistancesPairFeat(**kwargs)
            elif f == "sample_to_target_optional_ca_dists":
                return SampleToTargetOptionalCaPairDistancesPairFeat(**kwargs)
            elif f == "target_to_target_optional_ca_dists":
                return TargetToTargetOptionalCaPairDistancesPairFeat(**kwargs)
            elif f == "cross_seq_xsc_pair_dists":
                return CrossSequenceXscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "cross_seq_optional_ca_dists":
                return CrossSequenceOptionalCaPairDistancesPairFeat(**kwargs)

            # Structural and orientation features
            elif f == "x1_bb_pair_orientation":
                return RelativeResidueOrientationPairFeat(**kwargs)

            # Chain and indexing features
            elif f == "chain_idx_pair":
                return ChainIdxPairFeat(**kwargs)

            # Design and contact features
            elif f == "hotspot_mask_pair":
                return HotspotMaskPairFeat(**kwargs)
            elif f == "contact_type_pair":
                return ContactTypePairFeat(**kwargs)
            elif f == "target_mask_pair":
                return TargetMaskPairFeat(**kwargs)

            # Cross-sequence features
            elif f == "sample_to_target_pair_dists":
                return SampleToTargetPairwiseDistancesPairFeat(**kwargs)
            elif f == "target_to_sample_seq_sep":
                return TargetToSampleRelativeSequenceSeparationPairFeat(**kwargs)
            elif f == "target_to_sample_chain_idx":
                return TargetToSampleChainIndexPairFeat(**kwargs)
            elif f == "target_to_target_seq_sep":
                return TargetToTargetRelativeSequenceSeparationPairFeat(**kwargs)
            elif f == "target_to_target_chain_idx":
                return TargetToTargetChainIndexPairFeat(**kwargs)
            elif f == "cross_seq_rel_sep":
                return CrossSequenceRelativeSequenceSeparationPairFeat(**kwargs)
            elif f == "cross_seq_chain_idx":
                return CrossSequenceChainIndexPairFeat(**kwargs)

            else:
                raise OSError(f"Pair feature {f} not implemented.")

        else:
            raise OSError(f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'.")

    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode in ["seq", "target"]:
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise OSError(f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'.")

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(fcreator(batch))  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode
        # Concatenate features and mask
        features = torch.cat(feature_tensors, dim=-1)  # [b, n, dim_f] or [b, n, n, dim_f]
        if self.mode == "target":
            mask = batch["seq_target_mask"]
        else:
            mask = batch["mask"]
        features = self.apply_padding_mask(features, mask)  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_proc = self.ln_out(self.linear_out(features))  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(features_proc, mask)  # [b, n, dim_f] or [b, n, n, dim_f]


class ConcatFeaturesFactory(torch.nn.Module):
    """Factory for creating concat features from motif and/or target."""

    def __init__(
        self,
        enable_motif: bool = False,
        enable_target: bool = False,
        enable_ligand: bool = False,
        dim_feats_out: int = 256,
        use_ln_out: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.enable_motif = enable_motif
        self.enable_target = enable_target
        self.enable_ligand = enable_ligand

        if not (enable_motif or enable_target or enable_ligand):
            raise ValueError("At least one of enable_motif or enable_target or enable_ligand must be True")

        self.feature_creators = torch.nn.ModuleList()

        if enable_motif:
            self.feature_creators.append(MotifConcatSeqFeat(**kwargs))
            logger.info("Enabled motif concat features")
        elif enable_target:
            self.feature_creators.append(TargetConcatSeqFeat(**kwargs))
            logger.info("Enabled target concat features")
        elif enable_ligand:
            self.feature_creators.append(LigandConcatSeqFeat(**kwargs))
            logger.info("Enabled ligand concat features")

        # Calculate total input dimension
        total_dim = sum(creator.get_dim() for creator in self.feature_creators)

        # Create projection layers
        self.linear_out = torch.nn.Linear(total_dim, dim_feats_out, bias=False)
        self.ln_out = torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()

        # logger.info(f"ConcatFeaturesFactory: input feat dim {total_dim} -> output feat dim {dim_feats_out}")

    def forward(self, batch, seq_repr, seq_mask):
        """
        Args:
            batch: Input batch dictionary
            seq_repr: Original sequence representation [b, n, dim]
            seq_mask: Original sequence mask [b, n]

        Returns:
            extended_seq_repr: Extended sequence representation [b, n + n_concat, dim]
            extended_mask: Extended mask [b, n + n_concat]
        """
        # Get concat features
        all_feats = []
        all_masks = []

        for creator in self.feature_creators:
            feats, mask = creator(batch)  # [b, n_i, d_i], [b, n_i]
            all_feats.append(feats)
            all_masks.append(mask)

        # Concatenate features if we have any
        if len(all_feats) == 0:
            # No concat features, return original
            return seq_repr, seq_mask

        if len(all_feats) == 1:
            combined_feats = all_feats[0]  # [b, n_concat, d_total]
            combined_masks = all_masks[0]  # [b, n_concat]
        else:
            # Concatenate along sequence dimension (motif residues + target residues)
            combined_feats = torch.cat(all_feats, dim=1)  # [b, n_concat, d_total]
            combined_masks = torch.cat(all_masks, dim=1)  # [b, n_concat]

        # Apply linear projection to match seq_repr dimension
        projected_feats = self.ln_out(self.linear_out(combined_feats))  # [b, n_concat, dim_feats_out]
        projected_feats = projected_feats * combined_masks[..., None]  # Apply mask

        # Concatenate with original sequence representation
        extended_seq_repr, extended_mask = concat_padded_tensor(
            a=seq_repr, b=projected_feats, mask_a=seq_mask, mask_b=combined_masks
        )  # [b, pad_len, dim], pad_len = max(n_i + m_i)
        if extended_seq_repr.shape[1] < seq_repr.shape[1]:
            print(
                "\n\nERROR: feature_factory.py ln 3445 Issue in padding or data sample is not contiguous so breaks concat pad merging"
            )
            print(seq_mask.sum(-1), extended_seq_repr.shape, seq_repr.shape)
            raise ValueError("Issue in padding or data sample is not contiguous so breaks concat pad merging")
            #! TODO something weird is happening with the SAIR data
        # logger.info(f"ConcatFeaturesFactory: seq_repr {seq_repr.shape} + projected_feats {projected_feats.shape} -> extended_seq_repr{extended_seq_repr.shape}")
        # logger.info(f"ConcatFeaturesFactory: seq_mask {seq_mask.sum(-1)} + combined_masks {combined_masks.sum(-1)} -> extended_mask {extended_mask.sum(-1)}")
        return extended_seq_repr, extended_mask


class ConcatPairFeaturesFactory(torch.nn.Module):
    """Factory for creating extended pair representations using cross-sequence features."""

    def __init__(
        self,
        enable_target: bool = False,
        enable_motif: bool = False,
        enable_ligand: bool = False,
        dim_pair_out: int = 256,
        use_ln_out: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.enable_target = enable_target
        self.enable_motif = enable_motif
        self.enable_ligand = enable_ligand
        self.dim_pair_out = dim_pair_out

        if not (enable_target or enable_motif or enable_ligand):
            raise ValueError("At least one of enable_target or enable_motif or enable_ligand must be True")

        if enable_ligand and not enable_target and not enable_motif:
            # TODO:
            # Target-based features
            # Upper right: Sample-to-target features (using generic cross-sequence features)
            # self.upper_right_seq_sep = CrossSequenceRelativeSequenceSeparationPairFeat(
            #     seq1_key="residue_type",
            #     seq2_key="seq_target",
            #     idx1_key="residue_pdb_idx",
            #     idx2_key="target_pdb_idx",
            #     **kwargs,
            # )
            self.upper_right_xt_dist = CrossSequenceBackboneAtomPairDistancesPairFeat(
                coords1_key="x_t_coord",
                mask1_key="x_t_coord_mask",
                coords2_key="x_target",
                mask2_key="target_mask",
                **kwargs,
            )
            # self.upper_right_xsc_dist = CrossSequenceXscBBCAPairwiseDistancesPairFeat(
            #     coords1_key="x_t_coord", coords2_key="x_target", **kwargs
            # )
            # self.upper_right_optional_ca = CrossSequenceOptionalCaPairDistancesPairFeat(
            #     coords1_key="coords_nm", coords2_key="x_target", **kwargs
            # )
            # self.upper_right_chain = CrossSequenceChainIndexPairFeat(
            #     chain1_key="chains", chain2_key="target_chains", **kwargs
            # )
            # self.upper_right_hotspots = CrossSequenceHotspotMaskPairFeat(
            #     hotspot_mask1_key="hotspot_mask", hotspot_mask2_key="target_hotspot_mask", **kwargs
            # )

            # Lower left: NOT COMPUTED - just transpose of upper right for efficiency!

            # Lower right: Target-to-target features (using generic cross-sequence features)
            # self.lower_right_seq_sep = CrossSequenceRelativeSequenceSeparationPairFeat(
            #     seq1_key="seq_target",
            #     seq2_key="seq_target",
            #     idx1_key="target_pdb_idx",
            #     idx2_key="target_pdb_idx",
            #     **kwargs,
            # )
            self.lower_right_bond_mask = BondMaskPairFeat(key="target_bond_mask", **kwargs)
            self.lower_right_bond_order = BondOrderPairFeat(key="target_bond_order", **kwargs)

            self.lower_right_xt_dist = TargetAtomPairwiseDistancesPairFeat(
                **kwargs,
            )
            # self.lower_right_xsc_dist = CrossSequenceXscBBCAPairwiseDistancesPairFeat(
            #     coords1_key="x_target", coords2_key="x_target", **kwargs
            # )
            # self.lower_right_optional_ca = CrossSequenceOptionalCaPairDistancesPairFeat(
            #     coords1_key="x_target", coords2_key="x_target", **kwargs
            # )
            # self.lower_right_chain = CrossSequenceChainIndexPairFeat(
            #     chain1_key="target_chains", chain2_key="target_chains", **kwargs
            # )
            # self.lower_right_hotspots = CrossSequenceHotspotMaskPairFeat(
            #     hotspot_mask1_key="target_hotspot_mask", hotspot_mask2_key="target_hotspot_mask", **kwargs
            # )

            # Calculate total input dimension for cross-sequence features
            total_cross_seq_dim = (
                # self.upper_right_seq_sep.get_dim()
                self.upper_right_xt_dist.get_dim()
                # + self.upper_right_xsc_dist.get_dim()
                # + self.upper_right_optional_ca.get_dim()
                # + self.upper_right_chain.get_dim()
                # + self.upper_right_hotspots.get_dim()
            )
            total_lower_right_dim = (
                self.lower_right_xt_dist.get_dim()
                + self.lower_right_bond_mask.get_dim()
                + self.lower_right_bond_order.get_dim()
            )
            # Create projection layers to match original pair representation dimension
            self.linear_out = torch.nn.Linear(total_cross_seq_dim, dim_pair_out, bias=False)
            self.ln_out = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()

            self.linear_out_lower_right = torch.nn.Linear(total_lower_right_dim, dim_pair_out, bias=False)
            self.ln_out_lower_right = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()

            self.linear_out_in_dim = total_cross_seq_dim
            self.linear_out_lower_right_in_dim = total_lower_right_dim
            self.coords_key = "x_target"
            self.mask_key = "target_mask"
            logger.info(
                f"Enabled target-to-sample cross-sequence pair features: input feat dim {total_cross_seq_dim} -> output feat dim {dim_pair_out}"
            )
        # Create dedicated feature classes for each quadrant
        elif enable_target and not enable_motif:
            # Target-based features
            # Upper right: Sample-to-target features (using generic cross-sequence features)
            self.upper_right_seq_sep = CrossSequenceRelativeSequenceSeparationPairFeat(
                seq1_key="residue_type",
                seq2_key="seq_target",
                idx1_key="residue_pdb_idx",
                idx2_key="target_pdb_idx",
                **kwargs,
            )
            self.upper_right_xt_dist = CrossSequenceBackbonePairDistancesPairFeat(
                coords1_key="x_t_coord",
                mask1_key="x_t_coord_mask",
                coords2_key="x_target",
                mask2_key="target_mask",
                **kwargs,
            )
            # self.upper_right_xsc_dist = CrossSequenceXscBBCAPairwiseDistancesPairFeat(
            #     coords1_key="x_t_coord", coords2_key="x_target", **kwargs
            # )
            # self.upper_right_optional_ca = CrossSequenceOptionalCaPairDistancesPairFeat(
            #     coords1_key="coords_nm", coords2_key="x_target", **kwargs
            # )
            self.upper_right_chain = CrossSequenceChainIndexPairFeat(
                chain1_key="chains", chain2_key="target_chains", **kwargs
            )
            self.upper_right_hotspots = CrossSequenceHotspotMaskPairFeat(
                hotspot_mask1_key="hotspot_mask",
                hotspot_mask2_key="target_hotspot_mask",
                **kwargs,
            )

            # Lower left: NOT COMPUTED - just transpose of upper right for efficiency!

            # Lower right: Target-to-target features (using generic cross-sequence features)
            self.lower_right_seq_sep = CrossSequenceRelativeSequenceSeparationPairFeat(
                seq1_key="seq_target",
                seq2_key="seq_target",
                idx1_key="target_pdb_idx",
                idx2_key="target_pdb_idx",
                **kwargs,
            )
            self.lower_right_xt_dist = CrossSequenceBackbonePairDistancesPairFeat(
                coords1_key="x_target",
                mask1_key="target_mask",
                coords2_key="x_target",
                mask2_key="target_mask",
                **kwargs,
            )
            # self.lower_right_xsc_dist = CrossSequenceXscBBCAPairwiseDistancesPairFeat(
            #     coords1_key="x_target", coords2_key="x_target", **kwargs
            # )
            # self.lower_right_optional_ca = CrossSequenceOptionalCaPairDistancesPairFeat(
            #     coords1_key="x_target", coords2_key="x_target", **kwargs
            # )
            self.lower_right_chain = CrossSequenceChainIndexPairFeat(
                chain1_key="target_chains", chain2_key="target_chains", **kwargs
            )
            self.lower_right_hotspots = CrossSequenceHotspotMaskPairFeat(
                hotspot_mask1_key="target_hotspot_mask",
                hotspot_mask2_key="target_hotspot_mask",
                **kwargs,
            )

            # Calculate total input dimension for cross-sequence features
            total_cross_seq_dim = (
                self.upper_right_seq_sep.get_dim()
                + self.upper_right_xt_dist.get_dim()
                # + self.upper_right_xsc_dist.get_dim()
                # + self.upper_right_optional_ca.get_dim()
                + self.upper_right_chain.get_dim()
                + self.upper_right_hotspots.get_dim()
            )

            # Create projection layers to match original pair representation dimension
            self.linear_out_in_dim = total_cross_seq_dim
            self.linear_out = torch.nn.Linear(total_cross_seq_dim, dim_pair_out, bias=False)
            self.ln_out = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()

            self.coords_key = "x_target"
            self.mask_key = "target_mask"
            logger.info(
                f"Enabled target-to-sample cross-sequence pair features: {total_cross_seq_dim} -> {dim_pair_out}"
            )

        elif enable_motif and not enable_target:
            # Motif-based features (using same generic cross-sequence features, just with motif keys)
            # Upper right: Sample-to-motif features (using generic cross-sequence features)
            self.upper_right_seq_sep = CrossSequenceRelativeSequenceSeparationPairFeat(
                seq1_key="residue_type",
                seq2_key="seq_motif",
                idx1_key="residue_pdb_idx",
                idx2_key="motif_pdb_idx",
                **kwargs,
            )
            self.upper_right_xt_dist = CrossSequenceBackbonePairDistancesPairFeat(
                coords1_key="x_t_coord",
                mask1_key="x_t_coord_mask",
                coords2_key="x_motif",
                mask2_key="motif_mask",
                **kwargs,
            )
            self.upper_right_xsc_dist = CrossSequenceXscBBCAPairwiseDistancesPairFeat(
                coords1_key="x_t_coord", coords2_key="x_motif", **kwargs
            )
            self.upper_right_optional_ca = CrossSequenceOptionalCaPairDistancesPairFeat(
                coords1_key="coords_nm", coords2_key="x_motif", **kwargs
            )
            self.upper_right_chain = CrossSequenceChainIndexPairFeat(
                chain1_key="chains", chain2_key="motif_chains", **kwargs
            )

            # Lower left: NOT COMPUTED - just transpose of upper right for efficiency!

            # Lower right: Motif-to-motif features (using generic cross-sequence features)
            self.lower_right_seq_sep = CrossSequenceRelativeSequenceSeparationPairFeat(
                seq1_key="seq_motif",
                seq2_key="seq_motif",
                idx1_key="motif_pdb_idx",
                idx2_key="motif_pdb_idx",
                **kwargs,
            )
            self.lower_right_xt_dist = CrossSequenceBackbonePairDistancesPairFeat(
                coords1_key="x_motif",
                mask1_key="motif_mask",
                coords2_key="x_motif",
                mask2_key="motif_mask",
                **kwargs,
            )
            self.lower_right_xsc_dist = CrossSequenceXscBBCAPairwiseDistancesPairFeat(
                coords1_key="x_motif", coords2_key="x_motif", **kwargs
            )
            self.lower_right_optional_ca = CrossSequenceOptionalCaPairDistancesPairFeat(
                coords1_key="x_motif", coords2_key="x_motif", **kwargs
            )
            self.lower_right_chain = CrossSequenceChainIndexPairFeat(
                chain1_key="motif_chains", chain2_key="motif_chains", **kwargs
            )

            # Calculate total input dimension for cross-sequence features
            total_cross_seq_dim = (
                self.upper_right_seq_sep.get_dim()
                + self.upper_right_xt_dist.get_dim()
                + self.upper_right_xsc_dist.get_dim()
                + self.upper_right_optional_ca.get_dim()
                + self.upper_right_chain.get_dim()
            )

            # Create projection layers to match original pair representation dimension
            self.linear_out_in_dim = total_cross_seq_dim
            self.linear_out = torch.nn.Linear(total_cross_seq_dim, dim_pair_out, bias=False)
            self.ln_out = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()

            self.coords_key = "x_motif"
            self.mask_key = "motif_mask"
            logger.info(
                f"Enabled motif-to-sample cross-sequence pair features: {total_cross_seq_dim} -> {dim_pair_out}"
            )

        else:
            raise NotImplementedError("Both target and motif enabled not yet implemented")

    def ligand_forward(self, batch, orig_pair_rep, orig_seq_mask):
        b, n_orig, _, pair_dim = orig_pair_rep.shape
        orig_pair_rep.device

        # Prepare batch with target chain information
        batch_with_chains = self._prepare_batch_with_ligand(batch)

        # Get dimensions by computing one feature
        sample_feat = self.upper_right_xt_dist(batch_with_chains)
        n_concat = sample_feat.shape[
            2
        ]  # target sequence length (cross-sequence features are [b, n_orig, n_target, dim])

        # if n_concat == 0:
        #     return orig_pair_rep

        # Compute all features for each quadrant
        # Upper right: [b, n_orig, n_concat, total_dim]
        # upper_right_seq_sep = self.upper_right_seq_sep(batch_with_chains)
        upper_right_xt_dist = self.upper_right_xt_dist(batch_with_chains)
        # upper_right_xsc_dist = self.upper_right_xsc_dist(batch_with_chains)
        # upper_right_optional_ca = self.upper_right_optional_ca(batch_with_chains)
        # upper_right_chain = self.upper_right_chain(batch_with_chains)
        # upper_right_hotspots = self.upper_right_hotspots(batch_with_chains)
        # upper_right_combined = torch.cat(
        #     [
        #         upper_right_seq_sep,
        #         upper_right_xt_dist,
        #         # upper_right_xsc_dist,
        #         # upper_right_optional_ca,
        #         upper_right_chain,
        #         upper_right_hotspots,
        #     ],
        #     dim=-1,
        # )
        upper_right_combined = upper_right_xt_dist

        # Apply linear projection to upper right features
        upper_right_projected = self.ln_out(
            self.linear_out(upper_right_combined)
        )  # [b, n_orig, n_concat, dim_pair_out]

        # Lower left: [b, n_concat, n_orig, dim_pair_out] - transpose of upper right
        # No separate computation needed - just transpose the projected features!
        lower_left_projected = upper_right_projected.transpose(1, 2)

        # Lower right: [b, n_concat, n_concat, total_dim]
        # lower_right_seq_sep = self.lower_right_seq_sep(batch_with_chains)
        lower_right_bond_mask = self.lower_right_bond_mask(batch_with_chains)
        lower_right_bond_order = self.lower_right_bond_order(batch_with_chains)
        lower_right_xt_dist = self.lower_right_xt_dist(batch_with_chains)

        # lower_right_xsc_dist = self.lower_right_xsc_dist(batch_with_chains)
        # lower_right_optional_ca = self.lower_right_optional_ca(batch_with_chains)
        # lower_right_chain = self.lower_right_chain(batch_with_chains)
        # lower_right_hotspots = self.lower_right_hotspots(batch_with_chains)
        lower_right_combined = torch.cat(
            [
                # lower_right_seq_sep,
                lower_right_xt_dist,
                # lower_right_xsc_dist,
                # lower_right_optional_ca,
                # lower_right_chain,
                # lower_right_hotspots,
                lower_right_bond_mask,
                lower_right_bond_order,
            ],
            dim=-1,
        )

        # Apply linear projection to lower right features
        lower_right_projected = self.ln_out_lower_right(
            self.linear_out_lower_right(lower_right_combined)
        )  # [b, n_concat, n_concat, dim_pair_out]

        # Verify dimension consistency with original pair representation
        if self.dim_pair_out != pair_dim:
            raise ValueError(
                f"Configured output dimension {self.dim_pair_out} does not match original pair representation dimension {pair_dim}. Please set dim_pair_out={pair_dim} in config."
            )

        # # Construct extended pair representation as block matrix
        # n_extended = n_orig + n_concat
        # extended_pair_rep = torch.zeros(
        #     b, n_extended, n_extended, pair_dim, device=device
        # )

        # # Upper left: original pair representation
        # extended_pair_rep[:, :n_orig, :n_orig, :] = orig_pair_rep

        # # Upper right: sample-to-target features (projected)
        # extended_pair_rep[:, :n_orig, n_orig:, :] = upper_right_projected

        # # Lower left: target-to-sample features (transposed from upper right)
        # extended_pair_rep[:, n_orig:, :n_orig, :] = lower_left_projected

        # # Lower right: target-to-target features (projected)
        # extended_pair_rep[:, n_orig:, n_orig:, :] = lower_right_projected

        concat_mask = batch[self.mask_key].bool()  # .sum(dim=-1).bool()   # [b, n_concat]

        # [b, n_orig, n_orig, pair_dim], [b, n_concat, n_orig, pair_dim] -> [b, pad_len, n_orig, pair_dim]
        orig_pair_rep = orig_pair_rep * orig_seq_mask[:, :, None, None] * orig_seq_mask[:, None, :, None]
        lower_left_projected = lower_left_projected * concat_mask[:, :, None, None] * orig_seq_mask[:, None, :, None]
        extended_pair_rep_left, extended_mask = concat_padded_tensor(
            a=orig_pair_rep,
            b=lower_left_projected,
            mask_a=orig_seq_mask,
            mask_b=concat_mask,
        )  # [b, pad_len, n_orig, pair_dim], [b, pad_len], pad_len = max(n_i + m_i)

        # [b, n_orig, n_concat, pair_dim], [b, n_concat, n_concat, pair_dim] -> [b, pad_len, n_concat, pair_dim]
        upper_right_projected = upper_right_projected * orig_seq_mask[:, :, None, None] * concat_mask[:, None, :, None]
        lower_right_projected = lower_right_projected * concat_mask[:, :, None, None] * concat_mask[:, None, :, None]
        extended_pair_rep_right, extended_mask = concat_padded_tensor(
            a=upper_right_projected,
            b=lower_right_projected,
            mask_a=orig_seq_mask,
            mask_b=concat_mask,
        )  # [b, pad_len, n_concat, pair_dim], [b, pad_len], pad_len = max(n_i + m_i)

        # [b, n_orig, pad_len, pair_dim], [b, n_concat, pad_len, pair_dim] -> [b, pad_len, pad_len, pair_dim]
        extended_pair_rep, extended_mask = concat_padded_tensor(
            a=extended_pair_rep_left.transpose(1, 2),
            b=extended_pair_rep_right.transpose(1, 2),
            mask_a=orig_seq_mask,
            mask_b=concat_mask,
        )
        extended_pair_rep = extended_pair_rep.transpose(1, 2)  # [b, pad_len, pad_len, pair_dim], [b, pad_len]
        extended_pair_rep = extended_pair_rep * extended_mask[:, :, None, None] * extended_mask[:, None, :, None]

        return extended_pair_rep

    def forward(self, batch, orig_pair_rep, orig_seq_mask):
        """
        Args:
            batch: Input batch dictionary
            orig_pair_rep: Original pair representation [b, n_orig, n_orig, pair_dim]
            orig_seq_mask: Original sequence mask [b, n_orig]

        Returns:
            extended_pair_rep: Extended pair representation [b, n_extended, n_extended, pair_dim]
        """
        # Check if we have the required data
        if self.coords_key not in batch or self.mask_key not in batch:
            #! in the upstream branch this was done via the transform to get zero target features
            #! this is needed as otherwise DDP will complain
            B, N, _, _ = orig_pair_rep.shape
            blank = torch.zeros(self.linear_out_in_dim, device=orig_pair_rep.device)
            if self.enable_ligand:
                orig_pair_rep = (
                    orig_pair_rep
                    + 0
                    * self.ln_out_lower_right(
                        self.linear_out_lower_right(
                            torch.zeros(
                                self.linear_out_lower_right_in_dim,
                                device=orig_pair_rep.device,
                            )
                        )
                    )[None, None, None, :]
                )
            return orig_pair_rep + 0 * self.ln_out(self.linear_out(blank))[None, None, None, :]

        if self.enable_ligand:
            return self.ligand_forward(batch, orig_pair_rep, orig_seq_mask)

        b, n_orig, _, pair_dim = orig_pair_rep.shape
        orig_pair_rep.device

        # Prepare batch with target chain information
        batch_with_chains = self._prepare_batch_with_chains(batch)

        # Get dimensions by computing one feature
        sample_feat = self.upper_right_xt_dist(batch_with_chains)
        n_concat = sample_feat.shape[
            2
        ]  # target sequence length (cross-sequence features are [b, n_orig, n_target, dim])

        # if n_concat == 0:
        #     return orig_pair_rep

        # Compute all features for each quadrant
        # Upper right: [b, n_orig, n_concat, total_dim]
        upper_right_seq_sep = self.upper_right_seq_sep(batch_with_chains)
        upper_right_xt_dist = self.upper_right_xt_dist(batch_with_chains)
        # upper_right_xsc_dist = self.upper_right_xsc_dist(batch_with_chains)
        # upper_right_optional_ca = self.upper_right_optional_ca(batch_with_chains)
        upper_right_chain = self.upper_right_chain(batch_with_chains)
        upper_right_hotspots = self.upper_right_hotspots(batch_with_chains)
        upper_right_combined = torch.cat(
            [
                upper_right_seq_sep,
                upper_right_xt_dist,
                # upper_right_xsc_dist,
                # upper_right_optional_ca,
                upper_right_chain,
                upper_right_hotspots,
            ],
            dim=-1,
        )

        # Apply linear projection to upper right features
        upper_right_projected = self.ln_out(
            self.linear_out(upper_right_combined)
        )  # [b, n_orig, n_concat, dim_pair_out]

        # Lower left: [b, n_concat, n_orig, dim_pair_out] - transpose of upper right
        # No separate computation needed - just transpose the projected features!
        lower_left_projected = upper_right_projected.transpose(1, 2)

        # Lower right: [b, n_concat, n_concat, total_dim]
        lower_right_seq_sep = self.lower_right_seq_sep(batch_with_chains)
        lower_right_xt_dist = self.lower_right_xt_dist(batch_with_chains)
        # lower_right_xsc_dist = self.lower_right_xsc_dist(batch_with_chains)
        # lower_right_optional_ca = self.lower_right_optional_ca(batch_with_chains)
        lower_right_chain = self.lower_right_chain(batch_with_chains)
        lower_right_hotspots = self.lower_right_hotspots(batch_with_chains)
        lower_right_combined = torch.cat(
            [
                lower_right_seq_sep,
                lower_right_xt_dist,
                # lower_right_xsc_dist,
                # lower_right_optional_ca,
                lower_right_chain,
                lower_right_hotspots,
            ],
            dim=-1,
        )

        # Apply linear projection to lower right features
        lower_right_projected = self.ln_out(
            self.linear_out(lower_right_combined)
        )  # [b, n_concat, n_concat, dim_pair_out]

        # Verify dimension consistency with original pair representation
        if self.dim_pair_out != pair_dim:
            raise ValueError(
                f"Configured output dimension {self.dim_pair_out} does not match original pair representation dimension {pair_dim}. Please set dim_pair_out={pair_dim} in config."
            )

        # # Construct extended pair representation as block matrix
        # n_extended = n_orig + n_concat
        # extended_pair_rep = torch.zeros(
        #     b, n_extended, n_extended, pair_dim, device=device
        # )

        # # Upper left: original pair representation
        # extended_pair_rep[:, :n_orig, :n_orig, :] = orig_pair_rep

        # # Upper right: sample-to-target features (projected)
        # extended_pair_rep[:, :n_orig, n_orig:, :] = upper_right_projected

        # # Lower left: target-to-sample features (transposed from upper right)
        # extended_pair_rep[:, n_orig:, :n_orig, :] = lower_left_projected

        # # Lower right: target-to-target features (projected)
        # extended_pair_rep[:, n_orig:, n_orig:, :] = lower_right_projected

        concat_mask = batch[self.mask_key].sum(dim=-1).bool()  # [b, n_concat]
        # [b, n_orig, n_orig, pair_dim], [b, n_concat, n_orig, pair_dim] -> [b, pad_len, n_orig, pair_dim]
        orig_pair_rep = orig_pair_rep * orig_seq_mask[:, :, None, None] * orig_seq_mask[:, None, :, None]
        lower_left_projected = lower_left_projected * concat_mask[:, :, None, None] * orig_seq_mask[:, None, :, None]
        extended_pair_rep_left, extended_mask = concat_padded_tensor(
            a=orig_pair_rep,
            b=lower_left_projected,
            mask_a=orig_seq_mask,
            mask_b=concat_mask,
        )  # [b, pad_len, n_orig, pair_dim], [b, pad_len], pad_len = max(n_i + m_i)

        # [b, n_orig, n_concat, pair_dim], [b, n_concat, n_concat, pair_dim] -> [b, pad_len, n_concat, pair_dim]
        upper_right_projected = upper_right_projected * orig_seq_mask[:, :, None, None] * concat_mask[:, None, :, None]
        lower_right_projected = lower_right_projected * concat_mask[:, :, None, None] * concat_mask[:, None, :, None]
        extended_pair_rep_right, extended_mask = concat_padded_tensor(
            a=upper_right_projected,
            b=lower_right_projected,
            mask_a=orig_seq_mask,
            mask_b=concat_mask,
        )  # [b, pad_len, n_concat, pair_dim], [b, pad_len], pad_len = max(n_i + m_i)

        # [b, n_orig, pad_len, pair_dim], [b, n_concat, pad_len, pair_dim] -> [b, pad_len, pad_len, pair_dim]
        extended_pair_rep, extended_mask = concat_padded_tensor(
            a=extended_pair_rep_left.transpose(1, 2),
            b=extended_pair_rep_right.transpose(1, 2),
            mask_a=orig_seq_mask,
            mask_b=concat_mask,
        )
        extended_pair_rep = extended_pair_rep.transpose(1, 2)  # [b, pad_len, pad_len, pair_dim], [b, pad_len]
        extended_pair_rep = extended_pair_rep * extended_mask[:, :, None, None] * extended_mask[:, None, :, None]

        return extended_pair_rep

    def _prepare_batch_with_chains(self, batch):
        """Prepare batch with target chain information for feature computation."""
        batch_copy = dict(batch)

        b, n_orig = batch["x_t"]["bb_ca"].shape[:2]
        batch_copy["x_t_coord"] = torch.zeros(b, n_orig, 37, 3, device=batch["x_t"]["bb_ca"].device)
        batch_copy["x_t_coord_mask"] = torch.zeros(b, n_orig, 37, device=batch["x_t"]["bb_ca"].device)
        batch_copy["x_t_coord"][:, :, 1, :] = batch["x_t"]["bb_ca"]
        if "mask" in batch:  # for inference
            batch_copy["x_t_coord_mask"][:, :, 1] = batch["mask"]
        else:
            batch_copy["x_t_coord_mask"][:, :, 1] = batch["coord_mask"][:, :, 1]

        # Add target chains if not present
        if "target_chains" not in batch_copy and "seq_target" in batch_copy:
            b, n_target = batch_copy["seq_target"].shape
            device = batch_copy["seq_target"].device
            # Assign target residues to a different chain ID than the main sequence
            if "chains" in batch_copy:
                max_chain = batch_copy["chains"].max().item()
                batch_copy["target_chains"] = torch.full((b, n_target), max_chain + 1, device=device)
            else:
                batch_copy["target_chains"] = torch.ones((b, n_target), device=device)

        # Add target pdb indices if not present (use sequential numbering)
        if "target_pdb_idx" not in batch_copy and "seq_target" in batch_copy:
            b, n_target = batch_copy["seq_target"].shape
            device = batch_copy["seq_target"].device
            # Continue numbering from main sequence
            if "residue_pdb_idx" in batch_copy:
                max_idx = batch_copy["residue_pdb_idx"].max().item()
                target_indices = torch.arange(max_idx + 1, max_idx + 1 + n_target, device=device)
                batch_copy["target_pdb_idx"] = target_indices.unsqueeze(0).expand(b, -1)
            else:
                batch_copy["target_pdb_idx"] = torch.arange(n_target, device=device).unsqueeze(0).expand(b, -1)

        return batch_copy

    def _prepare_batch_with_ligand(self, batch):
        """Prepare batch with target ligand information for feature computation."""
        batch_copy = dict(batch)

        b, n_orig = batch["x_t"]["bb_ca"].shape[:2]
        batch_copy["x_t_coord"] = torch.zeros(b, n_orig, 37, 3, device=batch["x_t"]["bb_ca"].device)
        batch_copy["x_t_coord_mask"] = torch.zeros(b, n_orig, 37, device=batch["x_t"]["bb_ca"].device)
        batch_copy["x_t_coord"][:, :, 1, :] = batch["x_t"]["bb_ca"]
        if "mask" in batch:  # for inference
            batch_copy["x_t_coord_mask"][:, :, 1] = batch["mask"]
        else:
            batch_copy["x_t_coord_mask"][:, :, 1] = batch["coord_mask"][:, :, 1]

        # Add target chains if not present
        # if "target_chains" not in batch_copy and "seq_target" in batch_copy:
        #     b, n_target = batch_copy["seq_target"].shape
        #     device = batch_copy["seq_target"].device
        #     # Assign target residues to a different chain ID than the main sequence
        #     if "chains" in batch_copy:
        #         max_chain = batch_copy["chains"].max().item()
        #         batch_copy["target_chains"] = torch.full(
        #             (b, n_target), max_chain + 1, device=device
        #         )
        #     else:
        #         batch_copy["target_chains"] = torch.ones((b, n_target), device=device)

        # Add target pdb indices if not present (use sequential numbering)
        # if "target_pdb_idx" not in batch_copy and "seq_target" in batch_copy:
        #     b, n_target = batch_copy["seq_target"].shape
        #     device = batch_copy["seq_target"].device
        #     # Continue numbering from main sequence
        #     if "residue_pdb_idx" in batch_copy:
        #         max_idx = batch_copy["residue_pdb_idx"].max().item()
        #         target_indices = torch.arange(
        #             max_idx + 1, max_idx + 1 + n_target, device=device
        #         )
        #         batch_copy["target_pdb_idx"] = target_indices.unsqueeze(0).expand(b, -1)
        #     else:
        #         batch_copy["target_pdb_idx"] = (
        #             torch.arange(n_target, device=device).unsqueeze(0).expand(b, -1)
        #         )

        return batch_copy


class CrossSequenceRelativeSequenceSeparationPairFeat(Feature):
    """Computes relative sequence separation between two sequences and returns rectangular feature of shape [b, n1, n2, seq_sep_dim]."""

    def __init__(
        self,
        seq1_key="residue_type",
        seq2_key="residue_type_2",
        idx1_key="residue_pdb_idx",
        idx2_key="residue_pdb_idx_2",
        seq_sep_dim=21,
        **kwargs,
    ):
        super().__init__(dim=seq_sep_dim)
        self.seq1_key = seq1_key
        self.seq2_key = seq2_key
        self.idx1_key = idx1_key
        self.idx2_key = idx2_key
        self.seq_sep_dim = seq_sep_dim

    def forward(self, batch):
        device = self.extract_device(batch)

        # Get sequence lengths
        if self.seq1_key in batch and self.seq2_key in batch:
            n1 = batch[self.seq1_key].shape[1]
            n2 = batch[self.seq2_key].shape[1]
            b = batch[self.seq1_key].shape[0]
        else:
            # Fallback to extracting from batch
            b, n1 = self.extract_bs_and_n(batch)
            if self.seq2_key in batch:
                n2 = batch[self.seq2_key].shape[1]
            else:
                n2 = n1

        # Get indices if available, otherwise use sequential
        if self.idx1_key in batch:
            indices1 = batch[self.idx1_key].float()  # [b, n1]
        else:
            indices1 = torch.arange(n1, device=device).float().unsqueeze(0).expand(b, -1)

        if self.idx2_key in batch:
            indices2 = batch[self.idx2_key].float()  # [b, n2]
        else:
            indices2 = torch.arange(n2, device=device).float().unsqueeze(0).expand(b, -1)

        # Compute cross-sequence separations: [b, n1, n2]
        seq_sep = indices1[:, :, None] - indices2[:, None, :]  # [b, n1, n2]

        # Bin the separations
        low = -(self.seq_sep_dim / 2.0 - 1)
        high = self.seq_sep_dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.seq_sep_dim - 1, device=device)

        if self.idx1_key == self.idx2_key:
            # Only compute sequence separation for the same part
            return bin_and_one_hot(seq_sep, bin_limits)  # [b, n1, n2, seq_sep_dim]
        else:
            return torch.zeros(b, n1, n2, self.seq_sep_dim, device=device)


class CrossSequenceChainIndexPairFeat(Feature):
    """Computes chain index features between two sequences and returns rectangular feature of shape [b, n1, n2, 1]."""

    def __init__(self, chain1_key="chains", chain2_key="chains_2", **kwargs):
        super().__init__(dim=1)
        self.chain1_key = chain1_key
        self.chain2_key = chain2_key

    def forward(self, batch):
        device = self.extract_device(batch)

        # Get chain indices
        if self.chain1_key in batch:
            chains1 = batch[self.chain1_key]  # [b, n1]
        else:
            b, n1 = self.extract_bs_and_n(batch)
            chains1 = torch.zeros(b, n1, device=device)

        if self.chain2_key in batch:
            chains2 = batch[self.chain2_key]  # [b, n2]
        else:
            # Assign different chain ID for sequence 2
            b, n1 = chains1.shape
            n2 = n1  # fallback assumption
            chains2 = torch.full((b, n2), chains1.max().item() + 1, device=device)

        # Compute pairwise chain features: [b, n1, n2, 1]
        chain_pairs = (chains1[:, :, None] != chains2[:, None, :]).float().unsqueeze(-1)
        # chain_pairs = torch.einsum("bi,bj->bij", chains1, chains2).unsqueeze(-1)

        return chain_pairs


class CrossSequenceHotspotMaskPairFeat(Feature):
    """Computes hotspot mask features between two sequences and returns rectangular feature of shape [b, n1, n2, 1]."""

    def __init__(
        self,
        hotspot_mask1_key="hotspot_mask",
        hotspot_mask2_key="hotspot_mask_2",
        **kwargs,
    ):
        super().__init__(dim=1)
        self.hotspot_mask1_key = hotspot_mask1_key
        self.hotspot_mask2_key = hotspot_mask2_key

    def forward(self, batch):
        device = self.extract_device(batch)

        # Get chain indices
        if self.hotspot_mask1_key in batch:
            hotspot_mask1 = batch[self.hotspot_mask1_key]  # [b, n1]
        else:
            b, n1 = self.extract_bs_and_n(batch)
            hotspot_mask1 = torch.zeros(b, n1, device=device)

        if self.hotspot_mask2_key in batch:
            hotspot_mask2 = batch[self.hotspot_mask2_key]  # [b, n2]
        else:
            # Assign different chain ID for sequence 2
            b, n1 = hotspot_mask1.shape
            n2 = n1  # fallback assumption
            hotspot_mask2 = torch.zeros(b, n2, device=device)

        # Compute pairwise chain features: [b, n1, n2, 1]
        hotspot_pairs = (hotspot_mask1[:, :, None] + hotspot_mask2[:, None, :]).float().unsqueeze(-1)
        # chain_pairs = torch.einsum("bi,bj->bij", chains1, chains2).unsqueeze(-1)

        return hotspot_pairs


class CrossSequenceXscBBCAPairwiseDistancesPairFeat(Feature):
    """Computes cross-sequence x_sc backbone CA pairwise distances."""

    def __init__(
        self,
        coords1_key="coords_nm",
        coords2_key="x_target",
        mode_key="x_sc",
        x_sc_pair_dist_dim=30,
        x_sc_pair_dist_min=0.1,
        x_sc_pair_dist_max=3.0,
        **kwargs,
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.coords1_key = coords1_key
        self.coords2_key = coords2_key
        self.mode_key = mode_key
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max
        self._has_logged = False

    def forward(self, batch):
        # Get coordinate dimensions first to ensure consistent shape
        if self.coords1_key in batch:
            if len(batch[self.coords1_key].shape) == 4:  # [b, n, 37, 3] format
                b, n1 = batch[self.coords1_key].shape[:2]
            else:  # [b, n, 3] format
                b, n1 = batch[self.coords1_key].shape[:2]
        else:
            # Fallback: try to get length from other keys or default
            b, n1 = self.extract_bs_and_n(batch)

        if self.coords2_key in batch:
            if len(batch[self.coords2_key].shape) == 4:  # [b, n, 37, 3] format
                b, n2 = batch[self.coords2_key].shape[:2]
            else:  # [b, n, 3] format
                b, n2 = batch[self.coords2_key].shape[:2]
        else:
            # Fallback: use same as first sequence
            n2 = n1

        device = self.extract_device(batch)

        # Check if self-conditioning data is available
        if self.mode_key in batch and "bb_ca" in batch[self.mode_key]:
            sc_coords = batch[self.mode_key]["bb_ca"]  # [b, n_orig, 3]

            # Get target coordinates if available
            if self.coords2_key in batch:
                if len(batch[self.coords2_key].shape) == 4:  # [b, n, 37, 3] format
                    target_coords = batch[self.coords2_key][:, :, 1, :]  # [b, n2, 3] - CA atoms
                else:  # [b, n, 3] format
                    target_coords = batch[self.coords2_key]  # [b, n2, 3]

                # For cross-sequence, we need to match dimensions properly
                # If coords1_key refers to target data, extract the right portion of sc_coords
                if self.coords1_key == "x_target" and sc_coords.shape[1] != n1:
                    # sc_coords is from main sequence but we want target portion
                    # In this case, we should use target coordinates directly
                    if len(batch[self.coords1_key].shape) == 4:
                        coords1 = batch[self.coords1_key][:, :, 1, :]  # [b, n1, 3] - CA atoms
                    else:
                        coords1 = batch[self.coords1_key]  # [b, n1, 3]
                else:
                    coords1 = sc_coords[:, :n1, :]  # [b, n1, 3]

                # Compute cross-sequence distances: [b, n1, n2]
                cross_dists = torch.norm(coords1[:, :, None, :] - target_coords[:, None, :, :], dim=-1)

                # Bin the distances
                bin_limits = torch.linspace(self.min_dist, self.max_dist, self.dim - 1, device=device)
                return bin_and_one_hot(cross_dists, bin_limits)  # [b, n1, n2, dim]
            else:
                # No target coordinates, return zeros with correct dimensions
                if not self._has_logged:
                    logger.warning(f"No {self.coords2_key} found for CrossSequenceXscBBCAPairwiseDistancesPairFeat")
                    self._has_logged = True
                return torch.zeros(b, n1, n2, self.dim, device=device)
        else:
            # No self-conditioning data, return zeros with correct dimensions
            if not self._has_logged:
                logger.warning(f"No {self.mode_key} data found for CrossSequenceXscBBCAPairwiseDistancesPairFeat")
                self._has_logged = True
            return torch.zeros(b, n1, n2, self.dim, device=device)


class CrossSequenceOptionalCaPairDistancesPairFeat(Feature):
    """Computes cross-sequence optional CA pairwise distances."""

    def __init__(self, coords1_key="coords_nm", coords2_key="x_target", **kwargs):
        super().__init__(dim=30)  # Standard CA pair distance dimension
        self.coords1_key = coords1_key
        self.coords2_key = coords2_key
        self.min_dist = 0.1
        self.max_dist = 3.0
        self._has_logged = False

    def forward(self, batch):
        # Get coordinate dimensions first to ensure consistent shape
        if self.coords1_key in batch:
            if len(batch[self.coords1_key].shape) == 4:  # [b, n, 37, 3] format
                b, n1 = batch[self.coords1_key].shape[:2]
            else:  # [b, n, 3] format
                b, n1 = batch[self.coords1_key].shape[:2]
        else:
            # Fallback: try to get length from other keys or default
            b, n1 = self.extract_bs_and_n(batch)

        if self.coords2_key in batch:
            if len(batch[self.coords2_key].shape) == 4:  # [b, n, 37, 3] format
                b, n2 = batch[self.coords2_key].shape[:2]
            else:  # [b, n, 3] format
                b, n2 = batch[self.coords2_key].shape[:2]
        else:
            # Fallback: use same as first sequence
            n2 = n1

        device = self.extract_device(batch)

        # Check if optional CA coordinates feature should be used
        if batch.get("use_ca_coors_nm_feature", False):
            # Get CA coordinates from coords_nm
            if self.coords1_key in batch and self.coords2_key in batch:
                if len(batch[self.coords1_key].shape) == 4:  # [b, n, 37, 3] format
                    ca_coords1 = batch[self.coords1_key][:, :, 1, :]  # [b, n1, 3] - CA atoms
                else:  # [b, n, 3] format
                    ca_coords1 = batch[self.coords1_key]  # [b, n1, 3]

                if len(batch[self.coords2_key].shape) == 4:  # [b, n, 37, 3] format
                    ca_coords2 = batch[self.coords2_key][:, :, 1, :]  # [b, n2, 3] - CA atoms
                else:  # [b, n, 3] format
                    ca_coords2 = batch[self.coords2_key]  # [b, n2, 3]

                # Compute cross-sequence distances: [b, n1, n2]
                cross_dists = torch.norm(ca_coords1[:, :, None, :] - ca_coords2[:, None, :, :], dim=-1)

                # Bin the distances
                bin_limits = torch.linspace(self.min_dist, self.max_dist, self.dim - 1, device=device)
                return bin_and_one_hot(cross_dists, bin_limits)  # [b, n1, n2, dim]
            else:
                if not self._has_logged:
                    logger.warning("Missing coordinates for CrossSequenceOptionalCaPairDistancesPairFeat")
                    self._has_logged = True
                return torch.zeros(b, n1, n2, self.dim, device=device)
        else:
            # Feature disabled, return zeros with correct dimensions
            if not self._has_logged:
                logger.warning("use_ca_coors_nm_feature disabled for CrossSequenceOptionalCaPairDistancesPairFeat")
                self._has_logged = True
            return torch.zeros(b, n1, n2, self.dim, device=device)
