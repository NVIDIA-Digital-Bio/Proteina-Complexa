import gzip
import os
import random

import torch
from loguru import logger
from torch_scatter import scatter_mean

from proteinfoundation.nn.feature_factory.base_feature import Feature
from proteinfoundation.nn.feature_factory.feature_utils import (
    get_index_embedding,
    get_time_embedding,
    indices_force_start_w_one,
    sinusoidal_encoding,
)
from proteinfoundation.utils.fold_utils import extract_cath_code_by_level


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
            inds = torch.arange(1, n + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1)  # [b, n]
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
            inds = torch.arange(1, n + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1)  # [b, n]
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
