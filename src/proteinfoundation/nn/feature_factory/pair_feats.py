import einops
import torch
from loguru import logger

from proteinfoundation.nn.feature_factory.base_feature import Feature
from proteinfoundation.nn.feature_factory.feature_utils import bin_and_one_hot, bin_pairwise_distances
from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle


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
            inds = torch.arange(1, n + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1)  # [b, n]

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
