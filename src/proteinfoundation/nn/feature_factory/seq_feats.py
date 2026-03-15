import einops
import torch
from loguru import logger
from openfold.data import data_transforms
from torch.nn import functional as F

from proteinfoundation.nn.feature_factory.base_feature import Feature
from proteinfoundation.nn.feature_factory.feature_utils import bin_and_one_hot
from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle


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
            idx = torch.arange(1, n + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1)  # [b, n]
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
            idx = torch.arange(1, n + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1)  # [b, n]
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
