import torch
from loguru import logger
from openfold.np.residue_constants import atom_types
from torch.nn.utils.rnn import pad_sequence

from proteinfoundation.nn.feature_factory.base_feature import Feature
from proteinfoundation.nn.feature_factory.feature_utils import bin_and_one_hot, bin_pairwise_distances
from proteinfoundation.nn.feature_factory.pair_feats import BackbonePairDistancesNanometerPairFeat
from proteinfoundation.nn.feature_factory.seq_cond_feats import HotspotMaskSeqFeat
from proteinfoundation.nn.feature_factory.seq_feats import (
    Atom37NanometersCoorsSeqFeat,
    BackboneTorsionAnglesSeqFeat,
    OpenfoldSideChainAnglesSeqFeat,
    ResidueTypeSeqFeat,
)


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
