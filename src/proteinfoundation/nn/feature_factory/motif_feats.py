import torch
from loguru import logger
from openfold.np.residue_constants import atom_types
from torch.nn.utils.rnn import pad_sequence

from proteinfoundation.nn.feature_factory.base_feature import Feature
from proteinfoundation.nn.feature_factory.pair_feats import (
    BackbonePairDistancesNanometerPairFeat,
    CrossSequenceBackbonePairDistancesPairFeat,
)
from proteinfoundation.nn.feature_factory.seq_feats import (
    Atom37NanometersCoorsSeqFeat,
    BackboneTorsionAnglesSeqFeat,
    OpenfoldSideChainAnglesSeqFeat,
    ResidueTypeSeqFeat,
)


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


class BulkAllAtomXmotifSeqFeat(Feature):
    """Computes feature from x_motif coordinates, seq feature of shape [b, n, 3] or [b, na, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)  # dim will be fixed
        self.const_coors_abs = Atom37NanometersCoorsSeqFeat(rel=False)
        self.const_coors_rel = Atom37NanometersCoorsSeqFeat(rel=True)
        self.const_seq = ResidueTypeSeqFeat()
        self.const_sc_angles = OpenfoldSideChainAnglesSeqFeat()
        self.const_torsion_angles = BackboneTorsionAnglesSeqFeat()

        dim = (
            self.const_coors_abs.dim
            + self.const_coors_rel.dim
            + self.const_seq.dim
            + self.const_sc_angles.dim
            + self.const_torsion_angles.dim
            + 37
        )
        self.dim = dim

    def forward(self, batch):
        if "x_motif" in batch:
            # Coordinates features
            batch_coors = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_coors_abs = self.const_coors_abs(batch_coors)  # [b, n, some #]
            feat_coors_rel = self.const_coors_rel(batch_coors)  # [b, n, some #]

            # Sequence features
            seq_mask = batch["motif_mask"].sum(-1).bool()  # [b, n]
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {
                    "residue_type": seq_mask,
                },
            }
            feat_seq = self.const_seq(batch_seq)  # [b, n, some #]

            # Side chain angles features
            batch_sc_angles = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "coords": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_sc_angles = self.const_sc_angles(batch_sc_angles)  # [b, n, some #]
            if "residue_pdb_idx" in batch:
                idx = batch["residue_pdb_idx"]  # [b, n]
            else:
                self.assert_defaults_allowed(batch, "Relative sequence separation pair")
                b, n = self.extract_bs_and_n(batch)
                device = self.extract_device(batch)
                idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(device)  # [b, n]
            # Torsion angle features
            batch_torsion_angles = {
                "coords": batch["x_motif"],  # [b, n, 37, 3]
                "residue_pdb_idx": idx,  # [b, n]
            }
            feat_torsion_angles = self.const_torsion_angles(batch_torsion_angles)  # [b, n, some #]

            # motif mask
            motif_mask = batch["motif_mask"] * 1.0  # [b, n, 37]

            # concatenate all features
            feat = torch.cat(
                [feat_coors_abs, feat_coors_rel, feat_seq, feat_sc_angles, feat_torsion_angles, motif_mask], dim=-1
            )  # [b, n, some # added up]
            feat = feat * seq_mask[..., None]  # [b, n, some # added up]

            return feat

        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, self.dim, device=device)
