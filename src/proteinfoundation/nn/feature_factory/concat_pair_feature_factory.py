import torch
from loguru import logger

from proteinfoundation.nn.feature_factory.ligand_feats import BondMaskPairFeat, BondOrderPairFeat
from proteinfoundation.nn.feature_factory.pair_feats import (
    CrossSequenceBackbonePairDistancesPairFeat,
    CrossSequenceChainIndexPairFeat,
    CrossSequenceHotspotMaskPairFeat,
    CrossSequenceOptionalCaPairDistancesPairFeat,
    CrossSequenceRelativeSequenceSeparationPairFeat,
    CrossSequenceXscBBCAPairwiseDistancesPairFeat,
)
from proteinfoundation.nn.feature_factory.target_feats import (
    CrossSequenceBackboneAtomPairDistancesPairFeat,
    TargetAtomPairwiseDistancesPairFeat,
)
from proteinfoundation.utils.tensor_utils import concat_padded_tensor


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
            self.upper_right_xt_dist = CrossSequenceBackboneAtomPairDistancesPairFeat(
                coords1_key="x_t_coord",
                mask1_key="x_t_coord_mask",
                coords2_key="x_target",
                mask2_key="target_mask",
                **kwargs,
            )

            self.lower_right_bond_mask = BondMaskPairFeat(key="target_bond_mask", **kwargs)
            self.lower_right_bond_order = BondOrderPairFeat(key="target_bond_order", **kwargs)

            self.lower_right_xt_dist = TargetAtomPairwiseDistancesPairFeat(
                **kwargs,
            )

            # Calculate total input dimension for cross-sequence features
            total_cross_seq_dim = self.upper_right_xt_dist.get_dim()
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

        upper_right_xt_dist = self.upper_right_xt_dist(batch_with_chains)
        upper_right_combined = upper_right_xt_dist

        # Apply linear projection to upper right features
        upper_right_projected = self.ln_out(
            self.linear_out(upper_right_combined)
        )  # [b, n_orig, n_concat, dim_pair_out]

        # Lower left: [b, n_concat, n_orig, dim_pair_out] - transpose of upper right
        # No separate computation needed - just transpose the projected features!
        lower_left_projected = upper_right_projected.transpose(1, 2)

        # Lower right: [b, n_concat, n_concat, total_dim]
        lower_right_bond_mask = self.lower_right_bond_mask(batch_with_chains)
        lower_right_bond_order = self.lower_right_bond_order(batch_with_chains)
        lower_right_xt_dist = self.lower_right_xt_dist(batch_with_chains)

        lower_right_combined = torch.cat(
            [
                lower_right_xt_dist,
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
        upper_right_chain = self.upper_right_chain(batch_with_chains)
        upper_right_hotspots = self.upper_right_hotspots(batch_with_chains)
        upper_right_combined = torch.cat(
            [
                upper_right_seq_sep,
                upper_right_xt_dist,
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
        lower_right_chain = self.lower_right_chain(batch_with_chains)
        lower_right_hotspots = self.lower_right_hotspots(batch_with_chains)
        lower_right_combined = torch.cat(
            [
                lower_right_seq_sep,
                lower_right_xt_dist,
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

        return batch_copy
