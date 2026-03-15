import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

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


class ConcatPairFeaturesFactory(torch.nn.Module):
    """Factory for creating extended pair representations using cross-sequence features.

    Args:
        enable_target: Whether to enable target cross-sequence features
        enable_motif: Whether to enable motif cross-sequence features
        enable_ligand: Whether to enable ligand cross-sequence features
        dim_pair_out: The dimension of the output pair features
        use_ln_out: Whether to use layer normalization
        **kwargs: Additional keyword arguments

    Note: This only works with motif, (ligand, motif) or (target, motif)
    """

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
        if enable_ligand and enable_target:
            raise ValueError("Cannot enable both ligand and target cross-sequence features for now")
        if enable_ligand:
            self.upper_right_xt_dist_ligand = CrossSequenceBackboneAtomPairDistancesPairFeat(
                coords1_key="x_t_coord",
                mask1_key="x_t_coord_mask",
                coords2_key="x_target",
                mask2_key="target_mask",
                **kwargs,
            )
            self.lower_right_bond_mask_ligand = BondMaskPairFeat(key="target_bond_mask", **kwargs)
            self.lower_right_bond_order_ligand = BondOrderPairFeat(key="target_bond_order", **kwargs)

            self.lower_right_xt_dist_ligand = TargetAtomPairwiseDistancesPairFeat(
                **kwargs,
            )

            # Calculate total input dimension for cross-sequence features
            total_cross_seq_dim_ligand = self.upper_right_xt_dist_ligand.get_dim()
            total_lower_right_dim_ligand = (
                self.lower_right_xt_dist_ligand.get_dim()
                + self.lower_right_bond_mask_ligand.get_dim()
                + self.lower_right_bond_order_ligand.get_dim()
            )
            # Create projection layers to match original pair representation dimension
            self.linear_out_ligand = torch.nn.Linear(total_cross_seq_dim_ligand, dim_pair_out, bias=False)
            self.ln_out_ligand = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()

            self.linear_out_lower_right_ligand = torch.nn.Linear(total_lower_right_dim_ligand, dim_pair_out, bias=False)
            self.ln_out_lower_right_ligand = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()

            self.linear_out_in_dim_ligand = total_cross_seq_dim_ligand
            self.linear_out_lower_right_in_dim_ligand = total_lower_right_dim_ligand
            self.coords_key = "x_target"
            self.mask_key = "target_mask"
            logger.info(
                f"Enabled target-to-sample cross-sequence pair features: input feat dim {total_cross_seq_dim_ligand} -> output feat dim {dim_pair_out}"
            )

        elif enable_target:
            # Target-based features
            # Upper right: Sample-to-target features (using generic cross-sequence features)
            self.upper_right_seq_sep_target = CrossSequenceRelativeSequenceSeparationPairFeat(
                seq1_key="residue_type",
                seq2_key="seq_target",
                idx1_key="residue_pdb_idx",
                idx2_key="target_pdb_idx",
                **kwargs,
            )
            self.upper_right_xt_dist_target = CrossSequenceBackbonePairDistancesPairFeat(
                coords1_key="x_t_coord",
                mask1_key="x_t_coord_mask",
                coords2_key="x_target",
                mask2_key="target_mask",
                **kwargs,
            )

            self.upper_right_chain_target = CrossSequenceChainIndexPairFeat(
                chain1_key="chains", chain2_key="target_chains", **kwargs
            )
            self.upper_right_hotspots_target = CrossSequenceHotspotMaskPairFeat(
                hotspot_mask1_key="hotspot_mask",
                hotspot_mask2_key="target_hotspot_mask",
                **kwargs,
            )

            # Lower left: NOT COMPUTED - just transpose of upper right for efficiency!

            # Lower right: Target-to-target features (using generic cross-sequence features)
            self.lower_right_seq_sep_target = CrossSequenceRelativeSequenceSeparationPairFeat(
                seq1_key="seq_target",
                seq2_key="seq_target",
                idx1_key="target_pdb_idx",
                idx2_key="target_pdb_idx",
                **kwargs,
            )
            self.lower_right_xt_dist_target = CrossSequenceBackbonePairDistancesPairFeat(
                coords1_key="x_target",
                mask1_key="target_mask",
                coords2_key="x_target",
                mask2_key="target_mask",
                **kwargs,
            )

            self.lower_right_chain_target = CrossSequenceChainIndexPairFeat(
                chain1_key="target_chains", chain2_key="target_chains", **kwargs
            )
            self.lower_right_hotspots_target = CrossSequenceHotspotMaskPairFeat(
                hotspot_mask1_key="target_hotspot_mask",
                hotspot_mask2_key="target_hotspot_mask",
                **kwargs,
            )

            # Calculate total input dimension for cross-sequence features
            total_cross_seq_dim_target = (
                self.upper_right_seq_sep_target.get_dim()
                + self.upper_right_xt_dist_target.get_dim()
                + self.upper_right_chain_target.get_dim()
                + self.upper_right_hotspots_target.get_dim()
            )

            total_lower_right_dim_target = (
                self.lower_right_seq_sep_target.get_dim()
                + self.lower_right_xt_dist_target.get_dim()
                + self.lower_right_chain_target.get_dim()
                + self.lower_right_hotspots_target.get_dim()
            )
            # Create projection layers to match original pair representation dimension
            self.linear_out_in_dim_target = total_cross_seq_dim_target
            self.linear_out_lower_right_in_dim_target = total_lower_right_dim_target
            self.linear_out_target = torch.nn.Linear(total_cross_seq_dim_target, dim_pair_out, bias=False)
            self.ln_out_target = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()
            self.linear_out_lower_right_target = torch.nn.Linear(total_lower_right_dim_target, dim_pair_out, bias=False)
            self.ln_out_lower_right_target = torch.nn.LayerNorm(dim_pair_out) if use_ln_out else torch.nn.Identity()

            self.coords_key = "x_target"
            self.mask_key = "target_mask"
            logger.info(
                f"Enabled target-to-sample cross-sequence pair features: {total_cross_seq_dim_target} -> {dim_pair_out}"
            )

        if enable_motif:
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
            # self.upper_right_chain_motif = CrossSequenceChainIndexPairFeat(
            #     chain1_key="chains", chain2_key="motif_chains", **kwargs
            # )

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
            # self.lower_right_chain_motif = CrossSequenceChainIndexPairFeat(
            #     chain1_key="motif_chains", chain2_key="motif_chains", **kwargs
            # )

            # Calculate total input dimension for cross-sequence features
            total_cross_seq_dim = (
                self.upper_right_seq_sep.get_dim()
                + self.upper_right_xt_dist.get_dim()
                + self.upper_right_xsc_dist.get_dim()
                + self.upper_right_optional_ca.get_dim()
                # + self.upper_right_chain.get_dim() TODO: enable again
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

    def ligand_forward(self, batch, orig_pair_rep, orig_seq_mask):
        b, n_orig, _, pair_dim = orig_pair_rep.shape
        device = orig_pair_rep.device

        # Prepare batch with target chain information
        batch_with_chains = self._prepare_batch_with_ligand(batch)

        # Get dimensions by computing one feature
        sample_feat = self.upper_right_xt_dist_ligand(batch_with_chains)
        n_concat = sample_feat.shape[2]
        upper_right_xt_dist = self.upper_right_xt_dist_ligand(batch_with_chains)
        upper_right_combined = upper_right_xt_dist

        # Apply linear projection to upper right features
        upper_right_projected = self.ln_out_ligand(
            self.linear_out_ligand(upper_right_combined)
        )  # [b, n_orig, n_concat, dim_pair_out]

        # Lower left: [b, n_concat, n_orig, dim_pair_out] - transpose of upper right
        # No separate computation needed - just transpose the projected features!
        lower_left_projected = upper_right_projected.transpose(1, 2)

        # Lower right: [b, n_concat, n_concat, total_dim]
        lower_right_bond_mask = self.lower_right_bond_mask_ligand(batch_with_chains)
        lower_right_bond_order = self.lower_right_bond_order_ligand(batch_with_chains)
        lower_right_xt_dist = self.lower_right_xt_dist_ligand(batch_with_chains)

        lower_right_combined = torch.cat(
            [
                lower_right_xt_dist,
                lower_right_bond_mask,
                lower_right_bond_order,
            ],
            dim=-1,
        )

        # Apply linear projection to lower right features
        lower_right_projected = self.ln_out_lower_right_ligand(
            self.linear_out_lower_right_ligand(lower_right_combined)
        )  # [b, n_concat, n_concat, dim_pair_out]

        # Verify dimension consistency with original pair representation
        if self.dim_pair_out != pair_dim:
            raise ValueError(
                f"Configured output dimension {self.dim_pair_out} does not match original pair representation dimension {pair_dim}. Please set dim_pair_out={pair_dim} in config."
            )

        # # Construct extended pair representation as block matrix
        concat_mask = batch["target_mask"].bool()  # .sum(dim=-1).bool()   # [b, n_concat]

        # Simple concatenation approach - build the extended matrix directly
        n_extended = n_orig + n_concat
        extended_pair_rep = torch.zeros(b, n_extended, n_extended, pair_dim, device=device)

        # Upper left: original pair representation
        extended_pair_rep[:, :n_orig, :n_orig, :] = orig_pair_rep

        # Upper right: sample-to-target features
        extended_pair_rep[:, :n_orig, n_orig:, :] = upper_right_projected

        # Lower left: target-to-sample features (transpose of upper right)
        extended_pair_rep[:, n_orig:, :n_orig, :] = lower_left_projected

        # Lower right: target-to-target features
        extended_pair_rep[:, n_orig:, n_orig:, :] = lower_right_projected

        # Apply masks
        # collapse dim2 of concat_mask via sum and bool
        extended_seq_mask = torch.cat([orig_seq_mask, concat_mask], dim=1)  # [b, n_extended]
        extended_pair_rep = (
            extended_pair_rep * extended_seq_mask[:, :, None, None] * extended_seq_mask[:, None, :, None]
        )

        return extended_pair_rep

    def motif_forward(self, batch, orig_pair_rep, orig_seq_mask, orig_pair_rep_lig=None):
        if orig_pair_rep_lig is None:
            orig_pair_rep_lig = orig_pair_rep
        b, n_orig, _, pair_dim = orig_pair_rep.shape
        device = orig_pair_rep.device

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
        upper_right_xsc_dist = self.upper_right_xsc_dist(batch_with_chains)
        upper_right_optional_ca = self.upper_right_optional_ca(batch_with_chains)
        # upper_right_chain = self.upper_right_chain_motif(batch_with_chains)
        # upper_right_hotspots = self.upper_right_hotspots(batch_with_chains)
        upper_right_combined = torch.cat(
            [
                upper_right_seq_sep,
                upper_right_xt_dist,
                upper_right_xsc_dist,
                upper_right_optional_ca,
                # TODO: enable againupper_right_chain,
                # upper_right_hotspots,
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
        # lower_right_seq_sep = self.lower_right_seq_sep(batch_with_chains)
        lower_right_xt_dist = self.lower_right_xt_dist(batch_with_chains)
        # mock sep separation feature with 0 for motif since we do not want to leak that (same shape as xt_dist)
        lower_right_seq_sep = torch.zeros(b, n_concat, n_concat, self.lower_right_seq_sep.get_dim(), device=device)
        lower_right_xsc_dist = self.lower_right_xsc_dist(batch_with_chains)
        lower_right_optional_ca = self.lower_right_optional_ca(batch_with_chains)
        # lower_right_chain = self.lower_right_chain(batch_with_chains)
        # lower_right_chain = torch.zeros(b, n_concat, n_concat, self.lower_right_chain_motif.get_dim(), device=device)
        # lower_right_hotspots = self.lower_right_hotspots(batch_with_chains)
        lower_right_combined = torch.cat(
            [
                lower_right_seq_sep,
                lower_right_xt_dist,
                lower_right_xsc_dist,
                lower_right_optional_ca,
                # lower_right_chain,
                # lower_right_hotspots,
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

        # extract three quadrants from orig_pair_rep_motif
        # n_concat_motif = lower_right_combined.shape[1]
        N = n_concat
        # extend the orig_pair_rep_lig with N zeros in both dimensions
        b, n_lig, _, feat_dim = orig_pair_rep_lig.shape
        #! note if we do not have a ligand or target then n_lig is the original pair representation length
        # First extend rows (add N rows of zeros)
        zeros_bottom = torch.zeros(b, N, n_lig, feat_dim, device=orig_pair_rep_lig.device)
        orig_pair_rep_lig_extended_rows = torch.cat(
            [orig_pair_rep_lig, zeros_bottom], dim=1
        )  # [b, n_lig+N, n_lig, feat_dim]

        # Then extend columns (add N columns of zeros)
        zeros_right = torch.zeros(b, n_lig + N, N, feat_dim, device=orig_pair_rep_lig.device)
        orig_pair_rep_lig_extended = torch.cat(
            [orig_pair_rep_lig_extended_rows, zeros_right], dim=2
        )  # [b, n_lig+N, n_lig+N, feat_dim]
        USE_MOTIF_ZERO_FEATS = True  #! we use this for compatability with LaProteina GitHub
        if USE_MOTIF_ZERO_FEATS:
            # add lower_right_combined to orig_pair_rep_lig_extended in the bottom right corner
            # Use original sequence length for indexing, not the extended ligand length
            orig_pair_rep_lig_extended[:, :n_orig, n_lig:, :] = upper_right_projected * 0  # [5, 233, 8, 256]
            orig_pair_rep_lig_extended[:, n_lig:, :n_orig, :] = lower_left_projected * 0  # [5, 8, 233, 256]
            orig_pair_rep_lig_extended[:, n_lig:, n_lig:, :] = lower_right_projected * 0  # [5, 8, 8, 256]
        else:
            orig_pair_rep_lig_extended[:, :n_orig, n_lig:, :] = upper_right_projected  # [5, 233, 8, 256]
            orig_pair_rep_lig_extended[:, n_lig:, :n_orig, :] = lower_left_projected  # [5, 8, 233, 256]
            orig_pair_rep_lig_extended[:, n_lig:, n_lig:, :] = lower_right_projected  # [5, 8, 8, 256]

        return orig_pair_rep_lig_extended

    def target_forward(self, batch, orig_pair_rep, orig_seq_mask):
        b, n_orig, _, pair_dim = orig_pair_rep.shape
        device = orig_pair_rep.device

        # Prepare batch with target chain information
        batch_with_chains = self._prepare_batch_with_chains(batch)

        # Get dimensions by computing one feature
        sample_feat = self.upper_right_xt_dist_target(batch_with_chains)
        n_concat = sample_feat.shape[
            2
        ]  # target sequence length (cross-sequence features are [b, n_orig, n_target, dim])

        # Compute all features for each quadrant
        # Upper right: [b, n_orig, n_concat, total_dim]
        upper_right_seq_sep = self.upper_right_seq_sep_target(batch_with_chains)
        upper_right_xt_dist = self.upper_right_xt_dist_target(batch_with_chains)
        # upper_right_xsc_dist = self.upper_right_xsc_dist(batch_with_chains)
        # upper_right_optional_ca = self.upper_right_optional_ca(batch_with_chains)
        upper_right_chain = self.upper_right_chain_target(batch_with_chains)
        upper_right_hotspots = self.upper_right_hotspots_target(batch_with_chains)
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
        upper_right_projected = self.ln_out_target(
            self.linear_out_target(upper_right_combined)
        )  # [b, n_orig, n_concat, dim_pair_out]

        # Lower left: [b, n_concat, n_orig, dim_pair_out] - transpose of upper right
        # No separate computation needed - just transpose the projected features!
        lower_left_projected = upper_right_projected.transpose(1, 2)

        # Lower right: [b, n_concat, n_concat, total_dim]
        lower_right_seq_sep = self.lower_right_seq_sep_target(batch_with_chains)
        lower_right_xt_dist = self.lower_right_xt_dist_target(batch_with_chains)
        # lower_right_xsc_dist = self.lower_right_xsc_dist(batch_with_chains)
        # lower_right_optional_ca = self.lower_right_optional_ca(batch_with_chains)
        lower_right_chain = self.lower_right_chain_target(batch_with_chains)
        lower_right_hotspots = self.lower_right_hotspots_target(batch_with_chains)
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
        lower_right_projected = self.ln_out_lower_right_target(
            self.linear_out_lower_right_target(lower_right_combined)
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

        if batch["target_mask"].ndim == 3:
            concat_mask = batch["target_mask"].sum(dim=-1).bool()
        else:
            concat_mask = batch["target_mask"].bool()
        # [b, n_orig, n_orig, pair_dim], [b, n_concat, n_orig, pair_dim] -> [b, pad_len, n_orig, pair_dim]
        # Simple concatenation approach - build the extended matrix directly
        n_extended = n_orig + n_concat
        extended_pair_rep = torch.zeros(b, n_extended, n_extended, pair_dim, device=device)

        # Upper left: original pair representation
        extended_pair_rep[:, :n_orig, :n_orig, :] = orig_pair_rep

        # Upper right: sample-to-target features
        extended_pair_rep[:, :n_orig, n_orig:, :] = upper_right_projected

        # Lower left: target-to-sample features (transpose of upper right)
        extended_pair_rep[:, n_orig:, :n_orig, :] = lower_left_projected

        # Lower right: target-to-target features
        extended_pair_rep[:, n_orig:, n_orig:, :] = lower_right_projected

        # Apply masks
        # collapse dim2 of concat_mask via sum and bool
        # concat_mask = concat_mask.sum(dim=-1).bool()
        extended_seq_mask = torch.cat([orig_seq_mask, concat_mask], dim=1)  # [b, n_extended]
        extended_pair_rep = (
            extended_pair_rep * extended_seq_mask[:, :, None, None] * extended_seq_mask[:, None, :, None]
        )

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
        # Handle ligand processing
        if self.enable_ligand:
            if "x_target" not in batch or "target_mask" not in batch:
                # No ligand data available
                B, N, _, _ = orig_pair_rep.shape
                blank = torch.zeros(self.linear_out_in_dim_ligand, device=orig_pair_rep.device)
                blank_lr = torch.zeros(
                    self.linear_out_lower_right_in_dim_ligand,
                    device=orig_pair_rep.device,
                )
                updated_pair_rep = (
                    orig_pair_rep + 0 * self.ln_out_ligand(self.linear_out_ligand(blank))[None, None, None, :]
                )
                updated_pair_rep = (
                    updated_pair_rep
                    + 0
                    * self.ln_out_lower_right_ligand(self.linear_out_lower_right_ligand(blank_lr))[None, None, None, :]
                )
            else:
                updated_pair_rep = self.ligand_forward(batch, orig_pair_rep, orig_seq_mask)
        elif self.enable_target:
            #! No target dropout for now
            updated_pair_rep = self.target_forward(batch, orig_pair_rep, orig_seq_mask)
        else:
            # No ligand enabled, use original
            updated_pair_rep = orig_pair_rep
            # # Add zero contributions from ligand layers to avoid unused parameters
            # if hasattr(self, 'linear_out_ligand'):
            #     blank = torch.zeros(self.linear_out_in_dim_ligand, device=orig_pair_rep.device)
            #     orig_pair_rep_lig = orig_pair_rep_lig + 0*self.ln_out_ligand(self.linear_out_ligand(blank))[None, None, None, :]
            # if hasattr(self, 'linear_out_lower_right_ligand'):
            #     blank_lr = torch.zeros(self.linear_out_lower_right_in_dim_ligand, device=orig_pair_rep.device)
            #     orig_pair_rep_lig = orig_pair_rep_lig + 0*self.ln_out_lower_right_ligand(self.linear_out_lower_right_ligand(blank_lr))[None, None, None, :]

        # Add zero contributions from motif layers to avoid unused parameters
        if not self.enable_motif and hasattr(self, "linear_out"):
            zero_motif = torch.zeros(1, 1, self.linear_out.in_features, device=orig_pair_rep.device)
            updated_pair_rep = updated_pair_rep + 0 * self.ln_out(self.linear_out(zero_motif))[0, 0, :]

        if self.enable_motif:
            return self.motif_forward(batch, orig_pair_rep, orig_seq_mask, updated_pair_rep)
        else:
            return updated_pair_rep

    def _prepare_batch_with_chains(self, batch):
        """Prepare batch with target chain information for feature computation."""
        batch_copy = dict(batch)

        (
            b,
            n_orig,
        ) = batch["x_t"]["bb_ca"].shape[:2]
        device = batch["x_t"]["bb_ca"].device
        batch_copy["x_t_coord"] = torch.zeros(b, n_orig, 37, 3, device=device)
        batch_copy["x_t_coord_mask"] = torch.zeros(b, n_orig, 37, device=device)
        batch_copy["x_t_coord"][:, :, 1, :] = batch["x_t"]["bb_ca"]
        if "mask" in batch:  # for inference
            batch_copy["x_t_coord_mask"][:, :, 1] = batch["mask"]
        else:
            batch_copy["x_t_coord_mask"][:, :, 1] = batch["coord_mask"][:, :, 1]

        #! This adds correct splicing of motif features to remove all padding
        if self.enable_motif:
            if "motif_mask" in batch:
                is_compact_mode = batch["motif_mask"].sum(dim=(-1)).bool().all(dim=1).all()
                if not is_compact_mode:
                    mask_feats = batch["motif_mask"] * 1.0  # [b, n, 37]
                    batch_size = b
                    concat_motif_mask = []
                    concat_seq_motif_mask = []
                    concat_x_motif = []
                    concat_seq_motif = []
                    for b in range(batch_size):
                        residue_mask = batch["seq_motif_mask"][b]  # [n]
                        if residue_mask.any():
                            selected_x_motif = batch["x_motif"][b][residue_mask]
                            selected_seq_motif_mask = batch["seq_motif_mask"][b][residue_mask]
                            selected_motif_mask = batch["motif_mask"][b][residue_mask]
                            selected_seq_motif = batch["seq_motif"][b][residue_mask]
                        else:
                            selected_x_motif = torch.zeros(0, 37, 3, device=batch["x_motif"].device)
                            selected_seq_motif_mask = torch.zeros(
                                0,
                                dtype=torch.bool,
                                device=batch["seq_motif_mask"].device,
                            )
                            selected_motif_mask = torch.zeros(0, 37, device=batch["motif_mask"].device)
                            selected_seq_motif = torch.zeros(0, dtype=torch.bool, device=batch["seq_motif"].device)

                        concat_x_motif.append(selected_x_motif)
                        concat_seq_motif_mask.append(selected_seq_motif_mask)
                        concat_motif_mask.append(selected_motif_mask)
                        concat_seq_motif.append(selected_seq_motif)

                    # Pad to same length
                    padded_concat_x_motif = pad_sequence(concat_x_motif, batch_first=True, padding_value=0.0)
                    padded_concat_seq_motif_mask = pad_sequence(
                        concat_seq_motif_mask, batch_first=True, padding_value=False
                    )
                    padded_concat_motif_mask = pad_sequence(concat_motif_mask, batch_first=True, padding_value=False)
                    padded_concat_seq_motif = pad_sequence(concat_seq_motif, batch_first=True, padding_value=0)
                else:
                    #! if its already in concat mode then we can just use it
                    padded_concat_x_motif = batch["x_motif"]
                    padded_concat_seq_motif_mask = batch["seq_motif_mask"]
                    padded_concat_motif_mask = batch["motif_mask"]
                    padded_concat_seq_motif = batch["seq_motif"]
            else:
                padded_concat_x_motif = torch.zeros(b, 0, 37, 3, device=device)
                padded_concat_seq_motif_mask = torch.zeros(b, 0, dtype=torch.bool, device=device)
                padded_concat_motif_mask = torch.zeros(b, 0, 37, device=device)
                padded_concat_seq_motif = torch.zeros(b, 0, dtype=torch.bool, device=device)
            batch_copy["x_motif"] = padded_concat_x_motif
            batch_copy["seq_motif_mask"] = padded_concat_seq_motif_mask
            batch_copy["motif_mask"] = padded_concat_motif_mask
            batch_copy["seq_motif"] = padded_concat_seq_motif
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
