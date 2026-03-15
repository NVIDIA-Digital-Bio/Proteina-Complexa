from typing import Literal

import torch
from loguru import logger

from proteinfoundation.nn.feature_factory.base_feature import ZeroFeat
from proteinfoundation.nn.feature_factory.motif_feats import (
    BulkAllAtomXmotifSeqFeat,
    MotifAbsoluteCoordsSeqFeat,
    MotifMaskSeqFeat,
    MotifRelativeCoordsSeqFeat,
    MotifSequenceSeqFeat,
    MotifSideChainAnglesSeqFeat,
    MotifTorsionAnglesSeqFeat,
    MotifToSamplePairwiseDistancesPairFeat,
    XmotifPairwiseDistancesPairFeat,
)
from proteinfoundation.nn.feature_factory.pair_feats import (
    BackbonePairDistancesNanometerPairFeat,
    CaCoorsNanometersPairwiseDistancesPairFeat,
    ChainIdxPairFeat,
    ContactTypePairFeat,
    CrossSequenceBackbonePairDistancesPairFeat,
    CrossSequenceChainIndexPairFeat,
    CrossSequenceOptionalCaPairDistancesPairFeat,
    CrossSequenceRelativeSequenceSeparationPairFeat,
    CrossSequenceXscBBCAPairwiseDistancesPairFeat,
    HotspotMaskPairFeat,
    OptionalCaCoorsNanometersPairwiseDistancesPairFeat,
    RelativeResidueOrientationPairFeat,
    SequenceSeparationPairFeat,
    XscBBCAPairwiseDistancesPairFeat,
    XtBBCAPairwiseDistancesPairFeat,
)
from proteinfoundation.nn.feature_factory.seq_cond_feats import (
    BinderCenterFeat,
    ChainBreakPerResidueSeqFeat,
    ChainIdxSeqFeat,
    ContactTypeSeqFeat,
    FoldEmbeddingSeqFeat,
    HotspotMaskSeqFeat,
    IdxEmbeddingSeqFeat,
    IdxEmbeddingSeqFeatGenie2,
    StochasticTranslationSeqFeat,
    TimeEmbeddingPairFeat,
    TimeEmbeddingSeqFeat,
    TimeEmbeddingSeqFeatGenie2,
)
from proteinfoundation.nn.feature_factory.seq_feats import (
    Atom37NanometersCoorsSeqFeat,
    BackboneBondAnglesSeqFeat,
    BackboneTorsionAnglesSeqFeat,
    CaCoorsNanometersSeqFeat,
    LatentVariableSeqFeat,
    OpenfoldSideChainAnglesSeqFeat,
    OptionalCaCoorsNanometersSeqFeat,
    OptionalResidueTypeSeqFeat,
    ResidueTypeSeqFeat,
    TryCaCoorsNanometersSeqFeat,
    XscBBCASeqFeat,
    XscLocalLatentsSeqFeat,
    XtBBCASeqFeat,
    XtLocalLatentsSeqFeat,
)
from proteinfoundation.nn.feature_factory.target_feats import (
    TargetAbsoluteCoordsSeqFeat,
    TargetMaskPairFeat,
    TargetMaskSeqFeat,
    TargetRelativeCoordsSeqFeat,
    TargetSequenceSeqFeat,
    TargetSideChainAnglesSeqFeat,
    TargetTorsionAnglesSeqFeat,
    TargetToSamplePairwiseDistancesPairFeat,
    XtargetPairwiseDistancesPairFeat,
)


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
            elif f == "bulk_all_atom_xmotif":
                return BulkAllAtomXmotifSeqFeat(**kwargs)

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
            # elif f == "sample_to_target_pair_dists": # function is not implemented
            #     return SampleToTargetPairwiseDistancesPairFeat(**kwargs)
            # elif f == "target_to_sample_seq_sep":
            #     return TargetToSampleRelativeSequenceSeparationPairFeat(**kwargs)
            # elif f == "target_to_sample_chain_idx":
            #     return TargetToSampleChainIndexPairFeat(**kwargs)
            # elif f == "target_to_target_seq_sep":
            #     return TargetToTargetRelativeSequenceSeparationPairFeat(**kwargs)
            # elif f == "target_to_target_chain_idx":
            #     return TargetToTargetChainIndexPairFeat(**kwargs)
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
