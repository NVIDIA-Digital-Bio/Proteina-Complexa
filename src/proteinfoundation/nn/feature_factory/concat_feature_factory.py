import torch
from loguru import logger

from proteinfoundation.nn.feature_factory.ligand_feats import LigandConcatSeqFeat
from proteinfoundation.nn.feature_factory.motif_feats import MotifConcatSeqFeat
from proteinfoundation.nn.feature_factory.target_feats import TargetConcatSeqFeat
from proteinfoundation.utils.tensor_utils import concat_padded_tensor


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

        logger.info(f"ConcatFeaturesFactory: input feat dim {total_dim} -> output feat dim {dim_feats_out}")

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
