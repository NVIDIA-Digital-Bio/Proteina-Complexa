import torch
from loguru import logger

from proteinfoundation.nn.feature_factory.ligand_feats import LigandConcatSeqFeat
from proteinfoundation.nn.feature_factory.motif_feats import MotifConcatSeqFeat
from proteinfoundation.nn.feature_factory.target_feats import TargetConcatSeqFeat


class ConcatFeaturesFactory(torch.nn.Module):
    """Factory for creating concat features from motif and/or target.

    Args:
        enable_motif: Whether to enable motif concat features
        enable_target: Whether to enable target concat features
        enable_ligand: Whether to enable ligand concat features
        dim_feats_out: The dimension of the output features
        use_ln_out: Whether to use layer normalization
        **kwargs: Additional keyword arguments

    Note: This only works with motif, (ligand, motif) or (target, motif)
    """

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
        if enable_ligand and enable_target:
            raise ValueError("Cannot enable both ligand and target concat features for now")

        self.feature_creators = torch.nn.ModuleList()
        ligand_dim, motif_dim, target_dim = 0, 0, 0
        if enable_ligand:
            self.feature_creators.append(LigandConcatSeqFeat(**kwargs))
            logger.info("Enabled ligand concat features")
            ligand_dim = self.feature_creators[-1].get_dim()

        elif enable_target:
            self.feature_creators.append(TargetConcatSeqFeat(**kwargs))
            logger.info("Enabled target concat features")
            target_dim = self.feature_creators[-1].get_dim()

        if enable_motif:
            self.feature_creators.append(MotifConcatSeqFeat(**kwargs))
            logger.info("Enabled motif concat features")
            motif_dim = self.feature_creators[-1].get_dim()

        # Calculate total input dimension
        sum(creator.get_dim() for creator in self.feature_creators)
        # Create projection layers
        if enable_ligand:
            self.linear_out_ligand = torch.nn.Linear(ligand_dim, dim_feats_out, bias=False)
            self.ln_out_ligand = torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()

        if enable_target:
            self.linear_out_target = torch.nn.Linear(target_dim, dim_feats_out, bias=False)
            self.ln_out_target = torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()

        if enable_motif:
            self.linear_out = torch.nn.Linear(motif_dim, dim_feats_out, bias=False)
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

        if "batch_size" not in batch:
            batch["batch_size"] = seq_repr.shape[0]
        for creator in self.feature_creators:
            feats, mask = creator(batch)  # [b, n_i, d_i], [b, n_i]
            all_feats.append(feats)
            all_masks.append(mask)
            # logger.info(
            #     f"Enabled {creator.__class__.__name__} features: feats shape: {feats.shape} mask shape: {mask.shape}"
            # )
        # Concatenate features if we have any
        if len(all_feats) == 0:
            # No concat features, return original
            return seq_repr, seq_mask

        if len(all_feats) == 1:
            # Only motif features available
            combined_feats = all_feats[0]  # [b, n_concat, d_total] - this is motif
            combined_masks = all_masks[0]  # [b, n_concat]

            if self.enable_motif:
                # Apply linear projection for motif
                projected_feats = self.ln_out(self.linear_out(combined_feats))  # [b, n_concat, dim_feats_out]
            elif self.enable_ligand:
                projected_feats = self.ln_out_ligand(
                    self.linear_out_ligand(combined_feats)
                )  # [b, n_concat, dim_feats_out]
            elif self.enable_target:
                projected_feats = self.ln_out_target(
                    self.linear_out_target(combined_feats)
                )  # [b, n_concat, dim_feats_out]
            else:
                raise ValueError(
                    "Did not specify which features to use. Only one of enable_motif, enable_ligand, or enable_target can be True to condition on single feature type"
                )
            projected_feats = projected_feats * combined_masks[..., None]  # Apply mask

            # # Add zero contribution from ligand layers to avoid unused parameters
            # if hasattr(self, 'ln_out_ligand'):
            #     zero_ligand = torch.zeros(1, 1, self.ln_out_ligand.in_features, device=combined_feats.device)
            #     projected_feats = projected_feats + 0 * self.ln_out_ligand(self.linear_out_ligand(zero_ligand))[0, 0, :]
            # if hasattr(self, 'ln_out_target'):
            #     zero_target = torch.zeros(1, 1, self.ln_out_target.in_features, device=combined_feats.device)
            #     projected_feats = projected_feats + 0 * self.ln_out_target(self.linear_out_target(zero_target))[0, 0, :]
        elif len(all_feats) == 2:
            # Both ligand and motif features available
            # project motif feats separately
            projected_feats_motif = self.ln_out(
                self.linear_out(all_feats[1])  #! 1 index is from the order of the feature creator append
            )
            projected_feats_motif = projected_feats_motif * all_masks[1][..., None]

            # ligands feat separate
            combined_feats = all_feats[0]
            combined_masks = all_masks[0]
            # Apply linear projection to match seq_repr dimension
            if self.enable_ligand:
                projected_feats = self.ln_out_ligand(
                    self.linear_out_ligand(combined_feats)
                )  # [b, n_concat, dim_feats_out]
                if hasattr(self, "ln_out_target"):
                    zero_target = torch.zeros(
                        1,
                        1,
                        self.ln_out_target.in_features,
                        device=combined_feats.device,
                    )
                    projected_feats = (
                        projected_feats + 0 * self.ln_out_target(self.linear_out_target(zero_target))[0, 0, :]
                    )
            if self.enable_target:
                projected_feats = self.ln_out_target(self.linear_out_target(combined_feats))
                if hasattr(self, "ln_out_ligand"):
                    zero_ligand = torch.zeros(
                        1,
                        1,
                        self.ln_out_ligand.in_features,
                        device=combined_feats.device,
                    )
                    projected_feats = (
                        projected_feats + 0 * self.ln_out_ligand(self.linear_out_ligand(zero_ligand))[0, 0, :]
                    )

            projected_feats = projected_feats * combined_masks[..., None]  # Apply mask
            # combine ligand with motif feats along sequence dimension
            projected_feats = torch.cat([projected_feats, projected_feats_motif], dim=1)
            combined_masks = torch.cat([combined_masks, all_masks[1]], dim=1)
        else:
            raise ValueError(
                f"Invalid number of features: {len(all_feats)}. Does not support ligand, motif, and protein targets"
            )

        extended_seq_repr = torch.cat([seq_repr, projected_feats], dim=1)
        extended_mask = torch.cat([seq_mask, combined_masks], dim=1)
        return extended_seq_repr, extended_mask
