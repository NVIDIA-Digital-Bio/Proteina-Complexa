import torch
from loguru import logger
from torch.nn import functional as F

from proteinfoundation.nn.feature_factory.base_feature import Feature
from proteinfoundation.nn.feature_factory.feature_utils import NUM_BOND_ORDERS
from proteinfoundation.nn.feature_factory.seq_feats import AtomisticCoorsSeqFeat


class TargetChargeSeqFeat(Feature):
    """
    Computes feature from ligand charge, feature of shape [b, n, 1].
    """

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        assert "target_charge" in batch, "`target_charge` not in batch, cannot compute LigandChargeSeqFeat"
        return batch["target_charge"][..., None]  # [b, n, 1]


class TargetAtomNameSeqFeat(Feature):
    """
    Computes feature from ligand charge, feature of shape [b, n, 1].
    """

    def __init__(self, **kwargs):
        super().__init__(dim=4 * 64)

    def forward(self, batch):
        assert "target_atom_name" in batch, "`target_atom_name` not in batch, cannot compute LigandChargeSeqFeat"
        return batch["target_atom_name"]


class TargetGraphPESeqFeat(Feature):
    """
    Computes feature from ligand charge, feature of shape [b, n, 1].
    """

    def __init__(self, **kwargs):
        super().__init__(dim=32)

    def forward(self, batch):
        assert "target_laplacian_pe" in batch, "`target_laplacian_pe` not in batch, cannot compute LigandChargeSeqFeat"
        return batch["target_laplacian_pe"]


class AtomTypeSeqFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 128].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, input_is_onehot=True, **kwargs):
        super().__init__(dim=128)
        self.input_is_onehot = input_is_onehot

    def forward(self, batch):
        assert "residue_type" in batch, "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        rpadmask = batch["mask_dict"]["residue_type"]  # [b, n] binary
        # [b, n], the -1 padding becomes 0
        if self.input_is_onehot:
            rtype = rtype * rpadmask[..., None]
            rtype_onehot = rtype
        else:
            rtype_onehot = F.one_hot(rtype, num_classes=self.dim)  # [b, n, dim]
            rtype_onehot = rtype_onehot * rpadmask[..., None]  # zero out padding rows just in case
        return rtype_onehot * 1.0


class BondMaskPairFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 128].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        return batch["target_bond_mask"][..., None]


class BondOrderPairFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 128].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=NUM_BOND_ORDERS)

    def forward(self, batch):
        return F.one_hot(batch["target_bond_order"].long(), num_classes=self.dim)


class LigandConcatSeqFeat(Feature):
    """Computes concat ligand features combining coordinates, sequence, and mask."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.coords_feat = AtomisticCoorsSeqFeat()
        self.seq_feat = AtomTypeSeqFeat(input_is_onehot=True)
        self.charge_feat = TargetChargeSeqFeat()
        self.atom_name_feat = TargetAtomNameSeqFeat()
        self.graph_pe_feat = TargetGraphPESeqFeat()

        self.dim = (
            self.coords_feat.dim
            + self.seq_feat.dim
            + self.charge_feat.dim
            + self.atom_name_feat.dim
            + self.graph_pe_feat.dim
            + 1
        )  # 1 for the mask
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
        target_residue_mask = batch["target_mask"]  # .sum(dim=-1).bool()  # [b, n]
        is_compact_mode = target_residue_mask.all(dim=1).any()  # Check if any batch has all True (compact mode)
        if is_compact_mode:
            # Compact mode: data is already extracted, use directly
            coords_feats = self.coords_feat(
                {
                    "coords_nm": batch["x_target"],  # [b, n_target, 3]
                    "coord_mask": batch["target_mask"],  # [..., None],  # [b, n_target]
                }
            )  # [b, n_target, 148]

            # Sequence features
            batch_seq = {
                "residue_type": batch["seq_target"],  # [b, n_target, 128] #! already one hotted
                "mask_dict": {"residue_type": batch["target_mask"]},  # [b, n_target]
            }
            seq_feats = self.seq_feat(batch_seq)  # [b, n_target, 128]

            # Target mask features
            mask_feats = batch["target_mask"][..., None] * 1.0  # [b, n_target, 1]

            charge_feats = self.charge_feat(batch)  # [b, n_target, 1]
            atom_name_feats = self.atom_name_feat(batch)  # [b, n_target, 4*64]
            graph_pe_feats = self.graph_pe_feat(batch)  # [b, n_target, 32]

            # Concatenate all features
            combined_feats = torch.cat(
                [
                    coords_feats,
                    seq_feats,
                    mask_feats,
                    charge_feats,
                    atom_name_feats,
                    graph_pe_feats,
                ],
                dim=-1,
            )  # [b, n_target, 558]
            combined_feats = combined_feats * batch["target_mask"][..., None]  # Apply mask

            # Return as-is since it's already compact
            return combined_feats, batch["target_mask"].bool()
        else:
            raise ValueError("LigandConcatSeqFeat only supports compact mode")
