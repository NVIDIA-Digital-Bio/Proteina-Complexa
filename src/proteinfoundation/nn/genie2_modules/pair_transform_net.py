from openfold.model.dropout import DropoutColumnwise, DropoutRowwise
from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from torch import nn


class PairTransformLayer(nn.Module):
    """
    Pair Transform Layer.

    Adapted from Evoformer, this module utilizes a triangular multiplicative
    update layer and a triangular attention layer (if specified) to refine
    pair representation.
    """

    def __init__(
        self,
        c_p,
        include_mul_update,
        include_tri_att,
        c_hidden_mul,
        c_hidden_tri_att,
        n_head_tri,
        tri_dropout,
        pair_transition_n,
    ):
        """
        Args:
                c_p:
                        Dimension of paired residue-residue (pair) representation.
                include_mul_update:
                        Flag on whether to use triangular multiplicative update layer.
                include_tri_att:
                        Flag on whether to use triangular attention layer.
                c_hidden_mul:
                        Number of hidden dimensions in triangular multiplicative update layer.
                c_hidden_tri_att:
                        Number of hidden dimensions in triangular attention layer.
                n_head_tri:
                        Number of heads in triangular attention layer.
                tri_dropout:
                        Dropout rate.
                pair_transition_n:
                        Number of pair transition layers.
        """
        super().__init__()

        # Layers for triangular multiplicative updates
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_p, c_hidden_mul) if include_mul_update else None
        self.tri_mul_in = TriangleMultiplicationIncoming(c_p, c_hidden_mul) if include_mul_update else None

        # Layers for triangular attention
        self.tri_att_start = (
            TriangleAttentionStartingNode(c_p, c_hidden_tri_att, n_head_tri) if include_tri_att else None
        )
        self.tri_att_end = TriangleAttentionEndingNode(c_p, c_hidden_tri_att, n_head_tri) if include_tri_att else None

        # Layer for pair transition
        self.pair_transition = PairTransition(c_p, pair_transition_n)

        # Layers for dropouts
        self.dropout_row_layer = DropoutRowwise(tri_dropout)
        self.dropout_col_layer = DropoutColumnwise(tri_dropout)

    def forward(self, inputs):
        """
        Args:
                inputs:
                        A tuple containing
                                p:
                                        [B, N, N, c_p] pair representation
                                pair_residue_mask:
                                        [B, N, N] pairwise residue mask.

        Returns:
                outputs:
                        A tuple containing
                                p:
                                        [B, N, N, c_p] updated pair representation
                                pair_residue_mask:
                                        [B, N, N] pairwise residue mask.
        """
        # if self.training: Genie2 had this
        #     chunk_size = None
        # else:
        #     chunk_size = 4
        chunk_size = None
        p, pair_residue_mask = inputs
        if self.tri_mul_out is not None:
            p = p + self.dropout_row_layer(self.tri_mul_out(p, pair_residue_mask))
            p = p + self.dropout_row_layer(self.tri_mul_in(p, pair_residue_mask))
        if self.tri_att_start is not None:
            p = p + self.dropout_row_layer(self.tri_att_start(p, pair_residue_mask, chunk_size=chunk_size))
            p = p + self.dropout_col_layer(self.tri_att_end(p, pair_residue_mask, chunk_size=chunk_size))
        p = p + self.pair_transition(p, pair_residue_mask, chunk_size=chunk_size)
        p = p * pair_residue_mask.unsqueeze(-1)
        outputs = (p, pair_residue_mask)
        return outputs


class PairTransformNet(nn.Module):
    """
    Pair Transform Network.

    Adapted from Evoformer, this module utilizes multiple pair transform
    layers to refine pair representations before using them in the
    structure module.
    """

    def __init__(
        self,
        c_p,
        n_pair_transform_layer,
        include_mul_update,
        include_tri_att,
        c_hidden_mul,
        c_hidden_tri_att,
        n_head_tri,
        tri_dropout,
        pair_transition_n,
    ):
        """
        Args:
                c_p:
                        Dimension of paired residue-residue (pair) representation.
                n_pair_transform_layer:
                        Number of pair transform layers.
                include_mul_update:
                        Flag on whether to use triangular multiplicative update layer.
                include_tri_att:
                        Flag on whether to use triangular attention layer.
                c_hidden_mul:
                        Number of hidden dimensions in triangular multiplicative update layer.
                c_hidden_tri_att:
                        Number of hidden dimensions in triangular attention layer.
                n_head_tri:
                        Number of heads in triangular attention layer.
                tri_dropout:
                        Dropout rate.
                pair_transition_n:
                        Number of pair transition layers.
        """
        super().__init__()

        # Create pair transform layers
        layers = [
            PairTransformLayer(
                c_p,
                include_mul_update,
                include_tri_att,
                c_hidden_mul,
                c_hidden_tri_att,
                n_head_tri,
                tri_dropout,
                pair_transition_n,
            )
            for _ in range(n_pair_transform_layer)
        ]

        # Create model
        self.net = nn.Sequential(*layers)

    def forward(self, p, residue_mask):
        """
        Args:
                p:
                        [B, N, N, c_p] Pair representation.
                features:
                        A batched feature dictionary with a batch size B, where each
                        structure is padded to the maximum sequence length N. It contains
                        the following information
                                -	aatype:
                                                [B, N, 20] one-hot encoding on amino acid types
                                -	num_chains:
                                                [B, 1] number of chains in the structure
                                -	num_residues:
                                                [B, 1] number of residues in the structure
                                -	num_residues_per_chain:
                                                [B, 1] an array of number of residues by chain
                                -	atom_positions:
                                                [B, N, 3] an array of Ca atom positions
                                -	residue_mask:
                                                [B, N] residue mask to indicate which residue position is masked
                                -	residue_index:
                                                [B, N] residue index (started from 0)
                                -	chain_index:
                                                [B, N] chain index (started from 0)
                                -	fixed_sequence_mask:
                                                [B, N] mask to indicate which residue contains conditional
                                                sequence information
                                -	fixed_structure_mask:
                                                [B, N, N] mask to indicate which pair of residues contains
                                                conditional structural information
                                -	fixed_group:
                                                [B, N] group index to indicate which group the residue belongs to
                                                (useful for specifying multiple functional motifs)
                                -	interface_mask:
                                                [B, N] deprecated and set to all zeros.

        Returns:
                [B, N, N, c_p] Updated pair representation.
        """

        # Pairwise residue mask
        # Shape: [B, N, N]
        pair_residue_mask = residue_mask.unsqueeze(1) * residue_mask.unsqueeze(2)

        # Update pair representations
        # Shape: [B, N, N, c_p]
        p, _ = self.net((p, pair_residue_mask))

        return p
