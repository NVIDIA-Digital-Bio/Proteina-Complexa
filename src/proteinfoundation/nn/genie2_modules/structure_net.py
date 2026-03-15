import torch
from openfold.model.structure_module import BackboneUpdate, InvariantPointAttention
from openfold.model.structure_module import StructureModuleTransition as StructureTransition
from torch import nn


class StructureLayer(nn.Module):
    """
    Structure Layer.

    This module utilizes Invariant Point Attention (IPA) to combine multiple
    modes of representations (single, pair and frame) and subsequently update
    single representation. These update single representations are then used
    to compute backbone update.
    """

    def __init__(
        self,
        c_s,
        c_p,
        c_hidden,
        n_head,
        n_qk_point,
        n_v_point,
        ipa_dropout,
        n_structure_transition_layer,
        structure_transition_dropout,
    ):
        """
        Args:
                c_s:
                        Dimension of per-residue (single) representation.
                c_p:
                        Dimension of paired residue-residue (pair) representation.
                c_hidden:
                        Number of hidden dimensions in IPA layer.
                n_head:
                        Number of heads in IPA layer.
                n_qk_point:
                        Number of query/key points in IPA layer.
                n_v_point:
                        Number of value points in IPA layer.
                ipa_dropout:
                        Dropout rate in IPA layer.
                n_structure_transition_layer:
                        Number of structure transition layers.
                structure_transition_dropout:
                        Dropout rate in structure transition layer.
        """
        super().__init__()

        # Invariant point attention
        self.ipa = InvariantPointAttention(c_s, c_p, c_hidden, n_head, n_qk_point, n_v_point)
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)

        # Built-in dropout and layer norm
        self.transition = StructureTransition(c_s, n_structure_transition_layer, structure_transition_dropout)

        # Backbone update
        self.bb_update = BackboneUpdate(c_s)

    def forward(self, inputs):
        """
        Args:
                inputs:
                        A tuple containing
                                s:
                                        [B, N, c_s] single representation
                                p:
                                        [B, N, N, c_p] pair representation
                                t:
                                        [B, N] frames
                                mask:
                                        [B, N] residue mask
                                states:
                                        a running list to keep track of intermediate
                                        single representations.

        Returns:
                outputs:
                        A tuple containing
                                s:
                                        [B, N, c_s] updated single representation
                                p:
                                        [B, N, N, c_p] pair representation
                                t:
                                        [B, N] frames
                                mask:
                                        [B, N] residue mask
                                states:
                                        a (updated) running list to keep track of
                                        intermediate single representations.
        """
        s, p, t, mask, states = inputs
        s = s + self.ipa(s, p, t, mask)
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = self.transition(s)
        states.append(s.unsqueeze(0))
        # t = t.compose(self.bb_update(s))
        rigid_update = self.bb_update(s)
        t = t.compose_q_update_vec(rigid_update, mask[..., None])  #! could the mask here be hurting?
        outputs = (s, p, t, mask, states)
        return outputs


class StructureNet(nn.Module):
    """
    Structure Netork.

    This module utilizes multiple structure layers (with/without recycles)
    to compute and apply frame updates based on input single representations,
    pair representations and frames.
    """

    def __init__(
        self,
        c_s,
        c_p,
        n_structure_layer,
        n_structure_block,
        c_hidden_ipa,
        n_head_ipa,
        n_qk_point,
        n_v_point,
        ipa_dropout,
        n_structure_transition_layer,
        structure_transition_dropout,
    ):
        """
        Args:
                c_s:
                        Dimension of per-residue (single) representation.
                c_p:
                        Dimension of paired residue-residue (pair) representation.
                n_structure_layer:
                        Number of structure layers.
                n_structure_block:
                        Number of recycles.
                c_hidden_ipa:
                        Number of hidden dimensions in IPA layer.
                n_head_ipa:
                        Number of heads in IPA layer.
                n_qk_point:
                        Number of query/key points in IPA layer.
                n_v_point:
                        Number of value points in IPA layer.
                ipa_dropout:
                        Dropout rate in IPA layer.
                n_structure_transition_layer:
                        Number of structure transition layers.
                structure_transition_dropout:
                        Dropout rate in structure transition layer.
        """
        super().__init__()
        self.n_structure_block = n_structure_block

        # Create structure layers
        layers = [
            StructureLayer(
                c_s,
                c_p,
                c_hidden_ipa,
                n_head_ipa,
                n_qk_point,
                n_v_point,
                ipa_dropout,
                n_structure_transition_layer,
                structure_transition_dropout,
            )
            for _ in range(n_structure_layer)
        ]

        # Create model
        self.net = nn.Sequential(*layers)

    def forward(self, s, p, ts, residue_mask):
        """
        Args:
                s:
                        [B, N, c_s] Single represnetation.
                p:
                        [B, N, N, c_p] Pair representation.
                ts:
                        [B, N] Frames.
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
                A tuple containing
                        states:
                                [1 + num_strucutre_block * num_structure_layer, B, N, c_s]
                                intermediate single representations
                        ts:
                                [B, N] updated frames.

        """
        states = [s.unsqueeze(0)]
        mask = residue_mask.int()
        for block_idx in range(self.n_structure_block):
            s, p, ts, mask, states = self.net((s, p, ts, mask, states))
        states = torch.concat(states, dim=0)
        return states, ts, s
