import torch
from openfold.utils.rigid_utils import rot_to_quat
from torch import nn

from proteinfoundation.nn.genie2_modules.utils.geo_utils import distance


class PairFeatureNet(nn.Module):
    """
    Pair Feature Network.

    This module generates paired residue-residue (pair) representations by
    integrating the following information
            -	outer sum of the (single) residue representations
            -	relative positional encodings of residue pairs
            -	template of the input noised structure, including pairwise
                    distances and orientations between frames
            -	template of the motif structure, including pairwise distances
                    between motif Ca atoms (masked accordingly based on the
                    fixed structure mask in the feature dictionary).
    """

    def __init__(
        self,
        c_s,
        c_p,
        relpos_k,
        template_dist_min,
        template_dist_step,
        template_dist_n_bin,
    ):
        """
        Args:
                c_s:
                        Dimension of per-residue (single) representation.
                c_p:
                        Dimension of paired residue-residue (pair) representation.
                relpos_k:
                        Window size used in relative positional encoding.
                template_dist_min:
                        Minimum distance for pairwise distance bin.
                template_dist_step:
                        Pairwise distance bin size.
                template_dist_n_bin:
                        Number of pairwise distance bins.
        """
        super().__init__()
        self.c_s = c_s
        self.c_p = c_p

        # Layers for outer sum of single representations
        self.linear_s_p_i = nn.Linear(c_s, c_p, bias=False)
        self.linear_s_p_j = nn.Linear(c_s, c_p, bias=False)

        # Parameters and layers for relative positional encoding
        self.relpos_k = relpos_k
        self.relpos_n_bin = 2 * relpos_k + 2
        self.linear_relpos = nn.Linear(self.relpos_n_bin + 1, c_p, bias=False)

        # Parameters and layers for templates
        self.template_dist_min = template_dist_min
        self.template_dist_step = template_dist_step
        self.template_dist_n_bin = template_dist_n_bin
        self.linear_template = nn.Linear(self.template_dist_n_bin + 6, c_p, bias=False)
        self.linear_motif_template = nn.Linear(self.template_dist_n_bin + 2, c_p, bias=False)

    def forward(self, s, ts, residue_mask, residue_index, chain_index):
        """
        Args:
                s:
                        [B, N, c_s] Per-residue (single) representation
                ts:
                        [B, N] Frames at a given timestep
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
                [B, N, N, c_p] Paired residue-residue (pair) representation
        """
        # Pairwise residue masks
        # Shape: [B, N, N, 1]
        pair_residue_mask = residue_mask.unsqueeze(1) * residue_mask.unsqueeze(2)

        # Linear projections of single representation
        # Shape: [B, N, c_p]
        p_i = self.linear_s_p_i(s)
        p_j = self.linear_s_p_j(s)

        # Outer sum of linear projections of single representation
        # Shape: [B, N, N, c_p]
        p = p_i[:, :, None, :] + p_j[:, None, :, :]

        # Aggregate pair representation with pairwise relative position
        # encoding and template-based encodings
        # Shape: [B, N, N, c_p]
        fixed_sequence_mask = torch.zeros(residue_mask.shape).to(residue_mask.device)
        b, n = residue_mask.shape
        fixed_structure_mask = torch.zeros((b, n, n)).to(residue_mask.device)
        atom_positions = torch.zeros((b, n, 3)).to(residue_mask.device)

        p += self._relpos(residue_index, chain_index)
        p += self.linear_template(
            torch.cat(
                [
                    self._encode_positions(ts.get_trans(), residue_mask),
                    self._encode_orientations(ts.get_rots().get_rot_mats(), residue_mask),
                    fixed_structure_mask.unsqueeze(-1),
                    fixed_structure_mask.unsqueeze(-1),
                ],
                axis=-1,
            )
        )
        p += self.linear_motif_template(
            torch.cat(
                [
                    self._encode_positions(atom_positions, fixed_sequence_mask) * fixed_structure_mask.unsqueeze(-1),
                    fixed_structure_mask.unsqueeze(-1),
                    fixed_structure_mask.unsqueeze(-1),
                ],
                axis=-1,
            )
        )

        return p * pair_residue_mask.unsqueeze(-1)

    ############################
    ###   Helper Functions   ###
    ############################

    def _relpos(self, residue_index, chain_index):
        """
        Compute relative position encoding based on residue indices
        (within the chain) and chain indices.

        This algorithm is adopted from AlphaFold 2 Algorithm 4 & 5
        and implemented based on OpenFold utils/tensor_utils.py.

        Args:
                features:
                        A batched feature dictionary with a batch size B, where each
                        structure is padded to the maximum sequence length N. It contains
                        the following information that relates to this function
                                -	residue_mask:
                                                [B, N] residue mask to indicate which residue
                                                position is masked
                                -	residue_index:
                                                [B, N] residue index (started from 0)
                                -	chain_index:
                                                [B, N] chain index (started from 0)

        Returns:
                [B, N, N, c_p] Pair representation based on pairwise relative positions
        """
        # residue_index = features['residue_index']
        # chain_index = features['chain_index']
        # residue_mask = residue_mask

        # Same chain mask
        # Denotes if two residues are in the same chain
        # Shape: [B, N, N]
        is_same_chain = chain_index[:, :, None] == chain_index[:, None, :]

        # Pairwise relative position matrix, offsetted by window size
        # Note that relative residue position across chains is capped at
        # relpos_k + 1, or 2 * relpos_k + 1 with offset
        # Shape: [B, N, N]
        d_same_chain = torch.clip(
            residue_index[:, :, None] - residue_index[:, None, :] + self.relpos_k,
            0,
            2 * self.relpos_k,
        )
        d_diff_chain = torch.ones_like(d_same_chain) * (2 * self.relpos_k + 1)
        d = d_same_chain * is_same_chain + d_diff_chain * ~is_same_chain

        # Pairwise relative position encoding
        # Shape: [B, N, N, n_bin]
        oh = nn.functional.one_hot(d.long(), num_classes=self.relpos_n_bin).float()

        # Project to given single representation dimension
        # Shape: [B, N, N, c_p]
        return self.linear_relpos(torch.cat([oh, is_same_chain.unsqueeze(-1)], axis=-1))

    def _encode_positions(self, coords, mask):
        """
        Encode pairwise distances for a sequence of coordinates.

        Args:
                coords:
                        [B, N, 3] A sequence of atom positions.
                mask:
                        [B, N] Mask to indicate which atom position is masked.

        Returns:
                [B, N, N, n_bin] Masked pairwise distance encoding.
        """

        # Pairwise distance matrix
        # Shape: [B, N, N]
        d = distance(
            torch.stack(
                [
                    coords.unsqueeze(2).repeat(1, 1, coords.shape[1], 1),
                    coords.unsqueeze(1).repeat(1, coords.shape[1], 1, 1),
                ],
                dim=-2,
            )
        )

        # Distane bins
        # Shape: [n_bin]
        v = torch.arange(0, self.template_dist_n_bin).to(coords.device)
        v = self.template_dist_min + v * self.template_dist_step

        # Reshaped distance bins
        # Shape: [1, 1, 1, n_bin]
        v_reshaped = v.view(*((1,) * len(d.shape) + (len(v),)))

        # Pairwise distance bin matrix
        # Shape: [B, N, N]
        b = torch.argmin(torch.abs(d.unsqueeze(-1) - v_reshaped), dim=-1)

        # Pairwise distance bin encoding
        # Shape: [B, N, N, n_bin]
        oh = nn.functional.one_hot(b, num_classes=len(v)).float()

        # Pairwise mask
        # Shape: [B, N, N]
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        return oh * pair_mask.unsqueeze(-1)

    def _encode_orientations(self, rots, mask):
        """
        Encode pairwise relative orientations for a sequence of frames.

        Args:
                rots:
                        [B, N, 3, 3] A sequence of orientations.
                mask:
                        [B, N] Mask to indicate which orientation is masked.

        Returns:
                [B, N, N, 4] Masked pairwise relative orientation encoding
                (in terms of quaternions).
        """
        # Pairwise rotation matrix
        # Shape: [B, N, N, 3, 3]
        r = torch.matmul(rots.unsqueeze(1), rots.unsqueeze(2))

        # Pairwise quaternion
        # Shape: [B, N, N, 4]
        q = rot_to_quat(r)

        # Pairwise mask
        # Shape: [B, N, N]
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        return q * pair_mask.unsqueeze(-1)
