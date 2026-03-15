# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import openfold.utils.rigid_utils as ru
import torch
from openfold.np.residue_constants import restype_num
from torch import nn

# from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory
from proteinfoundation.nn.genie2_modules.pair_feature_net import PairFeatureNet
from proteinfoundation.nn.genie2_modules.pair_transform_net import PairTransformNet
from proteinfoundation.nn.genie2_modules.structure_net import StructureNet
from proteinfoundation.nn.genie2_modules.utils.geo_utils import compute_frenet_frames
from proteinfoundation.nn.protein_transformer import PairReprBuilder


def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return ru.Rigid(rots=rots, trans=trans)


class Genie2Denoiser(nn.Module):
    """
    SE(3)-Equivariant Denoiser.

    Given a noisy structure at timestep t, the model predicts the noise that
    is added at timestep t. For further details, please refer to our Genie
    paper at https://arxiv.org/abs/2301.12485 and our Genie 2 paper at
    https://arxiv.org/abs/2405.15489.
    """

    def __init__(
        self,
        name,
        c_s,
        c_p,
        # n_timestep,
        rescale,
        # Parameters for single feature network
        c_pos_emb,
        c_chain_emb,
        c_timestep_emb,
        # max_n_res,
        # max_n_chain,
        # Parameters for pair feature network
        relpos_k,
        template_dist_min,
        template_dist_step,
        template_dist_n_bin,
        # Parameters for pair transform network
        n_pair_transform_layer,
        include_mul_update,
        include_tri_att,
        c_hidden_mul,
        c_hidden_tri_att,
        n_head_tri,
        tri_dropout,
        pair_transition_n,
        # Parameters for structure network
        n_structure_layer,
        n_structure_block,
        c_hidden_ipa,
        n_head_ipa,
        n_qk_point,
        n_v_point,
        ipa_dropout,
        n_structure_transition_layer,
        structure_transition_dropout,
        max_n_res,
        n_timestep,
        **kwargs,
    ):
        """
        Args:
                c_s:
                        Dimension of per-residue (single) representation.
                c_p:
                        Dimension of paired residue-residue (pair) representation.
                n_timestep:
                        Total number of diffusion timesteps.
                rescale:
                        Rescale factor for coordinate space.
                *:
                        Module-specfic parameters. Refer to its corresponding module
                        for further details.
        """
        super().__init__()
        self.rescale = rescale
        self.output_param = kwargs["output_parameterization"]
        #! Below are used for Genie2 Feature Factory Additions
        kwargs["c_pos_emb"] = c_pos_emb
        kwargs["max_n_res"] = max_n_res
        kwargs["n_timestep"] = n_timestep
        self.single_feature_net = FeatureFactory(
            feats=kwargs["feats_init_seq"],
            dim_feats_out=c_s,
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.pair_feature_net = PairFeatureNet(
            c_s=c_s,
            c_p=c_p,
            relpos_k=relpos_k,
            template_dist_min=template_dist_min,
            template_dist_step=template_dist_step,
            template_dist_n_bin=template_dist_n_bin,
        )

        if kwargs["use_pair_feature_factory"]:
            self.pair_repr_builder = PairReprBuilder(
                feats_repr=kwargs["feats_pair_repr"],
                feats_cond=kwargs["feats_pair_cond"],
                dim_feats_out=c_p,
                dim_cond_pair=kwargs["dim_cond"],
                **kwargs,
            )
        else:
            self.pair_repr_builder = None

        self.pair_transform_net = (
            PairTransformNet(
                c_p=c_p,
                n_pair_transform_layer=n_pair_transform_layer,
                include_mul_update=include_mul_update,
                include_tri_att=include_tri_att,
                c_hidden_mul=c_hidden_mul,
                c_hidden_tri_att=c_hidden_tri_att,
                n_head_tri=n_head_tri,
                tri_dropout=tri_dropout,
                pair_transition_n=pair_transition_n,
            )
            if n_pair_transform_layer > 0
            else None
        )

        self.structure_net = StructureNet(
            c_s=c_s,
            c_p=c_p,
            n_structure_layer=n_structure_layer,
            n_structure_block=n_structure_block,
            c_hidden_ipa=c_hidden_ipa,
            n_head_ipa=n_head_ipa,
            n_qk_point=n_qk_point,
            n_v_point=n_v_point,
            ipa_dropout=ipa_dropout,
            n_structure_transition_layer=n_structure_transition_layer,
            structure_transition_dropout=structure_transition_dropout,
        )

        if "res_type" in self.output_param:
            kwargs["token_dim"]
            self.aatype_pred_net = nn.Sequential(
                nn.Linear(c_s, c_s),
                nn.ReLU(),
                nn.Linear(c_s, c_s),
                nn.ReLU(),
                nn.Linear(c_s, restype_num),
            )

    def forward(self, batch_nn: dict[str, torch.Tensor]):
        """
        Runs the network.

        Args:
                batch_nn: dictionary with keys
                        - "x_t": tensor of shape [b, n, 3]
                        - "t": tensor of shape [b]
                        - "mask": binary tensor of shape [b, n]
                        - "x_sc" (optional): tensor of shape [b, n, 3]
                        - And potentially others... All in the data batch.

        Returns:
                Predicted clean coordinates, shape [b, n, 3].
        """
        trans_t = batch_nn["x_t"]["bb_ca"]
        batch_nn["t"]["bb_ca"]
        mask = batch_nn["mask"]
        b, num_res, device = mask.shape[0], mask.shape[1], mask.device
        if "chains" in batch_nn:
            chain_index = batch_nn["chains"]
        else:
            chain_index = torch.zeros(mask.shape).to(mask.device)  #! Assume single chain

        # [b, n_res]
        if "residue_pdb_idx" in batch_nn:
            residue_index = batch_nn["residue_pdb_idx"]
            residue_index = residue_index - residue_index[:, 0].unsqueeze(-1) * mask
        else:
            residue_index = torch.arange(num_res, dtype=torch.float32).to(device)[None].repeat([b, 1])

        rots_s = compute_frenet_frames(trans_t, chain_index, mask)
        # ts = T(rots_s, trans_t)
        # Initial rigids
        curr_rigids = create_rigid(
            rots_s,
            trans_t,
        )

        ts = curr_rigids
        curr_rigids = curr_rigids.scale_translation(self.rescale)
        # Predict
        s = self.single_feature_net(batch_nn)
        p = self.pair_feature_net(s, ts, mask, residue_index, chain_index)
        if self.pair_repr_builder is not None:
            p = p + self.pair_repr_builder(batch_nn)
        if self.pair_transform_net is not None:
            p = self.pair_transform_net(p, mask)
        states, curr_rigids, s = self.structure_net(s, p, ts, mask)

        # Descale
        curr_rigids = curr_rigids.scale_translation(1.0 / self.rescale)

        output = curr_rigids.get_trans()
        nn_out = {"bb_ca": {self.output_param["bb_ca"]: output}}
        if "res_type" in self.output_param:
            res_type_logits = self.aatype_pred_net(s) * mask[..., None]  # [b, n, 20]
            nn_out["res_type"] = {self.output_param["res_type"]: res_type_logits}
        return nn_out
