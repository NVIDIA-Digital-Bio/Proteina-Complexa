import torch
from openfold.np.residue_constants import RESTYPE_ATOM37_MASK

from proteinfoundation.nn.feature_factory.concat_feature_factory import ConcatFeaturesFactory
from proteinfoundation.nn.feature_factory.concat_pair_feature_factory import ConcatPairFeaturesFactory
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition, MultiheadCrossAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.protein_transformer import PairReprBuilder


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(dtype=torch.bool, device=device)  # [21, 37]


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.

    Now supports concat features for motif and/or target conditioning.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_tri_attn = kwargs.get("use_tri_attn", False)
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # Concat features for motif and/or target
        concat_config = kwargs.get("concat_features", {})
        self.use_concat = (
            concat_config.get("enable_motif", False)
            or concat_config.get("enable_target", False)
            or concat_config.get("enable_ligand", False)
        )

        if self.use_concat:
            self.concat_factory = ConcatFeaturesFactory(
                enable_motif=concat_config.get("enable_motif", False),
                enable_target=concat_config.get("enable_target", False),
                enable_ligand=concat_config.get("enable_ligand", False),
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                **kwargs,
            )

            # Check for advanced pair representation mode
            self.use_advanced_pair = (
                (concat_config.get("enable_motif", False) and concat_config.get("motif_pair_features", False))
                or (concat_config.get("enable_target", False) and concat_config.get("target_pair_features", False))
                or (concat_config.get("enable_ligand", False) and concat_config.get("ligand_pair_features", False))
            )

            if self.use_advanced_pair:
                # Initialize pair features factory for extended pair representations
                self.concat_pair_factory = ConcatPairFeaturesFactory(
                    enable_motif=concat_config.get("enable_motif", False),
                    enable_target=concat_config.get("enable_target", False),
                    enable_ligand=concat_config.get("enable_ligand", False),
                    **kwargs,
                )
                from loguru import logger

                logger.info("Enabled advanced pair representation with cross-sequence features")
        else:
            self.concat_factory = None
            self.use_advanced_pair = False

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        self.use_target_cross_attn = kwargs.get("use_target_cross_attn", False)
        if self.use_target_cross_attn:
            assert not self.use_concat, "use_target_cross_attn requires use_concat to be False"
            self.target_factory = FeatureFactory(
                feats=kwargs["feats_target"],
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                mode="target",
                **kwargs,
            )
            self.target2binder_cross_attention_layer = torch.nn.ModuleList(
                [
                    MultiheadCrossAttnAndTransition(
                        dim_token_a=kwargs["token_dim"],
                        dim_token_b=kwargs["token_dim"],
                        nheads=kwargs["nheads"],
                        dim_cond=kwargs["dim_cond"],
                        residual_mha=True,
                        residual_transition=True,
                        use_qkln=self.use_qkln,
                    )
                    for _ in range(self.nlayers)
                ]
            )

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            use_tri_attn=self.use_tri_attn,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

        # self.linear_out = torch.nn.Sequential(
        #     torch.nn.LayerNorm(self.token_dim),
        #     torch.nn.Linear(self.token_dim, kwargs["latent_dim"] + 3, bias=False),
        # )

    # @torch.compile
    def forward(self, input: dict) -> dict[str, dict[str, torch.Tensor]]:
        """
        Runs the network.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        mask = input["mask"]  # [b, n] boolean
        orig_mask = mask.clone()  # [b, n] boolean

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Initial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]
        # Store original dimensions
        b, n_orig, _ = seqs.shape
        # Extend sequence representation with concat features if available
        if self.use_concat:
            seqs, mask = self.concat_factory(
                input, seqs, mask
            )  # [b, n_extended, token_dim], [b, n_extended], [b, n_extended]
            n_extended = seqs.shape[1]
            n_concat = n_extended - n_orig

            # Extend conditioning with zeros for concat features
            if n_concat > 0:
                zero_cond = torch.zeros(b, n_concat, c.shape[-1], device=seqs.device)
                c = torch.cat([c, zero_cond], dim=1)  # [b, n_extended, dim_cond]
        else:
            n_extended = n_orig
            n_concat = 0

        # Compute pair representation
        if self.use_concat and self.use_advanced_pair:  # and n_concat > 0:
            # Advanced mode - compute pair features for extended sequence
            pair_rep = self.pair_repr_builder(input)  # [b, n_orig, n_orig, pair_dim]
            pair_rep = self.concat_pair_factory(input, pair_rep, orig_mask)  # [b, n_extended, n_extended, pair_dim]
        else:
            # Simple mode - compute pair representation for original sequence only
            pair_rep = self.pair_repr_builder(input)  # [b, n_orig, n_orig, pair_dim]

            # Extend pair representation with zeros if we have concat features
            if n_concat > 0:
                dim_pair = pair_rep.shape[-1]

                # Pad first dimension: [b, n_orig, n_orig, d] -> [b, n_extended, n_orig, d]
                zero_pad_1 = torch.zeros(b, n_concat, n_orig, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_1], dim=1)

                # Pad second dimension: [b, n_extended, n_orig, d] -> [b, n_extended, n_extended, d]
                zero_pad_2 = torch.zeros(b, n_extended, n_concat, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_2], dim=2)

        if self.use_target_cross_attn:
            target_rep = self.target_factory(input)  # [b, n_target, token_dim]
            target_mask = input["seq_target_mask"]
            target_rep = target_rep * target_mask[..., None]  # [b, n_target, token_dim]

        # if seqs.shape[1] != c.shape[1]: pdb has weird inter padding that we do not handle
        # Run trunk
        for i in range(self.nlayers):
            if self.use_target_cross_attn:
                seqs = self.target2binder_cross_attention_layer[i](
                    seqs, target_rep, c, mask, target_mask
                )  # [b, n_extended, token_dim], [b, n_target, token_dim]

            # try:
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)  # [b, n_extended, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n_extended, n_extended, pair_dim]

        # Get outputs
        local_latents_out = self.local_latents_linear(seqs) * mask[..., None]  # [b, n_extended, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n_extended, 3]

        # Trim back to original sequence length (remove concat features) if we extended
        if n_concat > 0:
            local_latents_out = local_latents_out[:, :n_orig, :] * orig_mask[:, :, None]

            ca_nm_out = ca_nm_out[:, :n_orig, :] * orig_mask[:, :, None]

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {self.output_param["local_latents"]: local_latents_out}
        return nn_out
