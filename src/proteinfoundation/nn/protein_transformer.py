import torch
from openfold.model.structure_module import InvariantPointAttention
from openfold.utils.rigid_utils import Rigid

from proteinfoundation.nn.feature_factory.concat_feature_factory import ConcatFeaturesFactory
from proteinfoundation.nn.feature_factory.concat_pair_feature_factory import ConcatPairFeaturesFactory

# from proteinfoundation.nn.feature_factory import FeatureFactory, ConcatFeaturesFactory, ConcatPairFeaturesFactory
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNorm
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition


class SequenceToCoordinates(torch.nn.Module):
    """Updates coordinates from sequence."""

    def __init__(self, dim_token):
        super().__init__()
        self.ln = torch.nn.LayerNorm(dim_token)
        self.linear = torch.nn.Linear(dim_token, 3)

    def forward(self, seqs, x_t, mask):
        """
        Args:
            seqs: Input sequence representation, shape [b, n, token_dim]
            x_t: Input coordinates, shape [b, n, 3]
            mask: binary mask, shape [b, n]

        Returns:
            Updated (masked) coordinates
        """
        delta_x = self.linear(self.ln(seqs))  # [b, n, 3]
        x_t = x_t + delta_x
        return x_t * mask[..., None]


class CoordinatesToSequenceLinear(torch.nn.Module):
    """Updates sequence using linear layer from coordiates."""

    def __init__(self, dim_token):
        super().__init__()
        self.linear = torch.nn.Linear(3, dim_token, bias=False)
        self.ln = torch.nn.LayerNorm(dim_token)

    def forward(self, seqs, x_t, mask):
        """
        Args:
            seqs: Input sequence representation, shape [b, n, token_dim]
            x_t: Input coordinates, shape [b, n, 3]
            mask: binary mask, shape [b, n]

        Returns:
            Updated (masked) sequence
        """
        delta_seq = self.ln(self.linear(x_t))
        seqs = seqs + delta_seq
        return seqs * mask[..., None]


class CoordinatesToSequenceIPA(torch.nn.Module):
    """Updates sequence using IPA from coordiates (and identity rotations)."""

    def __init__(self, dim_token, dim_pair, c_hidden=16, nheads=8, no_qk_points=8, no_v_points=12):
        super().__init__()
        self.ipa = InvariantPointAttention(
            c_s=dim_token,
            c_z=dim_pair,
            c_hidden=c_hidden,
            no_heads=nheads,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
        )

    # @torch.compile
    def forward(self, seqs, pair_rep, x_t, mask):
        """
        Args:
            seqs: Input sequence representation, shape [b, n, token_dim]
            pair_rep: Pair represnetation, shape [b, n, n, pair_dim]
            x_t: Input coordinates, shape [b, n, 3]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence, shape [b, n, dim_token]
        """
        x_t = x_t * mask[..., None]
        r = Rigid(trans=x_t, rots=None)  # Rotations default to identity
        delta_seqs = self.ipa(s=seqs, z=pair_rep, r=r, mask=mask * 1.0)
        seqs = seqs + delta_seqs
        return seqs * mask[..., None]  # [b, n, token_dim]


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, **kwargs):
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = FeatureFactory(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(dim=dim_feats_out, dim_cond=dim_cond_pair)

    def forward(self, batch_nn):
        mask = batch_nn["mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
            repr = self.adaln(repr, cond, pair_mask)
        return repr


class ProteinTransformerAF3(torch.nn.Module):
    """
    Final neural network mimicking the one used in AF3 diffusion. It consists of:

    (1) Input preparation
    (1.a) Initial sequence representation from features
    (1.b) Embed coordaintes and add to initial sequence representation
    (1.c) Conditioning variables from features

    (2) Main trunk
    (2.a) A sequence of layers similar to algorithm 23 of AF3 (multi head attn, transition) using adaptive layer norm
    and adaptive output scaling (also from adaptive layer norm paper)

    (3) Recovering 3D coordinates
    (3.a) A layer that takes as input tokens and produces coordinates
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        self.use_attn_pair_bias = kwargs["use_attn_pair_bias"]
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_coors_on_the_fly = kwargs.get("update_coors_on_the_fly", False)  # For backward compat
        self.update_seq_with_coors = kwargs.get(
            "update_seq_with_coors"
        )  # For backward compat, None, linear or IPA, only used if coors on the fly
        self.update_pair_repr = kwargs.get("update_pair_repr", False)  # For backward compat
        self.update_pair_repr_every_n = kwargs.get("update_pair_repr_every_n", 2)  # For backward compat
        self.use_tri_mult = kwargs.get("use_tri_mult", False)  # For backward compat
        self.use_tri_attn = kwargs.get("use_tri_attn", False)  # For backward compat
        self.feats_pair_cond = kwargs.get("feats_pair_cond", [])  # For backward compat
        self.use_qkln = kwargs.get("use_qkln", False)  # For backward compat
        self.num_buckets_predict_pair = kwargs.get("num_buckets_predict_pair")  # For backward compat
        self.output_param = kwargs["output_parameterization"]

        # Registers
        self.num_registers = kwargs.get("num_registers")  # For backward compat
        if self.num_registers is None or self.num_registers <= 0:
            self.num_registers = 0
            self.registers = None
        else:
            self.num_registers = int(self.num_registers)
            self.registers = torch.nn.Parameter(torch.randn(self.num_registers, self.token_dim) / 20.0)

        # To encode corrupted 3d positions
        # self.linear_3d_embed = torch.nn.Linear(3, kwargs["token_dim"], bias=False)
        # removed, now part of features
        # It is equivalent, it adds some more features that are fed through some linear layer...
        # We were doing the same in a different place

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_init_seq"],
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
        self.use_concat = concat_config.get("enable_motif", False) or concat_config.get("enable_target", False)

        if self.use_concat:
            self.concat_factory = ConcatFeaturesFactory(
                enable_motif=concat_config.get("enable_motif", False),
                enable_target=concat_config.get("enable_target", False),
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                **kwargs,
            )

            # Check for advanced pair representation mode
            self.use_advanced_pair = (
                concat_config.get("enable_motif", False) and concat_config.get("motif_pair_features", False)
            ) or (concat_config.get("enable_target", False) and concat_config.get("target_pair_features", False))

            if self.use_advanced_pair:
                # Initialize pair features factory for extended pair representations
                self.concat_pair_factory = ConcatPairFeaturesFactory(
                    enable_motif=concat_config.get("enable_motif", False),
                    enable_target=concat_config.get("enable_target", False),
                    dim_pair_out=kwargs["pair_repr_dim"],
                    **kwargs,
                )
                from loguru import logger

                logger.info("Enabled advanced pair representation with cross-sequence features")
        else:
            self.concat_factory = None
            self.use_advanced_pair = False

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        if self.use_attn_pair_bias:
            self.pair_repr_builder = PairReprBuilder(
                feats_repr=kwargs["feats_pair_repr"],
                feats_cond=kwargs["feats_pair_cond"],
                dim_feats_out=kwargs["pair_repr_dim"],
                dim_cond_pair=kwargs["dim_cond"],
                **kwargs,
            )
        else:
            # If no pair bias no point in having a pair representation
            self.update_pair_repr = False

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=kwargs["token_dim"],
                    dim_pair=kwargs["pair_repr_dim"],
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=kwargs["residual_mha"],
                    residual_transition=kwargs["residual_transition"],
                    parallel_mha_transition=kwargs["parallel_mha_transition"],
                    use_attn_pair_bias=kwargs["use_attn_pair_bias"],
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # # Coors on the fly - this will need more care when dealing with different data modalities
        # if self.update_coors_on_the_fly:
        #     self.seq_to_coors = torch.nn.ModuleList([
        #         SequenceToCoordinates(dim_token=kwargs["token_dim"]) for _ in range(self.nlayers)
        #     ])

        #     # Coors to seq
        #     if self.update_seq_with_coors == "linear" or self.update_seq_with_coors == "linearipa":
        #         self.coors_to_seq_linear = torch.nn.ModuleList([
        #             CoordinatesToSequenceLinear(dim_token=kwargs["token_dim"]) for _ in range(self.nlayers)
        #         ])

        #     if self.update_seq_with_coors == "ipa" or self.update_seq_with_coors == "linearipa":
        #         self.coors_to_seq_ipa = torch.nn.ModuleList([
        #             CoordinatesToSequenceIPA(dim_token=kwargs["token_dim"], dim_pair=kwargs["pair_repr_dim"]) for _ in range(self.nlayers)
        #         ])

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            tri_mult_c=kwargs.get("tri_mult_c", kwargs["pair_repr_dim"]),
                            dropout=kwargs.get("pair_update_dropout", 0),
                            use_tri_attn=kwargs.get("use_tri_attn", False),
                            use_checkpointing=kwargs.get("use_checkpointing", True),
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )
            # For distogram pair prediction
            if self.num_buckets_predict_pair is not None:
                self.pair_head_prediction = torch.nn.Sequential(
                    torch.nn.LayerNorm(kwargs["pair_repr_dim"]),
                    torch.nn.Linear(kwargs["pair_repr_dim"], self.num_buckets_predict_pair),
                )

        self.coors_3d_decoder = torch.nn.Sequential(
            torch.nn.LayerNorm(kwargs["token_dim"]),
            torch.nn.Linear(kwargs["token_dim"], 3, bias=False),
        )

    def _extend_w_registers(self, seqs, pair, mask, cond_seq):
        """
        Extends the sequence representation, pair representation, mask and indices with registers.

        Args:
            - seqs: sequence representation, shape [b, n, dim_token]
            - pair: pair representation, shape [b, n, n, dim_pair]
            - mask: binary mask, shape [b, n]
            - cond_seq: tensor of shape [b, n, dim_cond]

        Returns:
            All elements above extended with registers / zeros.
        """
        if self.num_registers == 0:
            return seqs, pair, mask, cond_seq  # Do nothing

        b, n, _ = seqs.shape
        dim_pair = pair.shape[-1]
        r = self.num_registers
        dim_cond = cond_seq.shape[-1]

        # Concatenate registers to sequence
        reg_expanded = self.registers[None, :, :]  # [1, r, dim_token]
        reg_expanded = reg_expanded.expand(b, -1, -1)  # [b, r, dim_token]
        seqs = torch.cat([reg_expanded, seqs], dim=1)  # [b, r+n, dim_token]

        # Extend mask
        true_tensor = torch.ones(b, r, dtype=torch.bool, device=seqs.device)  # [b, r]
        mask = torch.cat([true_tensor, mask], dim=1)  # [b, r+n]

        # Extend pair representation with zeros; pair has shape [b, n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        # [b, n, n, pair_dim] -> [b, r+n, n, pair_dim]
        zero_pad_top = torch.zeros(b, r, n, dim_pair, device=seqs.device)  # [b, r, n, dim_pair]
        pair = torch.cat([zero_pad_top, pair], dim=1)  # [b, r+n, n, dim_pair]
        # [b, r+n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        zero_pad_left = torch.zeros(b, r + n, r, dim_pair, device=seqs.device)  # [b, r+n, r, dim_pair]
        pair = torch.cat([zero_pad_left, pair], dim=2)  # [b, r+n, r+n, dim+pair]

        # Extend cond
        zero_tensor = torch.zeros(b, r, dim_cond, device=seqs.device)  # [b, r, dim_cond]
        cond_seq = torch.cat([zero_tensor, cond_seq], dim=1)  # [b, r+n, dim_cond]

        return seqs, pair, mask, cond_seq

    def _undo_registers(self, seqs, pair, mask):
        """
        Undoes register padding.

        Args:
            - seqs: sequence representation, shape [b, r+n, dim_token]
            - pair: pair representation, shape [b, r+n, r+n, dim_pair]
            - mask: binary mask, shape [b, r+n]

        Returns:
            All three elements with the register padding removed.
        """
        if self.num_registers == 0:
            return seqs, pair, mask
        r = self.num_registers
        return seqs[:, r:, :], pair[:, r:, r:, :], mask[:, r:]

    # @torch.compile
    def forward(self, batch_nn: dict[str, torch.Tensor]):
        """
        Runs the network.

        Args:
            batch_nn: dictionary with keys
                - "x_t": tensor of shape [b, n, 3]
                - "t": tensor of shape [b]
                - "mask": binary tensor of shape [b, n]
                - "x_sc" (optional): tensor of shape [b, n, 3]
                - "cath_code" (optional): list of cath codes [b, ?]
                - And potentially others... All in the data batch.

        Returns:
            Predicted clean coordinates, shape [b, n, 3].
        """
        mask = batch_nn["mask"]
        orig_mask = mask.clone()

        # Conditioning variables
        c = self.cond_factory(batch_nn)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(batch_nn)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        # Pair representation
        pair_rep = None
        if self.use_attn_pair_bias:
            pair_rep = self.pair_repr_builder(batch_nn)  # [b, n, n, pair_dim]

        # Store original dimensions
        b, n_orig, _ = seqs.shape
        # Extend sequence representation with concat features if available
        if self.use_concat:
            seqs, mask = self.concat_factory(
                batch_nn, seqs, mask
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
            pair_rep = self.concat_pair_factory(batch_nn, pair_rep, orig_mask)  # [b, n_extended, n_extended, pair_dim]
        else:
            # Simple mode - compute pair representation for original sequence only
            # Extend pair representation with zeros if we have concat features
            if n_concat > 0:
                dim_pair = pair_rep.shape[-1]

                # Pad first dimension: [b, n_orig, n_orig, d] -> [b, n_extended, n_orig, d]
                zero_pad_1 = torch.zeros(b, n_concat, n_orig, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_1], dim=1)

                # Pad second dimension: [b, n_extended, n_orig, d] -> [b, n_extended, n_extended, d]
                zero_pad_2 = torch.zeros(b, n_extended, n_concat, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_2], dim=2)

        # Apply registers
        seqs, pair_rep, mask, c = self._extend_w_registers(seqs, pair_rep, mask, c)

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)  # [b, n, token_dim]

            # # Coors on the fly
            # if self.update_coors_on_the_fly:
            #     coors_3d = self.seq_to_coors[i](seqs, coors_3d, mask)

            #     # Update sequence with coordinates
            #     if self.update_seq_with_coors == "linear" or self.update_seq_with_coors == "linearipa":
            #         seqs = self.coors_to_seq_linear[i](seqs, coors_3d, mask)

            #     if self.update_seq_with_coors == "ipa" or self.update_seq_with_coors == "linearipa":
            #         seqs = self.coors_to_seq_ipa[i](seqs, pair_rep, coors_3d, mask)

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)  # [b, n, n, pair_dim]

        # Undo registers
        seqs, pair_rep, mask = self._undo_registers(seqs, pair_rep, mask)

        # Get final coordinates
        final_coors = self.coors_3d_decoder(seqs) * mask[..., None]  # [b, n, 3]
        # if self.update_coors_on_the_fly:
        #     final_coors = final_coors * 0.0 + coors_3d  # Ignore coordinates from final seuqence [b, n, 3]
        nn_out = {}
        if self.update_pair_repr and self.num_buckets_predict_pair is not None:
            pair_pred = self.pair_head_prediction(pair_rep)
            final_coors = (
                final_coors + torch.mean(pair_pred) * 0.0
            )  # Does not affect loss but pytorch does not complain for unused params
            final_coors = final_coors * mask[..., None]
            nn_out["pair_pred_bb_ca"] = pair_pred

        # Trim back to original sequence length (remove concat features) if we extended
        if n_concat > 0:
            final_coors = final_coors[:, :n_orig, :] * orig_mask[:, :, None]
        # This will have to change when dealing with more
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: final_coors}
        return nn_out

    def nflops_computer(self, b, n):
        """Approximately how many flops used, for the main transformer layers. Protein length n.

        Final equation per layer per sample:
            12 * n * dim^2 + (dim + 1) * n^2 + 4 * n * 4 + [pair_dim * n^2]

        where the term in brackets [...] is used only with pair biased attn.

        The final number is obtained by multiplying by batch size and number of layers.

        Decomposing the number:
            - MHA:
                - Computing QKV with linear layers: 3n * dim^2
                - Computing attn logits: dim * n^2
                - Softmax: n^2
                - Attention: dim * n^2

            - Transition (feed forward, single hidden layer with expansion factor 4)
                - 1st linear layer: 4n * dim^2
                - activations: 4n * dim
                - 2nd linear layer: 4n * dim^2

            - Pair bias
                - (dim + 1) * n^2
        """
        # MHA
        # QKV
        nflops = 3 * n * self.token_dim**2
        # Attn logits
        nflops = nflops + self.token_dim * n**2
        # softmax
        nflops = nflops + n**2
        # attn
        nflops = nflops + self.token_dim * n**2
        # Some linear layer I might be missing

        # FF in transition, assumes ff has expansion_factor=4
        # Linear
        nflops = nflops + n * self.token_dim * 4 * self.token_dim
        # activation
        nflops = nflops + n * 4 * self.token_dim
        # Linear
        nflops = nflops + n * self.token_dim * 4 * self.token_dim

        # Pair bias
        if self.use_attn_pair_bias:
            # Computing and adding bias to attn logits
            nflops = nflops + (self.pair_repr_dim + 1) * n**2

        # Ignoring updating pair representation when used

        return b * nflops * self.nlayers  # Adjust for batch size and layers
