import torch
from jaxtyping import Float
from torch.utils.checkpoint import checkpoint

from proteinfoundation.nn.feature_factory import FeatureFactory, get_time_embedding
from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNormIdentical
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.protein_transformer import PairReprBuilder


class ProteinTransformerAF3Int(torch.nn.Module):
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
        self.feats_pair_cond = kwargs.get("feats_pair_cond", [])  # For backward compat
        self.use_qkln = kwargs.get("use_qkln", False)  # For backward compat
        self.num_buckets_predict_pair = kwargs.get("num_buckets_predict_pair")  # For backward compat
        self.output_param = kwargs["output_parameterization"]

        self.t_emb_dim_diff_rec = 128
        self.use_dr_seq = kwargs.get("use_dr_seq", False)
        self.use_dr_pair = kwargs.get("use_dr_pair", False)

        # Registers
        self.num_registers = kwargs.get("num_registers")  # For backward compat
        if self.num_registers is None or self.num_registers <= 0:
            self.num_registers = 0
            self.registers = None
        else:
            self.num_registers = int(self.num_registers)
            self.registers = torch.nn.Parameter(torch.randn(self.num_registers, self.token_dim) / 20.0)

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

        # Linear heads for each intermediate layer output
        self.linear_int = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(self.token_dim),
                    torch.nn.Linear(self.token_dim, 3, bias=False),
                )
                for _ in range(self.nlayers - 1)
            ]
        )

        # Linear heads for each intermediate layer recycling to sequence
        self.linear_int_dr_seq = torch.nn.ModuleList(
            [torch.nn.Linear(3, self.token_dim, bias=False) for _ in range(self.nlayers - 1)]
        )
        self.adaln_seq_dr = torch.nn.ModuleList(
            [
                AdaptiveLayerNormIdentical(dim=self.token_dim, dim_cond=self.t_emb_dim_diff_rec, mode="single")
                for _ in range(self.nlayers - 1)
            ]
        )

        # Linear heads for each intermediate layer recycling to pair
        self.linear_int_dr_pair = torch.nn.ModuleList(
            [torch.nn.Linear(39, kwargs["pair_repr_dim"], bias=False) for _ in range(self.nlayers - 1)]  # 39 bins
        )
        self.adaln_pair_dr = torch.nn.ModuleList(
            [
                AdaptiveLayerNormIdentical(
                    dim=kwargs["pair_repr_dim"],
                    dim_cond=self.t_emb_dim_diff_rec,
                    mode="pair",
                )
                for _ in range(self.nlayers - 1)
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

    def _get_recycling_seq_n_pair(
        self,
        input: dict,
        bb_ca_int: Float[torch.Tensor, "b n 3"],
        layer_num: int,
        t_bb_ca: Float[torch.Tensor, "b"],
    ) -> tuple[Float[torch.Tensor, "b n d"], Float[torch.Tensor, "b n n d"]]:
        """
        Produces recycling representations for the sequence and pair representations.
        It uses a linear layer followed by an adaptive layer norm.

        NOTE: This assumes the linear interpolant to go between v and clean sample prediction.

        Args:
            input: Dictionary with input
            bb_ca_int: intermediate predicted backbone coors, in nm, shape [b, n, 3]
            layer_num: layer index
            t_bb_ca: target backbone diffusion time, shape [b]

        Returns:
            rec_seq: recycling rep for sequence, shape [b, n, token_dim]
            rec_pair: recycling rep for pair, shape [b, n, n, pair_dim]
        """

        def _rbf_pair_dists(
            pair_dists_nm: Float[torch.Tensor, "b n n"],
        ) -> Float[torch.Tensor, "b n n d"]:
            """
            Radial basis function encoding applied to pair distances in nm.

            Uses 39 (d=39) Gaussian kernels centered at linspace(0.1, 5) (in nm).

            Args:
                pair_dists_nm: pairwise distances in nm, shape [b, n, n]

            Returns:
                Radial basis function applied to pair distances, shape [b, n, n, d]
            """
            centers = torch.linspace(0.1, 5, 39, device=pair_dists_nm.device)  # [39]
            centers = centers[None, None, None, :]  # [1, 1, 1, 39]
            pair_dists = pair_dists_nm[..., None]  # [b, n, n, 1]
            feats = torch.exp(-((pair_dists - centers) ** 2) / 0.1)  # [b, n, n, 39]
            return feats

        # Go to clean sample prediction for bb_ca if needed
        if self.output_param["bb_ca"] == "v":
            t_bb_ca_brc = t_bb_ca[..., None, None]  # [b, 1, 1]
            bb_ca_int = input["x_t"]["bb_ca"] + bb_ca_int * (1.0 - t_bb_ca_brc)

        # Go to clean sample prediction for local_latents if needed

        t_emb_cond = get_time_embedding(t_bb_ca, self.t_emb_dim_diff_rec)  # [b, t_emb_dim]

        # Pairwise distances in nm
        pair_mask = input["mask"][:, :, None] * input["mask"][:, None, :]  # [b, n, n]
        pair_dist_nm = torch.norm(bb_ca_int[:, :, None, :] - bb_ca_int[:, None, :, :], dim=-1) * pair_mask  # [b, n, n]
        rbf_pair_dist = _rbf_pair_dists(pair_dist_nm) * pair_mask[..., None]  # [b, n, n, 39]

        # Apply linear layers and adaln to clean sample predictions
        linear_out_seq = self.linear_int_dr_seq[layer_num](bb_ca_int)  # [b, n, token_dim]
        seq_rec = self.adaln_seq_dr[layer_num](linear_out_seq, t_emb_cond, input["mask"])  # [b, n, token_dim]

        # Apply linear layer and adaln to pair repr
        linear_out_pair = self.linear_int_dr_pair[layer_num](rbf_pair_dist)  # [b, n, n, token_dim]
        pair_rec = self.adaln_pair_dr[layer_num](linear_out_pair, t_emb_cond, pair_mask)  # [b, n, n, token_dim]

        return seq_rec, pair_rec

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

        # Apply registers
        seqs, pair_rep, mask, c = self._extend_w_registers(seqs, pair_rep, mask, c)

        # Run trunk
        bb_ca_int = []
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)  # [b, n, token_dim]

            if i < self.nlayers - 1:
                x_out = self.linear_int[i](seqs) * mask[..., None]
                bb_ca_int.append(x_out[..., :3])

                # Update sequence and pair representations with differentiable recycling.
                # Need to keep into account the output parameterization for each data mode (x_1 or v).
                # Could otherwise always do clean sample prediction for intermediate layers, but that may
                # be different to what we predict at the end, if we're using the v parameterization.
                # rec_seq, rec_pair = self._get_recycling_seq_n_pair(
                #     input=batch_nn,
                #     bb_ca_int=bb_ca_int[-1],
                #     layer_num=i,
                #     t_bb_ca=batch_nn["t"]["bb_ca"],
                # )
                rec_seq, rec_pair = checkpoint(
                    self._get_recycling_seq_n_pair,
                    *(batch_nn, bb_ca_int[-1], i, batch_nn["t"]["bb_ca"]),
                )

                # Set coefficients to zero if unused
                f_dr_seq = 1.0 if self.use_dr_seq else 0.0
                f_dr_pair = 1.0 if self.use_dr_pair else 0.0

                # Apply differentiable recycling to sequence and pair
                seqs = seqs + rec_seq * f_dr_seq
                pair_rep = pair_rep + rec_pair * f_dr_pair

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)  # [b, n, n, pair_dim]

        # Undo registers
        seqs, pair_rep, mask = self._undo_registers(seqs, pair_rep, mask)

        # Get final coordinates
        final_coors = self.coors_3d_decoder(seqs) * mask[..., None]  # [b, n, 3]

        nn_out = {}
        if self.update_pair_repr and self.num_buckets_predict_pair is not None:
            pair_pred = self.pair_head_prediction(pair_rep)

            final_coors = (
                final_coors + torch.mean(pair_pred) * 0.0
            )  # Does not affect loss but pytorch does not complain for unused params
            final_coors = final_coors * mask[..., None]

            # If we actually end up using this we'll have to change what the NN returns here... And use it in the loss computation
            nn_out["pair_pred"] = pair_pred

        # This will have to change when dealing with more
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: final_coors}
        nn_out["bb_ca_int"] = {self.output_param["bb_ca"]: bb_ca_int}  # List of tensors of shape [b, n, 3]
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
