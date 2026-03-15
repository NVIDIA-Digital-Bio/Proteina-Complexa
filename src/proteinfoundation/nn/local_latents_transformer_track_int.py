import torch
from jaxtyping import Float
from openfold.np.residue_constants import RESTYPE_ATOM37_MASK

from proteinfoundation.nn.feature_factory import FeatureFactory, get_time_embedding
from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNormIdentical
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.protein_transformer import PairReprBuilder


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(dtype=torch.bool, device=device)  # [21, 37]


class LocalLatentsTransformerInt(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
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
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]

        self.use_diff_rec = kwargs.get("use_diff_rec", False)
        self.t_emb_dim_diff_rec = 128

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

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

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

        # Linear heads for each intermediate layer output
        self.linear_int = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(self.token_dim),
                    torch.nn.Linear(self.token_dim, 3 + kwargs["latent_dim"], bias=False),
                )
                for _ in range(self.nlayers - 1)
            ]
        )

        # Linear heads for each intermediate layer recycling to sequence
        dim_cond_diff_rec = int(2 * self.t_emb_dim_diff_rec)
        self.linear_int_dr_seq = torch.nn.ModuleList(
            [torch.nn.Linear(3 + kwargs["latent_dim"], self.token_dim, bias=False) for _ in range(self.nlayers - 1)]
        )
        self.adaln_seq_dr = torch.nn.ModuleList(
            [
                AdaptiveLayerNormIdentical(dim=self.token_dim, dim_cond=dim_cond_diff_rec, mode="single")
                for _ in range(self.nlayers - 1)
            ]
        )

        # Linear heads for each intermediate layer recycling to pair
        self.linear_int_dr_pair = torch.nn.ModuleList(
            [torch.nn.Linear(39, kwargs["pair_repr_dim"], bias=False) for _ in range(self.nlayers - 1)]  # 39 bins
        )
        self.adaln_pair_dr = torch.nn.ModuleList(
            [
                AdaptiveLayerNormIdentical(dim=kwargs["pair_repr_dim"], dim_cond=dim_cond_diff_rec, mode="pair")
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

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

    def _get_recycling_seq_n_pair(
        self,
        input: dict,
        bb_ca_int: Float[torch.Tensor, "b n 3"],
        local_latents_int: Float[torch.Tensor, "b n d"],
        layer_num: int,
        t_bb_ca: Float[torch.Tensor, "b"],
        t_local_latents: Float[torch.Tensor, "b"],
    ) -> tuple[Float[torch.Tensor, "b n d"], Float[torch.Tensor, "b n n d"]]:
        """
        Produces recycling representations for the sequence and pair representations.
        It uses a linear layer followed by an adaptive layer norm.

        NOTE: This assumes the linear interpolant to go between v and clean sample prediction.

        Args:
            bb_ca_int: intermediate predicted backbone coors, in nm, shape [b, n, 3]
            local_latents_int: intermediate predicted local latents, shape [b, n, latent_dim]
            layer_num: layer index
            t_bb_ca: target backbone diffusion time, shape [b]
            t_local_latents: target local latents diffusion time, shape [b]

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
        if self.output_param["local_latents"] == "v":
            t_local_latents_brc = t_local_latents[..., None, None]
            local_latents_int = input["x_t"]["local_latents"] + local_latents_int * (1.0 - t_local_latents_brc)

        t_bb_ca_emb = get_time_embedding(t_bb_ca, self.t_emb_dim_diff_rec)  # [b, t_emb_dim]
        t_local_latents_emb = get_time_embedding(t_local_latents, self.t_emb_dim_diff_rec)  # [b, t_emb_dim]
        t_emb_cond = torch.cat([t_bb_ca_emb, t_local_latents_emb], dim=-1)  # [b, 2 * t_emb_dim]

        # Pairwise distances in nm
        pair_dist_nm = torch.norm(bb_ca_int[:, :, None, :] - bb_ca_int[:, None, :, :], dim=-1)  # [b, n, n]
        rbf_pair_dist = _rbf_pair_dists(pair_dist_nm)  # [b, n, n, 39]

        # Apply linear layers and adaln to clean sample predictions
        linear_in_seq = torch.cat([bb_ca_int, local_latents_int], dim=-1)  # [b, n, 3 + latent_dim]
        linear_out_seq = self.linear_int_dr_seq[layer_num](linear_in_seq)  # [b, n, token_dim]
        # seq_rec = linear_out_seq
        seq_rec = self.adaln_seq_dr[layer_num](linear_out_seq, t_emb_cond, input["mask"])  # [b, n, token_dim]

        # Apply linear layer and adaln to pair repr
        pair_mask = input["mask"][:, :, None] * input["mask"][:, None, :]  # [b, n, n]
        linear_out_pair = self.linear_int_dr_pair[layer_num](rbf_pair_dist)  # [b, n, n, token_dim]
        # pair_rec = linear_out_pair
        pair_rec = self.adaln_pair_dr[layer_num](linear_out_pair, t_emb_cond, pair_mask)  # [b, n, n, token_dim]

        return seq_rec, pair_rec

    def forward(self, input: dict) -> dict[str, torch.Tensor]:
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

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_repr_builder(input)  # [b, n, n, pair_dim]

        # Run trunk
        bb_ca_int = []
        local_latents_int = []
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)  # [b, n, token_dim]

            if i < self.nlayers - 1:
                x_out = self.linear_int[i](seqs) * mask[..., None]
                bb_ca_int.append(x_out[..., :3])
                local_latents_int.append(x_out[..., 3:])

                # Update sequence and pair representations with differentiable recycling.
                # Need to keep into account the output parameterization for each data mode (x_1 or v).
                # Could otherwise always do clean sample prediction for intermediate layers, but that may
                # be different to what we predict at the end, if we're using the v parameterization.
                rec_seq, rec_pair = self._get_recycling_seq_n_pair(
                    input=input,
                    bb_ca_int=bb_ca_int[-1],
                    local_latents_int=local_latents_int[-1],
                    layer_num=i,
                    t_bb_ca=input["t"]["bb_ca"],
                    t_local_latents=input["t"]["local_latents"],
                )
                seqs = seqs + rec_seq * 1.0  # Do we need this? Disable with 0.0 here
                pair_rep = pair_rep + rec_pair

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)  # [b, n, n, pair_dim]

        # Get outputs
        local_latents_out = self.local_latents_linear(seqs) * mask[..., None]  # [b, n, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n, 3]

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {self.output_param["local_latents"]: local_latents_out}
        nn_out["local_latents_int"] = {
            self.output_param["local_latents"]: local_latents_int
        }  # List of tensors of shape [b, n, latent_dim]
        nn_out["bb_ca_int"] = {self.output_param["bb_ca"]: bb_ca_int}  # List of tensors of shape [b, n, 3]
        return nn_out
