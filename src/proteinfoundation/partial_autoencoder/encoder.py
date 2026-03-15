import torch

# from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition


class EncoderTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        self.nlayers = kwargs["encoder"]["nlayers"]
        self.token_dim = kwargs["encoder"]["token_dim"]
        self.pair_repr_dim = kwargs["encoder"]["pair_repr_dim"]
        self.update_pair_repr = kwargs["encoder"]["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["encoder"]["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["encoder"]["use_tri_mult"]
        self.normalize_latent = kwargs["encoder"]["normalize_latent"]

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_seq"],
            dim_feats_out=kwargs["encoder"]["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs["encoder"],
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_cond_seq"],
            dim_feats_out=kwargs["encoder"]["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs["encoder"],
        )

        self.transition_c_1 = Transition(kwargs["encoder"]["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["encoder"]["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_rep_factory = FeatureFactory(
            feats=kwargs["encoder"]["feats_pair_repr"],
            dim_feats_out=kwargs["encoder"]["pair_repr_dim"],
            use_ln_out=False,
            mode="pair",
            **kwargs["encoder"],
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=kwargs["encoder"]["token_dim"],
                    dim_pair=kwargs["encoder"]["pair_repr_dim"],
                    nheads=kwargs["encoder"]["nheads"],
                    dim_cond=kwargs["encoder"]["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=kwargs["encoder"]["use_qkln"],
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
                            token_dim=kwargs["encoder"]["token_dim"],
                            pair_dim=kwargs["encoder"]["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.latent_decoder_mean_n_log_scale = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, int(2 * kwargs["latent_z_dim"]), bias=False),
        )

        # self.ln_z = torch.nn.LayerNorm(kwargs["latent_z_dim"], elementwise_affine=False) if self.normalize_latent else torch.nn.Identity()
        self.ln_z = torch.nn.Identity()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            batch: batch from dataset

        Returns:
            Dictionary
            ...
        """
        if "mask" not in batch:
            mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        else:
            mask = batch["mask"]

        # Conditioning variables
        c = self.cond_factory(batch)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(batch)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_rep_factory(batch)  # [b, n, n, pair_dim]

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)  # [b, n, n, pair_dim]

        # Get final coordinates
        flat_out = self.latent_decoder_mean_n_log_scale(seqs) * mask[..., None]  # [b, n, 2 * latent_dim]
        flat_out = flat_out * mask[..., None]
        mean, log_scale = torch.chunk(flat_out, chunks=2, dim=-1)  # [b, n, latent_dim] each

        # # Use same log_scale across all dims of latent_dim
        # first = log_scale[:, :, 0]  # [b, n]
        # log_scale = log_scale * 0.0 + first[..., None]

        z = mean + torch.randn_like(log_scale) * torch.exp(log_scale)  # [b, n, latent_dim]

        # scale = torch.nn.functional.softplus(log_scale)  # [b, n, latent_dim]
        # log_scale = torch.log(scale)  # [b, n, latent_dim]
        # z = mean + torch.randn_like(log_scale) * scale  # [b, n, latent_dim]

        z_pre_ln = z
        z = self.ln_z(z) * mask[..., None]  # [b, n, latent_dim]
        output = {
            "mean": mean,
            "log_scale": log_scale,
            "z_latent": z,
            "z_latent_pre_ln": z_pre_ln,
        }
        return output
