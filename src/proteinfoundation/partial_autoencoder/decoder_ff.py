import einops
import torch
from openfold.np.residue_constants import RESTYPE_ATOM37_MASK

# from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(dtype=torch.bool, device=device)  # [21, 37]


class ResidualLayer(torch.nn.Module):
    """Residual layer with pre-LN"""

    def __init__(self, dim):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.linear(out)
        out = self.softplus(out)
        return x + out


class DecoderFFLocal(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        nlayers = kwargs["decoder"]["nlayers"]
        token_dim = kwargs["decoder"]["token_dim"]  # hidden dim

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["decoder"]["feats_seq"],
            dim_feats_out=token_dim,
            use_ln_out=False,
            mode="seq",
            **kwargs["decoder"],
        )

        layers = []
        for _ in range(nlayers):
            layers.append(ResidualLayer(dim=token_dim))
        self.ff_nn = torch.nn.Sequential(*layers)

        self.logit_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(token_dim),
            torch.nn.Linear(token_dim, 20, bias=False),
        )
        self.struct_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(token_dim),
            torch.nn.Linear(token_dim, (37 * 3), bias=False),
        )

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            input: {
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
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
        ca_coors_nm = input["ca_coors_nm"]  # [b, n, 3]
        mask = input["residue_mask"]  # [b, n] boolean

        # Iinitial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]

        seqs = self.ff_nn(seqs) * mask[..., None]  # [b, n, token_dim]

        # Get logits
        logits_out = self.logit_linear(seqs) * mask[..., None]  # [b, n, 20]

        # Get coordinates
        coors_flat_nm = self.struct_linear(seqs) * mask[..., None]  # [b, n, 37 * 3]
        coors_a37_nm = einops.rearrange(coors_flat_nm, "b n (a t) -> b n a t", a=37, t=3)  # [b, n, 37, 3]
        coors_a37_nm[..., 1, :] = coors_a37_nm[..., 1, :] * 0.0 + ca_coors_nm

        # Get sequence
        aatype_max = torch.argmax(logits_out, dim=-1)  # [b, n]
        aatype_max = aatype_max * mask  # [b, n]

        # Get atom_mask
        aa_a37_mask = get_atom_mask(device=logits_out.device)  # [21, 37] boolean
        atom_mask = aa_a37_mask[aatype_max, :]  # [b, n, 37] boolean
        atom_mask = atom_mask * mask[..., None]  # [b, n, 37] boolean

        output = {
            "coors_nm": coors_a37_nm,  # [b, n, 37, 3]
            "seq_logits": logits_out,  # [b, n, 20]
            "residue_mask": mask,  # [b, n]
            "aatype_max": aatype_max,  # [b, n]
            "atom_mask": atom_mask,  # [b, n, 37]
        }
        return output
