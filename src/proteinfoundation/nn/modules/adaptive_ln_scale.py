from typing import Literal

import torch

# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch


class AdaptiveLayerNorm(torch.nn.Module):
    """Adaptive layer norm layer, where scales and biases are learned from some
    conditioning variables."""

    def __init__(self, *, dim, dim_cond):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cond = torch.nn.LayerNorm(dim_cond)

        self.to_gamma = torch.nn.Sequential(torch.nn.Linear(dim_cond, dim), torch.nn.Sigmoid())

        self.to_beta = torch.nn.Linear(dim_cond, dim, bias=False)

    def forward(self, x, cond, mask):
        """
        Args:
            x: input representation, shape [*, dim]
            cond: conditioning variables, shape [*, dim_cond]
            mask: binary, shape [*]

        Returns:
            Representation after adaptive layer norm, shape as input representation [*, dim].
        """
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        out = normed * gamma + beta
        return out * mask[..., None]


class AdaptiveOutputScale(torch.nn.Module):
    """Adaptive scaling of a representation given conditioning variables."""

    def __init__(self, *, dim, dim_cond, adaln_zero_bias_init_value=-2.0):
        super().__init__()

        adaln_zero_gamma_linear = torch.nn.Linear(dim_cond, dim)
        torch.nn.init.zeros_(adaln_zero_gamma_linear.weight)
        torch.nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = torch.nn.Sequential(adaln_zero_gamma_linear, torch.nn.Sigmoid())

    def forward(self, x, cond, mask):
        """
        Args:
            x: input sequence, shape [*, dim]
            cond: conditioning variables, shape [*, dim_cond]
            mask: binary, shape [*]

        Returns:
            Scaled input, shape [*, dim].
        """
        gamma = self.to_adaln_zero_gamma(cond)  # [*, dim]
        return x * gamma * mask[..., None]


class AdaptiveLayerNormIdentical(torch.nn.Module):
    """
    Adaptive layer norm layer, where scales and biases are learned from some
    conditioning variables.

    This assumes that cond does not depend on n in any way, so basically diffusion times, etc,
    which are the same for the full protein. This property can be used to make the AdaLN more efficient,
    as we reduce the number of Linear layers we apply.
    """

    def __init__(
        self,
        *,
        dim: int,
        dim_cond: int,
        mode: Literal["seq", "pair"],
        use_ln_cond: bool = False,
    ):
        """
        Args:
            dim: dimension of the input representation
            dim_cond: dimension of the conditioning variables
            mode: either "seq" or "pair"
            use_ln_cond: whether to apply layer norm to the conditioning variables
        """
        super().__init__()
        assert mode in [
            "single",
            "pair",
        ], f"Mode {mode} not valid for AdaptiveLayerNormIdentical"
        self.mode = mode
        self.use_ln_cond = use_ln_cond

        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        if use_ln_cond:
            self.norm_cond = torch.nn.LayerNorm(dim_cond)

        self.to_gamma = torch.nn.Sequential(torch.nn.Linear(dim_cond, dim), torch.nn.Sigmoid())

        self.to_beta = torch.nn.Linear(dim_cond, dim, bias=False)

    def forward(self, x, cond, mask):
        """
        Args:
            x: input representation, shape [b, n, dim] if single, [b, n, n, d] if pair
            cond: conditioning variables, shape [b, dim_cond]
            mask: binary, shape [b, n] if single, [b, n, n] if pair

        Returns:
            Representation after adaptive layer norm, shape as input representation [*, dim].
        """
        assert cond.dim() == 2, f"Expected tensor cond with shape [b, dim_cond], got {cond.shape}"

        if self.mode == "single":
            assert x.dim() == 3, f"Expected tensor x with shape [b, n, dim] for `single` mode, got {x.shape}"
            assert mask.dim() == 2, f"Expected 2D tensor mask with shape [b, n] for `single` mode, got {mask.shape}"

        if self.mode == "pair":
            assert x.dim() == 4, f"Expected tensor x with shape [b, n, n, d] for `pair` mode, got {x.shape}"
            assert mask.dim() == 3, f"Expected tensor mask with shape [b, n, n] for `pair` mode, got {mask.shape}"

        normed = self.norm(x)  # [*, n, dim] if single, [*, n, n, dim] if pair
        if self.use_ln_cond:
            normed_cond = self.norm_cond(cond)  # [*, dim_cond]
        else:
            normed_cond = cond  # [*, dim_cond]

        gamma = self.to_gamma(normed_cond)  # [*, dim_cond]
        beta = self.to_beta(normed_cond)  # [*, dim_cond]

        # Prepare broadcasting
        if self.mode == "single":
            gamma_brc = gamma[..., None, :]  # [*, 1, dim_cond]
            beta_brc = beta[..., None, :]  # [*, 1, dim_cond]
        else:
            gamma_brc = gamma[..., None, None, :]  # [*, 1, 1, dim_cond]
            beta_brc = beta[..., None, None, :]  # [*, 1, 1, dim_cond]

        # Apply adaptive LN
        out = normed * gamma_brc + beta_brc
        return out * mask[..., None]
