import torch
from torch.nn import functional as F


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def forward(self, x):
        """
        Args:
            x: input tensor, shape [..., d]

        Returns:
            Tensor of shape [..., d//2].
        """
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x
