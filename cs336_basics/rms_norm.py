import torch
from torch import nn, Tensor
from einops import einsum, reduce
from jaxtyping import Float, Int


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(d_model, dtype=dtype).to(device)
            )
        )
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x2 = torch.pow(x, 2)
        denom = 1. / torch.sqrt(
            reduce(x2, "... d_model -> ...", "mean") + self.eps
        )
        y = einsum(x, self.weight, denom, "... d_model, d_model, ... -> ... d_model")
        return y.to(in_dtype)