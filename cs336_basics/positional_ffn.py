import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float
from cs336_basics.linear import Linear

class SiLU(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, " ..."]) -> torch.Tensor:
        return torch.mul(x, torch.sigmoid(x))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.silu = SiLU()
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> torch.Tensor:
        a = self.silu(self.w1(x))
        b = self.w3(x)
        return self.w2(torch.mul(a, b))

