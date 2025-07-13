import torch
from torch import nn
from torch import Tensor
from einops import einsum
from jaxtyping import Float

class Linear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        Winit: Float[Tensor, "out_features in_features"] | None = None,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        if Winit is not None:
            assert Winit.shape == (out_features, in_features), "inconsistent shapes"
            self.weight = nn.Parameter(
                Winit.to(device)
            )
        else:
            self.weight = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.zeros(out_features, in_features, dtype=dtype).to(device)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")