import torch
from torch import nn
from einops import einsum, rearrange

class RoPE(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0
        f_k = int(d_k // 2)
        # precompute Rik
        position_ids = torch.arange(max_seq_len)  # n
        theta_power = torch.float_power(theta, torch.arange(f_k) * 2 / d_k)  # k
        theta_scaled = einsum(position_ids, 1. / theta_power, "n, k -> n k")  # n, k
        cos_theta_scaled = torch.cos(theta_scaled)
        sin_theta_scaled = torch.sin(theta_scaled)
        Rik = torch.zeros(size=(max_seq_len, f_k, 2, 2)).to(device)  # n, k, 2, 2
        Rik[:, :, 0, 0] = cos_theta_scaled
        Rik[:, :, 1, 1] = cos_theta_scaled
        Rik[:, :, 1, 0] = sin_theta_scaled
        Rik[:, :, 0, 1] = -sin_theta_scaled
        self.register_buffer(
            'Rik',
            Rik,
            persistent=False
        )


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_p = rearrange(x, "... seq_len (f u) -> ... seq_len f u", u=2)  # ... seq_len, k, 2
        Rik = self.Rik[token_positions] # ... seq_len, k, 2, 2
        x_p_pos_encoded = einsum(Rik, x_p, "... seq_len k i j, ... seq_len k j -> ... seq_len k i")  # ... seq_len, k 2
        out = rearrange(x_p_pos_encoded, "... seq_len f u -> ... seq_len (f u)")
        return out