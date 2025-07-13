from typing import Mapping, Any

import torch
from jaxtyping import Float, Int
from torch import Tensor
from einops import einsum, rearrange, repeat
import math
from torch import nn
from cs336_basics.linear import Linear


def softmax(x: Tensor, dim: int, tau: float = 1) -> Tensor:
    """Subtracts from largest dimension in dim before softmaxxing."""
    xmax, _ = x.max(dim=dim, keepdim=True)
    x = (x - xmax) / tau
    expx = torch.exp(x)
    denom = expx.sum(dim=dim, keepdim=True)
    return expx / denom


def scaled_dot_prod_attn(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Tensor:
    d_k = Q.shape[-1]
    dot_prod = einsum(
        Q, K, " ... queries d_k, ... keys d_k -> ... queries keys"
    ) / math.sqrt(d_k)
    if mask is not None:
        dot_prod[~mask] += -float("inf")
    # softmax over keys
    attn = softmax(dot_prod, dim=-1)
    return einsum(
        attn, V, " ... queries keys, ... keys d_v -> ... queries d_v"
    )


class MultiHeadSA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_emb: nn.Module | None = None,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.qkv_proj = Linear(
            in_features=d_model,
            out_features=d_model * 3,
            device=device,
            dtype=dtype
        )
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype
        )
        self.pos_emb = pos_emb

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        assert (
            f"{prefix}q_proj.weight" in state_dict and
            f"{prefix}k_proj.weight" in state_dict and
            f"{prefix}v_proj.weight" in state_dict
        ) or (
            f"{prefix}qkv_proj.weight" in state_dict
        )
        if f"{prefix}q_proj.weight" in state_dict:
            qkv_weights = torch.cat(
                [
                    state_dict[f"{prefix}q_proj.weight"],
                    state_dict[f"{prefix}k_proj.weight"],
                    state_dict[f"{prefix}v_proj.weight"]
                ],
                dim=0
            )
            del state_dict[f"{prefix}q_proj.weight"]
            del state_dict[f"{prefix}k_proj.weight"]
            del state_dict[f"{prefix}v_proj.weight"]
            state_dict[f"{prefix}qkv_proj.weight"] = qkv_weights

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )


    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        batch_dims = list(x.shape[:-2])
        seq_len = x.shape[-2]
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(
            qkv,
            dim=-1,
            split_size_or_sections=[self.d_model] * 3
        )
        q = rearrange(q, " ... sequence_length (h d_h) -> ... h sequence_length d_h", h=self.num_heads)
        k = rearrange(k, " ... sequence_length (h d_h) -> ... h sequence_length d_h", h=self.num_heads)
        v = rearrange(v, " ... sequence_length (h d_h) -> ... h sequence_length d_h", h=self.num_heads)

        if self.pos_emb is not None and token_positions is not None:
            q = self.pos_emb(q, token_positions)
            k = self.pos_emb(k, token_positions)

        mask = torch.triu(
            torch.ones(
                size=(seq_len, seq_len),
                dtype=torch.bool
            ),
            diagonal=0
        ).T
        rep_pattern = ' '.join([str(i) for i in batch_dims + [self.num_heads]])
        mask = repeat(mask, f"m n -> {rep_pattern} m n")
        attn_v = scaled_dot_prod_attn(
            q, k, v, mask
        )
        attn_v_concat = rearrange(
            attn_v,
            " ... h sequence_length d_h -> ... sequence_length (h d_h) ",
            h=self.num_heads
        )
        return self.output_proj(attn_v_concat)