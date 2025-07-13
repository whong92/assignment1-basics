from typing import Mapping, Any

from cs336_basics.linear import Linear
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.positional_ffn import SwiGLU
from cs336_basics.attention import MultiHeadSA, softmax
from cs336_basics.rope import RoPE
from cs336_basics.embedding import Embedding

import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import repeat

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device = None,
        dtype=None
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.attn = MultiHeadSA(
            d_model=d_model,
            num_heads=num_heads,
            pos_emb = RoPE(
                theta=theta,
                d_k=d_model // num_heads,
                max_seq_len=max_seq_len,
                device=device
            ),
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )


    def forward(self, x: Float[Tensor, "batch sequence_length d_model"]) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        token_pos = torch.arange(seq_len)
        y = x + self.attn(self.ln1(x), token_pos)
        return y + self.ffn(self.ln2(y))


def top_k_top_p_filtering(
    p: Float[Tensor, " ... vocab_size"],
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = 0,
) -> Float[Tensor, " ... vocab_size"]:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, p.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = p < torch.topk(p, top_k)[0][..., -1, None]
        p[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_p, sorted_indices = torch.sort(p, descending=True)
        cumulative_probs = torch.cumsum(
            sorted_p,
            dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs < top_p

        indices_to_remove = torch.zeros_like(p, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        p[indices_to_remove] = filter_value
    return p


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.layers = nn.Sequential(*[
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    def sample(
        self,
        x: Int[Tensor, "batch_size sequence_length_in"],
        tau: float = 1,
        top_p: float = 1,
        max_num_tokens: int = 4096
    ) -> Int[Tensor, "sequence_length_out"]:
        in_len = x.shape[1]
        for _ in range(max_num_tokens):
            y = self(x.to(self.device))[:, -1]  # batch_size, vocab_size
            p = softmax(y, dim=-1, tau=tau)
            p = top_k_top_p_filtering(p, top_p=top_p)
            tok = torch.multinomial(p, num_samples=1).to('cpu')
            x = torch.concat((x, tok), dim=-1)
        return x[:, in_len:]

