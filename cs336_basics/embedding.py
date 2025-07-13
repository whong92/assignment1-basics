import torch
from torch import nn
from torch import Tensor
from jaxtyping import Int

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(num_embeddings, embedding_dim, dtype=dtype)
            )
        ).to(device)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
        return self.weight[token_ids]