import torch
from jaxtyping import Float, Int
from torch import Tensor
from einops import reduce

def cross_entropy(
    logits: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
    reduce: bool = True
) -> Float[Tensor, ""]:
    # max along vocab size
    logits_max, _ = logits.max(dim=-1, keepdim=True)
    logits -= logits_max
    denom = torch.logsumexp(logits, dim=-1, keepdim=False)
    targets = torch.unsqueeze(targets, dim=-1)
    numer = torch.gather(logits, index=targets, dim=-1).squeeze(dim=-1)
    loss = -numer + denom
    if reduce:
        return loss.mean()
    return loss

def perplexity(loss: Float[Tensor, "... seq_len"]) -> Float[Tensor, ""]:
    """This assumes the last dimension is the seq len"""
    return torch.mean(
        torch.exp(
            reduce(loss, "... seq_len -> ...", reduction='mean')
        )
    )