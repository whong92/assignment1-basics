from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from torch import Tensor


def clip_grads(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    for param in parameters:
        grad = param.grad
        if grad is None:
            continue
        norm = torch.norm(grad, p=2, dim=None)
        scale = max_l2_norm / (norm + 1e-6)
        if norm > max_l2_norm:
            param.grad *= scale
    return

def lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos(
                math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
            )
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


class AdamWCosineSchedule(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        max_lr: float = 1e-3,
        min_lr: float = 1e-3,
        warmup_iters: int = 0,
        cosine_cycle_iters: int = 1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        max_grad_l2_norm: float = 1e3
    ):
        if min_lr < 0 or max_lr < 0 or min_lr > max_lr:
            raise ValueError(f"Invalid learning rates: {min_lr} {max_lr}")
        if cosine_cycle_iters <= warmup_iters:
            raise ValueError(f"Invalid schedule: {warmup_iters} {cosine_cycle_iters}")
        defaults = {
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_iters": warmup_iters,
            "cosine_iters": cosine_cycle_iters,
            "betas": betas,
            "eps": eps,
            "wd": weight_decay,
            "max_grad_l2_norm": max_grad_l2_norm
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]  # Get the learning rate.
            wd = group["wd"]
            eps = group["eps"]
            max_lr = group["max_lr"]
            min_lr = group["min_lr"]
            warmup_iters = group["warmup_iters"]
            cosine_iters = group["cosine_iters"]
            max_grad_l2_norm = group["max_grad_l2_norm"]

            clip_grads(group["params"], max_l2_norm=max_grad_l2_norm)
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                lr = lr_schedule(
                    t,
                    max_lr,
                    min_lr,
                    warmup_iters,
                    cosine_iters
                )

                with torch.no_grad():
                    grad = p.grad  # Get the gradient of loss with respect to p.

                    if t == 0:
                        state["m"] = torch.zeros_like(grad, requires_grad=False)
                        state["v"] = torch.zeros_like(grad, requires_grad=False)

                    state["m"] = state["m"] * beta1 + (1 - beta1) * grad
                    state["v"] = state["v"] * beta2 + (1 - beta2) * torch.pow(grad, 2)

                    m: Tensor = state["m"]
                    v: Tensor = state["v"]

                    lrt = lr * math.sqrt(1. - math.pow(beta2, t + 1)) / (1. - math.pow(beta1, t + 1))

                    p -= lrt * m / (torch.sqrt(v) + eps)
                    p -= lr * wd * p

                state["t"] = t + 1  # Increment iteration number.
        return loss