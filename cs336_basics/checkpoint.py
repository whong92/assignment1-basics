import json

from cs336_basics.adamw import AdamWCosineSchedule
from cs336_basics.transformer import TransformerLM
from cs336_basics.types import ExperimentConfig
import torch
import typing
import os


def init_from_config(config: ExperimentConfig) -> tuple[TransformerLM, AdamWCosineSchedule]:
    num_iters = config.training.num_iters
    cosine_cycle_iters = max(num_iters - config.opt.warmup_iters, 0)

    with open(config.dataset.vocab_path, 'r') as fp:
        vocab_size = len(
            json.load(fp)
        )

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=1000,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        rope_theta=config.model.rope_theta,
        device=config.device
    )

    optimizer = AdamWCosineSchedule(
        model.parameters(),
        max_lr=config.opt.max_lr,
        min_lr=config.opt.min_lr,
        warmup_iters=config.opt.warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters
    )

    return model, optimizer


def load_checkpoint_and_config(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> tuple[TransformerLM, torch.optim.Optimizer, int]:
    ckpt = torch.load(src)
    config = ExperimentConfig.model_validate(ckpt["config"])
    model, optimizer = init_from_config(config)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer, ckpt["iteration"]


def save_checkpoint(
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: ExperimentConfig | None = None
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config.model_dump(mode='json') if config else None,
        },
        out
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]