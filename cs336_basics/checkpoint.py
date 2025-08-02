import json

from cs336_basics.adamw import AdamWCosineSchedule
from cs336_basics.pretokenization_example import SPECIAL_TOKENS
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.config_types import ExperimentConfig
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
    map_location: str | None = None,
) -> tuple[TransformerLM, torch.optim.Optimizer, Tokenizer, int]:
    ckpt = torch.load(src, map_location=map_location)
    config = ExperimentConfig.model_validate(ckpt["config"])
    model, optimizer = init_from_config(config)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    tokenizer_serialized_params = ckpt["tokenizer"]
    tokenizer = Tokenizer.from_serialized_vocab(
        vocab_ser=tokenizer_serialized_params["vocab"],
        merges_ser=tokenizer_serialized_params["merges"],
        special_tokens=SPECIAL_TOKENS,
    )
    return model, optimizer, tokenizer, ckpt["iteration"]


def save_checkpoint(
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    val_loss: float | None = None,
    config: ExperimentConfig | None = None
) -> None:
    tokenizer_serialized = {
        "vocab": open(config.dataset.vocab_path).read(),
        "merges": open(config.dataset.merges_path).read(),
        "special_tokens": SPECIAL_TOKENS
    }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config.model_dump(mode='json') if config else None,
            "tokenizer": tokenizer_serialized,
            "val_loss": val_loss if val_loss is not None else None,
        },
        out
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | None = None,
) -> int:
    ckpt = torch.load(src, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]