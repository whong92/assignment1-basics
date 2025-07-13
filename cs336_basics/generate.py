import numpy as np
from cs336_basics.data import get_batch_from_dataset_random
from cs336_basics.transformer import TransformerLM
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.adamw import AdamWCosineSchedule
from cs336_basics.cross_entropy import cross_entropy, perplexity
from cs336_basics.tokenizer import Tokenizer, SPECIAL_TOKENS
import json
import os
import torch

def generate(
    prompt: str,
    tokenizer: Tokenizer,
    model: TransformerLM,
    tau: float = 1.,
    top_p: float = 1.,
    max_num_tokens: int = 256
) -> str:
    prompt_toks = torch.IntTensor([tokenizer.encode(prompt)]).to(model.device)
    output_toks = model.sample(
        prompt_toks,
        tau=tau,
        top_p=top_p,
        max_num_tokens=max_num_tokens
    )
    return tokenizer.decode(output_toks[0])

def generate_main(
    ckpt_path: str,
    vocab_path: str,
    merges_path: str,
) -> None:
    prompt = "Once"
    tau = 1.
    top_p = 1.
    max_num_tokens = 256
    tok = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=SPECIAL_TOKENS
    )
    model, _ = load_checkpoint(ckpt_path)
    generate(
        prompt=prompt,
        tokenizer=tok,
        model=model,
        tau=tau,
        top_p=top_p,
        max_num_tokens=max_num_tokens
    )