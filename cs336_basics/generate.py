from cs336_basics.transformer import TransformerLM
from cs336_basics.checkpoint import load_checkpoint_and_config, init_from_config
from cs336_basics.tokenizer import Tokenizer, SPECIAL_TOKENS
from cs336_basics.config_types import ExperimentConfig
import torch
from importlib.resources import as_file, files
import yaml

def generate(
    prompt: str,
    tokenizer: Tokenizer,
    model: TransformerLM,
    tau: float = 1.,
    top_p: float = 1.,
    max_num_tokens: int = 256
) -> str:
    prompt_toks = torch.IntTensor([tokenizer.encode(prompt)])
    output_toks = model.sample(
        prompt_toks,
        tau=tau,
        top_p=top_p,
        max_num_tokens=max_num_tokens
    )
    return tokenizer.decode(output_toks[0])

def generate_main(
    config_name: str,
    ckpt_path: str,
    vocab_path: str,
    merges_path: str,
) -> None:
    prompt = "Once"
    tau = 1.
    top_p = 1.
    max_num_tokens = 30
    tok = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=SPECIAL_TOKENS
    )

    model, _, _ = load_checkpoint_and_config(
        ckpt_path,
    )

    generated = generate(
        prompt=prompt,
        tokenizer=tok,
        model=model,
        tau=tau,
        top_p=top_p,
        max_num_tokens=max_num_tokens
    )
    print(generated)


generate_main(
    "local_test",
    "checkpoints/last.ckpt",
"tests/fixtures/gpt2_vocab.json",
    "tests/fixtures/gpt2_merges.txt",
)