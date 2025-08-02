from cs336_basics.transformer import TransformerLM
from cs336_basics.checkpoint import load_checkpoint_and_config
from cs336_basics.tokenizer import Tokenizer
import torch

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
    ckpt_path: str,
) -> None:
    prompt = " "
    tau = 1.
    top_p = 1.
    max_num_tokens = 30

    model, _, tok, _ = load_checkpoint_and_config(
        ckpt_path,
    )
    model.eval()
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
    "/Users/waihongong/github/assignment1-basics/checkpoints/last.ckpt",
)