import numpy as np
from cs336_basics.data import get_batch_from_dataset_random
from cs336_basics.transformer import TransformerLM
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.adamw import AdamWCosineSchedule
from cs336_basics.cross_entropy import cross_entropy
import json
import os
import torch
import wandb
from tqdm.auto import tqdm
from pydantic import BaseModel, computed_field
import logging
from importlib.resources import as_file, files
import yaml
import click

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)



def run_valid(
    valid_dataset: np.ndarray,
    num_iters: int,
    batch_size: int,
    context_length: int,
    device: str,
    model: torch.nn.Module
) -> float:
    val_loss = 0.
    for _ in tqdm(range(num_iters), position=1, leave=False):
        x, y = get_batch_from_dataset_random(
            dataset=valid_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        with torch.no_grad():
            ypred = model(x)
            val_loss += cross_entropy(ypred, y, reduce=True).cpu().item()
    return val_loss


class ModelConfig(BaseModel):
    num_layers: int = 4
    num_heads: int = 16
    d_head: int = 32
    d_ff: int = 1344
    rope_theta: float = 10000

    @computed_field
    @property
    def d_model(self) -> int:
        return int(
            self.num_heads * self.d_head
        )


class OptConfig(BaseModel):
    max_lr: float = 1e-3
    min_lr: float = 1e-3
    warmup_iters: int = 0


class TrainingConfig(BaseModel):
    context_length: int = 256
    batch_size: int = 32
    num_tokens: int = 327_000_000
    valid_num_tokens: int = 327_000

    @computed_field
    @property
    def num_iters(self) -> int:
        return int(
            self.num_tokens / self.batch_size / self.context_length
        )

    @computed_field
    @property
    def valid_num_iters(self) -> int:
        return int(
            self.valid_num_tokens / self.batch_size / self.context_length
        )

class DatasetConfig(BaseModel):
    train_dataset_path: str
    valid_dataset_path: str
    vocab_path: str


class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig = ModelConfig()
    opt: OptConfig = OptConfig()
    training: TrainingConfig = TrainingConfig()
    ckpt_dir: str
    device: str = "cpu"
    ckpt_every: int = 2000


class Logger:

    def __init__(
        self,
        use_wandb: bool = False,
        file: str | None = None
    ):
        self.use_wandb = use_wandb
        self.pbar = None
        self.file = file
        if self.file:
            # Create a file handler
            file_handler = logging.FileHandler(self.file)
            file_handler.setLevel(logging.INFO)  # Set the logging level for this handler
            # Create a formatter and add it to the handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            # Add the handler to the logger
            logger.addHandler(file_handler)

    def log_metrics(self, data: dict, iter: int) -> None:
        msg = f"iter={iter} data={json.dumps(data)}"
        if self.pbar:
            self.pbar.set_description(msg)
        if self.use_wandb:
            wandb.log(data, step=iter)
        if self.file:
            logger.info(msg)


def training_loop(
    config: ExperimentConfig,
    mylogger: Logger
) -> None:
    dataset = np.memmap(
        config.dataset.train_dataset_path,
        dtype=np.uint16,
        mode="readonly"
    )
    valid_dataset = np.memmap(
        config.dataset.valid_dataset_path,
        dtype=np.uint16,
        mode="readonly"
    )

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

    # load last ckpt if exists
    last_ckpt_path = os.path.join(config.ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        ckpt_iter = load_checkpoint(last_ckpt_path, model=model, optimizer=optimizer)
    else:
        ckpt_iter = 0

    pbar = tqdm(range(num_iters), position=0, leave=False)
    mylogger.pbar = pbar
    for i in pbar:
        if i < ckpt_iter:
            continue
        optimizer.zero_grad()
        x, y = get_batch_from_dataset_random(
            dataset=dataset,
            batch_size=config.training.batch_size,
            context_length=config.training.context_length,
            device=config.device,
        )
        ypred = model(x)
        loss = cross_entropy(ypred, y, reduce=True)
        loss.backward()
        optimizer.step()
        mylogger.log_metrics({'train/loss': loss.cpu().item()}, iter=i)

        if (i + 1) % config.ckpt_every == 0:
            save_checkpoint(
                model, optimizer, i, last_ckpt_path
            )
            ith_ckpt_path = os.path.join(config.ckpt_dir, f"iter-{i:05d}.ckpt")
            save_checkpoint(
                model, optimizer, i, ith_ckpt_path
            )
            val_loss = run_valid(
                valid_dataset=valid_dataset,
                num_iters=config.training.valid_num_iters,
                batch_size=config.training.batch_size,
                context_length=config.training.context_length,
                device=config.device,
                model=model
            )
            mylogger.log_metrics({'val/loss': val_loss}, iter=i)


@click.command()
@click.option(
    "--config_name", type=click.STRING
)
@click.option(
    "--use_wandb", type=click.BOOL, default=False
)
@click.option(
    "--logfile", type=click.STRING, default=None
)
def training_main(
    config_name: str,
    use_wandb: bool = False,
    logfile: str | None = None
) -> None:

    with as_file(files("cs336_basics.configs") / f"{config_name}.yaml") as path:
        with open(path, "r") as f:
            config = ExperimentConfig.model_validate(
                yaml.safe_load(f)
            )


    if use_wandb:
        wandb.login()
        wandb.init(
            project="cs336-assignment-1",  # Specify your project
            name=config_name,
            config=config.model_dump(mode='json'),
        )

    training_loop(
        config=config,
        mylogger=Logger(use_wandb=use_wandb, file=logfile),
    )


if __name__=="__main__":
    training_main()
