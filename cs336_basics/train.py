import numpy as np
import json
import os
import torch
import wandb
from tqdm.auto import tqdm
import logging
from importlib.resources import as_file, files
import yaml
import click
import shutil

from cs336_basics.types import ExperimentConfig, DatasetConfig
from cs336_basics.data import get_batch_from_dataset_random
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint, init_from_config
from cs336_basics.cross_entropy import cross_entropy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

TMP_DATA_DIR = "/tmp/temp-data-dir"

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


def map_dataset_to_local(dataset_config: DatasetConfig) -> DatasetConfig:
    os.makedirs(TMP_DATA_DIR, exist_ok=True)
    new_dataset_config = DatasetConfig(
        train_dataset_path = f"{TMP_DATA_DIR}/{os.path.basename(dataset_config.train_dataset_path)}",
        valid_dataset_path = f"{TMP_DATA_DIR}/{os.path.basename(dataset_config.valid_dataset_path)}",
        vocab_path = f"{TMP_DATA_DIR}/{os.path.basename(dataset_config.vocab_path)}",
    )
    shutil.copy(dataset_config.train_dataset_path, new_dataset_config.train_dataset_path)
    shutil.copy(dataset_config.valid_dataset_path, new_dataset_config.valid_dataset_path)
    shutil.copy(dataset_config.vocab_path, new_dataset_config.vocab_path)
    return new_dataset_config


def training_loop(
    config: ExperimentConfig,
    mylogger: Logger,
    sync_dataset_to_local: bool = False
) -> None:
    # for performance reasons we don't want to read directly to remote storage
    if sync_dataset_to_local:
        logger.info(f"Syncing remote dataset to local dataset start")
        local_data_config = map_dataset_to_local(config.dataset)
        logger.info(f"Syncing remote dataset to local dataset end")
    else:
        local_data_config = config.dataset

    dataset = np.memmap(
        local_data_config.train_dataset_path,
        dtype=np.uint16,
        mode="readonly"
    )
    valid_dataset = np.memmap(
        local_data_config.valid_dataset_path,
        dtype=np.uint16,
        mode="readonly"
    )

    num_iters = config.training.num_iters
    # always initialize from config
    model, optimizer = init_from_config(config)

    # load from last checkpoint if exists
    os.makedirs(config.ckpt_dir, exist_ok=True)
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
                last_ckpt_path, model, optimizer, i, config
            )
            ith_ckpt_path = os.path.join(config.ckpt_dir, f"iter-{i:05d}.ckpt")
            save_checkpoint(
                ith_ckpt_path, model, optimizer, i, config
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
    "--use_wandb", is_flag=True
)
@click.option(
    "--logfile", type=click.STRING, default=None
)
@click.option(
    "--sync_dataset_to_local", is_flag=True
)
def training_main(
    config_name: str,
    use_wandb: bool = False,
    logfile: str | None = None,
    sync_dataset_to_local: bool = False,
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
        sync_dataset_to_local=sync_dataset_to_local,
    )


if __name__=="__main__":
    training_main()
