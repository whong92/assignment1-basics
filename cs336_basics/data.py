import torch
from torch import Tensor
import numpy.typing as npt
import numpy as np

def get_batch_from_dataset_random(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    assert len(dataset) > context_length
    max_start_idx = len(dataset) - context_length - 1
    start_idxs = np.random.randint(
        low=0, high=max_start_idx + 1, size=batch_size
    )
    input_idxs = np.arange(context_length)[None, :] + start_idxs[:, None]
    output_idxs = input_idxs + 1

    input_seq = torch.LongTensor(dataset[input_idxs]).to(device)
    output_seq = torch.LongTensor(dataset[output_idxs]).to(device)
    return input_seq.to(device), output_seq.to(device)