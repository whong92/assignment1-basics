import os
from typing import BinaryIO
import regex as re
from collections import Counter
from dataclasses import dataclass

from itertools import pairwise
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

SPECIAL_TOKENS = [
    "<|endoftext|>"
]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS_RE = "|".join([re.escape(s) for s in SPECIAL_TOKENS])
CHUNK_TOKEN = "<|endoftext|>"


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@dataclass
class Pretoken:
    string: str
    indices: list[int]
    pairs_set: set[tuple[int, int]]
    count: int = 1  # how often this pretoken appears in corpus

def count_pretokens(
    fname: str,
    chunk_start: int,
    chunk_end: int,
    special_tokens: list[str]
) -> dict[str, int]:
    with open(fname, "rb") as f:
        f.seek(chunk_start)
        chunk = f.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")
        pretoken_counts = Counter()
        special_tokens_re = "|".join([re.escape(s) for s in special_tokens])
        # remove special tokens and count
        for c in tqdm(re.split(special_tokens_re, chunk)):
            pretoken_counts.update(Counter(re.findall(PAT, c)))
    return pretoken_counts

def proc_pretokens(
    pretoken_counts: list[tuple[str, int]], special_token_offset: int
) -> list[Pretoken]:
    ret = []
    for pretoken_string, pretoken_count in pretoken_counts:
        pretoken_indices = list(map(int, pretoken_string.encode("utf-8")))
        pretoken_indices = [i + special_token_offset for i in pretoken_indices]
        ret.append(
            Pretoken(
                string = pretoken_string,
                indices = pretoken_indices,
                count = pretoken_count,
                pairs_set = set(pairwise(pretoken_indices))
            )
        )
    return ret


def split_list(l: list, n: int) -> list[list]:
    arr = np.array(l, dtype=object)
    return [ arr_split.tolist() for arr_split in np.array_split(arr, n)]


def chunk_and_pretokenize(
    fname: str,
    special_tokens: list[int],
    n_chunks: int,
    n_workers: int = 10,
) -> list[Pretoken]:
    ## Usage
    with open(fname, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, n_chunks, CHUNK_TOKEN.encode("utf-8")
        )

    pretoken_counts = Counter()
    pool = Pool(n_workers)
    jobs = pool.starmap(
        count_pretokens,
        [
            (
                fname, start, end, special_tokens
            )
            for start, end in pairwise(boundaries)
        ]
    )
    for chunk_pretoken_count in tqdm(jobs, total=n_chunks):
        pretoken_counts.update(chunk_pretoken_count)

    return proc_pretokens(list(pretoken_counts.items()), special_token_offset=len(special_tokens))
