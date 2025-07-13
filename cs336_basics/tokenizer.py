import os
from dataclasses import dataclass
from collections import Counter
from cs336_basics.pretokenization_example import Pretoken, chunk_and_pretokenize, PAT, find_chunk_boundaries, \
    CHUNK_TOKEN, SPECIAL_TOKENS
from itertools import pairwise
import regex as re
import json
from functools import lru_cache
from typing import Iterator, Iterable
import click
import numpy as np
from tqdm import tqdm
import uuid
from multiprocessing import Pool

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges:  list[tuple[int, int, int]]  # index1,index2 -> new_index
    special_tokens: list[str]


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


def merge_and_count_pair_change_per_doc(
    indices: list[int],
    pair: tuple[int, int],
    new_index: int,
    pair_count: dict[tuple[int, int], int] | None,
    doc_count: int = 1
) -> tuple[list[int], bool]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    if pair_count is not None:
        orig_pairs = pairwise(indices)
        new_pairs = pairwise(new_indices)
        for pair in orig_pairs:
            pair_count[pair] -= doc_count
        for pair in new_pairs:
            pair_count[pair] += doc_count
    return new_indices


def count_token_pairs(pretoks: list[Pretoken]) -> dict[tuple[int, int], int]:
    counts = Counter()
    for i, pretok in enumerate(pretoks):
        count_per_pretok = Counter(pairwise(pretok.indices))
        for pair, count in count_per_pretok.items():
            counts[pair] += count * pretok.count
    return counts


def merge_and_count_pair_change(
    pretoks: list[Pretoken],
    pair_to_merge: tuple[int, int],
    new_index: int,
    pair_counts: dict[[int, int], int] | None = None
) -> list[Pretoken]:
    bla = 0
    for pretok in pretoks:
        if pair_to_merge in pretok.pairs_set:
            bla += 1
            new_indices = merge_and_count_pair_change_per_doc(
                indices=pretok.indices,
                pair=pair_to_merge,
                new_index=new_index,
                pair_count=pair_counts,
                doc_count=pretok.count
            )
            pretok.indices = new_indices
            pretok.pairs_set = set(pairwise(new_indices))
    return pretoks


def train_bpe_pretokenize(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    pretok_n_chunks: int,
    pretok_n_workers: int = 1
) -> BPETokenizerParams:
    pretokens = chunk_and_pretokenize(
        input_path, special_tokens, pretok_n_chunks, pretok_n_workers
    )
    token_counts = count_token_pairs(pretokens)

    merges: list[tuple[int, int, int]] = []
    vocab: dict[int, bytes] = {i: t.encode("utf-8") for i, t in enumerate(special_tokens)}
    ofs = len(special_tokens)
    vocab.update({x: bytes([x - ofs]) for x in range(ofs, ofs + 256)})

    num_merges = max(0, vocab_size - len(vocab))
    for i in tqdm(range(num_merges)):
        # Find the most common pair, for ties, take lexicographically largest
        pair = max(token_counts, key=lambda k: (token_counts[k], vocab[k[0]], vocab[k[1]]))  # @inspect pair
        index1, index2 = pair

        # Merge that pair.
        new_index = len(vocab)
        merges.append((index1, index2, new_index))
        vocab[new_index] = vocab[index1] + vocab[index2]

        pretokens = merge_and_count_pair_change(
            pretokens,
            pair,
            new_index,
            pair_counts=token_counts
        )

    return BPETokenizerParams(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens
    )


# Copied from tests/common.py
@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]

    d = dict(zip(bs, characters))
    return d


class Tokenizer:

    def __init__(
        self,
        params: BPETokenizerParams
    ):
        self.params = params
        self.vocab = self.params.vocab
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.special_tokens = set(self.params.special_tokens) if self.params.special_tokens else []
        self.special_tokens_re = "(" + ("|".join([
            re.escape(s) for s in sorted(self.special_tokens)[::-1]
            # reverse lex order since want to match overlapping special tokens
            # we match the biggest overlap first
        ]))  + ")"
        self.pretok_to_idx_cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Copied from test_tokenizer.py"""
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        vocab_inv = {v: k for k, v in vocab.items()}
        merges_int = [
            (vocab_inv[b1], vocab_inv[b2], vocab_inv[b1 + b2])
            for b1, b2 in merges
        ]
        return Tokenizer(
            BPETokenizerParams(
                vocab=vocab,
                merges=merges_int,
                special_tokens=special_tokens
            )
        )


    def encode(self, text: str) -> list[int]:
        # remove special tokens and count
        pretoken_strs: list[str] = []
        if self.special_tokens:
            for c in re.split(self.special_tokens_re, text):
                if c in self.special_tokens:
                    pretoken_strs.append(c)
                else:
                    pretoken_strs.extend(
                        re.findall(PAT, c)
                    )
        else:
            pretoken_strs = re.findall(PAT, text)

        pretokens = []
        for pretoken_string in pretoken_strs:
            if pretoken_string in self.special_tokens:
                pretokens.append(
                    Pretoken(
                        string=pretoken_string,
                        indices=[self.vocab_inv[pretoken_string.encode("utf-8")]],
                        pairs_set={}
                    )
                )
            else:
                pretoken_bytes = [bytes([b]) for b in pretoken_string.encode("utf-8")]
                pretoken_indices = [self.vocab_inv[b] for b in pretoken_bytes]
                pretokens.append(
                    Pretoken(
                        string=pretoken_string,
                        indices=pretoken_indices,
                        pairs_set=set(pairwise(pretoken_indices))
                    )
                )

        for pretoken in pretokens:
            if pretoken.string in self.pretok_to_idx_cache:
                pretoken.indices = self.pretok_to_idx_cache[pretoken.string]
                pretoken.pairs_set = set(pairwise(pretoken.indices))
            else:
                for m, merge in enumerate(self.params.merges):
                    i1, i2, i3 = merge
                    merge_and_count_pair_change(
                        [pretoken],
                        (i1, i2),
                        i3,
                        pair_counts=None
                    )
                self.pretok_to_idx_cache[pretoken.string] = pretoken.indices

        output_idx = []
        for pretoken in pretokens:
            output_idx.extend(pretoken.indices)

        return output_idx

    def decode(self, ids: list[int]) -> str:
        bytelist = [self.vocab[index] for index in ids]
        out = b""
        for b in bytelist:
            out += b
        return out.decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for i, text in enumerate(iterable):
            for idx in self.encode(text):
                yield idx


def _run_tokenize_on_chunk(
    idx: int,
    vocab_fpath: str,
    merges_fpath: str,
    input_fpath: str,
    boundary: tuple[int, int],
    output_path: str,
    temp_output_path: str = '/tmp/tokenize',
    special_tokens: list[str] | None = None,
) -> int:

    tok = Tokenizer.from_files(
        vocab_filepath=vocab_fpath,
        merges_filepath=merges_fpath,
        special_tokens=special_tokens
    )
    temp_output_path = f'{temp_output_path}/{uuid.uuid4()}'
    os.makedirs(temp_output_path, exist_ok=True)
    # read input chunk into it's own temp copy
    input_temp_chunk = f'{temp_output_path}/input_temp_chunk.txt'
    start, end = boundary
    with open(input_fpath, "rb") as fin, open(input_temp_chunk, "wb") as fout:
        fin.seek(start)
        fout.write(
            fin.read(end - start)
        )

    # write chunks
    chunk_size = 1e7
    chunk_idx = 0
    chunk = []
    with open(input_temp_chunk) as f:
        for i in tqdm(tok.encode_iterable(f), position=idx):
            chunk.append(i)
            if len(chunk) == chunk_size:
                np.save(
                    f'{temp_output_path}/chunk-{chunk_idx:03d}.npy',
                    np.array(chunk, dtype=np.uint16)
                )
                chunk_idx += 1
                chunk = []

        if len(chunk) > 0:
            np.save(
                f'{temp_output_path}/chunk-{chunk_idx:03d}.npy',
                np.array(chunk, dtype=np.uint16)
            )
            tot_size = int(chunk_idx * chunk_size + len(chunk))

    num_chunks = chunk_idx + 1
    final_out = np.memmap(output_path, dtype=np.uint16, mode='write', shape=(tot_size, ))
    for chunk_idx in range(num_chunks):
        chunk = np.load(f'{temp_output_path}/chunk-{chunk_idx:03d}.npy',)
        s = int(chunk_idx * chunk_size)
        e = int(s + len(chunk))
        final_out[s:e] = chunk

    return tot_size


@click.group()
def cli():
    """A simple CLI for managing files."""
    pass

@cli.command()
@click.option("--vocab_filepath", type=click.STRING)
@click.option("--merges_filepath", type=click.STRING)
@click.option("--input_fpath", type=click.STRING)
@click.option("--output_path", type=click.STRING)
@click.option("--temp_output_path", default='/tmp/tokenizer', type=click.STRING)
@click.option("--n_chunks", default=1, type=click.INT)
def tokenize_main(
    vocab_filepath: str,
    merges_filepath: str,
    input_fpath: str,
    output_path: str,
    temp_output_path: str = '/tmp/tokenizer',
    n_chunks: int = 1,
) -> None:
    with open(input_fpath, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_chunks, CHUNK_TOKEN.encode("utf-8"))

    pool = Pool(n_chunks)

    output_sizes = pool.starmap(
        _run_tokenize_on_chunk,
        [
            (
                idx,
                vocab_filepath,
                merges_filepath,
                input_fpath,
                boundary,
                f"{output_path}.{boundary[0]}.{boundary[1]}",
                temp_output_path,
                SPECIAL_TOKENS
            )
            for idx, boundary in enumerate(pairwise(boundaries))
        ]
    )

    tot_size = sum(output_sizes)
    final_out = np.memmap(output_path, dtype=np.uint16, mode='write', shape=(tot_size,))
    i = 0
    for boundary in pairwise(boundaries):
        chunk = np.memmap(
            f"{output_path}.{boundary[0]}.{boundary[1]}",
            dtype=np.uint16
        )
        for c in chunk:
            final_out[i] = c
            i += 1

@cli.command()
@click.option("--input_path", type=click.STRING)
@click.option("--vocab_size", type=click.INT)
@click.option("--output_path", type=click.STRING)
@click.option("--pretok_n_chunks", type=click.INT)
@click.option("--pretok_n_workers", type=click.INT)
def train_bpe_main(
    input_path: str,
    vocab_size: int,
    output_path: str,
    pretok_n_chunks: int,
    pretok_n_workers: int = 10,
):
    bpe_params = train_bpe_pretokenize(
        input_path = input_path,
        vocab_size = vocab_size,
        special_tokens = SPECIAL_TOKENS,
        pretok_n_chunks=pretok_n_chunks,
        pretok_n_workers=pretok_n_workers
    )
    os.makedirs(output_path, exist_ok=True)
    bytes_to_printable = gpt2_bytes_to_unicode()

    vocab_printable = {}
    for k, v in bpe_params.vocab.items():
        if v in SPECIAL_TOKENS:
            vocab_printable[v] = k
        else:
            vocab_printable["".join([bytes_to_printable[t] for t in v])] = k

    with open(f"{output_path}/vocab.json", "w") as fp:
        json.dump(vocab_printable, fp, ensure_ascii=False, indent=4)

    vocab_printable_inv = {v: k for k, v in vocab_printable.items()}
    merges_bytes = [
        (vocab_printable_inv[i1], vocab_printable_inv[i2])
        for i1, i2, _ in bpe_params.merges
    ]
    with open(f"{output_path}/merges.txt", "w") as fp:
        for bytes1, bytes2 in merges_bytes:
            fp.write(bytes1 + " " + bytes2 + "\n")


if __name__ == "__main__":

    # vocab_filepath="/home/ong/personal/standford-cs336-2025/assignment1-basics/tests/fixtures/gpt2_vocab.json"
    # merges_filepath="/home/ong/personal/standford-cs336-2025/assignment1-basics/tests/fixtures/gpt2_merges.txt"
    # input_fpath = '/home/ong/personal/standford-cs336-2025/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    # output_path = '/home/ong/personal/standford-cs336-2025/assignment1-basics/data/TinyStoriesV2-GPT4-train.tok.npy'
    # temp_output_path = '/tmp/tokenize'
    # n_chunks = 5

    # data/TinyStoriesV2-GPT4-val.txt

    cli()
