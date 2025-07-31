Train BPE
```bash
uv run python cs336_basics/tokenizer.py \
  train-bpe-main --input_path data/TinyStoriesV2-GPT4-train.txt \
  --output_path data/TinyStoriesTrain_10k \
  --vocab_size 10000 \
  --pretok_n_chunks 10 \
  --pretok_n_workers 10
```

Tokenize
```bash
uv run python cs336_basics/tokenizer.py \
  tokenize-main \
  --vocab_filepath data/TinyStoriesTrain_10k/vocab.json \
  --merges_filepath data/TinyStoriesTrain_10k/merges.txt \
  --input_fpath data/TinyStoriesV2-GPT4-train.txt \
  --output_path data/TinyStoriesTrain_10k/TinyStoriesV2-GPT4-train.tok.npy \
  --n_chunks 10
```

Train Local
```bash
uv run python cs336_basics/train.py \
  --config_name local_test \
  --logfile debug.log
```

Train Remote
```bash
uv run python cs336_basics/train.py \
  --config_name default_full \
  --use_wandb \
  --sync_dataset_to_local
```