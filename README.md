# LLM From Scratch

Building a GPT-2-class language model end to end: tokenizer, attention, transformer,
training loop — then pretraining it myself and comparing against OpenAI's released
GPT-2 weights.

Full plan, schedule, and reasoning: [STUDY_PLAN.md](STUDY_PLAN.md).

## Setup

```
uv sync
uv run pytest -v
```

Requires Python 3.12+. Uses `torch` (MPS on Apple Silicon locally, CUDA/Colab for
training in later weeks), `tiktoken`, `numpy`, `matplotlib`, `pytest`.

## Structure

```
llm/            # the model, module by module, built up week by week
tests/          # one test file per module, the acceptance test for that week
scripts/        # standalone scripts (data prep, benchmarking, eval) — later weeks
notebooks/      # thin Colab entrypoints — later weeks
notes/          # week-by-week write-ups
```

## Progress

- [x] **Week 1** — Environment + micro-autograd. `llm/autograd.py`'s `Value` class
      matches PyTorch's autograd gradients to `1e-6` on a multi-op expression
      (`tests/test_autograd.py`). `Neuron`/`Layer`/`MLP` built on top of `Value`
      and verified against PyTorch the same way. `scripts/train_toy_mlp.py`
      trains an `MLP(3, [4, 4, 1])` on the toy 4-example dataset with plain
      gradient descent; loss decreases smoothly toward 0.
- [x] **Week 2** — Tokenization and byte-pair encoding. `llm/tokenizer.py` has a
      character-level warm-up and a full byte-level BPE tokenizer: byte
      pre-tokenization, iterative most-frequent-pair merging into a learned
      `merges` table, rank-ordered `encode`/`decode` against that table, and
      merge reversal — verified including nested (merge-of-merge) cases.
      `scripts/train_bpe.py` trains on a slice of `tinyshakespeare.txt` and
      plots vocab size vs. average tokens-per-word (`notes/week2/`). It also
      loads GPT-2's real `vocab.json`/`merges.txt` and reproduces GPT-2's regex
      pre-tokenizer (`pretokenize`), BPE-encoding each regex chunk independently
      so merges never cross a chunk boundary — `encode_with_gpt2_vocab` is
      byte-identical to `tiktoken.get_encoding("gpt2")` and
      `decode_with_gpt2_vocab(encode_with_gpt2_vocab(x)) == x` across thousands
      of held-out lines of `tinyshakespeare.txt`, including emoji/multi-byte
      UTF-8 (`tests/test_tokenizer.py`). Known follow-up: per-chunk encoding
      re-scans the full 50,000-rule merge table per chunk, ~5x slower than the
      pre-pretokenizer version — noted in `llm/tokenizer.py`, not yet optimized.
- [ ] **Week 3** — Data pipeline and a bigram baseline. `llm/data.py` caches a
      tokenized corpus to disk as a `uint16` shard, memory-maps it, and serves
      `(input, target)` sliding windows via a shard-aware `TokenDataset` —
      windows never cross a shard boundary (`tests/test_data.py`).
      `scripts/train_bigram.py`'s `BigramLM` (embedding-as-logits, no
      attention) trains against a small custom BPE vocab (`scripts/
      tokenize_bigram_corpus.py`, 2000 merges → vocab 2256, `data/bpe2k/`) —
      real GPT-2's 50257 vocab makes `nn.Embedding(vocab_size, vocab_size)`
      blow past 16GB RAM. Untrained loss matches `ln(2256) ≈ 7.72`; trained
      loss reaches `4.06` (train) / `5.74` (val), confirming real bigram
      statistics are learned (`tests/test_bigram.py`). Still open: token +
      learned positional embeddings (`[batch, seq, n_embd]`) called for by
      this week's plan haven't been built yet — the bigram floor doesn't need
      them, but week 4's attention will consume them.

