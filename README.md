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

Full week-by-week write-up with implementation details: [PROGRESS.md](PROGRESS.md).

- **Micro-autograd** — a from-scratch reverse-mode autograd engine
  (`llm/autograd.py`) whose gradients match PyTorch's to `1e-6`, with an MLP built
  on top of it that trains to convergence via plain gradient descent.
- **Byte-level BPE tokenizer** — `llm/tokenizer.py` reproduces GPT-2's tokenizer
  exactly: byte-identical to `tiktoken.get_encoding("gpt2")` on thousands of held-out
  lines (including emoji/multi-byte UTF-8), with lossless round-tripping.
- **Sharded token data pipeline** — `llm/data.py` memory-maps a tokenized corpus and
  serves shard-safe sliding-window `(input, target)` batches for training.
- **Bigram language model baseline** — `scripts/train_bigram.py` trains a
  `vocab_size × vocab_size` embedding-as-logits model; loss drops from the
  `ln(vocab_size)` untrained baseline (`7.72`) to `4.06` train / `5.74` val, confirming
  it learns real token transition statistics.
- **Token + positional embeddings** — `llm/embeddings.py` decouples embedding
  dimension from vocab size, laying the groundwork for attention (week 4).
- **MLP character-level name generator** (side exercise, Karpathy's `makemore`) —
  `scripts/make_more.py` learns 10-dim character embeddings, uses batch
  normalization (with running mean/std for eval) and Kaiming-style init for stable
  training, and samples novel plausible names.

