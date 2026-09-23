# Progress

Detailed week-by-week write-up. See [README.md](README.md) for the project overview and a
condensed version of these achievements.

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
- [x] **Week 3** — Data pipeline and a bigram baseline. `llm/data.py` caches a
      tokenized corpus to disk as a `uint16` shard, memory-maps it, and serves
      `(input, target)` sliding windows via a shard-aware `TokenDataset` —
      windows never cross a shard boundary (`tests/test_data.py`).
      `scripts/train_bigram.py`'s `BigramLM` (embedding-as-logits, no
      attention) trains against a small custom BPE vocab (`scripts/
      tokenize_bigram_corpus.py`, 2000 merges → vocab 2256, `data/bpe2k/`) —
      real GPT-2's 50257 vocab makes `nn.Embedding(vocab_size, vocab_size)`
      blow past 16GB RAM. Untrained loss matches `ln(2256) ≈ 7.72`; trained
      loss reaches `4.06` (train) / `5.74` (val), confirming real bigram
      statistics are learned (`tests/test_bigram.py`). `llm/embeddings.py`'s
      `TokenAndPositionalEmbedding` adds a learned token table
      (`vocab_size × n_embd`) to a learned positional table
      (`context_length × n_embd`), decoupling `n_embd` from `vocab_size` for
      the first time — verified by output shape `[batch, seq, n_embd]` and by
      the same token id at different positions producing different combined
      vectors (`tests/test_embeddings.py`). Not consumed by the bigram floor
      itself (position is irrelevant when only one prior token matters), but
      ready for week 4's attention.

## Side exercises

Not part of the main week-by-week curriculum, but reinforce the same fundamentals from a
different angle.

- **`scripts/make_more.py`** — a character-level MLP name generator (Karpathy's
  `makemore`), applied to `data/make_more/names.txt`. Learns a 10-dim embedding per
  character (`n_embd=10`) over a 3-character context window, feeding a 200-unit hidden
  layer with `tanh` activations. Uses batch normalization on the hidden pre-activations
  (per-batch mean/std at train time, an EMA-tracked running mean/std at eval/sampling
  time) to keep activations well-scaled without a bias term, plus Kaiming-style weight
  init (`5/3 / sqrt(fan_in)`) to keep pre-activation variance ~1 at initialization.
  Trained for 200k steps with step-decayed learning rate; samples novel,
  plausible-looking names by autoregressively feeding predictions back into the context
  window.
