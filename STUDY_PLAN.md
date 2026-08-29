# Study Plan: Build an LLM from Scratch

## Context

`llm_from_scratch/` is currently an empty directory. You want to go from nothing to a
GPT-2-class model you understand end to end: your own tokenizer, your own attention,
your own transformer — then **pretrained by you** on real data, and cross-checked against
OpenAI's released GPT-2 weights.

**Revised 2026-08-29** after you confirmed access to Google Colab. The original 12-week
plan was sized for an M1 laptop, which capped training at a ~15M-param toy model and made
"load OpenAI's weights" the only route to a coherent model. With a GPU in reach, actually
pretraining GPT-2 124M becomes the destination, and Karpathy's CUDA optimization material
— unusable on MPS — becomes a full week of content. That adds 3 weeks: **15 total.**

Constraints that shape this plan:

- **Time**: ~6-8 hrs/week over 15 weeks (~105 hours). One milestone per week.
- **Compute**: hybrid. Weeks 1-10 are local on the M1 Pro (16 GB, MPS) — all development,
  all tests, all debugging. Weeks 11-14 add **Colab Pro** for training runs only.
- **Money**: ~$20 total, i.e. two months of Colab Pro. See the budget section — it's the
  one real cost in the project and it drives the back half of the schedule.
- **Style**: build-first. Each week is a coding milestone with a concrete **acceptance
  test** that either passes or doesn't. Books and videos are support material, not the spine.
- **Your starting point**: fluent in Python (`uv`, `pyproject.toml`), shipped LLM-API
  projects (`predicate_gen_mcp`, `field_criteria_gen`), no from-scratch-training experience.

The organising principle: **you never move on from a component until a test proves your
version matches a reference implementation.** That's what separates "I watched a video
about attention" from "I know attention."

---

## What you end up with

Five things. The first is the headline, and it's a *comparison* rather than a model.

### 1. Two GPT-2 124M models, and the experiment between them

Identical architecture — the same `GPTModel` class, the same 124,439,808 parameters, the
same eval harness. Only the weights differ.

| | Yours (`checkpoints/mine`) | OpenAI's (`gpt2-124m`) |
|---|---|---|
| Weights from | your weeks 12-13 run | OpenAI, 2019 |
| Training data | ~2B tokens, FineWeb-Edu | ~10B unique tokens, WebText, multiple epochs |
| Compute | ~100 L4-hours, ~$20 | orders of magnitude more |
| Val loss *(est.)* | ~3.6-3.7 | ~3.29 |
| HellaSwag *(est.)* | ~27-28% | ~29.5% (random = 25%) |

Yours loses, and both sit close to random on HellaSwag, because 124M is genuinely small.
That's the expected result, not a failure. What you own at the end is the ability to say
exactly *why* and *by how much* — which is week 14's actual deliverable.

### 2. A working implementation — ~2,000 lines you typed

Twelve modules in `llm/`, three scripts, one thin notebook. No `transformers` in the model
path: it appears once, in week 9, purely as a weight source and a test oracle.

### 3. A test suite that proves it — ~15 tests, 5 hard gates

Not "it runs." `uv run pytest` green means the architecture is provably correct against
reference implementations, the causal mask provably masks, and the KV cache provably
doesn't change outputs.

### 4. `notes/` — the write-up, and the real artifact

The measured optimization ladder (your numbers, not Karpathy's), both loss curves, the
residual-connection gradient plot, the failure-mode catalogue, `gpt2-to-llama.md`, and the
head-to-head. In two years the code will be stale; this is what you'll still reason from.

### 5. Two runnable commands

```
uv run python -m llm.chat --weights gpt2-124m         # OpenAI's weights, your code
uv run python -m llm.chat --weights checkpoints/mine  # your weights, your code
```

**And the output that isn't a file:** you'll be able to read any LLM paper's architecture
section and know precisely where it slots into your `model.py`. Week 13's RoPE swap tests
exactly that — not whether you can write RoPE, but whether the codebase in your head is
flexible enough to absorb a modification you've never seen.

**What this is *not*:** a chatbot. Both models complete text; neither answers questions.
That gap is instruction tuning — see *What comes after*.

---

## The compute budget

This is the arithmetic that shapes weeks 11-14. Work it out once, up front, so the
schedule isn't a surprise.

Colab Pro gives **100 compute units/month**. Burn rates and GPT-2 124M throughput
(bf16 + `torch.compile` + FlashAttention, all of which you'll build in week 11):

| GPU | Units/hr | Hours per 100 units | ~tok/s | Tokens per month |
|-----|----------|---------------------|--------|------------------|
| A100 40GB | ~13 | ~8 | ~15,000 | ~430M |
| **L4 24GB** | **~2** | **~50** | **~5,500** | **~1.0B** |

**Use the L4.** It is the slower card and the better deal by ~2.3× — the A100 is ~2.7×
faster but costs ~6.5× more per hour. This is counter-intuitive enough that it's worth
verifying yourself in week 11 rather than taking on faith.

**Target: ~2B tokens over two months of Pro (~$20).** Chinchilla-optimal for 124M params
is ~2.5B tokens, so that lands in the right neighbourhood.

**Set expectations honestly now:** Karpathy's run that *beats* OpenAI's GPT-2 124M used
10B tokens. At 2B you should expect val loss around 3.6-3.7 against OpenAI's ~3.29 on
comparable data. Your model will be genuinely trained and clearly undertrained — in the
same league, not ahead. The interesting part is that you'll train on FineWeb-Edu, which
is much better data than GPT-2's WebText, so the comparison is a real experiment rather
than a foregone loss. Measuring that gap *is* week 14.

**Google Drive is the other budget.** Free tier is 15 GB. 2B tokens as `uint16` is 4 GB;
each checkpoint (weights + AdamW moments) is ~1.5 GB. Keep at most two checkpoints and
you'll sit around 7 GB. Plan for it in week 12 rather than discovering it at 2am.

---

## Target repo structure

Build this up incrementally — don't scaffold it all in week 1.

```
llm_from_scratch/
├── pyproject.toml           # uv-managed
├── llm/
│   ├── autograd.py          # wk 1  micro-autograd (throwaway, but keep it)
│   ├── tokenizer.py         # wk 2  BPE trainer + encoder/decoder
│   ├── data.py              # wk 3  dataset, sliding-window batching
│   ├── attention.py         # wk 4  causal multi-head attention
│   ├── layers.py            # wk 5  LayerNorm, GELU, FeedForward, Block
│   ├── model.py             # wk 6  GPTModel + configs
│   ├── train.py             # wk 7  training loop, LR schedule, checkpoints
│   ├── generate.py          # wk 8  sampling: temperature, top-k, top-p
│   ├── load_gpt2.py         # wk 9  HF weight → your model mapping
│   ├── kvcache.py           # wk 10 incremental decoding
│   ├── chat.py              # wk 10 REPL
│   └── rope.py              # wk 13 rotary embeddings (the modern-arch swap)
├── scripts/
│   ├── prepare_fineweb.py   # wk 12 tokenize + shard FineWeb-Edu to Drive
│   ├── benchmark.py         # wk 11 the optimization ladder
│   └── eval_hellaswag.py    # wk 14 yours vs OpenAI's
├── notebooks/
│   └── train_colab.ipynb    # wk 11 thin: clone, install, invoke scripts/train.py
├── tests/                   # one test file per module, written *with* the module
├── data/  checkpoints/      # (gitignored)
└── notes/                   # week-by-week write-ups — this is where learning sticks
```

**Push to GitHub from week 1.** Colab sessions are disposable; the workflow in weeks 11+
is `git clone` into a fresh runtime every time. Never edit code inside Colab — the
notebook stays thin and the learning stays in the repo.

---

## Schedule at a glance

Weeks 1-10 are local. Weeks 11-14 use Colab. **Bold** rows are hard gates.

| Wk | Where | Milestone | Acceptance test |
|----|-------|-----------|-----------------|
| 1 | local | Setup + micro-autograd | Gradients match PyTorch to 1e-6 |
| 2 | local | BPE tokenizer | Matches `tiktoken` GPT-2 IDs exactly on held-out text |
| 3 | local | Data pipeline + bigram | Val loss beats the `ln(vocab_size)` baseline |
| 4 | local | Causal multi-head attention | Matches `nn.MultiheadAttention` to 1e-5 |
| 5 | local | Transformer block | LayerNorm matches torch; residuals fix gradient decay |
| **6** | local | **Full GPT model** | Param count **exactly** 124,439,808 |
| 7 | local | Training loop | Val loss descends; samples are English-shaped |
| 8 | local | Decoding strategies | Seeded reproducibility; top-k/top-p restrict candidates |
| **9** | local | **Load real GPT-2 weights** | Logits match HF `GPT2LMHeadModel` to <1e-4 |
| **10** | local | **KV cache + chat REPL** | Cached output token-identical to uncached, ≥5× faster |
| **11** | Colab | **The optimization ladder** | ≥10× throughput, loss curve provably unchanged |
| **12** | Colab | **Data at scale + resume** | Kill mid-run, resume, loss curve has no discontinuity |
| 13 | Colab | The long run + modern arch | Run grinds unattended; RoPE implemented locally |
| 14 | Colab | Your GPT-2 vs OpenAI's | Perplexity + HellaSwag, both models, documented honestly |
| 15 | local | Write-up | A reader can explain a transformer from `notes/` alone |

---

## Week-by-week

### Week 1 — Environment + backpropagation from first principles

You cannot debug a training run you don't understand mechanically. Start here.

**Build**
- `uv init`, then `uv add torch tiktoken numpy matplotlib pytest`. Verify
  `torch.backends.mps.is_available()` and benchmark a matmul on `mps` vs `cpu` so you know
  when MPS is actually worth it (spoiler: not for tiny tensors).
- **Initialise git and push to GitHub now.** Weeks 11+ clone this repo into Colab.
- `llm/autograd.py`: a scalar `Value` class with `+`, `*`, `**`, `tanh`, `exp`, and a
  topological-sort `backward()`. Then `Neuron` → `Layer` → `MLP` on top of it, trained to
  fit a toy dataset.

**Acceptance test** — build the same expression graph in your `Value` class and in
PyTorch; assert gradients agree to 1e-6.

**Read**: Karpathy, "The spelled-out intro to neural networks and backpropagation"
(micrograd). Watch it *after* attempting the class yourself.

> Deliberately low-tech week. The point is that in week 7, when loss goes to NaN, you know
> what a gradient actually is.

---

### Week 2 — Tokenization and byte-pair encoding

**Build**
- Character-level tokenizer first (30 min) — establishes the `encode`/`decode` contract.
- `llm/tokenizer.py`: BPE from scratch. Byte-level pre-tokenization, iterative
  most-frequent-pair merging, a learned `merges` table, `encode`/`decode`, and
  `<|endoftext|>` handling.
- Train your BPE on a few MB of text. Plot vocab size vs. average tokens-per-word.
- Then load OpenAI's actual GPT-2 `vocab.json` + `merges.txt` into *your* implementation.

**Acceptance test** — with GPT-2's merge table loaded, your `encode()` produces
byte-identical IDs to `tiktoken.get_encoding("gpt2")` across a few thousand lines of
held-out text, and `decode(encode(x)) == x` for arbitrary UTF-8 including emoji.

**Read**: Karpathy, "Let's build the GPT Tokenizer" (minbpe). Sennrich et al. 2015.

> The emoji / multi-byte UTF-8 case is where most from-scratch tokenizers quietly break.
> Put it in the test.

---

### Week 3 — Data pipeline and a bigram baseline

**Build**
- `llm/data.py`: tokenize a corpus once, cache to disk as a `uint16` numpy array, yield
  `(input, target)` pairs by sliding window with configurable `context_length` and
  `stride`. Deterministic train/val split.
- Token embeddings + learned positional embeddings. Be able to say out loud what each axis
  of `[batch, seq, n_embd]` is.
- Train a bigram model (embedding → logits, no attention) as your floor.

**Acceptance test** — untrained val loss ≈ `ln(50257)` ≈ 10.82; trained bigram lands
below ~6.5. If the untrained number is off, your pipeline is misaligned — almost always an
off-by-one between inputs and targets.

**Corpus**: tinyshakespeare (~1 MB) or a Gutenberg book. Iterates in seconds.

> Design `data.py` so the array is **memory-mapped and shard-aware** from the start. In
> week 12 you'll point the same class at 4 GB of FineWeb-Edu shards on Drive, and it should
> need no rewrite.

---

### Week 4 — Attention, built in four stages

The core week. Do **not** jump straight to the batched multi-head version.

**Build** `llm/attention.py` in this exact order, keeping every stage:
1. `attention_v1` — one query attending over keys via dot products, explicit Python loops.
   Slow and obviously correct. This is what you reason against when stage 2 misbehaves.
2. `SelfAttention` — vectorised, with `/ sqrt(d_k)` scaling. Verify the scaling
   empirically: plot softmax entropy with and without it at `d_k = 768`.
3. `CausalAttention` — upper-triangular `-inf` mask plus attention dropout.
4. `MultiHeadAttention` — reshape into heads, batched, output projection.

**Acceptance test** — two checks. Copy your weights into `torch.nn.MultiheadAttention`
(`batch_first=True`, `is_causal=True`) and assert outputs match to 1e-5. Then assert that
for any position `t`, gradients w.r.t. inputs at positions `> t` are exactly zero. The
second is a real proof the mask works and catches bugs eyeballing the attention matrix won't.

**Read**: "Attention Is All You Need", sections 3.1-3.2 only. Alammar's *Illustrated
Transformer* for visuals.

> Keep stage 4 written the naive way (explicit matmul + softmax). In week 11 you'll swap
> it for `F.scaled_dot_product_attention` and measure what FlashAttention buys — you need
> the slow version to compare against.

---

### Week 5 — The transformer block

**Build** `llm/layers.py`:
- `LayerNorm` from scratch — mean/var with `unbiased=False`, learnable scale and shift,
  `eps=1e-5`.
- `GELU` — GPT-2 uses the tanh approximation. Implement it, plot against ReLU.
- `FeedForward`: `Linear(768→3072) → GELU → Linear(3072→768)`. Know why 4×.
- `TransformerBlock` — **pre-**LayerNorm ordering with residuals: `x = x + attn(ln1(x))`,
  then `x = x + ff(ln2(x))`.

**Acceptance test** — your `LayerNorm` matches `torch.nn.LayerNorm` to 1e-6. Then the one
that matters: stack 12 blocks, run backward, print the gradient norm at each layer **with
and without** residual connections. Without them, early-layer gradients collapse toward
zero; with them they stay comparable. Save the plot — it's the entire reason residuals exist.

---

### Week 6 — Assemble the full GPT ⭐

**Build** `llm/model.py`:
- `GPTConfig` dataclass and `GPT_CONFIG_124M`: `vocab_size=50257`, `context_length=1024`,
  `emb_dim=768`, `n_heads=12`, `n_layers=12`, `drop_rate=0.1`, `qkv_bias=True`.
- `GPTModel` — token + positional embeddings → dropout → 12 blocks → final LayerNorm →
  output head. Weight tying between the token embedding and the head.
- `.num_params()` and a memory-footprint estimator.

**Gate — an exact integer**

```
assert model.num_params() == 124_439_808

  38,597,376   token embedding    50257 × 768
     786,432   positional         1024 × 768
  85,054,464   12 × block         7,087,872 each
       1,536   final LayerNorm
```

If you get `163,037,184`, your output head isn't sharing weights with the embedding.
Also assert `[2, 1024]` int64 input returns `[2, 1024, 50257]`.

> **Forward warning for week 11.** You will later pad the vocab from 50257 to **50304**
> (= 128 × 393) because ugly numbers are slow on tensor cores. That changes the count to
> `124,475,904`. Keep 50257 as the canonical config and make padding a *training-time
> option*, or this gate will start failing and you'll think you broke something.

Also define the small config for week 7's local run: `emb_dim=384`, `n_heads=6`,
`n_layers=6`, `context_length=256` (~15M params).

---

### Week 7 — The training loop

This is a **correctness** run, not a scale run. Scale is weeks 11-13.

**Build** `llm/train.py`:
- Cross-entropy over flattened `[B*T, vocab]` logits. Understand why loss and perplexity
  relate by `exp()`.
- AdamW with weight decay on matrices but **not** biases or LayerNorm params.
- Linear warmup → cosine decay; gradient clipping at norm 1.0.
- Periodic validation, a sample at every eval, **checkpoint save/resume**.
- Pretrain the ~15M model on tinyshakespeare on MPS — 1-3 hours.

**Acceptance test** — val loss decreases (modulo noise); samples contain real words. Then
deliberately break things and record what each failure *looks like* in
`notes/failure-modes.md`: LR at 1e-1 (divergence to NaN), no clipping on a spiky batch, no
warmup, shuffled targets. Recognising these on sight is worth more than the successful run.

> Build checkpoint/resume properly now even though nothing here needs it. In week 12 it
> becomes load-bearing — Colab will disconnect mid-run and you'll lose days without it.

---

### Week 8 — Decoding strategies

**Build** `llm/generate.py`: greedy/argmax, temperature, top-k, top-p (nucleus), and a
seeded-reproducible path. Repetition penalty last — it's the one most likely to mask a bug
in the others.

**Acceptance test** — same seed + prompt + `temperature=0` gives byte-identical output
across runs. For `top_k=5`, instrument the sampler and assert the sampled token was in the
top 5 logits at every step over ~1000 steps. Write up what `temperature=0.1` vs `1.5` does
to one prompt, with actual samples.

---

### Week 9 — Load the real GPT-2 weights ⭐

This is the week your code stops being a toy.

**Build** `llm/load_gpt2.py`:
- `uv add transformers safetensors` — used *only* as a weight source and reference oracle,
  never in your model path.
- Download the `gpt2` 124M checkpoint and map it into your state dict. The traps, ordered
  by how much time they'll cost you:
  - HF stores attention/MLP weights in `Conv1D` layers, **transposed** relative to
    `nn.Linear` — you need `.T` on `c_attn`, `c_proj`, `c_fc`.
  - `c_attn` packs Q, K, V into one `[768, 2304]` tensor — split along dim 1.
  - Names are `h.{i}.attn.c_attn.weight`, `h.{i}.ln_1.weight`, `wte`, `wpe`, `ln_f`.
  - Dropout off (`model.eval()`) on both sides before comparing.
- Assert every parameter was assigned and no HF tensor went unused.

**Gate** — `torch.allclose(my_logits, hf_logits, atol=1e-4)`.

This retroactively proves weeks 2-6, because a bug anywhere breaks it. If it fails, bisect
layer by layer with forward hooks — the first layer that diverges is your bug.

**Then sample from it.** A coherent language model running entirely through your code.

---

### Week 10 — KV cache and a chat REPL ⭐

Moved ahead of the Colab work deliberately: you get a working chat experience on real
GPT-2 weights *before* the long training grind, and it's the last purely-local week.

**Build**
- `llm/kvcache.py` — cache K and V per layer, so generating token `n+1` is O(n) work
  instead of O(n²) recomputation over the prefix.
- `llm/chat.py` — a REPL with streaming output, adjustable sampling, context-window
  management (what to drop at 1024 tokens), and a prompt template.

**Gate — two hard requirements**

```
cached_tokens == uncached_tokens          # identical, same seed
cached_latency ≤ uncached_latency / 5     # 200-token continuation
```

The identity check catches off-by-one errors in cache position indexing, which is where
nearly every KV-cache bug lives. Plot per-token latency both ways.

> Base GPT-2 is a *completion* model, not a chat model. It continues your text rather than
> answering you. That's expected — instruction tuning is what closes the gap.

---

### Week 11 — Colab bring-up and the optimization ladder ⭐

The single best week of new material, and the reason lifting the local-only constraint was
worth it. None of this is possible on MPS.

**Build**
- `notebooks/train_colab.ipynb` — deliberately thin: mount Drive, `git clone` your repo,
  `pip install`, invoke `scripts/train.py`. **Never edit model code in Colab.**
- Confirm the runtime is an **L4**, not an A100 (Runtime → Change runtime type). Verify the
  unit economics from the budget table yourself before committing to it.
- `scripts/benchmark.py` — apply these one at a time to GPT-2 124M and measure each:

  | Step | Change | Expected |
  |------|--------|----------|
  | 0 | Naive fp32 baseline | baseline |
  | 1 | `torch.set_float32_matmul_precision("high")` (TF32) | ~3× |
  | 2 | `torch.autocast(dtype=torch.bfloat16)` | ~1.5-2× |
  | 3 | `torch.compile(model)` | ~1.5-2× |
  | 4 | `F.scaled_dot_product_attention` (FlashAttention) | ~1.3× |
  | 5 | Vocab 50257 → **50304** | ~1.04× |
  | 6 | `AdamW(..., fused=True)` | small |
  | 7 | Gradient accumulation → 0.5M-token batch | (correctness, not speed) |

- Understand *why* for each. Step 5 is the most instructive: 50304 = 128 × 393, and a
  vocab dimension divisible by 128 keeps tensor cores fed. Free speed from an ugly constant.
- Step 2: know why **bf16 beats fp16** here — same exponent range as fp32, so no
  `GradScaler` and no loss-scaling failures.

**Gate — two requirements**

```
total_speedup ≥ 10×            vs the naive fp32 baseline
loss_curve_optimized ≈ loss_curve_baseline    # over ~200 steps, same seed
```

The second matters more than the first. Every step above is a **speed** change, not a math
change — if the loss curve moves, you've changed semantics and introduced a bug.
Optimizing without altering results is the actual skill.

**Read**: Karpathy, "Let's reproduce GPT-2" — the optimization segment specifically.

---

### Week 12 — Data at scale, and surviving disconnection ⭐

**Build**
- `scripts/prepare_fineweb.py` — stream the FineWeb-Edu 10B sample from HuggingFace,
  tokenize with your week-2 tokenizer (multiprocess — this is CPU-bound and slow),
  write `uint16` shards of ~100M tokens each to Drive. Target ~2B tokens ≈ 4 GB.
- Point your week-3 `data.py` at the shards. If you built it memory-mapped and shard-aware
  as advised, this is a config change, not a rewrite.
- Checkpoint/resume hardening: save weights **plus optimizer moments, LR-schedule step,
  RNG state, and shard/offset position**. Rotate — keep the last two only (Drive is 15 GB).
- Launch the real run.

**Gate — the one that saves your project**

```
1. start a run, train ~500 steps
2. kill the Colab session mid-run
3. fresh runtime, resume from checkpoint
4. assert the loss curve has no visible discontinuity at the resume point
```

A jump at the seam means you dropped optimizer state or restarted the data stream — both
silently waste days of compute. Test this deliberately *before* the long run, not after.

> Colab **will** disconnect. Treat every run as interruptible and this is a non-event.

---

### Week 13 — The long run, and the modern-architecture delta

The run grinds unattended in Colab. Local work fills the week — this overlap is why the
plan is 15 weeks and not 17.

**Colab (passive)** — training continues toward ~2B tokens. Check in daily: loss curve,
compute units remaining, checkpoint rotation. Expect to renew Pro for a second month here.

**Local (active)** — read the Llama 2/3 papers and write `notes/gpt2-to-llama.md` on the
four changes that matter:
- **RoPE** instead of learned positional embeddings
- **RMSNorm** instead of LayerNorm
- **SwiGLU** instead of the GELU MLP
- **Grouped-query attention** instead of full MHA

Implement **one** — `llm/rope.py`, since RoPE is the most instructive — as a swappable
option, and confirm it still trains on the week-7 local setup.

**Acceptance test** — RoPE model trains to a comparable loss on tinyshakespeare, and you
can explain in `notes/` why rotary embeddings extrapolate to longer contexts than learned
positional embeddings do.

---

### Week 14 — Your GPT-2 versus OpenAI's ⭐

The payoff week. You now have two 124M models with identical architecture and different
weights: yours, and OpenAI's.

**Build**
- `scripts/eval_hellaswag.py` — the standard zero-shot completion benchmark. OpenAI's
  GPT-2 124M scores ~29.5%; random is 25%.
- Perplexity for both models on the same held-out set.
- A comparison table in `notes/`: val loss, HellaSwag, tokens trained, wall-clock hours,
  dollars spent — and sample generations from each on identical prompts.

**Acceptance test** — an honest, documented comparison. **Yours will probably lose**, and
that's the expected result at 2B tokens against OpenAI's WebText run. What matters is that
you can say precisely *why* and *by how much*: undertrained relative to Chinchilla, offset
partly by FineWeb-Edu being better data than WebText. Quantifying your own model's
shortfall is a more valuable skill than winning.

---

### Week 15 — Write up

**Build** `notes/README.md` — the full write-up: architecture diagram, every acceptance
test and its result, the optimization ladder table, both loss curves, the failure-mode
catalogue, the head-to-head comparison, total cost, and what surprised you.

**Acceptance test** — a reader who hasn't seen your code can explain how a transformer
works from `notes/` alone.

---

## Reference material

Support, not the spine. Reach for these when stuck, after you've attempted the build.

- **Karpathy, "Let's reproduce GPT-2"** — now a primary source for weeks 11-14, not just
  background. The optimization segment maps directly onto week 11.
- **Karpathy, Neural Networks: Zero to Hero** — micrograd (wk 1), tokenizer (wk 2),
  "Let's build GPT" (wk 4-7).
- **Sebastian Raschka, *Build a Large Language Model (From Scratch)*** — closest match to
  weeks 2-9.
- **nanoGPT** and **build-nanogpt** — read *after* week 6 as a check on your structure;
  `build-nanogpt` is the reference for weeks 11-12's training infrastructure.
- **Papers**: Attention Is All You Need (wk 4); GPT-2 (wk 6); Chinchilla (wk 12); FlashAttention
  (wk 11); Llama 2/3 (wk 13); HellaSwag (wk 14).
- **FineWeb-Edu dataset card** (HuggingFace) — read the filtering methodology in week 12;
  it's why your 2B tokens go further than GPT-2's WebText did.

---

## Working rules

1. **Write the test before the module.** Every acceptance check is a real `pytest` test.
   `uv run pytest` stays green as you go.
2. **No copy-paste.** Type every line. If you consult a reference, close it and reimplement
   from memory.
3. **Never advance on a red test.** If week 4's attention doesn't match torch, week 5 is
   built on sand — and you'll debug it in week 9 at ten times the cost.
4. **Never edit code in Colab.** The notebook clones your repo and calls scripts. Code that
   only exists in a notebook cell is lost when the runtime dies.
5. **Every run is interruptible.** Assume disconnection. Checkpoint accordingly.
6. **Keep `notes/` per week** — what you built, what broke, what you'd forgotten seven days
   later. This is the real artifact; the code is a side effect.
7. **Time-box debugging to 90 minutes**, then check the reference.
8. **Budget** ~1 hr reading, ~4-5 hrs building, ~1 hr testing and notes.

---

## Verification

The plan is on track if these hold:

- `uv run pytest` passes at every week boundary, tests accumulating (~15 by the end).
- **Week 6**: `model.num_params() == 124_439_808`. An exact integer — no ambiguity.
- **Week 9**: `torch.allclose(your_logits, hf_logits, atol=1e-4)`. Proves weeks 2-6 correct.
- **Week 10**: cached and uncached generation produce identical token sequences.
- **Week 11**: ≥10× throughput over naive fp32, with the loss curve provably unchanged.
- **Week 12**: kill and resume mid-run leaves no discontinuity in the loss curve.
- **Week 14**: a documented head-to-head against OpenAI's GPT-2 124M on the same harness.
- End-to-end, both models through your own code:
  ```
  uv run python -m llm.chat --weights gpt2-124m        # OpenAI's
  uv run python -m llm.chat --weights checkpoints/mine  # yours
  ```

## What comes after

The natural continuation — roughly another 8 weeks — is classification fine-tuning → LoRA
→ instruction tuning (SFT) → DPO, applied to **your own** pretrained base model rather than
OpenAI's. That's what turns the week-14 completion model into something that follows
instructions, and it's the point where the project stops being a study exercise. Karpathy's
**nanochat** is the reference for that arc.
