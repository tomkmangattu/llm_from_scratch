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
