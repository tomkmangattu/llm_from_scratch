import math
import torch
import pytest

from scripts.train_bigram import BigramLM, BigramLMNaive

VOCAB= 6

def test_naive_matches_vectorized():
    torch.manual_seed(0)
    idx = torch.randint(0, VOCAB, (2, 4))
    targets = torch.randint(0, VOCAB, (2, 4))

    fast = BigramLM(VOCAB)
    naive = BigramLMNaive(VOCAB)

    naive.load_state_dict(fast.state_dict())

    _, loss_fast = fast(idx, targets)
    _, loss_naive = naive(idx, targets)

    assert torch.allclose(loss_fast, loss_naive, atol=1e-5)

# untrained model produces a loss close to ln(vocab_size)
def test_untrained_bigram_matches_uniform_baseline():
    torch.manual_seed(0)
    model = BigramLM(VOCAB)
    idx = torch.randint(0, VOCAB, (8, 16))
    targets = torch.randint(0, VOCAB, (8, 16))

    _, loss = model(idx, targets)

    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.2)