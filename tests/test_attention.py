import math
import torch
from llm.attention import attention_v1, SelfAttention

def test_attention_v1():
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    values = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

    out = attention_v1(query, keys, values)
    print(out)
    assert torch.allclose(out, torch.tensor([6.33, 3.66]), atol=1e-2)

def test_selfAttention():
    torch.manual_seed(0)
    model = SelfAttention(4, 2)
    x = torch.Tensor([[1, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0]])
    out = model(x)

    assert out.shape == (3, 2)

    # cross-check against attention_v1: each output row should equal
    # attention_v1 run with that position's own Q against the shared K/V.
    # attention_v1 has no 1/sqrt(d_out) scaling, so pre-scale Q to compensate.
    Q, K, V = model.W_q(x), model.W_k(x), model.W_v(x)
    Q_scaled = Q / math.sqrt(model.d_out)
    for t in range(x.shape[0]):
        expected = attention_v1(Q_scaled[t], K, V)
        assert torch.allclose(out[t], expected, atol=1e-5)