import torch

from llm.embeddings import TokenAndPositionalEmbedding

VOCAB_SIZE = 6
CONTEXT_LENGTH = 4
N_EMBD = 3

def test_output_shape():
    torch.manual_seed(0)
    model = TokenAndPositionalEmbedding(VOCAB_SIZE, CONTEXT_LENGTH, N_EMBD)
    idx = torch.randint(0, VOCAB_SIZE, (2, CONTEXT_LENGTH))

    out = model(idx)

    assert out.shape == (2, CONTEXT_LENGTH, N_EMBD)

# same token at every position -> token part is identical each time, so any
# difference across positions must come from the positional embedding
def test_same_token_differs_by_position():
    torch.manual_seed(0)
    model = TokenAndPositionalEmbedding(VOCAB_SIZE, CONTEXT_LENGTH, N_EMBD)
    idx = torch.full((1, CONTEXT_LENGTH), fill_value=2, dtype=torch.long)

    out = model(idx)

    for t in range(CONTEXT_LENGTH):
        for other_t in range(t + 1, CONTEXT_LENGTH):
            assert not torch.allclose(out[0, t], out[0, other_t])
