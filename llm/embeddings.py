import torch.nn as nn
import torch

class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, n_embd: int) -> None:
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(context_length, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        position = torch.arange(T)
        return self.token_embd(idx) + self.pos_embd(position)


# token embedding
# row 0: [ 0.1, -0.2,  0.4]
# row 1: [ 0.3,  0.0, -0.1]
# row 2: [-0.5,  0.2,  0.3]   <- token id 2
# row 3: [ 0.0,  0.6, -0.2]
# row 4: [ 0.2, -0.3,  0.1]
# row 5: [-0.1,  0.4,  0.0]   <- token id 5

# position embedding
# row 0: [ 1.0,  0.0,  0.0]   <- position 0
# row 1: [ 0.0,  1.0,  0.0]   <- position 1
# row 2: [ 0.0,  0.0,  1.0]   <- position 2
# row 3: [ 0.5,  0.5,  0.5]   <- position 3

# idx = torch.tensor([[2, 5, 1, 3]])

# self.token_embd(idx) gathers rows 2, 5, 1, 3
# [ [-0.5, 0.2, 0.3],   # token 2
#   [-0.1, 0.4, 0.0],   # token 5
#   [ 0.3, 0.0, -0.1],  # token 1
#   [ 0.0, 0.6, -0.2] ] # token 3

# position = torch.arange(4) = [0, 1, 2, 3]
# [ [1.0, 0.0, 0.0],
#   [0.0, 1.0, 0.0],
#   [0.0, 0.0, 1.0],
#   [0.5, 0.5, 0.5] ]

# Adding them (the (4,3) broadcasts against (1,4,3)) gives the final (1, 4, 3) output: