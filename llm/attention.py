from typing import Any

import torch
import torch.nn as nn
import math

# query  = [1.0, 0.0]

# keys = [
#   [1.0, 0.0],   # position 0
#   [0.0, 1.0],   # position 1
#   [1.0, 1.0],   # position 2
# ]

# values = [
#   [10.0, 0.0],  # position 0
#   [ 0.0, 10.0], # position 1
#   [ 5.0, 5.0],  # position 2
# ]

# query - position asking the question
# key - what does each position advertise about itself, for matching purposes?
# value - what does each position actually contribute once selected?

def attention_v1(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    T, _ = keys.shape
    scores = []
    for i in range(T):
        # i=0: dot([1,0], [1,0]) = 1*1 + 0*0 = 1.0
        scores.append(torch.dot(query, keys[i]))
    attn_weights = torch.softmax(torch.stack(scores), dim=0)

    weighted_values = []
    # 0.422 * [10, 0]  = [4.220, 0.000]
    # 0.155 * [0, 10]  = [0.000, 1.550]
    # 0.422 * [5, 5]   = [2.110, 2.110]
    # -----------------------------------
    # sum              = [6.330, 3.660]
    for i in range(len(values)):
        weighted_values.append(attn_weights[i] * values[i])

    return torch.stack(weighted_values).sum(dim=0)

# example for SelfAttention
# d_in =  4
# d_out = 2
# T =     3

# W_q.weight = [[1, 0, 0, 1],    # produces output dim 0
#               [0, 1, 1, 0]]    # produces output dim 1

# x = [[1, 0, 1, 0],   # position 0
#      [0, 2, 0, 1],   # position 1
#      [1, 1, 0, 0]]   # position 2

# Q[0] = [1, 1] = [ x[0] * W_q.weight[0], x[0] * W_q.weight[1]]

# Q = [[1, 1],
#      [1, 2],
#      [1, 1]]


class SelfAttention(nn.Module):

    def __init__(self, d_in : int, d_out : int) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        Q = self.W_q(x); K = self.W_k(x); V = self.W_v(x)
        scores = Q @ K.T
        scores *= 1/math.sqrt(self.d_out)
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_weights @ V


