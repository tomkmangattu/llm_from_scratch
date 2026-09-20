from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from llm.data import shard_paths, SHAKESPEARE_PATTERN, load_shards, TokenDataset

data_dir = Path(__file__).parent.parent / "data" / "bpe2k"

class BigramLM(nn.Module):
    #     weight = [
    # [ 0.1, -0.2,  0.0,  0.3, -0.1,  0.2],   # row 0: "what follows token 0"
    # [ 0.4,  0.1, -0.3,  0.0,  0.2, -0.1],   # row 1
    # [-0.2,  0.3,  0.1, -0.1,  0.0,  0.4],   # row 2
    # [ 0.0, -0.1,  0.2,  0.4, -0.3,  0.1],   # row 3
    # [ 0.2,  0.0, -0.1,  0.1,  0.3, -0.2],   # row 4
    # [-0.1,  0.2,  0.3, -0.2,  0.1,  0.0],   # row 5
    # ]
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        # nn.Embedding defaults to std=1, too spread out for logits
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    # idx = torch.tensor([[2, 5, 1, 3]])   # shape (1, 4) — token ids, e.g. "First Citizen:\n"-ish
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor| None]:
        logits = self.token_embedding(idx) # This just gathers rows 2, 5, 1, 3 out of weight, in that order:

        loss = None
        if targets is not None:
            # logits.shape == (1, 4, 6)   # (B, T, vocab_size)
            # logits[0, 0] == weight[2]   # what comes after token 2? -> [-0.2, 0.3, 0.1, -0.1, 0.0, 0.4]
            # logits[0, 1] == weight[5]   # what comes after token 5?
            B, T, vocab_size = logits.shape
            # TokenDataset gives you y = x shifted by one: targets = torch.tensor([[5, 1, 3, ??]])
            # logits.shape        == (1, 4, 6)
            # logits.view(4, 6)   == (4, 6)
            # targets.shape == (1, 4)
            # targets == [[5, 1, 3, 8]]  
            loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T))
        return logits, loss


class BigramLMNaive(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor| None]:
        weight = self.token_embedding.weight
        B, T = idx.shape
        logits = weight[idx]

        loss = None
        if targets is not None:
            losses = []
            for b in range(B):
                for t in range(T):
                    row = logits[b, t]
                    target = targets[b, t]

                    expo = torch.exp(row)
                    probs =  expo / sum(expo)
                    loss_bt = - torch.log(probs[target])
                    losses.append(loss_bt)

            loss = torch.stack(losses).mean()
        
        return logits, loss

# VOCAB_SIZE = 50257
VOCAB_SIZE = 2256

if __name__ == "__main__":
    # bpe2k vocab (2256), not real GPT-2 (50257): vocab_size^2 embedding blows past 16GB RAM at GPT-2 scalewh
    shard_files = shard_paths(data_dir, SHAKESPEARE_PATTERN)
    shardList = load_shards(shard_files)

    shard = shardList[0]
    split = int(0.9 * len(shard))
    print(f'split {split}')
    train_shard, val_shard = shard[:split], shard[split:]

    train_ds = TokenDataset([train_shard], context_length=128, stride=128) #type: ignore
    val_ds = TokenDataset([val_shard], context_length=128, stride=128) #type: ignore

    model = BigramLM(VOCAB_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)

    global_step = 0
    for epoch in range(100):
        for step, (xb, yb) in enumerate(train_loader):
            logits, loss = model(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    xb_val, yb_val = next(iter(val_loader))
                    _, val_loss = model(xb_val, yb_val)

                print(f'step {global_step} train loss {loss} validation loss {val_loss}')
                model.train()
            
            if global_step >= 2000:
                break

            global_step += 1

        if global_step >= 2000:
            break
    