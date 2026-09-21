import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import random

WORDS_PATH = merge_path = Path(__file__).parent.parent / "data" / "make_more" / "names.txt"

words = open(WORDS_PATH, "r").read().splitlines()

def getCharVsTokens():
    chars = sorted(list(set("".join(words))))
    soti = {s : i + 1 for i, s in enumerate(chars)}
    soti["."] = 0
    return soti

block_size = 3

# soti = getCharVsTokens()
soti = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g':7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 0}
itos = {i: s for s, i in soti.items()}
vocab_size = len(soti)

def build_data_set(words) -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []

    for word in words:
        context = [0] * block_size
        for chrt in word + ".":
            x = context
            y = soti[chrt]
            X.append(x)
            Y.append(y)
            context = context[1:] + [y]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return (X, Y)

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_data_set(words=words[:n1])
Xdev, Ydev = build_data_set(words=words[n1:n2])
Xte, Yte = build_data_set(words=words[n2:])

n_embd = 10 # char embedding dimension
n_hidden = 200 # no of nurens in hidden layer

C = torch.randn((vocab_size, n_embd))
W1 = torch.randn((n_embd * block_size, n_hidden))
b1 = torch.randn(n_hidden)

W2 = torch.randn((n_hidden, vocab_size))
b2 = torch.randn(vocab_size)

parameters = [C, W1, b1, W2, b2]
for param in parameters:
    param.requires_grad = True

print(f'Total no of parameters {len(parameters)}')

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construct
    # (low, high, size)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @W2 + b2
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += - lr * p.grad # type: ignore

    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

plt.plot(lossi)
# plt.show()

@torch.no_grad()
def split_loss(split):
    x, y = {
        "val" : (Xdev, Ydev),
        "test" : (Xte, Yte)
    }[split]

    emb = C[x]
    embcat = emb.view(x.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('val')
split_loss('test')

@torch.no_grad()
def sample_names(n=20):
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            h = torch.tanh(embcat @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))

sample_names()
