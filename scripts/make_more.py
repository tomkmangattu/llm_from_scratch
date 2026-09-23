import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import random

WORDS_PATH = Path(__file__).parent.parent / "data" / "make_more" / "names.txt"

words = open(WORDS_PATH, "r").read().splitlines()

# builds soti dynamically from the corpus; kept as an alternative to the hardcoded table below
def getCharVsTokens():
    chars = sorted(list(set("".join(words))))
    soti = {s : i + 1 for i, s in enumerate(chars)}
    soti["."] = 0
    return soti

block_size = 3 # no of preceding characters used as context to predict the next one

# soti = getCharVsTokens()
# hardcoded so token ids stay stable across runs regardless of which words are in the corpus
soti = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g':7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 0}
itos = {i: s for s, i in soti.items()}
vocab_size = len(soti)

# turns a list of words into (context, next_char) training pairs using a sliding window of block_size
def build_data_set(words) -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []

    for word in words:
        context = [0] * block_size # '.' padding before the first character
        for chrt in word + ".": # '.' also marks end of word
            x = context
            y = soti[chrt]
            X.append(x)
            Y.append(y)
            context = context[1:] + [y] # slide the window forward
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
# Kaiming-style init: gain (5/3 for tanh) / sqrt(fan_in) keeps preact variance ~1 at init
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3) / ((n_embd * block_size) ** 0.5)
# b1 = torch.randn(n_hidden) * 0.01 # redundant with batchnorm bias below, so left unused

W2 = torch.randn((n_hidden, vocab_size)) * 0.01 # small init so initial logits/loss aren't overconfident
b2 = torch.randn(vocab_size) * 0

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

# running estimates of batchnorm mean/std, updated via EMA during training and used at eval/inference time
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2] # [C, W1, b1, W2, b2]
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
    hpreact = embcat @ W1 # + b1

    # batchnorm: normalize preactivations using this batch's stats, then scale/shift
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        # EMA update of running stats, used later at eval time instead of per-batch stats
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    h = torch.tanh(hpreact)
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
    lossi.append(loss.log10().item()) # log10 so the loss curve is easier to read on a linear plot

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
    hpreact = embcat @ W1 # + b1

    # use running batchnorm stats (not batch stats) since eval isn't done in mini-batches
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('val')
split_loss('test')

# generating sample names

@torch.no_grad()
def sample_names(n=20):
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            hpreact = embcat @ W1
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias

            h = torch.tanh(hpreact)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item() # sample next char id from the distribution
            context = context[1:] + [ix] # slide context window forward, feeding prediction back in
            out.append(ix)
            if ix == 0: # '.' means end of generated word
                break
        print(''.join(itos[i] for i in out))

sample_names()
