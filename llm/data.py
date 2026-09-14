from pathlib import Path
from typing import Any
import numpy as np
import os
import torch
import warnings

from llm.tokenizer import encode_with_gpt2_vocab

shakespeare_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
shakespeare_out_path = Path(__file__).parent.parent / "data" / "tinyshakespeare_000000.bin"
data_dir = Path(__file__).parent.parent / "data"
SHAKESPEARE_PATTERN = "tinyshakespeare_*.bin"  # per-corpus: a bare *.bin would mix corpora
TOKEN_DTYPE = np.uint16
TOKEN_ITEMSIZE = np.dtype(TOKEN_DTYPE).itemsize  # bytes per token on disk
TOKEN_MAX = np.iinfo(TOKEN_DTYPE).max            # largest id the dtype can hold

def tokenize_to_bin(text_path: Path, out_path: Path) -> int:
    """Encode text_path with the GPT-2 tokenizer and cache it as a token shard.
    """
    if out_path.exists():
        return out_path.stat().st_size // TOKEN_ITEMSIZE
    with open(text_path, encoding="utf-8") as file:
        text = file.read()
    gpt2_tokens = encode_with_gpt2_vocab(text)
    if len(gpt2_tokens) < 1:
        raise ValueError("Empty string")

    if max(gpt2_tokens) > TOKEN_MAX:
        raise ValueError(f"token id {max(gpt2_tokens)} does not fit in {np.dtype(TOKEN_DTYPE).name}")
    tmp_path = out_path.with_suffix(".bin.tmp")
    np.array(gpt2_tokens, dtype=TOKEN_DTYPE).tofile(tmp_path)
    os.replace(tmp_path, out_path)
    return len(gpt2_tokens)

def load_shards(paths: list[Path]) -> list[np.memmap]:
    """Open each shard read-only. Order is significant: it defines the token stream."""
    if not paths:
        raise ValueError("no shards given")
    return [np.memmap(p, dtype=TOKEN_DTYPE, mode="r") for p in paths]
            
def shard_paths(directory: Path, pattern: str = "*.bin") -> list[Path]:
    """Deterministic, sorted shard list. Glob order is filesystem order — never use it raw."""
    return sorted(directory.glob(pattern))

# Worked example -- two shards, so the flat index has to be resolved to a shard.
#
#   shard 0 = [5962, 22307, 25, 198, 8421, 356, 5120]          7 tokens
#             'First Citizen:\nBefore we proceed'
#   shard 1 = [597, 2252, 11, 3285, 502, 2740, 13, 198, 198]   9 tokens
#             ' any further, hear me speak.\n\n'
#   context_length T = 4     (tokens per window)
#   stride         = 2     (token distance between window starts)
#
# __init__ counts windows per shard, then accumulates:
#   _windows_per_shard = [2, 3]      max(0, (7-4-1)//2+1) = 2,  (9-4-1)//2+1 = 3
#   _shard_bounds      = [0, 2, 5]   shard 0 owns flat [0,2); shard 1 owns [2,5)
#   len(dataset)       = 5           = _shard_bounds[-1]
#
# __getitem__ maps flat index -> (shard, local window) -> token offset:
#
#   index | shard local start | x                       y
#   ------+-------------------+----------------------------------------------
#     0   |   0     0     0   | 'First Citizen:\n'       ' Citizen:\nBefore'
#     1   |   0     1     2   | ':\nBefore we'           '\nBefore we proceed'
#     2   |   1     0     0   | ' any further, hear'     ' further, hear me'
#     3   |   1     1     2   | ', hear me speak'        ' hear me speak.'
#     4   |   1     2     4   | ' me speak.\n'           ' speak.\n\n'
#
# Note index 2: shard flips to 1 and both local and start reset to 0 -- windows
# never span two shards. y is always x advanced by exactly one token.

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, shards: list[np.memmap], context_length: int, stride: int) -> None:
        super().__init__()
        if context_length < 1:
            raise ValueError(f'context length {context_length} is less than 1')
        if stride < 1:
            raise ValueError(f'Stride {stride} is less than 1')
        self.shards = shards
        self.T = context_length
        self.stride = stride

        self._windows_per_shard = []
        for s in self.shards:
            self._windows_per_shard.append(max(0, (len(s) - self.T - 1) // self.stride + 1))
        self._shard_bounds = [0]
        for c in self._windows_per_shard:
            self._shard_bounds.append(self._shard_bounds[-1] + c)
        # coverage check
        total_tokens = sum(len(s) for s in self.shards)
        if total_tokens < 1:
            raise ValueError("Total number of tokens less than 1")
        self.coverage = len(self) * self.T / total_tokens
        if self.stride > self.T :
            warnings.warn('There are gaps between windows, so tokens are skipped entirely')

    # number of chunks/ x and y pairs
    def __len__(self):
        return self._shard_bounds[-1]


    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} is greater than {len(self)}') # Python's legacy iteration protocol calls __getitem__ with 0, 1, 2, … and stops on IndexError
        shard = np.searchsorted(self._shard_bounds, index, side="right") - 1 # which shard

        # where in the shard 
        local = index - self._shard_bounds[shard] # this is the 0th / 1st / 2nd window of this shard.
        start = local * self.stride # begin reading at token N.

        s = self.shards[shard]
        x = s[start: start + self.T].astype(np.int64)
        y = s[start + 1 : start + self.T + 1].astype(np.int64)

        return (torch.from_numpy(x), torch.from_numpy(y))


if __name__ == "__main__":
    print(tokenize_to_bin(shakespeare_path, shakespeare_out_path))
    gpt2_tokens : np.memmap = np.memmap(shakespeare_out_path, dtype=TOKEN_DTYPE, mode="r")
    print(gpt2_tokens)
    shard_files = shard_paths(data_dir, SHAKESPEARE_PATTERN)
    print(load_shards(shard_files)[0])

    shards: list[np.memmap] = [gpt2_tokens[0: 10], gpt2_tokens[10: 20]]  # type: ignore
    print(shards)
    dt = TokenDataset(shards, 4, 3)
    print(dt[0])
    print(dt[1])