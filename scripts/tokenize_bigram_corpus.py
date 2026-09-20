from pathlib import Path
import json
import numpy as np

from llm.tokenizer import encode, count_adjacent_pairs, find_most_frequent_pair, merge

with open("data/tinyshakespeare.txt",  encoding="utf-8") as f:
    text = f.read()

merge_path = Path(__file__).parent.parent / "data" / "bpe2k" / "merges.json"
shakespeare_sample = Path(__file__).parent.parent / "data" / "bpe2k" / "tinyshakespeare_000000.bin"
merge_path.parent.mkdir(parents=True, exist_ok=True)

NUM_MERGES = 2000
TOKEN_DTYPE = np.uint16
TOKEN_MAX = np.iinfo(TOKEN_DTYPE).max

token_list = encode(text=text)
merge_pair : dict[tuple[int, int], int] = {}
next_token = 256

for i in range(NUM_MERGES):
    adjacent_pair_count = count_adjacent_pairs(token_list)
    most_freq_pair = find_most_frequent_pair(adjacent_pair_count)

    if most_freq_pair is None:
        break

    merge(token_list, most_freq_pair, next_token)
    merge_pair[most_freq_pair] = next_token
    next_token += 1

    if i % 100 == 0:
        print(i)

with open(merge_path, "w") as file:
    json.dump(list(merge_pair.items()), file)

if len(token_list) < 1:
    raise ValueError("Empty string")

if max(token_list) > TOKEN_MAX:
    raise ValueError(f"token id {max(token_list)} does not fit in {np.dtype(TOKEN_DTYPE).name}")

np.array(token_list, dtype=TOKEN_DTYPE).tofile(shakespeare_sample)

print(f'final vocab size: {next_token}')
print(f'tokens: {len(token_list)} started as {len(encode(text))} raw bytes')