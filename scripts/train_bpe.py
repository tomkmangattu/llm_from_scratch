import matplotlib.pyplot as plt

from llm.tokenizer import encode, count_adjacent_pairs, merge, find_most_frequent_pair

SLICE_SIZE = 300_000
NUM_MERGES = 300
SNAPSHOT_EVERY = 25 

with open("data/tinyshakespeare.txt", encoding="utf-8") as f:
    text = f.read()[:SLICE_SIZE]

num_words = len(text.split())
history: list[tuple[int, float]] = []
token_list = encode(text)
merged_pair: dict[tuple[int, int], int] = {}
next_token = 256

for i in range(NUM_MERGES):
    adjacent_pairs_count = count_adjacent_pairs(token_list)
    most_freq_pair = find_most_frequent_pair(adjacent_pairs_count)
    if most_freq_pair is None:
        break
    merge(token_list, most_freq_pair, next_token)
    merged_pair[most_freq_pair] = next_token
    next_token += 1

    if (i+1) % SNAPSHOT_EVERY == 0:
        history.append((next_token, len(token_list) / num_words))

print(f'final vocab size: {next_token}')
print(f'tokens: {len(token_list)} started as {len(encode(text))} raw bytes')

vocab_sizes, avg_tokens_per_word = zip(*history)
plt.plot(vocab_sizes, avg_tokens_per_word)

plt.xlabel("vocab_size")
plt.ylabel("avg_tokens_per_word")
plt.title("vocab-size vs avg-tokens-per-word")
plt.savefig("notes/week2/vocab_vs_tokens_per_word.png")