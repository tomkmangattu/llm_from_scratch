import json
from pathlib import Path
import regex

vocab_path = Path(__file__).parent.parent / "data/gpt2_vocab.json"
merge_path = Path(__file__).parent.parent / "data/gpt2_merges.txt"

gpt2_pattern = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

with open(vocab_path) as file:
    gpt2_vocab = json.load(file)

gpt2_rev_vocab = {value:key for key, value in gpt2_vocab.items()}

with open(merge_path) as file:
    gpt2_merges = []
    for line in file:
        if line.startswith("#version:"):
            continue
        pair = line.strip().split()
        gpt2_merges.append((pair[0], pair[1]))

merge_to_actual = { gpt2_vocab[merge[0] + merge[1]] : (gpt2_vocab[merge[0]], gpt2_vocab[merge[1]]) for merge in gpt2_merges}

# byte level pretokenizer start
def encode(text: str) -> list[int]:
    # base vocab is the 256 byte values, so any UTF-8 string decomposes with no OOV
    return list(text.encode("utf-8"))

def decode(byte_values: list[int]) -> str:
    # only valid once every merged id has been expanded back to raw bytes via expand_merges
    return bytes(byte_values).decode("utf-8")

def count_adjacent_pairs(token_list: list[int]) -> dict[tuple[int, int], int]:
    # idx += 1 (not += 2) so overlapping runs like "aaaa" count all 3 adjacent pairs, not just 2
    adjacent_pair_count: dict[tuple[int, int], int] = {}
    idx = 0
    while idx < len(token_list) - 1:
        pair = token_list[idx], token_list[idx+1]
        if pair in adjacent_pair_count:
            adjacent_pair_count[pair] += 1
        else:
            adjacent_pair_count[pair] = 1
        idx += 1
    return adjacent_pair_count

def merge(token_list : list[int], pair: tuple[int, int], new_id : int) -> None:
    # pop() already shifts everything left by one, so idx += 1 (not += 2) reaches the next unconsumed position
    idx = 0
    while idx < len(token_list) - 1:
        if token_list[idx] == pair[0] and token_list[idx + 1] == pair[1]:
            token_list[idx] = new_id
            token_list.pop(idx + 1)
        idx += 1

def find_most_frequent_pair(adjacent_token_count : dict[tuple[int, int], int]) -> tuple[int, int] | None:
    freq_pair : tuple[int, int] | None = None
    freq = 0
    for key, value in adjacent_token_count.items():
        if not freq_pair:
            freq_pair = key; freq = value
        elif value > freq:
            freq_pair = key; freq = value
    # freq > 1 stops training once no pair repeats, so we never merge a one-off occurrence
    if freq > 1 and freq_pair:
        return freq_pair
    return None

def expand_merges(token_list : list[int], merged_pair : dict[tuple[int, int], int]) -> None:
    new_token_merged = {new_token: pair for pair, new_token in merged_pair.items()}
    idx = 0
    while idx < len(token_list):
        if token_list[idx] > 255 :
            # idx does NOT advance here: actual_pair[0] may itself be a merged id (nested merge), so re-check the same slot
            actual_pair = new_token_merged[token_list[idx]]
            token_list[idx] = actual_pair[0]
            token_list.insert(idx + 1, actual_pair[1])
        else:
            idx += 1

def encode_with_bpe(text: str, merged_pair : dict[tuple[int, int], int]) -> list[int]:
    encoded_list = encode(text)
    code_vs_pair = {code: pair for pair, code in merged_pair.items()}
    # new_id was assigned in increasing order during training, so ascending code order == learned-first order
    sorted_code = sorted(code_vs_pair.keys())

    adjacent_pair_count = count_adjacent_pairs(encoded_list)
    for code in sorted_code:
        current_pair = code_vs_pair[code]
        if adjacent_pair_count.get(current_pair):
            merge(encoded_list, current_pair, code)
            # recompute after every merge since pair positions shift once tokens are consumed
            adjacent_pair_count = count_adjacent_pairs(encoded_list)

    return encoded_list

def byte_level_tokenizer():
    text = "Hi my name is tom. I am from India"
    # text = "aaaaaaaa"
    merged_pair : dict[tuple[int, int], int] = {}
    next_token = 256
    token_list = encode(text)
    print(f'Embeded text: {token_list}')
    adjacent_token_count = count_adjacent_pairs(token_list)
    print(f'Adjacent pair count : {adjacent_token_count}')

    most_freq_element = find_most_frequent_pair(adjacent_token_count)
    while most_freq_element:
        # print(f'Most freq element {most_freq_element}')
        merge(token_list, most_freq_element, next_token)
        merged_pair[most_freq_element] = next_token
        next_token += 1
        adjacent_token_count = count_adjacent_pairs(token_list)
        most_freq_element = find_most_frequent_pair(adjacent_token_count)

    print(f'Embeded text after freq replace: {token_list}')
    print(f'Merged pairs {merged_pair}')
    expand_merges(token_list, merged_pair)
    print(f'Reversing merge {token_list}')
    str_decoded = decode(token_list)
    print(f'decoded text: {str_decoded}')

    new_text = "Hi am a software developer from india"
    text_emc = encode_with_bpe(new_text, merged_pair)
    print(f'Encoded text {text_emc}')
    expand_merges(text_emc, merged_pair)
    text_dec =  decode(text_emc)
    print(f'Decoded text {text_dec}')
# byte level pretokenizer end

# character level tokenizer start
def char_level_example():
    embedding :dict[str, int] = {}
    rev_embedding : dict[int, str] = {}

    def encode(text: str):
        word_list = list(text)
        sorted_words_set = sorted(set(word_list))
        for idx, word in enumerate(sorted_words_set):
            embedding[word] = idx
            rev_embedding[idx] = word
        return [embedding.get(word, 0) for word in word_list]

    def decode(nums: list[int]):
        letters = [rev_embedding.get(num, "") for num in nums]
        return "".join(letters)
    
    text = "Hello World!"
    str_embed = encode(text)
    print(f'Embeded text: {str_embed}')
    str_decoded = decode(str_embed)
    print(f'decoded text: {str_decoded}')

# character level tokenizer end

# GPT-2 vocab loading start
def byte_to_unicode() -> dict[int, str]:
    byte_vs_unicode : dict[int, str] = {}
    next_token = 256
    for idx in range(0, 256):
        if 33 <= idx <= 126 or 161 <= idx <= 172 or 174 <= idx <= 255 :
            ch = chr(idx)
            byte_vs_unicode[idx] = ch
        else:
            byte_vs_unicode[idx] = chr(next_token)
            next_token += 1
    return byte_vs_unicode

def unicode_to_byte() -> dict[str, int]:
    return {unicode: byte for byte , unicode in byte_to_unicode().items()}

def encode_chunk_with_gpt2_vocab(chunk: str, gpt2_merges, gpt2_vocab, byte_vs_unicode: dict[int, str]) -> list[int]:
    byte_list = encode(chunk) # text to byte
    char_list = [byte_vs_unicode[byte] for byte in byte_list]  # byte to unicode
    token_ids = [gpt2_vocab[char] for char in char_list] # unicode to gpt2_token

    # NOTE: runs the full 50,000-rule gpt2_merges scan per chunk (not per text), so per-chunk overhead
    # is much higher than before pretokenize() was added; fix later if this becomes a bottleneck
    adjacent_pair_count = count_adjacent_pairs(token_ids)
    for left, right in gpt2_merges:
        pair_ids = (gpt2_vocab[left], gpt2_vocab[right]) # merge pairs are unicode, converting to tokens
        if adjacent_pair_count.get(pair_ids):
            new_id = gpt2_vocab[left + right] # new token of merge pairs
            merge(token_ids, pair_ids, new_id)
            # recompute after every merge since pair positions shift once tokens are consumed
            adjacent_pair_count = count_adjacent_pairs(token_ids)
    return token_ids

def encode_with_gpt2_vocab(text: str, gpt2_merges = gpt2_merges, gpt2_vocab = gpt2_vocab) -> list[int]:
    # merges must never cross a pretokenize() chunk boundary, so each chunk is BPE-encoded independently
    byte_vs_unicode = byte_to_unicode()
    gpt2_tokens = []
    for chunk in pretokenize(text):
        gpt2_tokens.extend(encode_chunk_with_gpt2_vocab(chunk, gpt2_merges, gpt2_vocab, byte_vs_unicode))
    return gpt2_tokens

def decode_with_gpt2_vocab(token_ids: list[int]) -> str:
    token_ids = list(token_ids)  # avoid mutating the caller's list
    idx = 0
    while idx < len(token_ids):
        if token_ids[idx] in merge_to_actual:
            left, right = merge_to_actual[token_ids[idx]]
            token_ids[idx] = left
            # idx does NOT advance here: left may itself be a merged id (nested merge), so re-check the same slot
            token_ids.insert(idx + 1, right)
        else:
            idx += 1

    char_list = [gpt2_rev_vocab[token] for token in token_ids] # converting to chars
    unicode_vs_byte = unicode_to_byte()
    byte_list = [unicode_vs_byte[char] for char in char_list] # converting to byte array
    return decode(byte_list)

def pretokenize(text: str) -> list[str]:
    return gpt2_pattern.findall(text)

if __name__ == "__main__":
    # char_level_example()
    # byte_level_tokenizer()
    gpt2_tokens = encode_with_gpt2_vocab("Hello World! This is a test sentence for tokenization.")
    print(f'GPT2 token encoded : {gpt2_tokens}')
    decoded_line = decode_with_gpt2_vocab(gpt2_tokens)
    print(f'Decoded Line : {decoded_line}')
