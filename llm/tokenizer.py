
# byte level pretokenizer start
def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))

def decode(byte_values: list[int]) -> str:
    return bytes(byte_values).decode("utf-8")

def adjacent_token_counter(token_list: list[int]) -> dict[tuple[int, int], int]:
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
    idx = 0
    while idx < len(token_list) - 1:
        if token_list[idx] == pair[0] and token_list[idx + 1] == pair[1]:
            token_list[idx] = new_id
            token_list.pop(idx + 1)
        idx += 1

def most_freq_finder(adjacent_token_count : dict[tuple[int, int], int]) -> tuple[int, int] | None:
    freq_pair : tuple[int, int] | None = None
    freq = 0
    for key, value in adjacent_token_count.items():
        if not freq_pair:
            freq_pair = key; freq = value
        elif value > freq:
            freq_pair = key; freq = value
    if freq > 1 and freq_pair:
        return freq_pair
    return None

def decode_most_freq(token_list : list[int], merged_pair : dict[tuple[int, int], int]) -> None:
    new_token_merged = {new_token: pair for pair, new_token in merged_pair.items()}
    idx = 0
    while idx < len(token_list):
        if token_list[idx] > 255 :
            actual_pair = new_token_merged[token_list[idx]]
            token_list[idx] = actual_pair[0]
            token_list.insert(idx + 1, actual_pair[1])
        else:
            idx += 1

def encode_with_bpe(text: str, merged_pair : dict[tuple[int, int], int]) -> list[int]:
    encoded_list = encode(text)
    code_vs_pair = {code: pair for pair, code in merged_pair.items()}
    sorted_code = sorted(code_vs_pair.keys())

    adjacent_pair_count = adjacent_token_counter(encoded_list)
    for code in sorted_code:
        current_pair = code_vs_pair[code]
        if adjacent_pair_count.get(current_pair):
            merge(encoded_list, current_pair, code)
            adjacent_pair_count = adjacent_token_counter(encoded_list)

    return encoded_list

def byte_level_tokenizer():
    text = "Hi my name is tom. I am from India"
    # text = "aaaaaaaa"
    merged_pair : dict[tuple[int, int], int] = {}
    next_token = 256
    token_list = encode(text)
    print(f'Embeded text: {token_list}')
    adjacent_token_count = adjacent_token_counter(token_list)
    print(f'Adjacent pair count : {adjacent_token_count}')

    most_freq_element = most_freq_finder(adjacent_token_count)
    while most_freq_element:
        # print(f'Most freq element {most_freq_element}')
        merge(token_list, most_freq_element, next_token)
        merged_pair[most_freq_element] = next_token
        next_token += 1
        adjacent_token_count = adjacent_token_counter(token_list)
        most_freq_element = most_freq_finder(adjacent_token_count)

    print(f'Embeded text after freq replace: {token_list}')
    print(f'Merged pairs {merged_pair}')
    decode_most_freq(token_list, merged_pair)
    print(f'Reversing merge {token_list}')
    str_decoded = decode(token_list)
    print(f'decoded text: {str_decoded}')

    new_text = "Hi am a software developer from india"
    text_emc = encode_with_bpe(new_text, merged_pair)
    print(f'Encoded text {text_emc}')
    decode_most_freq(text_emc, merged_pair)
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

if __name__ == "__main__":
    # char_level_example()
    byte_level_tokenizer()
