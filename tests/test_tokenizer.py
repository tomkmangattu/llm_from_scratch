from llm.tokenizer import (
    adjacent_token_counter,
    decode,
    decode_most_freq,
    encode,
    encode_with_bpe,
    merge,
    most_freq_finder,
)


def test_encode_decode_round_trip_ascii():
    text = "Hello, world!"
    assert decode(encode(text)) == text


def test_encode_decode_round_trip_emoji():
    text = "Hi 🌍 from BPE"
    assert decode(encode(text)) == text
    assert isinstance(encode(text), list)


def test_adjacent_token_counter_counts_overlapping_pairs():
    # "aaaa" -> 3 overlapping (97, 97) pairs, not 2 (non-overlapping)
    assert adjacent_token_counter([97, 97, 97, 97]) == {(97, 97): 3}


def test_adjacent_token_counter_empty_and_single_token():
    assert adjacent_token_counter([]) == {}
    assert adjacent_token_counter([97]) == {}


def test_merge_replaces_all_non_overlapping_occurrences():
    tokens = [97, 97, 97, 97]
    merge(tokens, (97, 97), 256)
    assert tokens == [256, 256]


def test_merge_ignores_absent_pair():
    tokens = [1, 2, 3]
    merge(tokens, (9, 9), 256)
    assert tokens == [1, 2, 3]


def test_most_freq_finder_picks_highest_count():
    counts = {(1, 2): 1, (3, 4): 5, (5, 6): 2}
    assert most_freq_finder(counts) == (3, 4)


def test_most_freq_finder_returns_none_below_threshold():
    # a pair that only occurs once anywhere is not worth merging
    assert most_freq_finder({(1, 2): 1}) is None
    assert most_freq_finder({}) is None


def test_decode_most_freq_reverses_single_level_merge():
    tokens = [256, 256]
    decode_most_freq(tokens, {(97, 97): 256})
    assert tokens == [97, 97, 97, 97]


def test_decode_most_freq_reverses_nested_merge():
    # 257 = (256, 97), and 256 itself = (97, 97) -> must fully expand both levels
    tokens = [257]
    decode_most_freq(tokens, {(97, 97): 256, (256, 97): 257})
    assert tokens == [97, 97, 97]


def test_encode_with_bpe_merges_pair_at_index_zero():
    assert encode_with_bpe("am", {(97, 109): 256}) == [256]


def test_encode_with_bpe_merges_all_occurrences_of_a_pair():
    merged_pair = {(97, 109): 256}
    assert encode_with_bpe("am am am", merged_pair) == [256, 32, 256, 32, 256]


def test_encode_with_bpe_applies_multiple_learned_merges():
    merged_pair = {(97, 109): 256, (111, 109): 257, (32, 73): 258}
    assert encode_with_bpe("am I am", merged_pair) == [256, 258, 32, 256]


def test_bpe_encode_decode_round_trip():
    merged_pair = {(97, 109): 256, (111, 109): 257, (32, 73): 258}
    text = "Hi am a software developer from india"
    tokens = encode_with_bpe(text, merged_pair)
    decode_most_freq(tokens, merged_pair)
    assert decode(tokens) == text
