import pytest
import tiktoken

from llm.tokenizer import (
    byte_to_unicode,
    count_adjacent_pairs,
    decode,
    decode_with_gpt2_vocab,
    encode_chunk_with_gpt2_vocab,
    expand_merges,
    encode,
    encode_with_bpe,
    encode_with_gpt2_vocab,
    merge,
    find_most_frequent_pair,
    pretokenize,
    unicode_to_byte,
)

_tiktoken_enc = tiktoken.get_encoding("gpt2")


def test_encode_decode_round_trip_ascii():
    text = "Hello, world!"
    assert decode(encode(text)) == text


def test_encode_decode_round_trip_emoji():
    text = "Hi 🌍 from BPE"
    assert decode(encode(text)) == text
    assert isinstance(encode(text), list)


def test_count_adjacent_pairs_counts_overlapping_pairs():
    # "aaaa" -> 3 overlapping (97, 97) pairs, not 2 (non-overlapping)
    assert count_adjacent_pairs([97, 97, 97, 97]) == {(97, 97): 3}


def test_count_adjacent_pairs_empty_and_single_token():
    assert count_adjacent_pairs([]) == {}
    assert count_adjacent_pairs([97]) == {}


def test_merge_replaces_all_non_overlapping_occurrences():
    tokens = [97, 97, 97, 97]
    merge(tokens, (97, 97), 256)
    assert tokens == [256, 256]


def test_merge_ignores_absent_pair():
    tokens = [1, 2, 3]
    merge(tokens, (9, 9), 256)
    assert tokens == [1, 2, 3]


def test_find_most_frequent_pair_picks_highest_count():
    counts = {(1, 2): 1, (3, 4): 5, (5, 6): 2}
    assert find_most_frequent_pair(counts) == (3, 4)


def test_find_most_frequent_pair_returns_none_below_threshold():
    # a pair that only occurs once anywhere is not worth merging
    assert find_most_frequent_pair({(1, 2): 1}) is None
    assert find_most_frequent_pair({}) is None


def test_expand_merges_reverses_single_level_merge():
    tokens = [256, 256]
    expand_merges(tokens, {(97, 97): 256})
    assert tokens == [97, 97, 97, 97]


def test_expand_merges_reverses_nested_merge():
    # 257 = (256, 97), and 256 itself = (97, 97) -> must fully expand both levels
    tokens = [257]
    expand_merges(tokens, {(97, 97): 256, (256, 97): 257})
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
    expand_merges(tokens, merged_pair)
    assert decode(tokens) == text


# GPT-2 vocab tests start

def test_byte_to_unicode_is_a_bijection_over_all_256_bytes():
    byte_vs_unicode = byte_to_unicode()
    assert len(byte_vs_unicode) == 256
    assert len(set(byte_vs_unicode.values())) == 256


def test_byte_to_unicode_maps_printable_ascii_to_itself():
    byte_vs_unicode = byte_to_unicode()
    assert byte_vs_unicode[65] == "A"
    assert byte_vs_unicode[33] == "!"


def test_byte_to_unicode_remaps_non_printable_bytes_above_256():
    byte_vs_unicode = byte_to_unicode()
    assert ord(byte_vs_unicode[0]) >= 256


def test_unicode_to_byte_inverts_byte_to_unicode():
    byte_vs_unicode = byte_to_unicode()
    unicode_vs_byte = unicode_to_byte()
    assert all(unicode_vs_byte[ch] == byte for byte, ch in byte_vs_unicode.items())


def test_pretokenize_splits_contraction_from_its_stem():
    assert pretokenize("isn't") == ["isn", "'t"]


def test_pretokenize_keeps_leading_space_attached_to_word():
    assert pretokenize(" dog") == [" dog"]


def test_pretokenize_leaves_exactly_one_space_for_the_next_word_chunk():
    # "a  b" -> "a", one lone space chunk, then " b" (leading space attaches to "b")
    assert pretokenize("a  b") == ["a", " ", " b"]


def test_pretokenize_splits_punctuation_from_preceding_word():
    assert pretokenize("Hello World!") == ["Hello", " World", "!"]


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox jumps over the lazy dog.",
        "To stale 't a little more.",
        "isn't that a lovely   dog, right?",
        "unicode: café, naïve, 日本語 \U0001f600\U0001f680",
        "",
        "   ",
        "####hashtag####",
    ],
)
def test_encode_with_gpt2_vocab_matches_tiktoken(text):
    assert encode_with_gpt2_vocab(text) == _tiktoken_enc.encode(text)


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox jumps over the lazy dog.",
        "isn't that a lovely   dog, right?",
        "unicode: café, naïve, 日本語 \U0001f600\U0001f680",
    ],
)
def test_decode_with_gpt2_vocab_round_trips(text):
    assert decode_with_gpt2_vocab(encode_with_gpt2_vocab(text)) == text


def test_decode_with_gpt2_vocab_does_not_mutate_caller_list():
    token_ids = encode_with_gpt2_vocab("isn't that a lovely dog?")
    token_ids_before = list(token_ids)
    decode_with_gpt2_vocab(token_ids)
    assert token_ids == token_ids_before


def test_encode_chunk_with_gpt2_vocab_matches_tiktoken_for_single_chunk():
    from llm.tokenizer import gpt2_merges, gpt2_vocab

    chunk = " dog"
    byte_vs_unicode = byte_to_unicode()
    assert encode_chunk_with_gpt2_vocab(
        chunk, gpt2_merges, gpt2_vocab, byte_vs_unicode
    ) == _tiktoken_enc.encode(chunk)

# GPT-2 vocab tests end
