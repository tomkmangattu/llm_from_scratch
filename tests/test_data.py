import warnings
import os
from typing import cast

import numpy as np
import pytest

from llm.data import tokenize_to_bin
from llm.tokenizer import decode_with_gpt2_vocab

# Multi-byte UTF-8 on purpose: week 2 proved this is where tokenizers break.
SAMPLE = "First Citizen:\nBefore we proceed 🌍, hear me spéak.\n"


def test_roundtrip_through_disk(tmp_path):
    # text -> ids -> uint16 on disk -> memmap -> text must be byte-identical.
    txt = tmp_path / "corpus.txt"
    txt.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "corpus.bin"

    tokenize_to_bin(txt, out)
    tokens = np.memmap(out, dtype=np.uint16, mode="r")

    assert tokens.dtype == np.uint16
    assert decode_with_gpt2_vocab(tokens.tolist()) == SAMPLE


def test_returns_token_count(tmp_path):
    # The return value is the token count, not a max id or a flag.
    txt = tmp_path / "corpus.txt"
    txt.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "corpus.bin"

    n = tokenize_to_bin(txt, out)

    assert n == np.memmap(out, dtype=np.uint16, mode="r").size
    # A flat uint16 file has no header, so bytes // 2 is the count.
    assert n == out.stat().st_size // 2


def test_cache_hit_does_not_retokenize(tmp_path):
    # Second call must skip the work and still report the same count.
    txt = tmp_path / "corpus.txt"
    txt.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "corpus.bin"

    first = tokenize_to_bin(txt, out)
    mtime = os.stat(out).st_mtime_ns

    second = tokenize_to_bin(txt, out)

    assert second == first
    assert os.stat(out).st_mtime_ns == mtime


def test_cache_hit_reads_the_path_it_was_given(tmp_path):
    # Guards against reading a module-level default instead of out_path.
    short = tmp_path / "short.txt"
    short.write_text("hello", encoding="utf-8")
    short_bin = tmp_path / "short.bin"

    long = tmp_path / "long.txt"
    long.write_text(SAMPLE * 20, encoding="utf-8")
    long_bin = tmp_path / "long.bin"

    tokenize_to_bin(short, short_bin)
    tokenize_to_bin(long, long_bin)

    assert tokenize_to_bin(short, short_bin) == short_bin.stat().st_size // 2
    assert tokenize_to_bin(long, long_bin) == long_bin.stat().st_size // 2
    assert tokenize_to_bin(short, short_bin) != tokenize_to_bin(long, long_bin)


def test_empty_corpus_raises(tmp_path):
    txt = tmp_path / "empty.txt"
    txt.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        tokenize_to_bin(txt, tmp_path / "empty.bin")


def test_token_id_too_large_for_uint16_raises(tmp_path, monkeypatch):
    # numpy would wrap 70000 to 4464 silently; the guard must stop it instead.
    monkeypatch.setattr("llm.data.encode_with_gpt2_vocab", lambda text: [1, 70000, 2])

    txt = tmp_path / "corpus.txt"
    txt.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "corpus.bin"

    with pytest.raises(ValueError):
        tokenize_to_bin(txt, out)
    assert not out.exists()


def test_crash_mid_write_leaves_no_usable_cache(tmp_path, monkeypatch):
    # Colab will disconnect. A half-written .bin must never satisfy the cache check.
    txt = tmp_path / "corpus.txt"
    txt.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "corpus.bin"

    def die(*args):
        raise KeyboardInterrupt("simulated kill between write and rename")

    monkeypatch.setattr("llm.data.os.replace", die)
    with pytest.raises(KeyboardInterrupt):
        tokenize_to_bin(txt, out)
    assert not out.exists()

    monkeypatch.undo()
    n = tokenize_to_bin(txt, out)
    assert n == np.memmap(out, dtype=np.uint16, mode="r").size
    assert decode_with_gpt2_vocab(np.memmap(out, dtype=np.uint16, mode="r").tolist()) == SAMPLE


# --- load_shards / shard_paths -------------------------------------------------

from llm.data import TOKEN_DTYPE, load_shards, shard_paths


def write_shard(path, values):
    path.write_bytes(np.array(values, dtype=TOKEN_DTYPE).tobytes())
    return path


def test_open_shards_in_caller_order(tmp_path):
    # The list order IS the token stream order; the function must not re-sort.
    a = write_shard(tmp_path / "a.bin", [1, 2, 3])
    b = write_shard(tmp_path / "b.bin", [40, 50])

    shards = load_shards([b, a])

    assert [s.size for s in shards] == [2, 3]
    assert [int(s[0]) for s in shards] == [40, 1]
    assert all(isinstance(s, np.memmap) for s in shards)
    assert all(s.dtype == TOKEN_DTYPE for s in shards)


def test_shards_are_read_only(tmp_path):
    # mode="r" is what stops a stray assignment from editing the corpus on disk.
    shard = load_shards([write_shard(tmp_path / "a.bin", [1, 2, 3])])[0]

    assert not shard.flags.writeable
    with pytest.raises(ValueError):
        shard[0] = 99


def test_no_shards_raises(tmp_path):
    # The one corrupt input numpy cannot catch: a glob that matched nothing.
    with pytest.raises(ValueError):
        load_shards([])


def test_missing_shard_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_shards([tmp_path / "absent.bin"])


def test_zero_length_shard_raises(tmp_path):
    (tmp_path / "empty.bin").write_bytes(b"")

    with pytest.raises(ValueError):
        load_shards([tmp_path / "empty.bin"])


def test_truncated_shard_raises(tmp_path):
    # An odd byte count cannot be a whole number of uint16 tokens.
    (tmp_path / "odd.bin").write_bytes(b"\x01\x02\x03")

    with pytest.raises(ValueError):
        load_shards([tmp_path / "odd.bin"])


def test_shard_paths_are_sorted(tmp_path):
    # Created in reverse; glob order is filesystem order, so the sort must fix it.
    for i in reversed(range(12)):
        write_shard(tmp_path / f"corpus_{i:06d}.bin", [i])

    found = shard_paths(tmp_path, "corpus_*.bin")

    assert [p.name for p in found] == [f"corpus_{i:06d}.bin" for i in range(12)]
    # Zero-padding is what makes lexical sort agree with numeric order past 9.
    assert [int(s[0]) for s in load_shards(found)] == list(range(12))


def test_shard_paths_filters_by_pattern(tmp_path):
    write_shard(tmp_path / "fineweb_000000.bin", [1])
    write_shard(tmp_path / "fineweb_000001.bin", [2])
    write_shard(tmp_path / "tinyshakespeare_000000.bin", [3])
    (tmp_path / "notes.txt").write_text("not a shard", encoding="utf-8")

    assert [p.name for p in shard_paths(tmp_path, "fineweb_*.bin")] == [
        "fineweb_000000.bin",
        "fineweb_000001.bin",
    ]
    # The bare default would silently mix two corpora into one token stream.
    assert len(shard_paths(tmp_path)) == 3


def test_shard_paths_ignores_interrupted_writes(tmp_path):
    # A killed tokenize_to_bin leaves .bin.tmp behind; it must never be globbed.
    write_shard(tmp_path / "corpus_000000.bin", [1, 2])
    write_shard(tmp_path / "corpus_000001.bin.tmp", [9, 9])

    assert [p.name for p in shard_paths(tmp_path)] == ["corpus_000000.bin"]


def test_shard_paths_empty_directory(tmp_path):
    # Returns [] rather than raising -- load_shards is where that becomes an error.
    assert shard_paths(tmp_path) == []
    with pytest.raises(ValueError):
        load_shards(shard_paths(tmp_path))


def test_tokenize_then_find_then_open(tmp_path):
    # The whole step-1 pipeline: encode -> shard on disk -> find -> memmap.
    txt = tmp_path / "corpus.txt"
    txt.write_text(SAMPLE, encoding="utf-8")
    n = tokenize_to_bin(txt, tmp_path / "corpus_000000.bin")

    shards = load_shards(shard_paths(tmp_path, "corpus_*.bin"))

    assert len(shards) == 1
    assert shards[0].size == n
    assert decode_with_gpt2_vocab(shards[0].tolist()) == SAMPLE




# --- TokenDataset --------------------------------------------------------------

import torch
from torch.utils.data import DataLoader

from llm.data import TokenDataset


def shard(start, n) -> np.memmap:
    """n distinct tokens beginning at `start`, so a window's shard is identifiable.

    A plain ndarray stands in for a memmap: TokenDataset only slices and len()s
    its shards, and np.memmap is an ndarray subclass. The cast keeps the type
    checker quiet without writing a real file per fixture.
    """
    return cast(np.memmap, np.arange(start, start + n, dtype=TOKEN_DTYPE))


def test_len_counts_windows_per_shard():
    # 12 tokens at T=4,stride=4 is 2 windows, not 3: the buggy formula
    # (len-T)//stride+1 says 3, and only a length where (len-T) divides evenly
    # by stride tells the two apart.
    assert len(TokenDataset([shard(0, 12)], 4, 4)) == 2
    # Summing tokens first and applying the formula once gives 5: each shard
    # loses its own tail, and the tails do not join across shards.
    assert len(TokenDataset([shard(0, 12), shard(100, 12)], 4, 4)) == 4


def test_len_clamps_shards_too_short_for_a_window():
    # A 3-token shard yields no window at T=4; it must contribute 0, not a
    # negative count that silently steals windows from its neighbour.
    assert len(TokenDataset([shard(0, 12), shard(100, 3)], 4, 1)) == 8


def test_shard_bounds_accumulate():
    ds = TokenDataset([shard(0, 7), shard(100, 9)], 4, 2)

    assert ds._windows_per_shard == [2, 3]
    assert ds._shard_bounds == [0, 2, 5]
    assert len(ds) == 5


def test_init_rejects_bad_parameters():
    with pytest.raises(ValueError):
        TokenDataset([shard(0, 12)], 0, 4)
    with pytest.raises(ValueError):
        TokenDataset([shard(0, 12)], 4, 0)


def test_init_rejects_empty_shards():
    # Otherwise len()==0 and a training run completes instantly having learned
    # nothing. Must be a ValueError, not a ZeroDivisionError from coverage.
    with pytest.raises(ValueError):
        TokenDataset([], 4, 4)


def test_getitem_shapes_and_dtype():
    x, y = TokenDataset([shard(0, 12)], 4, 4)[0]

    assert x.shape == (4,) and y.shape == (4,)
    # nn.Embedding rejects uint16 indices; the int64 cast is not optional.
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_getitem_target_is_input_shifted_by_one():
    x, y = TokenDataset([shard(0, 12)], 4, 4)[0]

    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]
    assert torch.equal(y[:-1], x[1:])   # the invariant, independent of literals
    assert not torch.equal(x, y)        # y == x means the model learns to copy


def test_getitem_walks_every_window_in_order():
    ds = TokenDataset([shard(0, 12), shard(100, 12)], 4, 4)

    assert [ds[i][0].tolist() for i in range(len(ds))] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [100, 101, 102, 103],
        [104, 105, 106, 107],
    ]
    assert [ds[i][1].tolist() for i in range(len(ds))] == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [101, 102, 103, 104],
        [105, 106, 107, 108],
    ]


def test_getitem_never_crosses_a_shard_boundary():
    # Disjoint value ranges: a window mixing them was assembled from two shards.
    ds = TokenDataset([shard(0, 12), shard(100, 12)], 4, 4)

    for i in range(len(ds)):
        x, y = ds[i]
        values = x.tolist() + y.tolist()
        assert max(values) - min(values) < 50, f"window {i} spans shards: {values}"


def test_getitem_respects_stride():
    # 14 tokens, T=4, stride=3 -> starts 0, 3, 6, 9. Catches `start = local`
    # (stride honoured in the count but ignored in the placement).
    ds = TokenDataset([shard(0, 14)], 4, 3)

    assert len(ds) == 4
    assert [ds[i][0][0].item() for i in range(len(ds))] == [0, 3, 6, 9]


def test_getitem_uses_the_shard_offset():
    # Catches `start = index * stride`, which ignores which shard we are in and
    # reads past the end of every shard after the first.
    ds = TokenDataset([shard(0, 12), shard(100, 12)], 4, 4)

    assert ds[2][0].tolist() == [100, 101, 102, 103]   # local resets to 0
    assert ds[3][0].tolist() == [104, 105, 106, 107]


def test_getitem_raises_index_error_out_of_range():
    # DataLoader trusts __len__; beyond it must raise, not return a short window.
    # Negative indices are the silent case: without an explicit bounds check they
    # index backwards through _shard_bounds and return EMPTY tensors, which
    # DataLoader happily collates into a zero-width batch.
    ds = TokenDataset([shard(0, 12)], 4, 4)

    for bad in (len(ds), len(ds) + 10, -1, -len(ds) - 1):
        with pytest.raises(IndexError):
            ds[bad]


def test_dataloader_produces_batched_tensors():
    ds = TokenDataset([shard(0, 12), shard(100, 12)], 4, 4)
    xb, yb = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))

    assert xb.shape == (2, 4) and yb.shape == (2, 4)
    assert xb.dtype == torch.int64 and yb.dtype == torch.int64
    assert xb.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert yb.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_coverage_is_one_for_a_clean_partition():
    ds = TokenDataset([shard(0, 1200)], 4, 4)

    assert ds.coverage == pytest.approx(1.0, abs=0.01)


def test_coverage_reports_overlap():
    ds = TokenDataset([shard(0, 1200)], 4, 2)

    assert ds.coverage == pytest.approx(2.0, abs=0.01)


def test_coverage_warns_when_discarding_tokens():
    # stride > context_length silently drops corpus; it is legal but must be loud.
    with pytest.warns(UserWarning):
        ds = TokenDataset([shard(0, 1200)], 4, 8)

    assert ds.coverage == pytest.approx(0.5, abs=0.01)


def test_no_warning_for_a_clean_partition():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TokenDataset([shard(0, 1200)], 4, 4)


def test_windows_read_real_text_in_order(tmp_path):
    # End to end: tokenize -> shard -> memmap -> windows must stitch back to the
    # original text (minus the tail that only ever appears as a target).
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    txt = tmp_path / "corpus.txt"
    txt.write_text(text, encoding="utf-8")
    tokenize_to_bin(txt, tmp_path / "corpus_000000.bin")
    ds = TokenDataset(load_shards(shard_paths(tmp_path, "corpus_*.bin")), 4, 4)

    joined = []
    for i in range(len(ds)):
        joined.extend(ds[i][0].tolist())
    assert text.startswith(decode_with_gpt2_vocab(joined))
    assert len(joined) == len(ds) * 4
