"""Unit tests for true-suffix continuation code length (contract 3.7)."""
import math

from evaluator.continuation_code_length import (
    canonicalize_seq,
    prefix_cut_points,
    suffix_code_length_bpn,
    continuation_bpn_grid,
)

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


class _NucTokenizer:
    vocab_size = 4

    def encode(self, seq):
        return [BASE_TO_IDX[c] for c in seq]


def _uniform_probs(ctx):
    return [0.25, 0.25, 0.25, 0.25]


def _cyclic_next_probs(ctx):
    # deterministic transition A->C->G->U->A given the last token
    if not ctx:
        return [0.25, 0.25, 0.25, 0.25]
    nxt = (ctx[-1] + 1) % 4
    probs = [0.0, 0.0, 0.0, 0.0]
    probs[nxt] = 1.0
    return probs


def test_cut_points_are_multiples_of_6():
    cuts = prefix_cut_points(100, ratios=(0.10, 0.25, 0.50))
    assert cuts[0.10] == 6      # 6*floor(10/6)
    assert cuts[0.25] == 24     # 6*floor(25/6)
    assert cuts[0.50] == 48     # 6*floor(50/6)
    assert all(p % 6 == 0 for p in cuts.values())


def test_cut_points_skip_no_suffix():
    # seq_len 6 with ratio 0.9 -> p=0 (no interior multiple of 6); skip
    cuts = prefix_cut_points(6, ratios=(0.9,))
    assert cuts == {}


def test_uniform_continuation_is_2_bits():
    seq = "ACGU" * 25
    bpn = suffix_code_length_bpn(seq, 24, _NucTokenizer(), _uniform_probs)
    assert abs(bpn - 2.0) < 1e-9, bpn


def test_deterministic_continuation_is_zero():
    seq = "ACGU" * 25  # cyclic, fully predictable by _cyclic_next_probs
    bpn = suffix_code_length_bpn(seq, 24, _NucTokenizer(), _cyclic_next_probs)
    assert bpn < 1e-9, bpn


def test_nan_when_prefix_covers_whole_seq():
    seq = "ACGUACGU"
    assert math.isnan(suffix_code_length_bpn(seq, len(seq), _NucTokenizer(), _uniform_probs))
    assert math.isnan(suffix_code_length_bpn(seq, 0, _NucTokenizer(), _uniform_probs))


def test_prefix_and_suffix_encode_independently():
    # A tokenizer whose encode records the string length; verify prefix and
    # suffix are encoded separately (no single encode of the whole seq).
    class _RecordingTokenizer:
        vocab_size = 4

        def __init__(self):
            self.calls = []

        def encode(self, s):
            self.calls.append(s)
            return [BASE_TO_IDX[c] for c in s]

    tk = _RecordingTokenizer()
    seq = "ACGU" * 10
    suffix_code_length_bpn(seq, 20, tk, _uniform_probs)
    # exactly two independent encodes: prefix (20 nt) and suffix (20 nt)
    assert tk.calls == [seq[:20], seq[20:]]


def test_grid_matches_individual():
    seq = "ACGU" * 25
    cuts = prefix_cut_points(len(seq), ratios=(0.10, 0.50))
    grid = continuation_bpn_grid(seq, cuts, _NucTokenizer(), _uniform_probs)
    for r, p in cuts.items():
        assert abs(grid[r] - suffix_code_length_bpn(seq, p, _NucTokenizer(), _uniform_probs)) < 1e-12


def test_canonicalize_t_to_u():
    assert canonicalize_seq("acgt") == "ACGU"