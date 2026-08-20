"""CPU unit fixtures for contract 3.7 continuation cut (6*floor(ratio*n/6))."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p4_generate import continuation_cut  # noqa: E402


def test_cut_is_multiple_of_six():
    # n=100, ratio=0.5 -> 50 -> floor 50//6=8 -> 48 (a multiple of 6)
    assert continuation_cut(100, 0.5) == 48
    assert continuation_cut(100, 0.5) % 6 == 0
    # n=200, ratio=0.1 -> 20 -> 20//6=3 -> 18
    assert continuation_cut(200, 0.1) == 18
    assert continuation_cut(200, 0.1) % 6 == 0


def test_cut_keeps_at_least_one_nt_suffix():
    # prefix must be < n so at least 1 suffix nt remains
    for n in range(6, 100, 7):
        k = continuation_cut(n, 0.25)
        if k is not None:
            assert 1 <= k < n, (n, k)


def test_cut_none_when_no_suffix_or_no_prefix():
    # n too short / huge ratio -> no usable cut
    assert continuation_cut(5, 0.5) is None          # n*frac < 6 -> k=0
    assert continuation_cut(6, 1.0) is None          # k == n -> no suffix
    assert continuation_cut(3, 0.5) is None
    # exact multiple floor still leaves a suffix
    assert continuation_cut(6, 0.5) == 0 or continuation_cut(6, 0.5) is None