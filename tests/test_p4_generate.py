"""Tests for the Phase 4 generation protocol runner (contract 3.6).

CPU-safe tests of the deterministic parts: decoder grid definition, validity
check, and the seeded generation-loop helper run against a lightweight
stub adapter (no GPU, no real model). GPU end-to-end generation is covered by
the existing GPU adapter tests; this guards the protocol constants + loop.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from p4_generate import (
    DECODER_GRID, PRE_REGISTERED, GENERATION_SEEDS, VALID_PER_SEED,
    _is_valid, _hash,
)


def test_decoder_grid_has_pre_registered_points():
    assert "conservative" in DECODER_GRID
    assert "balanced" in DECODER_GRID
    assert "exploratory" in DECODER_GRID
    # contract 3.6 pre-registered points
    assert DECODER_GRID["conservative"] == {"temperature": 0.8, "top_p": 0.95}
    assert DECODER_GRID["balanced"] == {"temperature": 1.0, "top_p": 0.95}
    assert DECODER_GRID["exploratory"] == {"temperature": 1.1, "top_p": 1.0}
    assert set(PRE_REGISTERED) == {"conservative", "balanced", "exploratory"}


def test_generation_seeds_and_target():
    assert len(GENERATION_SEEDS) == 5
    assert len(set(GENERATION_SEEDS)) == 5
    assert VALID_PER_SEED == 2000


def test_is_valid():
    assert _is_valid("ACGUACGU")
    assert _is_valid("acguacgu")  # canonicalized
    assert not _is_valid("ACXNACGU")  # invalid IUPAC char
    assert not _is_valid("AC KGU")


def test_hash_deterministic_and_order_independent():
    a = _hash(["ACGU", "ACGUACGU"])
    b = _hash(["ACGUACGU", "ACGU"])
    assert a == b
    c = _hash(["ACGU", "ACGUACGU", "UGCA"])
    assert c != a


class _StubAdapter:
    """Deterministic stub returning a fixed valid sequence regardless of params."""
    def __init__(self, out: str = "ACGU" * 8):
        self._out = out
        self.guard = type("G", (), {"cpu_fallback_count": 0})()

    def generate(self, prefix, n, temperature=1.0, top_p=1.0):
        out = (prefix + self._out)[: (len(prefix) + n)]
        return out[len(prefix):]


def test_generate_cell_returns_target_valid():
    torch = pytest.importorskip("torch")  # generate_cell seeds torch global RNG
    from p4_generate import generate_cell
    adapter = _StubAdapter()
    valid, invalid = generate_cell(adapter, temp=1.0, top_p=0.95,
                                   gen_seed=100, valid_per_seed=50,
                                   target_len=32)
    assert len(valid) == 50
    assert all(_is_valid(s) for s in valid)
    assert invalid == 0


def test_generate_cell_handles_invalid():
    torch = pytest.importorskip("torch")
    from p4_generate import generate_cell
    call = {"n": 0}

    class _Flaky(_StubAdapter):
        def generate(self, prefix, n, temperature=1.0, top_p=1.0):
            call["n"] += 1
            if call["n"] % 3 == 0:
                return "XX"  # invalid
            return "ACGU" * 4

    valid, invalid = generate_cell(_Flaky(), temp=1.0, top_p=1.0,
                                   gen_seed=100, valid_per_seed=20,
                                   target_len=16)
    assert len(valid) == 20
    assert invalid > 0


def test_continuation_metrics_wired():
    """evaluate_continuation computes edit_dist/nt_acc/kmer_recovery from
    (generated_suffix, true_suffix) pairs (contract 3.6 continuation)."""
    from evaluator.eval_continuation import evaluate_continuation
    # exact match -> nt_acc 1.0, edit_dist 0, kmer_recovery 1.0
    res = evaluate_continuation([("ACGUACGU", "ACGUACGU", 0.25)])
    assert res.count == 1 and res.prefix_frac == 0.25
    assert res.mean_edit_dist() == 0.0
    assert res.mean_nt_acc() == 1.0
    assert res.mean_kmer_recovery() == 1.0
    # total mismatch -> nt_acc 0.0 (pred all U, ref ACGU: 1/4 positional match)
    res2 = evaluate_continuation([("UUUU", "ACGU", 0.5)])
    assert abs(res2.mean_nt_acc() - 0.25) < 1e-9
    assert res2.mean_edit_dist() == 3.0
    # empty -> NaN guarded
    res3 = evaluate_continuation([])
    assert res3.count == 0
    assert res3.mean_nt_acc() != res3.mean_nt_acc()  # NaN