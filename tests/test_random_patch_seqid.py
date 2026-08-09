"""Tests for the P2 random-patch sequence-specificity fix (contract 3.2).

Random patch boundaries must be decided by (sequence_id + seed) with prefix
consistency. `model.dataset._seq_id` provides a stable content-based id used by
both the training data pipeline and the BLT inference adapter, so train and
eval reproduce the identical boundary pattern for the same sequence.
Pure / CPU-safe (no GPU required).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import _seq_id  # noqa: E402
from model.patch import PatchPolicy  # noqa: E402

DIST = [0.25, 0.25, 0.25, 0.25]  # pmf over lengths 1..4


def test_seq_id_deterministic_and_content_based():
    a, b = "ACGUACGUACGUACGU", "ACGUACGUACGUACGG"  # differ in last base
    assert _seq_id(a) == _seq_id(a)
    assert isinstance(_seq_id(a), int)
    assert _seq_id(a) != _seq_id(b)  # content-based, not a constant


def test_random_boundary_is_seq_id_dependent():
    policy = PatchPolicy(kind="random", seed=17, length_dist=DIST)
    seq = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"  # 48 nt
    b0 = policy.boundary(seq, len(seq), _seq_id(seq))
    # different seq_id -> different pattern for the same content is impossible,
    # so instead check that two DIFFERENT sequences with different ids differ
    b_other = policy.boundary(seq, len(seq), _seq_id(seq) ^ 0x1)  # forced diff id
    assert b0 != b_other


def test_random_boundary_prefix_consistent():
    policy = PatchPolicy(kind="random", seed=17, length_dist=DIST)
    seq = "ACGUACGUACGUACGUACGUACGU"
    sid = _seq_id(seq)
    full = policy.boundary(seq, len(seq), sid)
    prefix = policy.boundary(seq[:12], 12, sid)
    assert prefix == full[:12]


def test_random_boundary_seed_dependent():
    p1 = PatchPolicy(kind="random", seed=17, length_dist=DIST)
    p2 = PatchPolicy(kind="random", seed=29, length_dist=DIST)
    seq = "ACGUACGUACGUACGUACGUACGU"
    assert p1.boundary(seq, len(seq), _seq_id(seq)) != p2.boundary(
        seq, len(seq), _seq_id(seq))
