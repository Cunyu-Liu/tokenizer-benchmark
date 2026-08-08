"""Tests for temporal OOD extraction (Phase 1, releases 23-26)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_time_ood import classify_temporal


def test_new_accession_kept():
    r22 = {"URS0000000001"}
    train = {"deadbeef"}
    keep, reason = classify_temporal("URS0000000002", "ACGUTTGCA", "tRNA", r22, train)
    assert keep is True
    assert reason == "new"


def test_accession_in_release22_dropped():
    r22 = {"URS0000000001"}
    keep, reason = classify_temporal("URS0000000001", "ACGUACGU", "tRNA", r22, set())
    assert keep is False
    assert reason == "in_release22"


def test_ambiguous_sequence_dropped():
    r22 = set()
    keep, reason = classify_temporal("URS0000000002", "ACGUNACGU", "tRNA", r22, set())
    assert keep is False
    assert reason == "ambiguous"


def test_exact_overlap_with_train_dropped():
    r22 = set()
    # canonical of "ACGUTTGCA" = "ACGUUUGCA"
    canon = "ACGUUUGCA"
    from data.canonicalize import sha256_hex
    train = {sha256_hex(canon)}
    keep, reason = classify_temporal("URS0000000002", "ACGUTTGCA", "tRNA", r22, train)
    assert keep is False
    assert reason == "exact_overlap_train"
