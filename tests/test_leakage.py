"""Unit tests for leakage report logic."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.leakage_report import canonical_overlap, cluster_split_crossing


def _df(rows):
    return pl.DataFrame(rows, schema={"canonical_sequence_hash": pl.Utf8,
                                      "split_membership": pl.Utf8,
                                      "cluster_id": pl.Utf8})


def test_no_exact_overlap():
    df = _df([
        {"canonical_sequence_hash": "h1", "split_membership": "train", "cluster_id": "c1"},
        {"canonical_sequence_hash": "h2", "split_membership": "test", "cluster_id": "c2"},
    ])
    ov = canonical_overlap(df, "canonical_sequence_hash")
    assert all(v == 0 for v in ov.values())


def test_exact_overlap_detected():
    df = _df([
        {"canonical_sequence_hash": "h1", "split_membership": "train", "cluster_id": "c1"},
        {"canonical_sequence_hash": "h1", "split_membership": "test", "cluster_id": "c2"},
    ])
    ov = canonical_overlap(df, "canonical_sequence_hash")
    assert list(ov.values()) == [1]


def test_no_cluster_crossing():
    df = _df([
        {"canonical_sequence_hash": "h1", "split_membership": "train", "cluster_id": "c1"},
        {"canonical_sequence_hash": "h2", "split_membership": "test", "cluster_id": "c2"},
    ])
    assert cluster_split_crossing(df) == 0


def test_cluster_crossing_detected():
    df = _df([
        {"canonical_sequence_hash": "h1", "split_membership": "train", "cluster_id": "c1"},
        {"canonical_sequence_hash": "h2", "split_membership": "test", "cluster_id": "c1"},
    ])
    assert cluster_split_crossing(df) == 1
