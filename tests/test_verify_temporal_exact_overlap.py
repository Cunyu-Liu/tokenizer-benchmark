"""CPU unit fixtures for data.verify_temporal_exact_overlap.exact_overlap."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl


def _write_split(path, memberships, hashes):
    pl.DataFrame({"split_membership": memberships, "canonical_sequence_hash": hashes}
                 ).write_parquet(path)


def test_overlap_detected(tmp_path):
    from data.verify_temporal_exact_overlap import exact_overlap
    split = str(tmp_path / "split.parquet")
    clean = str(tmp_path / "clean.parquet")
    _write_split(split, ["train"] * 4, ["a", "b", "c", "d"])
    _write_split(clean, ["test"] * 2, ["d", "e"])
    r = exact_overlap(split, clean)  # d is in train AND clean -> overlap 1
    assert r["exact_overlap_count"] == 1 and r["pass"] is False


def test_zero_overlap_pass(tmp_path):
    from data.verify_temporal_exact_overlap import exact_overlap
    split = str(tmp_path / "split.parquet")
    clean = str(tmp_path / "clean.parquet")
    _write_split(split, ["train"] * 3, ["a", "b", "c"])
    _write_split(clean, ["test"] * 2, ["x", "y"])
    r = exact_overlap(split, clean)
    assert r["exact_overlap_count"] == 0 and r["pass"] is True


def test_only_train_membership_counts(tmp_path):
    from data.verify_temporal_exact_overlap import exact_overlap
    split = str(tmp_path / "split.parquet")
    clean = str(tmp_path / "clean.parquet")
    # clean hash appears only as a train-row in split; test-row should not count
    _write_split(split, ["train", "test", "train"], ["h", "h", "i"])
    _write_split(clean, ["test"], ["h"])
    # train covers h only once -> overlap 1
    r = exact_overlap(split, clean)
    assert r["exact_overlap_count"] == 1