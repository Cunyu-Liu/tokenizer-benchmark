"""Unit fixtures for p4_eval_codec --subsample loading (CPU-only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import pyarrow.parquet as pq


def _write_subsample(path):
    table = pa.table({
        "canonical_sequence": pa.array(["ACGU" * 5, "ACGUACGU", "CCGG" * 4]),
        "cluster_id": pa.array(["c1", "c2", "c3"]),
    })
    pq.write_table(table, path)
    return path


def test_load_seqs_from_subsample():
    from p4_eval_codec import _load_seqs
    tmp = "/tmp/test_codec_subsample.parquet"
    _write_subsample(tmp)
    seqs = _load_seqs("ignored_split.parquet", "test", subsample=tmp)
    assert seqs == ["ACGU" * 5, "ACGUACGU", "CCGG" * 4]


def test_load_seqs_subssample_respects_n():
    from p4_eval_codec import _load_seqs
    tmp = "/tmp/test_codec_subsample.parquet"
    _write_subsample(tmp)
    seqs = _load_seqs("ignored_split.parquet", "test", n=2, subsample=tmp)
    assert seqs == ["ACGU" * 5, "ACGUACGU"]


def test_load_seqs_fallback_first_n():
    from p4_eval_codec import _load_seqs
    split = "/tmp/test_codec_split.parquet"
    table = pa.table({
        "split_membership": pa.array(["train", "test", "test"]),
        "canonical_sequence": pa.array(["AAAA", "CCCC", "UUUU"]),
    })
    pq.write_table(table, split)
    seqs = _load_seqs(split, "test", n=1)
    assert seqs == ["CCCC"]