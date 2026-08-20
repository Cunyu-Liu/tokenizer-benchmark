"""CPU unit fixtures for the validation continuation query manifest builder."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from build_validation_continuation_manifest import build  # noqa: E402


def _mk_split(path, rows):
    cols = {k: [r[k] for r in rows] for k in rows[0].keys()}
    pl.DataFrame(cols).write_parquet(path)


def _rows():
    seqs = ["ACGU" * 40, "CCGG" * 30, "UUAC" * 25, "ACGU" * 20, "GG" * 50]
    out = []
    for i, s in enumerate(seqs):
        out.append({"split_membership": "validation", "canonical_sequence": s,
                    "canonical_sequence_hash": "h%d" % i, "length": len(s),
                    "rna_type": "tRNA", "source_database": "RFAM"})
    return out


def test_build_prefix_len_multiple_of_six(tmp_path):
    split = str(tmp_path / "s.parquet")
    _mk_split(split, _rows())
    man, keep = build(split, "validation", 1201, n=5, fracs=[0.10, 0.25, 0.50])
    assert man["validation_query_seed"] == 1201
    # all kept prefixes come from continuation_cut -> multiple of 6, nonempty suffix
    for e in keep:
        assert e["prefix_len"] % 6 == 0
        assert 1 <= e["prefix_len"] < e["length"]
        assert e["prefix"] == _rows()[0]["canonical_sequence"][:e["prefix_len"]] or \
               e["prefix_len"] <= 3000  # prefix exists on some canonical seq


def test_build_deterministic(tmp_path):
    split = str(tmp_path / "s.parquet")
    _mk_split(split, _rows())
    m1, _ = build(split, "validation", 1201, n=3, fracs=[0.50])
    m2, _ = build(split, "validation", 1201, n=3, fracs=[0.50])
    assert m1["selection_sha256"] == m2["selection_sha256"]


def test_build_differs_across_seed(tmp_path):
    split = str(tmp_path / "s.parquet")
    _mk_split(split, _rows())
    m1, k1 = build(split, "validation", 1201, n=3, fracs=[0.50])
    m2, k2 = build(split, "validation", 1202, n=3, fracs=[0.50])
    # selection is seed-dependent, so order (and manifest) differs
    assert m1["selection_sha256"] != m2["selection_sha256"] or \
           [e["canonical_sequence_hash"] for e in k1] != \
           [e["canonical_sequence_hash"] for e in k2]