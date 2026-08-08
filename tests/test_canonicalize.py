"""Tests for canonicalize module (Phase 1, §5.4 mandatory tests)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.canonicalize import (
    canonicalize_one,
    parse_header,
    iter_fasta,
)


def test_t_to_u_and_uppercase():
    res = canonicalize_one("URS0000000001", "acgtUGCA", "tRNA")
    assert res.canonical_seq == "ACGUUGCA"
    assert res.alphabet_status == "primary"


def test_ambiguous_detected():
    res = canonicalize_one("URS0000000002", "ACGNUR", "tRNA")
    assert res.alphabet_status == "ambiguous"


def test_length_bins():
    assert canonicalize_one("u1", "AAAA", "tRNA").length_bin == "<16"
    s16 = "A" * 16
    assert canonicalize_one("u2", s16, "tRNA").length_bin == "16-4096"
    s4097 = "A" * 4097
    assert canonicalize_one("u3", s4097, "tRNA").length_bin == "4097-16384"
    sbig = "A" * 16385
    assert canonicalize_one("u4", sbig, "tRNA").length_bin == ">16384"


def test_hash_stability():
    a = canonicalize_one("u", "ACGU", "tRNA")
    b = canonicalize_one("u", "ACGU", "tRNA")
    assert a.canonical_hash == b.canonical_hash
    assert len(a.canonical_hash) == 64


def test_header_parse():
    acc, rna = parse_header("URS0000612EFC tmRNA from 1 species")
    assert acc == "URS0000612EFC"
    assert rna == "tmRNA"


def test_iter_fasta_gz_roundtrip():
    import gzip

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.fa.gz"
        with gzip.open(p, "wt") as f:
            f.write(">URS0000000001 tRNA from 1 species\nACGU\nACGU\n")
            f.write(">URS0000000002 rRNA from 2 species\nAAAA\n")
        recs = list(iter_fasta(p))
        assert recs[0][0] == "URS0000000001 tRNA from 1 species"
        assert recs[0][1] == "ACGUACGU"
        assert recs[1][1] == "AAAA"


def test_roundtrip_byte_consistency():
    """canonicalization is deterministic and reproducible byte-for-byte."""
    res = canonicalize_one("u", "acgtUGCA", "tRNA")
    res2 = canonicalize_one("u", "acgtUGCA", "tRNA")
    # canonical sequence + hash identical across two runs
    assert res.canonical_seq == res2.canonical_seq
    assert res.canonical_hash == res2.canonical_hash


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{failures} failures")
    sys.exit(1 if failures else 0)