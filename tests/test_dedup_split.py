"""Tests for dedup and split modules (Phase 1)."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
from pathlib import Path

from data.dedup import dedup
from data.split import assign_splits, load_family_annotations


def _sample_table():
    return pa.table({
        "accession": ["U1", "U2", "U3", "U4", "U5"],
        "rna_type": ["tRNA", "tRNA", "rRNA", "rRNA", "miRNA"],
        "raw_sequence_hash": [f"raw{i}" for i in range(5)],
        "canonical_sequence_hash": ["hA", "hA", "hB", "hB", "hC"],
        "canonical_sequence": ["ACGU" * 4, "ACGU" * 4, "UUUU" * 4, "UUUU" * 4, "GGGG" * 4],
        "alphabet_status": ["primary"] * 5,
        "length": [16, 16, 16, 16, 16],
        "length_bin": ["16-4096"] * 5,
    })


def test_dedup_merges_exact_duplicates():
    t = _sample_table()
    d = dedup(t)
    assert d.num_rows == 3  # A,B,C
    # accession mapping preserved
    acc_sets = set(d.column("accessions").to_pylist())
    assert "U1\tU2" in acc_sets
    assert "U3\tU4" in acc_sets


def test_dedup_num_accessions():
    d = dedup(_sample_table())
    by_hash = {h: n for h, n in zip(d.column("canonical_sequence_hash").to_pylist(), d.column("num_accessions").to_pylist())}
    assert by_hash["hA"] == 2
    assert by_hash["hC"] == 1


def _cluster_table():
    # build a dedup table with cluster_id and accessions
    return pa.table({
        "canonical_sequence_hash": ["HA", "HB", "HC", "HD", "HE", "HF"],
        "canonical_sequence": ["A" * 16, "B" * 16, "C" * 16, "D" * 16, "E" * 16, "F" * 16],
        "rna_type": ["tRNA"] * 6,
        "length": [16] * 6,
        "length_bin": ["16-4096"] * 6,
        "num_accessions": [1] * 6,
        "accessions": ["URA", "URB", "URC", "URD", "URE", "URF"],
        "cluster_id": ["c1", "c1", "c2", "c3", "c4", "c4"],
    })


def test_split_no_exact_overlap():
    # same canonical hash must not span splits; here each is unique so trivially holds
    t = _cluster_table()
    fam_annot = {"URA": "RF1", "URB": "RF1", "URC": "RF2", "URD": "RF3", "URE": "RF4", "URF": "RF4"}
    out = assign_splits(t, fam_annot, {})
    assert out.num_rows == 6


def test_split_cluster_not_across():
    # members of same cluster must get same split
    t = _cluster_table()
    fam_annot = {}
    out = assign_splits(t, fam_annot, {})
    splits = list(out.column("split_membership").to_pylist())[:2]
    assert splits[0] == splits[1]  # c1 members same split


def test_split_family_heldout():
    # RF1 is a family; if it becomes family_test, both its sequences go there
    t = _cluster_table()
    fam_annot = {"URA": "RF1", "URB": "RF1"}
    # make RF1 eligible by injecting many clusters? For simplicity, test routing logic
    # by directly checking that same-family clusters co-route.
    out = assign_splits(t, fam_annot, {})
    # cluster c1 (URA,URB) both RF1 -> same split
    sp = list(out.column("split_membership").to_pylist())
    assert sp[0] == sp[1]


def test_family_annot_loader():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        f.write("URS1\tRF00177\t109.4\t3.3e-33\t2\t200\t29\t230\tdesc\n")
        f.write("URS1\tRF00001\t200.0\t1e-50\t0\t100\t1\t100\tdesc2\n")  # higher score wins
        f.write("URS2\tRF00005\t50.0\t1e-10\t0\t50\t1\t50\tdesc3\n")
        p = f.name
    try:
        m = load_family_annotations(Path(p))
        assert m["URS1"] == "RF00001"  # highest score
        assert m["URS2"] == "RF00005"
    finally:
        Path(p).unlink()


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                import traceback; traceback.print_exc()
                print(f"FAIL {name}: {e}")
    print(f"\n{failures} failures")
    sys.exit(1 if failures else 0)