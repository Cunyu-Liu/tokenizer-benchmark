"""Tests for the BLT sealed-test subsample tool (contract 3.5, user decision).

Pure / CPU-safe. Verifies deterministic cluster-stratified selection and
manifest composition.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from p4_subsample import pick_subsample, selection_hash, strata_composition  # noqa: E402


def _row(hash_, cluster, rna="rRNA", length_bin="1000-4096", family="RF00001",
         clan="CL00001", eligible=True):
    return {
        "canonical_sequence_hash": hash_, "canonical_sequence": "ACGU" * 10,
        "rna_type": rna, "length": 40, "length_bin": length_bin,
        "cluster_id": cluster, "family_annotation": family, "clan_annotation": clan,
        "eligible_family": eligible, "split_membership": "test",
    }


def test_stratified_no_cluster_oversampling():
    # cluster A has 5 rows, cluster B has 3, cluster C has 2
    by_cluster = {
        "A": [_row(f"hA{i}", "A") for i in range(5)],
        "B": [_row(f"hB{i}", "B") for i in range(3)],
        "C": [_row(f"hC{i}", "C") for i in range(2)],
    }
    rows = pick_subsample(by_cluster, n=100, n_per_cluster=1)
    assert len(rows) == 3  # all clusters, 1 each (only 3 clusters available)
    assert len({r["cluster_id"] for r in rows}) == 3


def test_stratified_target_cap():
    by_cluster = {
        "A": [_row(f"hA{i}", "A") for i in range(5)],
        "B": [_row(f"hB{i}", "B") for i in range(5)],
    }
    rows = pick_subsample(by_cluster, n=3, n_per_cluster=2)
    assert len(rows) == 3


def test_deterministic_and_hash_stable():
    by_cluster = {
        "A": [_row(f"hA{i}", "A") for i in range(5)],
        "B": [_row(f"hB{i}", "B") for i in range(3)],
    }
    r1 = pick_subsample(by_cluster, n=100, n_per_cluster=1)
    r2 = pick_subsample(by_cluster, n=100, n_per_cluster=1)
    assert selection_hash(r1) == selection_hash(r2)
    assert selection_hash(r1) == selection_hash(r2[::-1])  # order-independent


def test_composition_counts():
    by_cluster = {
        "A": [_row("hA0", "A", rna="tRNA"), _row("hA1", "A", rna="tRNA")],
        "B": [_row("hB0", "B", rna="rRNA")],
    }
    rows = pick_subsample(by_cluster, n=100, n_per_cluster=1)
    comp = strata_composition(rows)
    assert comp["n_clusters"] == 2
    assert comp["column_counts"]["rna_type"]["tRNA"] == 1
    assert comp["column_counts"]["rna_type"]["rRNA"] == 1