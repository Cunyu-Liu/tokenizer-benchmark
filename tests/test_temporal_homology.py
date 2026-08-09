"""Unit tests for temporal-OOD 80/80 homology removal logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.build_temporal_homology import kept_temporal_indices


def test_no_homology_keeps_all():
    # 2 train (ids 0,1), 2 temporal (ids 2,3), each temporal its own cluster
    member_to_rep = {0: 0, 1: 1, 2: 2, 3: 3}
    kept = kept_temporal_indices(2, 2, member_to_rep)
    assert kept == [0, 1]


def test_temporal_clustered_with_train_removed():
    # train ids 0,1 ; temporal ids 2,3
    # temporal 2 (member) clusters with train 0 (rep=0) -> removed
    # temporal 3 its own cluster (rep=3) -> kept
    member_to_rep = {0: 0, 1: 1, 2: 0, 3: 3}
    kept = kept_temporal_indices(2, 2, member_to_rep)
    assert kept == [1]


def test_temporal_self_representative_in_train_cluster_removed():
    # temporal id 2 is its own rep, but shares rep with train member 0
    member_to_rep = {0: 2, 1: 1, 2: 2, 3: 3}
    kept = kept_temporal_indices(2, 2, member_to_rep)
    assert kept == [1]


def test_all_temporal_in_train_cluster():
    member_to_rep = {0: 0, 1: 0, 2: 0, 3: 0}
    kept = kept_temporal_indices(2, 2, member_to_rep)
    assert kept == []
