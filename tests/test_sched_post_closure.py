"""Unit tests for p4_auto_schedule post-closure 3-GPU training cap."""
import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import p4_auto_schedule as ps


class _G(dict):
    def __getattr__(self, k):
        return self[k]


def _gpu(idx, used=0, total=40960, util=100, mig=False):
    g = _G(index=idx, used=used, total=total, free=total - used, mig=mig)
    return g


def test_post_closure_max_gpus_constant():
    assert ps.POST_CLOSURE_MAX_GPUS == 3


def test_core_closed_false_when_empty():
    assert ps.core_closed({}) is False


def test_core_closed_true_when_all_done():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    assert ps.core_closed(ex) is True


def test_core_closed_false_when_running():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    ex[(ps.ARMS[0], ps.SEEDS[0])] = "RUNNING"
    assert ps.core_closed(ex) is False


def test_free_gpus_cap_limits_distinct_cards(monkeypatch):
    """With max_gpus=3 and 6 free non-MIG cards, at most 3 placements allowed."""
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]

    def fake_table():
        return cards

    monkeypatch.setattr(ps, "gpu_table", fake_table)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})

    free = ps.free_gpus(max_gpus=3)
    distinct = {c["index"] for c in free}
    assert len(free) == 3
    assert len(distinct) == 3


def test_free_gpus_no_cap_without_max(monkeypatch):
    """Without max_gpus (closure not done), all free cards get placements."""
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3)]

    def fake_table():
        return cards

    monkeypatch.setattr(ps, "gpu_table", fake_table)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})

    free = ps.free_gpus()
    assert len(free) == 4


def test_free_gpus_cap_reuses_active_cards(monkeypatch):
    """A card already hosting a run is counted first; new cards fill to cap."""
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4)]

    def fake_table():
        return cards

    # GPU 0, 1 already host runs.
    monkeypatch.setattr(ps, "gpu_table", fake_table)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {0: {111}, 1: {222}})

    free = ps.free_gpus(max_gpus=3)
    distinct = {c["index"] for c in free}
    # active 0,1 + one new card = 3 distinct
    assert len(distinct) == 3
    assert 0 in distinct and 1 in distinct