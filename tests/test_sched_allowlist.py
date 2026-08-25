"""Unit tests for p4_auto_schedule post-closure 3-GPU cap + train-GPU allowlist."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import p4_auto_schedule as ps


class _G(dict):
    def __getattr__(self, k):
        return self[k]


def _gpu(idx, used=0, total=40960, util=100, mig=False):
    return _G(index=idx, used=used, total=total, free=total - used, mig=mig)


# A relaxed allowlist (all 6 non-MIG cards) used by cap-only tests so the
# max_gpus logic is tested independently of the hard {2,4} reservation.
ALL6 = {0, 1, 2, 3, 4, 5}


def test_train_gpu_allowlist_excludes_eval_cards():
    assert ps.TRAIN_GPU_ALLOWLIST == {2, 4}


def test_free_gpus_never_uses_outside_allowlist(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    free = ps.free_gpus()
    assert {c["index"] for c in free} <= ps.TRAIN_GPU_ALLOWLIST
    assert {c["index"] for c in free} == {2, 4}


def test_free_gpus_allowlist_with_active(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    # a run on GPU2 already (in-allowlist); GPU0/1/3/5 outside must be skipped
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {2: {111}})
    free = ps.free_gpus()
    assert {c["index"] for c in free} <= ps.TRAIN_GPU_ALLOWLIST


def test_free_gpus_cap_still_respects_allowlist(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    free = ps.free_gpus(max_gpus=3)
    # hard allowlist {2,4} dominates: only those 2 cards may be used
    assert {c["index"] for c in free} <= ps.TRAIN_GPU_ALLOWLIST
    assert len(free) == 2
    assert {c["index"] for c in free} == {2, 4}


def test_core_closed_true_when_all_done():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    assert ps.core_closed(ex) is True


def test_core_closed_false_when_running():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    ex[(ps.ARMS[0], ps.SEEDS[0])] = "RUNNING"
    assert ps.core_closed(ex) is False


# ---- cap-only tests decoupled from the hard allowlist ----

def test_free_gpus_cap_limits_distinct_cards(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    monkeypatch.setattr(ps, "TRAIN_GPU_ALLOWLIST", ALL6)
    free = ps.free_gpus(max_gpus=3)
    distinct = {c["index"] for c in free}
    assert len(free) == 3
    assert len(distinct) == 3


def test_free_gpus_cap_reuses_active_cards(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "TRAIN_GPU_ALLOWLIST", ALL6)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {0: {111}, 1: {222}})
    free = ps.free_gpus(max_gpus=3)
    distinct = {c["index"] for c in free}
    assert len(distinct) == 3
    assert 0 in distinct and 1 in distinct


def test_free_gpus_no_cap_without_max(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    monkeypatch.setattr(ps, "TRAIN_GPU_ALLOWLIST", ALL6)
    free = ps.free_gpus()
    assert {c["index"] for c in free} == ALL6