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


def test_train_gpu_allowlist_excludes_eval_cards():
    assert ps.TRAIN_GPU_ALLOWLIST == {1, 2, 4}


def test_free_gpus_never_uses_outside_allowlist(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    free = ps.free_gpus()
    assert {c["index"] for c in free} <= ps.TRAIN_GPU_ALLOWLIST
    assert {c["index"] for c in free} == {1, 2, 4}


def test_free_gpus_allowlist_with_active(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    # a run on GPU1 already (in-allowlist); GPU0 outside must still be skipped
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {1: {111}})
    free = ps.free_gpus()
    assert {c["index"] for c in free} <= ps.TRAIN_GPU_ALLOWLIST


def test_free_gpus_cap_still_respects_allowlist(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    free = ps.free_gpus(max_gpus=3)
    assert {c["index"] for c in free} <= ps.TRAIN_GPU_ALLOWLIST
    assert len(free) == 3


def test_core_closed_true_when_all_done():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    assert ps.core_closed(ex) is True


def test_core_closed_false_when_running():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    ex[(ps.ARMS[0], ps.SEEDS[0])] = "RUNNING"
    assert ps.core_closed(ex) is False