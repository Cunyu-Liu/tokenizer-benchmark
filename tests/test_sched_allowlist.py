"""Unit tests for p4_auto_schedule single-GPU cap + train-GPU allowlist."""
import builtins
import io
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
    assert ps.TRAIN_GPU_ALLOWLIST == {0, 2, 4}


def test_owner_training_limits_are_single_job_single_gpu():
    assert ps.MAX_CONCURRENT == 1
    assert ps.MAX_PER_GPU == 1
    assert ps.POST_CLOSURE_MAX_GPUS == 1


def test_free_gpus_never_uses_outside_allowlist(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    free = ps.free_gpus()
    assert {c["index"] for c in free} <= ps.TRAIN_GPU_ALLOWLIST
    assert {c["index"] for c in free} == {0, 2, 4}


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
    assert len(free) == 3
    assert {c["index"] for c in free} == {0, 2, 4}


def test_core_closed_true_when_all_done():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    assert ps.core_closed(ex) is True


def test_core_closed_false_when_running():
    ex = {(a, s): "DONE" for a in ps.ARMS for s in ps.SEEDS}
    ex[(ps.ARMS[0], ps.SEEDS[0])] = "RUNNING"
    assert ps.core_closed(ex) is False


def test_one_pass_does_not_backfill_until_all_existing_workers_finish(monkeypatch):
    existing = {(ps.ARMS[0], ps.SEEDS[0]): "RUNNING"}
    launches = []
    monkeypatch.setattr(ps, "existing_run_statuses", lambda: existing)
    monkeypatch.setattr(ps, "free_gpus", lambda max_gpus=None: [_gpu(0), _gpu(2)])
    monkeypatch.setattr(ps, "launch", lambda arm, seed, gpu: launches.append((arm, seed, gpu)))
    monkeypatch.setattr(ps, "log", lambda _msg: None)

    assert ps.one_pass() == 0
    assert launches == []


def test_one_pass_launches_only_one_worker_after_drain(monkeypatch):
    launches = []
    monkeypatch.setattr(ps, "existing_run_statuses", lambda: {})
    monkeypatch.setattr(ps, "free_gpus", lambda max_gpus=None: [_gpu(0), _gpu(2), _gpu(4)])
    monkeypatch.setattr(ps, "launch", lambda arm, seed, gpu: launches.append((arm, seed, gpu)))
    monkeypatch.setattr(ps, "log", lambda _msg: None)

    assert ps.one_pass() == 1
    assert len(launches) == 1


def test_launch_does_not_apply_wall_clock_timeout(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ps, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(ps, "PROJ", str(tmp_path))
    monkeypatch.setattr(ps, "PYTHON", "/test/python")
    monkeypatch.setattr(ps.time, "strftime", lambda _fmt: "20260827T120000")
    monkeypatch.setattr(ps, "log", lambda _msg: None)
    monkeypatch.setattr(
        ps.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)),
    )

    ps.launch("F6", 43, 2)

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd.startswith("nohup bash -lc ")
    assert "timeout" not in cmd
    assert "p4_train.py --arm F6 --seed 43" in cmd
    assert kwargs["shell"] is True
    assert kwargs["check"] is True


def test_direct_worker_gpu_mapping_reads_process_environment(monkeypatch):
    environ = io.BytesIO(b"PATH=/usr/bin\0CUDA_VISIBLE_DEVICES=4\0")
    monkeypatch.setattr(builtins, "open", lambda *_args, **_kwargs: environ)

    gpu = ps._physical_gpu_for_process(
        "1234",
        "/test/python -u p4_train.py --arm F1 --seed 29",
    )

    assert gpu == 4


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
    assert len(distinct) == 1
    assert distinct.isdisjoint({0, 1})


def test_free_gpus_no_cap_without_max(monkeypatch):
    cards = [_gpu(0), _gpu(1), _gpu(2), _gpu(3), _gpu(4), _gpu(5)]
    monkeypatch.setattr(ps, "gpu_table", lambda: cards)
    monkeypatch.setattr(ps, "_our_gpu_pids", lambda: {})
    monkeypatch.setattr(ps, "TRAIN_GPU_ALLOWLIST", ALL6)
    free = ps.free_gpus()
    assert {c["index"] for c in free} == ALL6
