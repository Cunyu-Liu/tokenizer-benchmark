"""Tests for the Phase 4 run aggregation / finalize tool (contract 2.3, 4).

Exercises the pure-Python acceptance logic of p4_finalize without needing a
GPU or any trained checkpoint:
  - validate_manifest() enforces the Phase 4 acceptance invariants.
  - read_run() distinguishes DONE vs RUNNING_PRE_MANIFEST.
  - RUN_RE recognizes only real science-run bundle names.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import p4_finalize as pf


def _valid_manifest(**over):
    m = {
        "run_id": "tokbench_F1_100M_s17_nt16384_17",
        "phase": 4,
        "arm": "F1",
        "seed": 17,
        "status": "DONE",
        "final_nt": 2_000_000_000,
        "budget_nt": 2_000_000_000,
        "best_val_loss": 1.2345,
        "best_checkpoint": "/mnt/.../ckpt_nt0200000000_step123456.pt",
        "params": 100_000_000,
        "cpu_fallback_count": 0,
        "config": {"budget_nt": 2_000_000_000, "batch_nt": 16384},
    }
    m.update(over)
    return m


def test_valid_manifest_passes():
    assert pf.validate_manifest(_valid_manifest(), "F1", 17) == []


def test_rejects_status_not_done():
    issues = pf.validate_manifest(_valid_manifest(status="RUNNING"), "F1", 17)
    assert any("status!=DONE" in i for i in issues)


def test_rejects_budget_not_reached():
    issues = pf.validate_manifest(_valid_manifest(final_nt=1_000_000_000), "F1", 17)
    assert any("final_nt" in i and "<" in i for i in issues)


def test_rejects_cpu_fallback():
    issues = pf.validate_manifest(_valid_manifest(cpu_fallback_count=3), "F1", 17)
    assert any("cpu_fallback_count" in i for i in issues)


def test_rejects_missing_best_checkpoint():
    issues = pf.validate_manifest(_valid_manifest(best_checkpoint=None), "F1", 17)
    assert any("best_checkpoint" in i for i in issues)


def test_rejects_arm_seed_mismatch():
    issues = pf.validate_manifest(_valid_manifest(), "F2", 17)
    assert any("arm mismatch" in i for i in issues)
    issues = pf.validate_manifest(_valid_manifest(), "F1", 29)
    assert any("seed mismatch" in i for i in issues)


def test_read_run_missing_manifest(tmp_path):
    d = tmp_path / "phase4_F1_s17_20260810T000000"
    d.mkdir()
    (d / "run.log").write_text("line1\nline2\n")
    entry = pf.read_run(str(d))
    assert entry["status"] == "RUNNING_PRE_MANIFEST"
    assert entry["manifest"] is False


def test_read_run_done(tmp_path):
    d = tmp_path / "phase4_F1_s17_20260810T000000"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(_valid_manifest()))
    entry = pf.read_run(str(d))
    assert entry["status"] == "DONE"
    assert entry["accept"] is True


def test_run_re_matches_science_bundles_only():
    assert pf.RUN_RE.match("phase4_F1_s17_20260810T020817")
    assert pf.RUN_RE.match("phase4_P3_s43_20260810T000000")
    assert not pf.RUN_RE.match("phase4_launch.log")
    assert not pf.RUN_RE.match("phase4_smoke_F1")
    assert not pf.RUN_RE.match("phase4_FX_s17_20260810T000000")
