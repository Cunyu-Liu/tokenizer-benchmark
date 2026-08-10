"""Tests for the Phase 4 main-table aggregator (contract 3.7).

Pure / CPU-safe. Uses synthetic run dirs + eval outputs to verify that the
aggregator correctly selects best-checkpoint rows, computes per-arm means, and
reports missing cells.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from p4_main_table import build_table_a, METRIC_BY_ARM  # noqa: E402


def _manifest(arm, seed, best_val, params, thr, vram, fallback=0, status="DONE"):
    return {
        "arm": arm, "seed": seed, "status": status,
        "best_checkpoint": "/mnt/ck_%s_%d.pt" % (arm, seed),
        "best_val_loss": best_val, "final_nt": 2_000_000_000,
        "params": params, "non_embedding_params": params - 1_000_000,
        "throughput_nt_s": thr, "peak_vram_mb": vram,
        "cpu_fallback_count": fallback,
    }


def _likelihood(arm, seed, bpn, metric, split="test"):
    return {"arm": arm, "seed": seed, "split": split, "metric": metric,
            "bpn": bpn, "n_sequences": 100, "valid_nt_count": 40000}


def _gen(arm, seed, uniq=0.9, val=1.0):
    return {"arm": arm, "seed": seed, "uniqueness": uniq, "validity_rate": val}


def _write_runs(tmp_path):
    runs = tmp_path / "runs"
    specs = {
        "F1": [(17, 1.04, 100_000_000, 4600, 18100),
               (29, 1.05, 100_000_000, 4550, 18200),
               (43, 1.03, 100_000_000, 4700, 18000)],
        "F2": [(17, 3.60, 100_000_000, 3600, 14100)],
    }
    for arm, seeds in specs.items():
        for seed, val, params, thr, vram in seeds:
            d = runs / ("phase4_%s_s%d_20260810T000000" % (arm, seed))
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(json.dumps(_manifest(arm, seed, val, params, thr, vram)))
    return runs


def _write_results(tmp_path, specs):
    res = tmp_path / "results"
    res.mkdir(parents=True)
    for arm, rows in specs.items():
        for seed, bpn in rows:
            (res / ("%s_%d_test.json" % (arm, seed))).write_text(
                json.dumps(_likelihood(arm, seed, bpn, METRIC_BY_ARM[arm])))
            (res / ("gen_%s_%d.json" % (arm, seed))).write_text(
                json.dumps(_gen(arm, seed)))
    return res


def test_table_a_means_and_metric(tmp_path):
    runs = _write_runs(tmp_path)
    res = _write_results(tmp_path, {"F1": [(17, 1.0), (29, 1.1), (43, 0.9)]})
    t = build_table_a(str(runs), str(res), "test")
    f1 = [c for c in t["arms"] if c["arm"] == "F1"][0]
    assert f1["metric"] == "next_base_BPN"
    assert abs(f1["mean_likelihood_bpn"] - 1.0) < 1e-9  # (1.0+1.1+0.9)/3
    assert abs(f1["mean_best_val_loss"] - 1.04) < 1e-9
    assert abs(f1["mean_params"] - 100_000_000) < 1
    assert abs(f1["mean_throughput_nt_s"] - 4616.666) < 1
    assert len(f1["seeds"]) == 3


def test_table_a_missing_likelihood_reported(tmp_path):
    runs = _write_runs(tmp_path)
    res = _write_results(tmp_path, {"F1": [(17, 1.0)]})  # only seed 17 has results
    t = build_table_a(str(runs), str(res), "test")
    f1 = [c for c in t["arms"] if c["arm"] == "F1"][0]
    # mean over the 1 available seed, others None-filtered
    assert abs(f1["mean_likelihood_bpn"] - 1.0) < 1e-9
    # seed 29/43 rows have no likelihood_bpn key
    r29 = [s for s in f1["seeds"] if s["seed"] == 29][0]
    assert "likelihood_bpn" not in r29


def test_table_a_incomplete_run_dir(tmp_path):
    runs = tmp_path / "runs"
    # only F1 s17 has a dir; F2 has none
    (runs / "phase4_F1_s17_20260810T000001").mkdir(parents=True)
    (runs / "phase4_F1_s17_20260810T000001" / "manifest.json").write_text(
        json.dumps(_manifest("F1", 17, 1.0, 100, 1, 1)))
    res = _write_results(tmp_path, {"F1": [(17, 1.0)]})
    t = build_table_a(str(runs), str(res), "test")
    f2 = [c for c in t["arms"] if c["arm"] == "F2"][0]
    assert all(s["status"] == "MISSING_RUN_DIR" for s in f2["seeds"])