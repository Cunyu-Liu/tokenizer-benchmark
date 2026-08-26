"""CPU unit fixtures for the Phase-4 closure ledger."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import p4_closure_ledger as ledger

RUN_RE = ledger.RUN_RE


def test_run_re_matches_track_r_and_b1():
    assert RUN_RE.match("phase4_F1_s17_20260810T020817").groups() == ("F1", "17")
    assert RUN_RE.match("phase4_B1_s29_20260820T152207").groups() == ("B1", "29")
    assert RUN_RE.match("phase4_P3_s43_20260818T000000").groups() == ("P3", "43")


def test_run_re_rejects_unknown_arm():
    assert RUN_RE.match("phase4_F8_s17_20260810T000000") is None  # F8 not an arm


def test_run_re_seed_is_any_digits_and_accepts_corrected_retry_suffix():
    assert RUN_RE.match("phase4_F1_s9_20260810T000000") is not None
    assert RUN_RE.match("phase4_P2_s17_20260820T124621_restart") is not None


def test_scan_counts_done_corrected_retry_and_preserves_it(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "_live_cells", lambda: set())

    retry = tmp_path / "phase4_P2_s17_20260820T124621_restart"
    retry.mkdir()
    (retry / "manifest.json").write_text(json.dumps({
        "status": "DONE", "arm": "P2", "seed": 17,
        "final_nt": 2_000_000_000,
        "best_checkpoint": "/retry-final.pt",
    }))

    stale = tmp_path / "phase4_P2_s17_20260821T000000"
    stale.mkdir()
    (stale / "manifest.json").write_text(json.dumps({
        "status": "FAIL_CLOSED_WITH_EVIDENCE", "arm": "P2", "seed": 17,
    }))

    result = ledger.scan(str(tmp_path))
    row = next(r for r in result["rows"]
               if r["arm"] == "P2" and r["seed"] == 17)
    assert row["status"] == "DONE"
    assert row["final_nt"] == 2_000_000_000
    assert row["best_checkpoint"] == "/retry-final.pt"
