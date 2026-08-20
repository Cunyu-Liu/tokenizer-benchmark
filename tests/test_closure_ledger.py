"""CPU unit fixtures for the Phase-4 closure ledger (dir-name parsing)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p4_closure_ledger import RUN_RE


def test_run_re_matches_track_r_and_b1():
    assert RUN_RE.match("phase4_F1_s17_20260810T020817").groups() == ("F1", "17")
    assert RUN_RE.match("phase4_B1_s29_20260820T152207").groups() == ("B1", "29")
    assert RUN_RE.match("phase4_P3_s43_20260818T000000").groups() == ("P3", "43")


def test_run_re_rejects_unknown_arm():
    assert RUN_RE.match("phase4_F8_s17_20260810T000000") is None  # F8 not an arm


def test_run_re_seed_is_any_digits_and_restart_needs_scan_exclusion():
    # the regex takes any seed ((\d+)); seed 9 and the `_restart` suffix both
    # match at the regex level, so the scan() layer excludes restart dirs via an
    # explicit "_restart" in dir check.
    assert RUN_RE.match("phase4_F1_s9_20260810T000000") is not None
    assert RUN_RE.match("phase4_P2_s17_20260820T124621_restart") is not None