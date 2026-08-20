"""CPU unit fixtures for p4_eval_codec_table codec-table driver (discovery)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p4_eval_codec_table import discover_done  # noqa: E402


def _mk_run(runs, arm, seed, status, ckpts_nt):
    d = "phase4_%s_s%d_20260810T000000" % (arm, seed)
    os.makedirs(os.path.join(runs, d), exist_ok=True)
    cks = []
    for nt in ckpts_nt:
        cks.append({"nt": nt, "path": "/mnt/%s/%s_ckpt_%d.pt" % (d, arm, nt)})
    man = {
        "arm": arm, "seed": seed, "status": status,
        "checkpoints": [{"nt": c["nt"], "path": c["path"]} for c in cks],
    }
    with open(os.path.join(runs, d, "manifest.json"), "w") as fh:
        json.dump(man, fh)


def test_discover_done_takes_final_checkpoint(tmp_path):
    for arm, seed in [("F1", 17), ("F4", 29)]:
        _mk_run(str(tmp_path), arm, seed, "DONE", [1_000_000_000, 1_500_000_000, 1_900_006_453])
    done = discover_done(str(tmp_path))
    assert [(a, s) for a, s, _ in done] == [("F1", 17), ("F4", 29)]
    for arm, seed, ckpt in done:
        assert ckpt.endswith("_ckpt_1900006453.pt"), ckpt  # last (final) checkpoint


def test_discover_skips_non_done_and_restart(tmp_path):
    _mk_run(str(tmp_path), "F1", 17, "DONE", [1_000_000_000])
    _mk_run(str(tmp_path), "F2", 17, "RUNNING", [500_000_000])
    # a restart dir (name with _restart) must be excluded
    d = "phase4_F3_s17_20260812T000000_restart"
    os.makedirs(os.path.join(str(tmp_path), d), exist_ok=True)
    json.dump({"arm": "F3", "seed": 17, "status": "DONE",
               "checkpoints": [{"nt": 1, "path": "/x"}]},
              open(os.path.join(str(tmp_path), d, "manifest.json"), "w"))
    done = discover_done(str(tmp_path))
    assert [(a, s) for a, s, _ in done] == [("F1", 17)]


def test_discover_empty_when_no_done(tmp_path):
    _mk_run(str(tmp_path), "F1", 17, "RUNNING", [100_000])
    assert discover_done(str(tmp_path)) == []