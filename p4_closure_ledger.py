"""Phase-4 33-run closure ledger (contract run_manifest / Phase 4-G registry).

Scans all (arm, seed) cells of the core 33-run matrix and emits an authoritative
status ledger (DONE / RUNNING / NOT_STARTED / FAIL) with per-cell final_nt,
best_val_loss, throughput and checkpoint path. Sources of truth:
  - live p4_train processes (ps) mark RUNNING;
  - run manifest.json `status` marks DONE / FAIL_CLOSED_WITH_EVIDENCE;
  - otherwise NOT_STARTED.

Read-only over artifacts; writes only the ledger JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter

ARMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "P1", "P2", "P3", "B1"]
SEEDS = [17, 29, 43]
RUN_RE = re.compile(r"^phase4_(F[1-7]|P[1-3]|B1)_s(\d+)_\d{8}T\d{6}")
DEFAULT_RUNS = "/mnt/cunyuliu/tokenizer-benchmark/runs"


def _live_cells() -> set[tuple[str, int]]:
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True,
                         check=True).stdout
    live = set()
    for m in re.finditer(r"p4_train\.py --arm (F[1-7]|P[1-3]|B1) --seed (\d+)", out):
        live.add((m.group(1), int(m.group(2))))
    return live


def scan(runs_dir: str) -> dict:
    live = _live_cells()
    best = {}
    for d in sorted(__import__("os").listdir(runs_dir)):
        m = RUN_RE.match(d)
        if not m or "_restart" in d:
            continue
        import os
        mf = os.path.join(runs_dir, d, "manifest.json")
        if not os.path.isfile(mf):
            continue
        with open(mf) as fh:
            man = json.load(fh)
        key = (m.group(1), int(m.group(2)))
        st = man.get("status")
        if st and best.get(key) != "DONE":
            # keep DONE over conflicting stale; prefer highest final_nt within
            prev = best.get(key)
            if prev not in (None, "DONE") or st == "DONE":
                best[key] = {
                    "status": st, "final_nt": man.get("final_nt"),
                    "best_val_loss": man.get("best_val_loss"),
                    "throughput_nt_s": man.get("throughput_nt_s"),
                    "best_checkpoint": man.get("best_checkpoint"),
                }
    rows = []
    for a in ARMS:
        for s in SEEDS:
            if (a, s) in live:
                status = "RUNNING"
            else:
                r = best.get((a, s))
                status = r["status"] if r else "NOT_STARTED"
            rows.append({
                "arm": a, "seed": s, "status": status,
                "final_nt": best.get((a, s), {}).get("final_nt"),
                "best_val_loss": best.get((a, s), {}).get("best_val_loss"),
                "throughput_nt_s": best.get((a, s), {}).get("throughput_nt_s"),
                "best_checkpoint": best.get((a, s), {}).get("best_checkpoint"),
            })
    counts = Counter(r["status"] for r in rows)
    return {"matrix": "Track R (F1-F7,P1-P3) + B1 bridge x seeds 17/29/43",
            "cell_count": len(rows), "counts": dict(counts), "rows": rows,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ledger = scan(args.runs)
    with open(args.out, "w") as fh:
        json.dump(ledger, fh, indent=2)
    print("counts:", ledger["counts"])
    for r in ledger["rows"]:
        if r["status"] == "NOT_STARTED":
            print("  %s s%d NOT_STARTED" % (r["arm"], r["seed"]))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()