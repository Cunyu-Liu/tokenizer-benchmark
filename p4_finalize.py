"""Phase 4 run aggregation / finalize (contract 2.3, 4, 5.4).

Consolidates every phase4_* science-run manifest under /mnt into a single
auditable status registry and validates each COMPLETED run against the Phase 4
acceptance invariants:

  - status == DONE
  - cumulative valid target nt reached the 2.0B budget (>= budget_nt)
  - cpu_fallback_count == 0 (GPU-only, no silent CPU fallback)
  - a best validation checkpoint was selected (best_val_loss finite, path set)
  - parameters recorded (> 0)
  - arm / seed are one of the 10x3 scientific matrix (no silent seed swap)

Also reports which (arm, seed) cells are complete / running / pending so the
remaining 100M matrix can be scheduled as GPUs free.

Read-only: never writes to run dirs. Writes only the aggregate registry JSON.
Usage:
  conda activate toktokenbench
  python p4_finalize.py --runs /mnt/cunyuliu/tokenizer-benchmark/runs \
      --out /mnt/cunyuliu/tokenizer-benchmark/runs/phase4_registry.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

ARMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "P1", "P2", "P3"]
SEEDS = [17, 29, 43]
RUN_RE = re.compile(r"^phase4_(F[1-7]|P[1-3])_s(\d+)_\d{8}T\d{6}$")


def validate_manifest(m: dict, arm: str, seed: int) -> list[str]:
    """Return a list of acceptance violations (empty == pass)."""
    issues = []
    if m.get("status") != "DONE":
        issues.append("status!=DONE (%s)" % m.get("status"))
    if m.get("arm") != arm:
        issues.append("arm mismatch (%s)" % m.get("arm"))
    if m.get("seed") != seed:
        issues.append("seed mismatch (%s)" % m.get("seed"))
    budget = (m.get("config") or {}).get("budget_nt")
    final_nt = m.get("final_nt")
    if budget is None or final_nt is None:
        issues.append("missing budget_nt/final_nt")
    elif final_nt < budget:
        issues.append("final_nt %s < budget %s" % (final_nt, budget))
    if m.get("cpu_fallback_count") != 0:
        issues.append("cpu_fallback_count=%s" % m.get("cpu_fallback_count"))
    bv = m.get("best_val_loss")
    if bv is None:
        issues.append("no best_val_loss")
    if not m.get("best_checkpoint"):
        issues.append("no best_checkpoint")
    if not (m.get("params") or 0) > 0:
        issues.append("no params")
    return issues


def read_run(path: str) -> dict:
    mf = os.path.join(path, "manifest.json")
    if not os.path.exists(mf):
        return {"dir": os.path.basename(path), "manifest": False,
                "status": "RUNNING_PRE_MANIFEST"}
    with open(mf) as fh:
        m = json.load(fh)
    entry = {
        "dir": os.path.basename(path),
        "run_id": m.get("run_id"),
        "arm": m.get("arm"),
        "seed": m.get("seed"),
        "status": m.get("status"),
        "final_nt": m.get("final_nt"),
        "budget_nt": (m.get("config") or {}).get("budget_nt"),
        "batch_nt": (m.get("config") or {}).get("batch_nt"),
        "best_val_loss": m.get("best_val_loss"),
        "best_checkpoint": m.get("best_checkpoint"),
        "n_checkpoints": len(m.get("checkpoints", [])),
        "final_loss": m.get("final_loss"),
        "throughput_nt_s": m.get("throughput_nt_s"),
        "peak_vram_mb": m.get("peak_vram_mb"),
        "params": m.get("params"),
        "non_embedding_params": m.get("non_embedding_params"),
        "cpu_fallback_count": m.get("cpu_fallback_count"),
        "wall_seconds": m.get("wall_seconds"),
        "end_utc": m.get("end_utc"),
    }
    if m.get("status") == "DONE":
        entry["issues"] = validate_manifest(m, m.get("arm"), m.get("seed"))
        entry["accept"] = len(entry["issues"]) == 0
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/mnt/cunyuliu/tokenizer-benchmark/runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = sorted(
        d for d in os.listdir(args.runs)
        if RUN_RE.match(d) and os.path.isdir(os.path.join(args.runs, d)))

    cells = {}
    entries_by_cell = {}
    for arm in ARMS:
        for s in SEEDS:
            cells[(arm, s)] = {"state": "PENDING"}
    running = []
    completed = []
    for d in dirs:
        m = RUN_RE.match(d)
        arm, seed = m.group(1), int(m.group(2))
        key = (arm, seed)
        entry = read_run(os.path.join(args.runs, d))
        entry["cell"] = "%s_s%d" % (arm, seed)
        entries_by_cell[(arm, seed)] = entry
        if entry.get("status") == "DONE":
            cells[key] = {"state": "DONE" if entry.get("accept") else "FAIL",
                          "best_val_loss": entry.get("best_val_loss"),
                          "issues": entry.get("issues", [])}
            completed.append(entry)
        elif entry.get("status") == "RUNNING_PRE_MANIFEST":
            cells[key] = {"state": "RUNNING"}
            running.append(entry)
        else:
            cells[key] = {"state": "RUNNING"}

    registry = {
        "phase": 4,
        "matrix": "10 arms x 3 seeds = 30 runs",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": {"%s_s%d" % k: v for k, v in cells.items()},
        "completed": completed,
        "running": running,
        "counts": {
            "done": sum(1 for c in cells.values() if c["state"] == "DONE"),
            "fail": sum(1 for c in cells.values() if c["state"] == "FAIL"),
            "running": sum(1 for c in cells.values() if c["state"] == "RUNNING"),
            "pending": sum(1 for c in cells.values() if c["state"] == "PENDING"),
            "total": len(cells),
        },
    }

    out = args.out or os.path.join(args.runs, "phase4_registry.json")
    with open(out, "w") as fh:
        json.dump(registry, fh, indent=2, default=str)

    # Summary table
    print("%-9s %-8s %-10s %-10s %s" % ("CELL", "STATE", "BEST_VAL", "NT/BUDGET", "NOTES"))
    for d in dirs:
        m = RUN_RE.match(d)
        arm, seed = m.group(1), int(m.group(2))
        e = entries_by_cell[(arm, seed)]
        state = cells[(arm, seed)]["state"]
        if e.get("status") != "DONE":
            print("%-9s %-8s" % (e["cell"], state))
            continue
        bv = "%.4f" % e["best_val_loss"] if e.get("best_val_loss") is not None else "-"
        nb = "%s/%s" % (e.get("final_nt"), e.get("budget_nt"))
        issues = e.get("issues", [])
        print("%-9s %-8s %-10s %-10s %s" % (
            e["cell"], state, bv, nb, "; ".join(issues) if issues else "OK"))
    c = registry["counts"]
    print("DONE=%d FAIL=%d RUNNING=%d PENDING=%d / %d" % (
        c["done"], c["fail"], c["running"], c["pending"], c["total"]))
    print("WROTE", out)


if __name__ == "__main__":
    main()
