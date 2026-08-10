"""Phase 4 read-only run monitor (contract 5.5 low-frequency monitoring).

Scans the /mnt run tree for phase4_* bundles and reports, per run:
  - status (RUNNING_PRE_MANIFEST / DONE / FAILED),
  - cumulative valid target nt vs 2.0B budget,
  - best validation loss and best checkpoint,
  - throughput (nt/s), peak VRAM, cpu_fallback_count.

Read-only: never writes, never touches checkpoints, never starts/stops jobs.
Usage:
  conda activate toktokenbench
  python p4_status.py --runs /mnt/cunyuliu/tokenizer-benchmark/runs
"""
from __future__ import annotations

import argparse
import json
import os
import time


def read_run(path: str) -> dict | None:
    mf = os.path.join(path, "manifest.json")
    if not os.path.exists(mf):
        # running but no manifest yet (before first validation checkpoint)
        log = os.path.join(path, "run.log")
        nlines = 0
        if os.path.exists(log):
            with open(log) as fh:
                nlines = sum(1 for _ in fh)
        return {"dir": os.path.basename(path), "status": "RUNNING_PRE_MANIFEST",
                "log_lines": nlines}
    with open(mf) as fh:
        m = json.load(fh)

    # Top-level summary fields (status, final_nt, best_val_loss, best_checkpoint)
    # are only written when a run completes (status=DONE). While a run is in
    # progress the manifest accumulates `checkpoints`/`validations` arrays
    # incrementally at each validation; derive live progress from those so the
    # monitor is useful during training (contract 5.5 low-frequency monitoring).
    ckpts = m.get("checkpoints", [])
    status = m.get("status")
    if status is None and ckpts:
        status = "RUNNING"
    final_nt = m.get("final_nt")
    if final_nt is None and ckpts:
        final_nt = max(c["nt"] for c in ckpts)
    best_val = m.get("best_val_loss")
    best_ckpt = m.get("best_checkpoint")
    if best_val is None and ckpts:
        best = min(ckpts, key=lambda c: c.get("val_loss", float("inf")))
        if "val_loss" in best:
            best_val = best["val_loss"]
            best_ckpt = best["path"]
    return {
        "dir": os.path.basename(path), "arm": m.get("arm"), "seed": m.get("seed"),
        "run_id": m.get("run_id"), "status": status,
        "final_nt": final_nt, "budget": m.get("config", {}).get("budget_nt"),
        "best_val_loss": best_val,
        "best_ckpt": os.path.basename(best_ckpt) if best_ckpt else None,
        "n_checkpoints": len(ckpts),
        "final_loss": m.get("final_loss"),
        "throughput_nt_s": round(m.get("throughput_nt_s") or 0),
        "peak_vram_mb": round(m.get("peak_vram_mb") or 0),
        "cpu_fallback": m.get("cpu_fallback_count"),
        "batch_nt": m.get("config", {}).get("batch_nt"),
        "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(mf))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/mnt/cunyuliu/tokenizer-benchmark/runs")
    args = ap.parse_args()

    # Only real run bundles: directories matching phase4_<arm>_s<seed>_<ts>.
    # Skips launch.log files and smoke/aux dirs that are not full science runs.
    import re
    pat = re.compile(r"^phase4_(F[1-7]|P[1-3])_s(\d+)_\d{8}T\d{6}$")
    dirs = sorted(
        d for d in os.listdir(args.runs)
        if pat.match(d) and os.path.isdir(os.path.join(args.runs, d)))
    if not dirs:
        print("no phase4 run dirs found under", args.runs)
        return
    print("%-34s %-22s %-9s %-12s %-10s %-10s %-9s %-9s %s" % (
        "DIR", "STATUS", "NT/BUDGET", "BEST_VAL", "FINAL_LOSS", "NT/S", "VRAM_MB", "FALLBACK", "BEST_CKPT"))
    for d in dirs:
        r = read_run(os.path.join(args.runs, d))
        if r is None:
            continue
        nt = r.get("final_nt")
        budget = r.get("budget")
        nt_s = "%d/%s" % (nt, budget) if (nt is not None and budget is not None) else "-"
        bv = "%.4f" % r["best_val_loss"] if r.get("best_val_loss") is not None else "-"
        fl = "%.4f" % r["final_loss"] if r.get("final_loss") is not None else "-"
        print("%-34s %-22s %-12s %-10s %-10s %-9s %-9s %-9s %s" % (
            r["dir"], r["status"], nt_s, bv, fl,
            r.get("throughput_nt_s") or "-", r.get("peak_vram_mb") or "-",
            r.get("cpu_fallback") if r.get("cpu_fallback") is not None else "-",
            r.get("best_ckpt") or "-"))


if __name__ == "__main__":
    main()
