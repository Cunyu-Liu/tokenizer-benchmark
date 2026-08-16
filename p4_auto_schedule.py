#!/usr/bin/env python3
"""Phase 4 automatic run scheduler (owner-approved: use all 6 GPUs).

Monitors GPU availability and dispatches the remaining Phase 4 science runs
(10 arms x 3 seeds = 30 total) in a fixed priority order, maximizing GPU
utilization across GPU0-5 (up to 6 concurrent 100M jobs; recorded as an
approved operational deviation from Appendix B's '<=2 parallel' on 2026-08-15).

Queue logic:
  - A run is "needed" if (arm, seed) is in the 30-cell matrix and no existing
    phase4_<arm>_s<seed>_* run dir has status DONE.
  - FAIL_CLOSED_WITH_EVIDENCE / RUNNING_PRE_MANIFEST (stale) dirs do NOT count
    as done; the same (arm, seed) is re-launched (new run dir).
  - Priority order (contract + owner): F7 s17 -> P2 s17 -> P3 s17 -> all other
    s17 -> s29 -> s43 (arm order F1..F7, P1..P3).

Launch uses the exact reliable env (conda python, CUDA_VISIBLE_DEVICES pin,
expandable_segments). MIG partitions (GPU6/7) and occupied GPUs are skipped.
Logs every dispatch to phase4_auto.log.

Usage:
  python p4_auto_schedule.py            # one pass: launch as many as free GPUs allow
  python p4_auto_schedule.py --loop     # daemon: poll every 300s
  python p4_auto_schedule.py --dry-run  # print the plan, do not launch
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

PYTHON = "/home/cunyuliu/miniconda3/envs/toktokenbench/bin/python"
PROJ = "/home/cunyuliu/tokenizer-benchmark"
RUNS = "/mnt/cunyuliu/tokenizer-benchmark/runs"
AUTO_LOG = f"{RUNS}/phase4_auto.log"
TIMEOUT_S = 650000
FREE_MIN_MiB = 20000
MAX_CONCURRENT = 6  # owner-approved deviation: use GPU0-5 fully

ARMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "P1", "P2", "P3"]
SEEDS = [17, 29, 43]
# Priority: within a seed, F7/P2/P3 first (still pending), then the rest.
PRIORITY_PREFIX = {"F7": 0, "P2": 1, "P3": 2}
ARM_RE = re.compile(r"^(F[1-7]|P[1-3])$")


def log(msg: str):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(AUTO_LOG, "a") as fh:
        fh.write(line + "\n")


def _occupied_uuids() -> set[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return set()
    return {u.strip() for u in out.strip().splitlines() if u.strip()}


def gpu_table() -> list[dict]:
    occ = _occupied_uuids()
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    raw = {}
    for line in out.strip().splitlines():
        idx, used, total, util = [c.strip() for c in line.split(",")]
        raw[int(idx)] = {"used": int(used), "total": int(total),
                         "mig": util == "[N/A]"}
    umap = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    idx2uuid = {}
    for line in umap.strip().splitlines():
        i, u = [c.strip() for c in line.split(",")]
        idx2uuid[int(i)] = u
    result = []
    for idx in sorted(raw):
        g = {"index": idx, **raw[idx]}
        g["free"] = g["total"] - g["used"]
        g["occupied"] = idx2uuid.get(idx) in occ
        result.append(g)
    return result


def _our_gpu_pids() -> dict:
    """Map physical GPU index -> set of our p4_train pids (via CUDA_VISIBLE_DEVICES)."""
    import re as _re
    gpu_of_pid = {}
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args="], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            m = _re.match(r"\s*(\d+)\s+(.*p4_train\.py.*)", line)
            if not m:
                continue
            pid = m.group(1)
            args = m.group(2)
            # CUDA_VISIBLE_DEVICES=<gpu> prefix in the bash -lc wrapper
            mm = _re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", args)
            if mm:
                gpu_of_pid.setdefault(int(mm.group(1)), set()).add(pid)
    except Exception:
        pass
    return gpu_of_pid


def free_gpus() -> list[dict]:
    """Full, non-MIG GPUs with enough free memory for a 100M run.

    Decision rule (2026-08-16, owner-approved: keep legacy mRNA_editflow
    processes running): a GPU is usable if it is non-MIG, has
    >= FREE_MIN_MiB free memory, and is NOT already hosting one of our
    p4_train training processes (no double-booking). Legacy non-benchmark
    processes occupy a few GiB on some cards but leave ample free memory.
    """
    our_gpus = _our_gpu_pids()
    cands = [g for g in gpu_table()
             if not g["mig"]
             and g["free"] >= FREE_MIN_MiB
             and not our_gpus.get(g["index"])]
    return sorted(cands, key=lambda g: (g["index"] == 0, -g["free"]))


def existing_run_statuses() -> dict:
    """{(arm, seed): status} from all phase4_<arm>_s<seed>_* run dirs.

    A cell counts as RUNNING if ANY phase4_<arm>_s<seed>_* directory has a
    live p4_train.py process, even before its manifest exists
    (RUNNING_PRE_MANIFEST). Otherwise we read the manifest status. FAIL /
    stale dirs do NOT satisfy the cell.
    """
    import json

    # live p4_train processes: { (arm, seed) } from the command line.
    live = set()
    try:
        out = subprocess.run(
            ["ps", "-eo", "args="], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            m = re.search(r"p4_train\.py --arm (F[1-7]|P[1-3]) --seed (17|29|43)", line)
            if m:
                live.add((m.group(1), int(m.group(2))))
    except Exception:
        pass

    status = {}
    if not os.path.isdir(RUNS):
        return {k: ("RUNNING" if k in live else "UNKNOWN") for k in live}
    for d in os.listdir(RUNS):
        m = re.match(r"phase4_(F[1-7]|P[1-3])_s(17|29|43)_", d)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        key = (arm, seed)
        if key in live:
            status[key] = "RUNNING"
            continue
        manifest = os.path.join(RUNS, d, "manifest.json")
        st = "UNKNOWN"
        if os.path.isfile(manifest):
            try:
                with open(manifest) as fh:
                    st = json.load(fh).get("status", "UNKNOWN")
            except Exception:
                st = "CORRUPT"
        # FAIL / RUNNING_PRE_MANIFEST / stale dirs do NOT satisfy the cell.
        if key not in status or st == "DONE":
            status[key] = st
    return status


def build_priority_queue(existing: dict) -> list[tuple[str, int]]:
    """Return the ordered list of (arm, seed) cells that still need a run.

    Cells already DONE or RUNNING (including RUNNING_PRE_MANIFEST, detected
    via live processes) are excluded; FAIL / stale cells are re-launched.
    """
    needed = []
    for seed in SEEDS:
        for arm in ARMS:
            st = existing.get((arm, seed))
            if st in ("DONE", "RUNNING", "RUNNING_PRE_MANIFEST"):
                continue  # satisfied or already training
            needed.append((arm, seed))
    # Stabilize with the owner priority prefix within each seed group.
    def key(item):
        arm, seed = item
        prio = PRIORITY_PREFIX.get(arm, 10)
        return (SEEDS.index(seed), prio, ARMS.index(arm))
    needed.sort(key=key)
    return needed


def running_count(existing: dict) -> int:
    """Count (arm, seed) cells currently RUNNING (or RUNNING_PRE_MANIFEST)."""
    return sum(1 for st in existing.values() if st in ("RUNNING", "RUNNING_PRE_MANIFEST"))


def launch(arm: str, seed: int, gpu: int) -> None:
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = f"{RUNS}/phase4_{arm}_s{seed}_{ts}"
    inner = (f"{PYTHON} -u p4_train.py --arm {arm} --seed {seed} "
             f"--device 0 --out-dir {out_dir}")
    prefix = (f"cd {PROJ} && CUDA_VISIBLE_DEVICES={gpu} "
              f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ")
    cmd = (f"nohup timeout {TIMEOUT_S} bash -lc {shlex.quote(prefix + inner)} "
           f">> {out_dir}/run.log 2>&1 &")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(cmd, shell=True, check=True, cwd=PROJ)
    log(f"LAUNCH {arm} s{seed} -> GPU {gpu} -> {out_dir}")


def one_pass(dry_run: bool = False) -> int:
    existing = existing_run_statuses()
    free = free_gpus()
    running = running_count(existing)
    needed = build_priority_queue(existing)

    log("pass: running=%d free=%d needed=%d" % (running, len(free), len(needed)))
    for (arm, seed) in needed:
        log("  pending %s s%d" % (arm, seed))

    launched = 0
    for g in free:
        if running + launched >= MAX_CONCURRENT:
            break
        if not needed:
            break
        arm, seed = needed.pop(0)
        if dry_run:
            log("DRY: would launch %s s%d on GPU%d" % (arm, seed, g["index"]))
        else:
            launch(arm, seed, g["index"])
        launched += 1
    return launched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", action="store_true", help="daemon: poll every 300s")
    ap.add_argument("--interval", type=int, default=300,
                    help="poll interval seconds when --loop")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.loop:
        log("auto-scheduler daemon start (interval=%ds)" % args.interval)
        while True:
            try:
                one_pass(dry_run=args.dry_run)
            except Exception as e:
                log("pass error: %r" % e)
            time.sleep(args.interval)
    else:
        one_pass(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
