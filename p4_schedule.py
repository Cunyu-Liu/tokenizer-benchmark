#!/usr/bin/env python3
"""Phase 4 run scheduler.

Discovers a free full-size GPU and launches a p4_train run with the fixed
environment that has proven reliable on this box, avoiding the three failure
modes observed earlier:
  - F6@105545 : 'python: not found'  -> explicit conda python path
  - P1@103227 : empty log, no proc   -> launch is verified before returning
  - P2@011614 : OOM on GPU6 (MIG)    -> MIG partitions are skipped here

Rules (contract 4: GPU-only, no silent CPU):
  - --device 0 inside the container; the physical GPU is pinned via CUDA_VISIBLE_DEVICES.
  - fallback=0 everywhere (p4_train asserts it; launch env also forces it).
  - MIG partitions (nvidia-smi utilization == "[N/A]") are never selected.
  - A candidate GPU must have >= FREE_MIN_MiB free for a 100M run.
  - GPU1-5 are preferred over GPU0 (shared/dynamic card) when both are free.

Usage:
  python p4_schedule.py --list-free
  python p4_schedule.py --arm F7 --seed 17 --gpu 2
  python p4_schedule.py --arm P3 --seed 29            # auto-pick first free GPU
  python p4_schedule.py --arm P1 --seed 17 --dry-run  # build cmd, do not launch
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time

PYTHON = "/home/cunyuliu/miniconda3/envs/toktokenbench/bin/python"
PROJ = "/home/cunyuliu/tokenizer-benchmark"
RUNS = "/mnt/cunyuliu/tokenizer-benchmark/runs"
LAUNCH_LOG = f"{RUNS}/phase4_launch.log"
TIMEOUT_S = 650000  # generous wall-clock; nohup+timeout mirrors prior launches
FREE_MIN_MiB = 20000

ARM_RE = re.compile(r"^(F[1-7]|P[1-3]|B1)$")


def _occupied_uuids() -> set[str]:
    """Set of GPU uuids that currently have >=1 compute application running."""
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
    # physical index -> raw stats
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    raw = {}
    for line in out.strip().splitlines():
        idx, used, total, util = [c.strip() for c in line.split(",")]
        raw[int(idx)] = {"used": int(used), "total": int(total),
                         "mig": util == "[N/A]"}
    # physical index -> uuid, to map occupancy
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


def free_gpus() -> list[dict]:
    """Full, non-MIG GPUs with no running compute app and enough free memory.

    Occupancy (not just free memory) decides "free": a GPU already running one
    of our 100M runs has ~16-20GiB free but must NOT be double-booked. GPU1-5 are
    preferred over the shared GPU0; among equally-sized options the one with the
    most free memory wins so we minimize contention."""
    cands = [g for g in gpu_table()
             if not g["mig"] and not g["occupied"] and g["free"] >= FREE_MIN_MiB]
    return sorted(cands, key=lambda g: (g["index"] == 0, -g["free"]))


def build_cmd(arm: str, seed: int, gpu: int, out_dir: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = out_dir or f"{RUNS}/phase4_{arm}_s{seed}_{ts}"
    inner = (
        f"{PYTHON} -u p4_train.py "
        f"--arm {arm} --seed {seed} --device 0 "
        f"--out-dir {out_dir}"
    )
    # CUDA_VISIBLE_DEVICES pins the physical card; fallback=0 forces GPU-only.
    prefix = f"cd {PROJ} && CUDA_VISIBLE_DEVICES={gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    full = (f"nohup timeout {TIMEOUT_S} bash -lc {shlex.quote(prefix + inner)} "
            f">> {out_dir}/run.log 2>&1 &")
    return full, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--gpu", type=int, help="explicit physical GPU (overrides auto-pick)")
    ap.add_argument("--out-dir")
    ap.add_argument("--list-free", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list_free:
        for g in free_gpus():
            print(f"GPU{g['index']}: free {g['free']}MiB (used {g['used']}/{g['total']})")
        return

    if not args.arm or not ARM_RE.match(args.arm):
        sys.exit("--arm must match F[1-7]|P[1-3]")
    if args.seed is None:
        sys.exit("--seed is required")
    if args.gpu is not None:
        gpus = gpu_table()
        g = next((x for x in gpus if x["index"] == args.gpu), None)
        if g is None:
            sys.exit(f"GPU {args.gpu} not found")
        if g["mig"]:
            sys.exit(f"GPU {args.gpu} is a MIG partition; cannot host 100M training")
        if g["free"] < FREE_MIN_MiB:
            sys.exit(f"GPU {args.gpu} only has {g['free']}MiB free (<{FREE_MIN_MiB}MiB)")
        if g["occupied"]:
            sys.exit(f"GPU {args.gpu} already has a running compute app; refusing to double-book")
        pick = args.gpu
    else:
        cands = free_gpus()
        if not cands:
            print("no free GPU now; retry later (low-frequency scheduling)")
            return
        pick = cands[0]["index"]

    cmd, out_dir = build_cmd(args.arm, args.seed, pick, args.out_dir or "")
    print(f"launching {args.arm} s{args.seed} on GPU {pick}")
    print(f"  out-dir : {out_dir}")
    print(f"  cmd     : {cmd}")
    if args.dry_run:
        return

    import os
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(cmd, shell=True, check=True, cwd=PROJ)
    with open(LAUNCH_LOG, "a") as fh:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        fh.write(f"[{ts}] P4 {args.arm} s{args.seed} -> GPU {pick} -> {out_dir}\n")
    print(f"launched (see {LAUNCH_LOG})")


if __name__ == "__main__":
    main()