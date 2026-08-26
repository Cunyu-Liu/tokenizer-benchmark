"""Phase 4 codec main-table driver (contract 3.6 headline canonical_code_length_BPN).

For every DONE (arm, seed) cell it discovers a final (budget-adjacent) checkpoint,
scores it with ``p4_eval_codec.py`` on the SAME frozen homology-stratified
subsample (--subsample), and aggregates the V3 headline ``canonical_code_length_BPN``
across seeds per arm into a JSON codec table, plus Uniform/Markov/PPM calibration
baselines (computed once on train, scored on the same test subset).

This is GPU-bound (per-prefix codec is O(T^2)), so it is intended to run on a
dedicated, non-contended GPU after Phase-4 training frees capacity. It shells out
to ``p4_eval_codec.py`` per arm so each per-arm result is an auditable JSON
artifact, then merges them.

Run (example):
  CUDA_VISIBLE_DEVICES=2 python -m p4_eval_codec_table --subsample <parquet> \\
      --n 1000 --device 0 --calib-n 2000 --out codec_main_table.json

CPU-only test of checkpoint discovery is in tests/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq

from evaluator.codec_scoring import calibration_bpns

RUN_RE = re.compile(
    r"^phase4_(F[1-7]|P[1-3]|B1)_s(\d+)_\d{8}T\d{6}(?:_restart)?$"
)
DEFAULT_RUNS = "/mnt/cunyuliu/tokenizer-benchmark/runs"
DEFAULT_SPLIT = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"


def discover_done(runs_dir: str) -> list[tuple[str, int, str]]:
    """(arm, seed, final_checkpoint) for every accepted DONE run cell.

    The final checkpoint is the LAST validated checkpoint in the manifest's
    ``checkpoints`` list (budget-adjacent final-budget checkpoint; Phase-4
    training breaks at the 2.0B budget before saving a 2.0B val checkpoint, so
    this is the last saved ~1.9-1.999B state, deterministic and consistent
    across arms). Corrected-retry directories are valid when their manifest is
    DONE. If more than one DONE bundle exists for a cell, prefer the bundle
    with the furthest checkpoint, then the later timestamped directory.
    """
    best: dict[tuple[str, int], tuple[float, str, str]] = {}
    for d in sorted(os.listdir(runs_dir)):
        m = RUN_RE.match(d)
        if not m:
            continue
        mf = os.path.join(runs_dir, d, "manifest.json")
        if not os.path.isfile(mf):
            continue
        with open(mf) as fh:
            man = json.load(fh)
        if man.get("status") != "DONE":
            continue
        cks = man.get("checkpoints") or []
        if not cks:
            continue
        ckpt = cks[-1].get("path")
        if not ckpt:
            continue
        nt = cks[-1].get("nt")
        rank_nt = float(nt) if isinstance(nt, (int, float)) else -1.0
        key = (m.group(1), int(m.group(2)))
        candidate = (rank_nt, d, ckpt)
        if key not in best or candidate[:2] > best[key][:2]:
            best[key] = candidate

    out = [(arm, seed, value[2])
           for (arm, seed), value in best.items()]
    # deterministic sort by (arm, seed)
    arm_order = {"F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4, "F6": 5, "F7": 6,
                 "P1": 7, "P2": 8, "P3": 9, "B1": 10}
    out.sort(key=lambda t: (arm_order.get(t[0], 99), t[1]))
    return out


def run_one(arm: str, seed: int, ckpt: str, subsample: str, n: int,
            device: int, dev, calib_n: int = 0) -> tuple[int, dict]:
    """Score one arm/seed via p4_eval_codec (subprocess); returns (rc, result)."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    cmd = [
        sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4_eval_codec.py"),
        "--arm", arm, "--seed", str(seed), "--ckpt", ckpt,
        "--subsample", subsample, "--n", str(n),
        "--calib-n", str(calib_n), "--device", str(device),
        "--out", tmp,
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(dev)
    print("RUN", arm, "s%d" % seed, flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    rc = proc.returncode
    result = None
    if rc == 0:
        with open(tmp) as fh:
            result = json.load(fh)
    else:
        # The table driver is normally redirected to a durable log. Surface
        # the child diagnostics there instead of reducing failures to rc=1.
        if proc.stderr:
            print("  STDERR(%s s%d):" % (arm, seed),
                  proc.stderr[-2000:], flush=True)
        if proc.stdout:
            print("  STDOUT(%s s%d):" % (arm, seed),
                  proc.stdout[-1000:], flush=True)
    os.remove(tmp)
    return rc, result


def codec_subsample_seqs(subsample: str) -> list[str]:
    pf = pq.ParquetFile(subsample)
    seqs: list[str] = []
    for batch in pf.iter_batches(batch_size=50_000, columns=["canonical_sequence"]):
        seqs.extend(batch.to_pydict()["canonical_sequence"])
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--subsample", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--device", type=int, default=0, help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--arms", default=None,
                    help="comma list to restrict (default: all DONE cells)")
    ap.add_argument("--calib-n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    done = discover_done(args.runs)
    if args.arms:
        allow = set(args.arms.split(","))
        done = [t for t in done if t[0] in allow]
    if not done:
        raise SystemExit("no DONE cells with a final checkpoint found")

    cells = []
    for arm, seed, ckpt in done:
        rc, res = run_one(arm, seed, ckpt, args.subsample, args.n, args.device,
                          args.device, calib_n=0)
        row = {"arm": arm, "seed": seed, "ckpt": ckpt, "rc": rc}
        if res:
            for k in ("canonical_code_length_BPN", "canonical_code_nll_BPN",
                      "quantized_cdf_nll_bits", "coded_bits", "valid_nt",
                      "n_sequences", "decoded_byte_identical", "nll_gate_ok_1e4",
                      "bits_gate_ok_64", "coder_overhead_bits",
                      "canonical_code_nll_BPN", "evaluator_status"):
                row[k] = res.get(k)
            row["decode_gate_method"] = res.get("decode_gate_method")
        cells.append(row)
        print("  %s s%d rc=%d BPN=%s decode_ok=%s" % (
            arm, seed, rc, row.get("canonical_code_length_BPN"),
            row.get("decoded_byte_identical")), flush=True)

    # Uniform/Markov/PPM baselines scored once on the same test subset.
    calibration = None
    if args.calib_n > 0:
        test_seqs = codec_subsample_seqs(args.subsample)[:args.n]
        train_df_rows = []
        pf = pq.ParquetFile(args.split)
        for batch in pf.iter_batches(batch_size=50_000,
                                     columns=["split_membership", "canonical_sequence"]):
            d = batch.to_pydict()
            for s, seq in zip(d["split_membership"], d["canonical_sequence"]):
                if s == "train":
                    train_df_rows.append(seq)
                    if len(train_df_rows) >= args.calib_n:
                        break
            if len(train_df_rows) >= args.calib_n:
                break
        calibration = calibration_bpns(train_df_rows, test_seqs, markov_order=3)

    table = {
        "headline": "canonical_code_length_BPN",
        "split": "test",
        "subsample": args.subsample,
        "n_requested": args.n,
        "cells": cells,
        "arms": sorted({c["arm"] for c in cells}),
        "calibration": calibration,
        "generated_utc": None,
    }
    with open(args.out, "w") as fh:
        json.dump(table, fh, indent=2, default=str)
    print("WROTE", args.out)

    # compact per-arm mean digest (only cells with a finished score)
    from collections import defaultdict
    by_arm = defaultdict(list)
    for c in cells:
        if c.get("canonical_code_length_BPN") is not None:
            by_arm[c["arm"]].append(c["canonical_code_length_BPN"])
    print("\nArm-mean canonical_code_length_BPN:")
    for arm in sorted(by_arm):
        xs = by_arm[arm]
        print("  %s mean=%.4f n=%d" % (arm, sum(xs) / len(xs), len(xs)))


if __name__ == "__main__":
    main()
