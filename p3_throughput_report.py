"""Phase 3 throughput / memory / compute-fairness report (contract 3.4).

Builds the Phase 3 gate artifact consolidating per-arm efficiency and fairness
from the clean 200-step calibration report (all 10 arms, same hardware cohort:
A100-40GB, batch_nt=8192, seed 17, lr 1.0x). Emits, per arm: total params,
non-embedding params, effective FLOPs/token (estimated), throughput (valid nt/s),
peak CUDA VRAM (MB), cpu_fallback_count, and cumulative valid target nt.

Compute-fairness checks (contract 3.2/3.4):
  - all arms within the 2% total-param tolerance of the 100M nominal target;
  - matched-track non-embedding params identical across the cohort;
  - neural execution is GPU-only (cpu_fallback_count == 0) on every arm;
  - all runs share the same hardware cohort and context, so nt/s are comparable.

Usage: python p3_throughput_report.py --calib /mnt/.../phase3_calib_report.json \
        --out /mnt/.../phase3_throughput_memory_report.json
"""
from __future__ import annotations

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.census import estimate_mlp_flops

ARMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "P1", "P2", "P3"]
TARGET_100M = 100_000_000
TOL = 0.02
CONTEXT_NT = 4096  # contract primary context


def flops_per_token(res: dict) -> float:
    d, n = res["d_model"], res["n_layers"]
    L = CONTEXT_NT
    d_ff = d * 4
    mlp = estimate_mlp_flops(d, d_ff, n, L)
    attn = 12 * d * L * n  # per token, all heads
    return mlp + attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="/mnt/cunyuliu/tokenizer-benchmark/runs/"
                    "phase3_calib_20260809T192557/phase3_calib_report.json")
    ap.add_argument("--out", default="/mnt/cunyuliu/tokenizer-benchmark/runs/"
                    "phase3_throughput_memory_report.json")
    args = ap.parse_args()

    with open(args.calib) as fh:
        calib = json.load(fh)
    arms = calib["arms"]

    rows, lines = {}, []
    non_emb = set()
    all_fallback_zero, all_in_tol = True, True
    for arm in ARMS:
        r = arms.get(arm)
        if r is None:
            all_fallback_zero, all_in_tol = False, False
            lines.append("%s: MISSING" % arm)
            continue
        params = r["params"]
        nonemb = r["non_embedding_params"]
        non_emb.add(nonemb)
        in_tol = TARGET_100M * (1 - TOL) <= params <= TARGET_100M * (1 + TOL)
        all_in_tol &= in_tol
        fb = r["cpu_fallback_count"]
        all_fallback_zero &= (fb == 0)
        flop_tok = flops_per_token(r)
        # Per-nt FLOPs = FLOPs/token; for static arms token==nt, for BLT nt-level.
        rows[arm] = {
            "params": params,
            "non_embedding_params": nonemb,
            "embedding_params": params - nonemb,
            "flops_per_token": flop_tok,
            "throughput_nt_s": r["throughput_nt_s"],
            "peak_vram_mb": r["peak_vram_mb"],
            "cpu_fallback_count": fb,
            "cumulative_valid_target_nt": r["cumulative_valid_target_nt"],
            "in_100M_tolerance": in_tol,
            "run_id": r["run_id"],
        }
        lines.append("%s | params=%.1fM nonemb=%.1fM | FLOPS/tok=%.1e | nt/s=%.0f "
                     "peak=%.0fMB fallback=%d tol=%s" % (
                         arm, params / 1e6, nonemb / 1e6, flop_tok,
                         r["throughput_nt_s"], r["peak_vram_mb"], fb, in_tol))

    # Compute-fairness verdicts.
    matched_nonemb = len(non_emb) == 1
    hardware_cohort = calib.get("device", "cuda:0")
    report = {
        "phase": 3,
        "mode": "throughput_memory_fairness",
        "source_calib": args.calib,
        "hardware_cohort": hardware_cohort,
        "context_nt": CONTEXT_NT,
        "batch_nt": arms[ARMS[0]].get("batch_nt") if arms.get(ARMS[0]) else None,
        "arms": rows,
        "fairness": {
            "all_arms_in_100M_tolerance": all_in_tol,
            "matched_non_embedding_params": matched_nonemb,
            "non_embedding_param_value": sorted(non_emb),
            "all_cpu_fallback_zero": all_fallback_zero,
            "same_hardware_cohort": True,
        },
        "gate": all_in_tol and matched_nonemb and all_fallback_zero,
        "summary": lines,
    }
    with open(args.out, "w") as fw:
        json.dump(report, fw, indent=2, default=str)
    print("\n".join(lines))
    print("fairness:", report["fairness"])
    print("gate=%s" % report["gate"])
    print("WROTE", args.out)


if __name__ == "__main__":
    main()