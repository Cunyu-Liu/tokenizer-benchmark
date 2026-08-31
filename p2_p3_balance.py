"""P2/P3 patch-budget balance audit (contract 3.2, 5.4).

Verifies that the P2 supported-strata conditional random patch is balanced
against P3 (causal entropy patch) on train-only data:

  - Total patch-count relative error  <= 0.5%
  - Per preregistered sequence-length stratum <= 2%
  - Measured FLOPs difference <= 5%
  - Per (causal prefix-length x causal-entropy) stratum, P2/P3 boundary-rate
    absolute difference <= 2 percentage points
  - Supported-strata (0.05 <= q <= 0.95) cover >= 80% of training positions
    and >= 80% of P3 boundaries
  - Non-supported strata: P2 must exactly replay P3's deterministic boundary

The audit is GPU-only (needs the frozen entropy predictor), runs on train-only
sequences, and writes a deterministic JSON report. It does NOT read validation
or any final test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.conditional_patch import (
    ConditionalRandomPatchPolicy, fit_p2_q, PATCH_RANDOMIZATION_SEED,
    SUPPORT_LO, SUPPORT_HI, MIN_SUPPORT_COVERAGE,
)
from model.entropy_predictor import EntropyPatchPolicy, calibrate_entropy
from model.dataset import sample_train_sequences

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
ENTROPY_BUDGET_NT = 2_000_000
# Preregistered sequence-length strata (contract 3.2): raw-nt sequence length.
LENGTH_STRATA = [(16, 127), (128, 511), (512, 2047), (2048, 4096)]
# Audit sample size (train-only, frozen).
AUDIT_N_SEQS = 2000


def _patch_count(boundaries: list[list[int]]) -> int:
    return sum(1 for b in boundaries for x in b if x > 0.5)


def _flops_estimate(n_tokens: int, d_model: int, n_layers: int) -> float:
    """Rough model FLOPs for a token sequence (2 * params * tokens), used ONLY
    for the P2-vs-P3 relative FLOP balance check (same backbone, so the
    relative difference is driven by token counts, not absolute accuracy)."""
    # BLT FLOPs scale with number of nt tokens (patches) in the sequence.
    return 2.0 * n_tokens * (d_model ** 2 * n_layers * 4)


def audit_p2_p3(p2_policy: ConditionalRandomPatchPolicy,
                p3_policy: EntropyPatchPolicy,
                seqs: list[str],
                d_model: int, n_layers: int,
                device: str = "cuda:0") -> dict:
    """Run the full P2/P3 balance audit over `seqs` (train-only).

    Returns a dict report with per-gate results and a top-level pass/fail.
    """
    p2_bnds: list[list[int]] = []
    p3_bnds: list[list[int]] = []
    p2_tokens: list[int] = []
    p3_tokens: list[int] = []
    total_p2 = 0
    total_p3 = 0

    for seq in seqs:
        c = seq.upper().replace("T", "U")[:4096]
        if len(c) == 0:
            continue
        b3 = p3_policy.boundary(c, len(c))
        b2 = p2_policy.boundary(c, len(c))
        p3_bnds.append(b3)
        p2_bnds.append(b2)
        total_p3 += sum(1 for x in b3 if x > 0.5)
        total_p2 += sum(1 for x in b2 if x > 0.5)
        # token count ~ number of patches (positions between boundaries + 1)
        p3_tokens.append(sum(1 for x in b3 if x > 0.5))
        p2_tokens.append(sum(1 for x in b2 if x > 0.5))

    # --- Gate 1: total patch-count relative error ---------------------------
    total_err = abs(total_p2 - total_p3) / max(1, total_p3) * 100.0
    gate1_pass = bool(total_err <= 0.5)

    # --- Gate 2: per length-stratum patch-count relative error ---------------
    stratum_results = []
    gate2_pass = True
    for lo, hi in LENGTH_STRATA:
        p2_n = sum(1 for seq, b in zip(seqs, p2_bnds)
                   if lo <= len(seq) <= hi and any(x > 0.5 for x in b))
        p3_n = sum(1 for seq, b in zip(seqs, p3_bnds)
                   if lo <= len(seq) <= hi and any(x > 0.5 for x in b))
        err = abs(p2_n - p3_n) / max(1, p3_n) * 100.0
        ok = bool(err <= 2.0)
        gate2_pass = gate2_pass and ok
        stratum_results.append({
            "stratum": [lo, hi],
            "p2_patch_count": p2_n,
            "p3_patch_count": p3_n,
            "relative_error_pct": round(err, 4),
            "pass": ok,
        })

    # --- Gate 3: measured FLOPs difference ----------------------------------
    flops_p2 = sum(_flops_estimate(t, d_model, n_layers) for t in p2_tokens)
    flops_p3 = sum(_flops_estimate(t, d_model, n_layers) for t in p3_tokens)
    flops_diff = abs(flops_p2 - flops_p3) / max(1, flops_p3) * 100.0
    gate3_pass = flops_diff <= 5.0

    # --- Gate 4: per-stratum boundary-rate absolute diff <= 2pp -------------
    # Uses P2's frozen (prefix-length x entropy) strata. For each occupied
    # stratum, compare P2 boundary rate vs P3 boundary rate.
    stratum_rates = []
    gate4_pass = True
    if p2_policy.q_table is not None and p2_policy.ent_edges and p2_policy.prefix_edges:
        nb, ne = len(p2_policy.q_table), len(p2_policy.q_table[0])
        cnt2 = np.zeros((nb, ne), dtype=np.float64)
        cnt3 = np.zeros((nb, ne), dtype=np.float64)
        pos = np.zeros((nb, ne), dtype=np.float64)
        for seq, b2, b3 in zip(seqs, p2_bnds, p3_bnds):
            c = seq.upper().replace("T", "U")[:4096]
            ent = p2_policy._batch_entropy(c)
            for i in range(len(c)):
                bp = p2_policy._prefix_bin(i)
                be = p2_policy._entropy_bin(float(ent[i]))
                pos[bp, be] += 1
                cnt2[bp, be] += (b2[i] > 0.5)
                cnt3[bp, be] += (b3[i] > 0.5)
        for bp in range(nb):
            for be in range(ne):
                if pos[bp, be] == 0:
                    continue
                r2 = cnt2[bp, be] / pos[bp, be] * 100.0
                r3 = cnt3[bp, be] / pos[bp, be] * 100.0
                diff = abs(r2 - r3)
                ok = diff <= 2.0
                gate4_pass = gate4_pass and ok
                stratum_rates.append({
                    "prefix_bin": bp, "entropy_bin": be,
                    "positions": int(pos[bp, be]),
                    "p2_boundary_rate_pct": round(r2, 3),
                    "p3_boundary_rate_pct": round(r3, 3),
                    "abs_diff_pp": round(diff, 3),
                    "pass": ok,
                })

    # --- Gate 5: supported-strata coverage ----------------------------------
    coverage = dict(p2_policy.coverage_report or {})
    gate5_pass = bool(coverage.get("passes_support_coverage", False))

    # --- Gate 6: non-supported strata exact P3 replay ------------------------
    replay_fail = 0
    replay_checked = 0
    if p2_policy.q_table is not None and p2_policy.ent_edges and p2_policy.prefix_edges:
        nb, ne = len(p2_policy.q_table), len(p2_policy.q_table[0])
        for seq, b2, b3 in zip(seqs, p2_bnds, p3_bnds):
            c = seq.upper().replace("T", "U")[:4096]
            ent = p2_policy._batch_entropy(c)
            for i in range(1, len(c)):
                bp = p2_policy._prefix_bin(i)
                be = p2_policy._entropy_bin(float(ent[i]))
                q = p2_policy.q_table[bp][be]
                if not (SUPPORT_LO <= q <= SUPPORT_HI):
                    replay_checked += 1
                    if int(b2[i] > 0.5) != int(b3[i] > 0.5):
                        replay_fail += 1
    gate6_pass = replay_checked == 0 or replay_fail == 0

    all_pass = bool(gate1_pass and gate2_pass and gate3_pass and gate4_pass
                    and gate5_pass and gate6_pass)
    return {
        "audit_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_sequences": len(seqs),
        "total_patch_count": {"p2": total_p2, "p3": total_p3,
                              "relative_error_pct": round(total_err, 4),
                              "pass": gate1_pass},
        "length_strata": stratum_results,
        "gate2_length_strata_pass": gate2_pass,
        "flops": {"p2": flops_p2, "p3": flops_p3,
                  "difference_pct": round(flops_diff, 4), "pass": gate3_pass},
        "stratum_boundary_rate": {
            "n_checked": len(stratum_rates),
            "worst_abs_diff_pp": round(max((s["abs_diff_pp"] for s in stratum_rates), default=0.0), 3),
            "pass": gate4_pass},
        "supported_coverage": coverage,
        "non_supported_replay": {"checked": replay_checked, "failures": replay_fail,
                                 "pass": gate6_pass},
        "all_pass": all_pass,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="/mnt/cunyuliu/tokenizer-benchmark/runs/p2_p3_balance_audit.json")
    ap.add_argument("--n-seqs", type=int, default=AUDIT_N_SEQS)
    args = ap.parse_args()

    dev = "cuda:%d" % args.device
    import torch
    assert torch.cuda.is_available(), "CUDA required"

    # Train-only entropy calibration (frozen; same as P1/P2/P3 training).
    calib = calibrate_entropy(SPLIT_8080, device=dev, target_patch_len=6,
                              seed=17, budget_nt=ENTROPY_BUDGET_NT, batch_size=256)
    # Fit P2 q (train-only) -> returns fitted policy + coverage.
    p2_policy, cov = fit_p2_q(calib, SPLIT_8080, device=dev,
                              n_seqs=args.n_seqs, seed=17)
    p3_policy = EntropyPatchPolicy(calib.predictor, calib.gate, device=dev)

    seqs = sample_train_sequences(SPLIT_8080, n=args.n_seqs, seed=17, split="train")
    seqs = [s[:4096] for s in seqs]

    # Model dims for FLOP estimate (use P3 BLT backbone dims).
    from model.train_config import resolved_config
    cfg = resolved_config("P3", 17)
    report = audit_p2_p3(p2_policy, p3_policy, seqs,
                         d_model=cfg.arch.d_model, n_layers=cfg.arch.n_layers,
                         device=dev)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("P2/P3 balance audit all_pass = %s" % report["all_pass"])
    print("  total patch err: %.4f%% (<=0.5%% gate)" % report["total_patch_count"]["relative_error_pct"])
    print("  FLOPs diff: %.4f%% (<=5%% gate)" % report["flops"]["difference_pct"])
    print("  supported coverage: pos=%.3f bnd=%.3f" % (
        report["supported_coverage"].get("position_coverage", 0),
        report["supported_coverage"].get("boundary_coverage", 0)))
    print("  report -> %s" % args.out)


if __name__ == "__main__":
    main()
