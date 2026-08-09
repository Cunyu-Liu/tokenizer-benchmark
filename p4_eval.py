"""Phase 4 sealed-test main-table scorer (contract 3.5, 5.4).

Loads a trained Phase 4 checkpoint and computes likelihood metrics over a
held-out split:

  - next_base_BPN     : exact per-base cross-entropy (F1 NUC, F4/F5 overlap,
                        P1/P2/P3 BLT) -- each real nt scored exactly once.
  - canonical_path_BPN: cumulative token NLL over the canonical token path
                        divided by raw canonical nt count (F2/F3 BPE/Unigram,
                        F6/F7 non-overlap).

Flat backbones (F1-F7) are scored with a SINGLE forward per sequence (batched
log-prob extraction), making the ~100K-sequence sealed test tractable. BLT arms
(P1/P2/P3) need an exact causal per-base conditional, so each position is
forwarded on its causal prefix (no future-nt leak within the open patch);
the checkpoint must carry the persisted entropy calibration for exact boundary
replay.

Reads the frozen split parquet, canonicalizes each sequence, and accumulates
ScoreSums via the shared scorer. GPU-only (adapter asserts CUDA); results are
written as an auditable JSON artifact keyed by (arm, seed, split).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq

from evaluator.scorer import ScoreSums, canonicalize_seq
from evaluator.internal_adapter import InternalFlatAdapter
from evaluator.blt_adapter import InternalBLTAdapter

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
# Flat arms that support exact next_base_BPN via full-prefix causal conditioning.
FLAT_EXACT_NEXT_BASE = {"F1"}


def _nats_to_bits(nats: float) -> float:
    return nats * math.log2(math.e)


def _sums_from_logprobs(lp: list[float], n_nt: int) -> ScoreSums:
    """ScoreSums from a precomputed per-position log-prob list (batched path)."""
    out = ScoreSums()
    nats = sum(lp)
    out.nll_nats_sum = -nats
    out.nll_bits_sum = _nats_to_bits(-nats)
    out.valid_nt_count = n_nt
    out.sequence_count = 1
    return out


def _seqs(split_path: str, split: str, n: int | None = None) -> list[str]:
    out = []
    pf = pq.ParquetFile(split_path)
    for batch in pf.iter_batches(batch_size=50_000, columns=["split_membership", "canonical_sequence"]):
        d = batch.to_pydict()
        for s, seq in zip(d["split_membership"], d["canonical_sequence"]):
            if s == split:
                out.append(seq)
                if n is not None and len(out) >= n:
                    return out
    return out


def _metric_for(adapter) -> tuple[str, str]:
    """Return (metric, scoring_mode). scoring_mode in
    {next_base_batched, next_base_blt, next_base_overlap, canonical_batched}."""
    arm_type = adapter.arm.tokenizer_type
    if adapter.arm.backbone == "blt":
        return "next_base_BPN", "next_base_blt"
    if arm_type == "overlap_mer":
        return "overlap_path_BPN", "next_base_overlap"
    if adapter.arm.id in FLAT_EXACT_NEXT_BASE:
        return "next_base_BPN", "next_base_batched"
    return "canonical_path_BPN", "canonical_batched"


def score_arm(adapter, seqs: list[str], split: str, max_len: int = 4096) -> dict:
    metric, mode = _metric_for(adapter)
    agg = ScoreSums()
    n_invalid = 0
    n_trunc = 0
    for seq in seqs:
        try:
            canon = canonicalize_seq(seq)
        except ValueError:
            agg.invalid_count += 1
            n_invalid += 1
            continue
        if len(canon) > max_len:
            canon = canon[:max_len]
            n_trunc += 1
        if mode == "next_base_batched":
            # F1 NUC: single forward -> all per-base conditionals.
            lp = adapter.all_log_probs_next_base(canon)
            sums = _sums_from_logprobs(lp, len(canon))
        elif mode == "next_base_overlap":
            # F4/F5 overlap: exact per-base via short-context k-mer model.
            sums, _ = _overlap_score(adapter, canon)
        elif mode == "next_base_blt":
            # P1/P2/P3: exact causal per-base, one forward per position.
            from evaluator.scorer import score_next_base
            sums = score_next_base(canon, adapter.log_prob_next_base)
        else:  # canonical_batched
            ids = adapter.encode(canon)
            lp = adapter.all_log_probs_token(ids)
            sums = _sums_from_logprobs(lp, len(canon))
        agg.add(sums)
    return {
        "arm": adapter.arm.id,
        "split": split,
        "metric": metric,
        "n_sequences": len(seqs),
        "n_invalid": n_invalid,
        "n_truncated": n_trunc,
        "nll_nats_sum": agg.nll_nats_sum,
        "nll_bits_sum": agg.nll_bits_sum,
        "valid_nt_count": agg.valid_nt_count,
        "bpn": agg.nll_bits_sum / max(1, agg.valid_nt_count),
        "cpu_fallback_count": adapter.guard.cpu_fallback_count,
    }


def _overlap_score(adapter, canon: str):
    """Overlap k-mer next_base scoring (exact, per-position short context)."""
    from evaluator.scorer import score_overlap_path
    return score_overlap_path(canon, adapter.k, adapter.log_prob_next_base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    device = "cuda:%d" % args.device
    from model import train_config as tc
    cfg = tc.resolved_config(args.arm, args.seed)
    if cfg.arm.backbone == "blt":
        adapter = InternalBLTAdapter(args.arm, args.seed, args.ckpt, device=device)
    else:
        adapter = InternalFlatAdapter(args.arm, args.seed, args.ckpt, device=device)
    seqs = _seqs(SPLIT_8080, args.split, args.n)
    result = score_arm(adapter, seqs, args.split)
    result["run_id"] = adapter.cfg.run_id
    result["ckpt"] = args.ckpt
    result["device"] = device
    assert result["cpu_fallback_count"] == 0, "cpu fallback in scoring %s" % args.arm
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("%s %s %s metric=%s bpn=%.4f n=%d invalid=%d trunc=%d fallback=%d" % (
        result["arm"], args.seed, result["split"], result["metric"], result["bpn"],
        result["n_sequences"], result["n_invalid"], result["n_truncated"],
        result["cpu_fallback_count"]))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
