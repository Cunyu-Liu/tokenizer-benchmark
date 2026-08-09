"""Phase 4 sealed-test main-table scorer (contract 3.5, 5.4).

Loads a trained Phase 4 checkpoint (flat backbones F1-F7) and computes the
likelihood metrics over a held-out split, one batched forward per sequence:

  - next_base_BPN     : exact per-base cross-entropy (F1 NUC, F4/F5 overlap)
  - canonical_path_BPN: cumulative token NLL over the canonical token path
                        divided by raw canonical nt count (F2/F3 BPE/Unigram,
                        F6/F7 non-overlap)

Reads the frozen split parquet, canonicalizes each sequence, encodes, and
accumulates ScoreSums via the shared scorer. GPU-only (adapter asserts CUDA);
results are written as an auditable JSON artifact keyed by (arm, seed, split).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq

from evaluator import scorer as S
from evaluator.internal_adapter import InternalFlatAdapter

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
# Arms that support exact next_base_BPN via full-prefix causal conditioning.
EXACT_NEXT_BASE = {"F1"}


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


def score_arm(adapter: InternalFlatAdapter, seqs: list[str], split: str,
              max_len: int = 4096) -> dict:
    arm_type = adapter.arm.tokenizer_type
    exact = adapter.arm.id in EXACT_NEXT_BASE
    if arm_type == "overlap_mer":
        metric = "overlap_path_BPN"
    elif exact:
        metric = "next_base_BPN"
    else:
        metric = "canonical_path_BPN"
    agg = S.ScoreSums()
    n_invalid = 0
    n_trunc = 0
    for seq in seqs:
        try:
            canon = adapter.canonicalize(seq)
        except ValueError:
            agg.invalid_count += 1
            n_invalid += 1
            continue
        if len(canon) > max_len:
            canon = canon[:max_len]
            n_trunc += 1
        if arm_type == "overlap_mer":
            sums, _ = S.score_overlap_path(canon, adapter.k, adapter.log_prob_next_base)
        elif exact:
            # one forward per sequence: encode full nt -> per-base next-base NLL
            sums = S.score_next_base(canon, adapter.log_prob_next_base)
        else:
            ids = adapter.encode(canon)
            # count raw canonical nt (compression-path view) excluding a BOS if any
            n_nt = len(canon)
            sums = S.score_canonical_path(ids, adapter.log_prob_token)
            sums.valid_nt_count = n_nt
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
