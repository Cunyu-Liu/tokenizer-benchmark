"""Phase 4 main-table canonical codec scorer (contract 3.6).

Computes the V3 headline `canonical_code_length_BPN` -- a REAL decodable
64-bit range-coded bitstream over the frozen canonical token path -- plus the
ideal-NLL diagnostics and the non-neural calibration baselines, for one trained
checkpoint on the primary cluster-held-out test (or a deterministic subsample).

Encode/decode discipline (byte-identical recovery):
  The batched single-forward path cannot be paired with an independent decoder
  (fp32 batched vs per-prefix differ by ~1 ULP -> CDF flips), so BOTH encode and
  decode use the same per-prefix fp32 forward (`log_probs_token(ctx, False)` for
  flat arms, `log_probs_next_base` for BLT arms): identical CDFs by
  construction. Each (model, seed, split) writes ONE continuous coder stream.

The codec is per-position, so the FULL ~100K-seq primary test is heavy on GPU;
use --n for fixtures / validation and the homology-stratified subsample
(p4_subsample.py) for the final table when budget requires.

GPU-only (adapter asserts CUDA); results written as an auditable JSON artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq

from evaluator.codec_scoring import codec_roundtrip, calibration_bpns

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
TRAIN_VIEW = "train"


def _load_seqs(split_path: str, split: str, n: int | None = None) -> list[str]:
    out = []
    pf = pq.ParquetFile(split_path)
    for batch in pf.iter_batches(batch_size=50_000,
                                 columns=["split_membership", "canonical_sequence"]):
        d = batch.to_pydict()
        for s, seq in zip(d["split_membership"], d["canonical_sequence"]):
            if s == split:
                out.append(seq)
                if n is not None and len(out) >= n:
                    return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None,
                    help="subsample size for the codec stream (None = full split)")
    ap.add_argument("--calib-n", type=int, default=2000,
                    help="train sequences used to fit Markov/PPM baselines")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda:%d" % args.device
    from model import train_config as tc
    cfg = tc.resolved_config(args.arm, args.seed)
    if cfg.arm.backbone == "blt":
        from evaluator.blt_adapter import InternalBLTAdapter
        adapter = InternalBLTAdapter(args.arm, args.seed, args.ckpt, device=device)
    else:
        from evaluator.internal_adapter import InternalFlatAdapter
        adapter = InternalFlatAdapter(args.arm, args.seed, args.ckpt, device=device)

    seqs = _load_seqs(SPLIT_8080, args.split, args.n)
    if not seqs:
        raise SystemExit("no sequences for split=%s" % args.split)

    result = codec_roundtrip(adapter, seqs)
    result["arm"] = args.arm
    result["seed"] = args.seed
    result["split"] = args.split
    result["n_sequences_loaded"] = len(seqs)
    result["run_id"] = adapter.cfg.run_id
    result["ckpt"] = args.ckpt
    result["device"] = device

    # calibration baselines: fit on a train subsample, score the same codec seqs
    if args.calib_n > 0:
        train_seqs = _load_seqs(SPLIT_8080, TRAIN_VIEW, args.calib_n)
        result["calibration"] = calibration_bpns(train_seqs, seqs, markov_order=3)

    assert getattr(adapter, "guard", None) is None or adapter.guard.cpu_fallback_count == 0
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("%s s%d split=%s BPN=%.4f nll_BPN=%.4f decode_ok=%s "
          "nll_gate=%s bits_gate_64=%s overhead=%s bits | n=%d nt=%d" % (
              args.arm, args.seed, args.split,
              result.get("canonical_code_length_BPN"),
              result.get("canonical_code_nll_BPN"),
              result.get("decoded_byte_identical"),
              result.get("nll_gate_ok_1e4"),
              result.get("bits_gate_ok_64"),
              result.get("coder_overhead_bits"),
              result.get("n_sequences"), result.get("valid_nt")))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
