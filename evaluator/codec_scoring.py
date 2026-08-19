"""Canonical codec scoring + compression baselines (contract 3.6).

Bridges evaluator/codec.py (actual decodable bitstream) and
calibration_baselines.py (non-neural references) so the Phase 4 scorer can
report the V3 headline plus calibration next to it:

  - canonical_code_length_BPN : real 64-bit range-coded bitstream bits / nt
  - canonical_code_nll_BPN    : ideal -log2 p over the same canonical path
  - uniform / order-k Markov / PPM baselines (fit train, score holdout)

The headline is the ACTUAL code length, not an idealized NLL (contract 3.6:
"actual canonical codec length", not unconditional compression ratio).
"""
from __future__ import annotations

import math

from .codec import RangeEncoder64, quantize_cdf, cdf_symbol_bits, CDF_TOTAL
from .calibration_baselines import (
    uniform_bits_per_nt,
    MarkovModel,
    PPMModel,
    canonicalize_seq,
)


def _probs_from_logprobs(logps: list[float]) -> list[float]:
    """Numerically-stable normalized probs from natural-log softmax."""
    m = max(logps)
    w = [math.exp(lp - m) for lp in logps]
    s = sum(w)
    return [x / s for x in w]


def _encode_with_cdf(enc, probs, tid, total) -> tuple[float, float]:
    """Encode one symbol and return (canonical_nll_bits, quantized_nll_bits)."""
    cdf = quantize_cdf(probs, total)
    enc.encode(cdf, tid, total)
    p = probs[tid]
    canon_bits = -math.log2(p) if p > 0 else float("inf")
    return canon_bits, cdf_symbol_bits(cdf, tid, total)


def canonical_codec_score(adapter, sequences) -> dict:
    """Actual canonical code length over a flat adapter's token paths.

    For each sequence we take the frozen canonical token path, get the model's
    full-vocabulary log-probs in one forward, quantize to a 2**24 integer CDF
    (min freq 1, largest-remainder), and encode through the frozen 64-bit range
    coder. Returns the actual coded bits and the NLL diagnostic plus the 64-bit
    consistency gate.
    """
    enc = RangeEncoder64()
    total = CDF_TOTAL
    canonical_nll_bits = 0.0
    quantized_nll_bits = 0.0
    valid_nt = 0
    n_sequences = 0
    for seq in sequences:
        canon = canonicalize_seq(seq)
        ids = adapter.encode(canon)
        full_lp = adapter.all_log_probs_full(ids)
        for i, tid in enumerate(ids):
            probs = _probs_from_logprobs(full_lp[i])
            cb, qb = _encode_with_cdf(enc, probs, tid, total)
            canonical_nll_bits += cb
            quantized_nll_bits += qb
        valid_nt += len(canon)
        n_sequences += 1
    bitstream = enc.finish()
    coded_bits = len(bitstream) * 8
    return {
        "coded_bits": coded_bits,
        "quantized_cdf_nll_bits": quantized_nll_bits,
        "canonical_nll_bits": canonical_nll_bits,
        "valid_nt": valid_nt,
        "n_sequences": n_sequences,
        "canonical_code_length_BPN": coded_bits / valid_nt if valid_nt else float("nan"),
        "canonical_code_nll_BPN": canonical_nll_bits / valid_nt if valid_nt else float("nan"),
        "coder_consistency_ok": abs(coded_bits - quantized_nll_bits) <= 64,
    }


def canonical_codec_score_blt(adapter, sequences) -> dict:
    """Actual canonical code length for a BLT arm (contract 3.6).

    BLT cannot use a single full-sequence forward (the open-patch embedding
    would leak future nt). Instead each position i is scored with the model's
    full 4-way causal distribution p(x_i | x_<i) from `log_probs_next_base`,
    using the same deterministic patch boundaries as training. The per-nt
    canonical path is encoded through the frozen 64-bit range coder.
    """
    from .codec import RangeEncoder64 as _Enc
    from .continuation_code_length import BASE_TO_IDX as _BTI

    enc = _Enc()
    total = CDF_TOTAL
    canonical_nll_bits = 0.0
    quantized_nll_bits = 0.0
    valid_nt = 0
    n_sequences = 0
    for seq in sequences:
        canon = canonicalize_seq(seq)
        for i, nxt in enumerate(canon):
            lp = adapter.log_probs_next_base(canon[:i])
            probs = _probs_from_logprobs(lp)
            cb, qb = _encode_with_cdf(enc, probs, _BTI[nxt], total)
            canonical_nll_bits += cb
            quantized_nll_bits += qb
        valid_nt += len(canon)
        n_sequences += 1
    bitstream = enc.finish()
    coded_bits = len(bitstream) * 8
    return {
        "coded_bits": coded_bits,
        "quantized_cdf_nll_bits": quantized_nll_bits,
        "canonical_nll_bits": canonical_nll_bits,
        "valid_nt": valid_nt,
        "n_sequences": n_sequences,
        "canonical_code_length_BPN": coded_bits / valid_nt if valid_nt else float("nan"),
        "canonical_code_nll_BPN": canonical_nll_bits / valid_nt if valid_nt else float("nan"),
        "coder_consistency_ok": abs(coded_bits - quantized_nll_bits) <= 64,
    }


def calibration_bpns(train_seqs, holdout_seqs, markov_order: int = 3) -> dict:
    """Non-neural calibration references (contract 3.6).

    uniform = 2 bits/nt (theoretical); order-k Markov and PPM are fit on train
    and scored on holdout. These anchor the actual scale of a 1% BPN difference.
    """
    train = [canonicalize_seq(s) for s in train_seqs]
    hold = [canonicalize_seq(s) for s in holdout_seqs]
    return {
        "uniform_BPN": uniform_bits_per_nt(),
        "markov_order%d_BPN" % markov_order:
            MarkovModel(order=markov_order).fit(train, budget_nt=None).bits_per_nt(hold),
        "ppm_BPN": PPMModel(order=5).fit(train, budget_nt=None).bits_per_nt(hold),
    }