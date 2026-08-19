"""True-suffix continuation code length (contract 3.7 required secondary).

Main Evidence B: for the same raw prefix, the true suffix's actual code length
is the required main-text secondary (alongside the headline canonical codec).

Contract 3.7 rules implemented here:
  - prefix cut points are raw-nucleotide based: 6 * floor((ratio * len) / 6),
    making every model see the exact same raw prefix; only samples with >= 1 nt
    of suffix are kept.
  - the tokenizer is applied to the observed prefix and the hidden suffix
    INDEPENDENTLY (no token may cross the prefix/suffix boundary).
  - the suffix is scored as a conditional code length: probs_fn receives the
    prefix tokens followed by the suffix prefix, so the true suffix is not
    "seen" during prediction (autoregressive).

The denominator is the suffix's canonical nucleotide count (suffix nt), so this
is a per-suffix-position bits-per-nt. EOS is NOT folded in (reported separately
elsewhere).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


def canonicalize_seq(seq: str) -> str:
    s = seq.upper().replace("T", "U")
    for ch in s:
        if ch not in BASE_TO_IDX:
            raise ValueError(f"non-primary IUPAC char {ch!r}")
    return s


def prefix_cut_points(seq_len: int, ratios: Sequence[float] = (0.10, 0.25, 0.50)
                      ) -> dict[float, int]:
    """Raw-nt prefix cut points, clamped to multiples of 6 (contract 3.7).

    Returns {ratio: prefix_nt} for ratios that leave at least 1 nt of suffix.
    """
    cuts = {}
    for r in ratios:
        p = 6 * (int(r * seq_len) // 6)
        if 0 < p < seq_len:
            cuts[r] = p
    return cuts


def suffix_code_length_bpn(
    canonical_seq: str,
    prefix_nt: int,
    tokenizer,
    probs_fn: Callable[[Sequence[int]], Sequence[float]],
) -> float:
    """Conditional code length of the true suffix given the observed prefix.

    tokenizer.encode is applied to prefix and suffix independently (no cross
    token). probs_fn maps a token-id context (prefix tokens + suffix prefix) to
    a full-vocabulary probability vector. Returns bits per suffix nt.
    """
    seq = canonicalize_seq(canonical_seq)
    if not 0 < prefix_nt < len(seq):
        return float("nan")
    prefix = seq[:prefix_nt]
    suffix = seq[prefix_nt:]
    prefix_tokens = tokenizer.encode(prefix)
    suffix_tokens = tokenizer.encode(suffix)
    suffix_nt = len(suffix)
    if suffix_nt == 0:
        return float("nan")

    total_bits = 0.0
    ctx = list(prefix_tokens)
    for tid in suffix_tokens:
        probs = probs_fn(ctx)
        p = probs[tid] if 0 <= tid < len(probs) else 0.0
        if p <= 0:
            total_bits += float("inf")
        else:
            total_bits += -math.log2(p)
        ctx.append(tid)
    return total_bits / suffix_nt


def continuation_bpn_grid(
    canonical_seq: str,
    ratio_to_nt: dict[float, int],
    tokenizer,
    probs_fn: Callable[[Sequence[int]], Sequence[float]],
) -> dict[float, float]:
    """Return {ratio: suffix_code_length_bpn} over a frozen cut-point map."""
    return {r: suffix_code_length_bpn(canonical_seq, p, tokenizer, probs_fn)
            for r, p in ratio_to_nt.items()}