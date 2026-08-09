"""TokBench-RNA shared scorer.

Implements 3.5 unified likelihood accounting:
  - ScoreSums: only sums + counts, never a single average scalar.
  - next_base_BPN: exact per-nucleotide cross-entropy, each real nt counted once.
  - canonical_path_BPN: cumulative token NLL over the canonical token path,
    divided by raw canonical nt count (compression-path view, not marginal).
  - overlap_path_BPN: full-vocabulary overlap path score (each step adds one nt).

All methods are pure Python (CPU-safe); the neural side is supplied by an
adapter's log_prob_next_base / log_prob_token callbacks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

ALPHABET = tuple("ACGU")
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}


@dataclass
class ScoreSums:
    nll_nats_sum: float = 0.0
    nll_bits_sum: float = 0.0
    valid_nt_count: int = 0
    eos_nll_sum: float = 0.0
    sequence_count: int = 0
    invalid_count: int = 0
    truncation_count: int = 0

    def add(self, other: "ScoreSums") -> "ScoreSums":
        self.nll_nats_sum += other.nll_nats_sum
        self.nll_bits_sum += other.nll_bits_sum
        self.valid_nt_count += other.valid_nt_count
        self.eos_nll_sum += other.eos_nll_sum
        self.sequence_count += other.sequence_count
        self.invalid_count += other.invalid_count
        self.truncation_count += other.truncation_count
        return self

    def next_base_bpn(self) -> float:
        if self.valid_nt_count <= 0:
            return float("nan")
        return self.nll_bits_sum / self.valid_nt_count

    def to_dict(self) -> dict:
        return {
            "nll_nats_sum": self.nll_nats_sum,
            "nll_bits_sum": self.nll_bits_sum,
            "valid_nt_count": self.valid_nt_count,
            "eos_nll_sum": self.eos_nll_sum,
            "sequence_count": self.sequence_count,
            "invalid_count": self.invalid_count,
            "truncation_count": self.truncation_count,
            "next_base_bpn": self.next_base_bpn(),
        }


def _nats_to_bits(nats: float) -> float:
    return nats * math.log2(math.e)


def canonicalize_seq(seq: str) -> str:
    s = seq.upper().replace("T", "U")
    for ch in s:
        if ch not in BASE_TO_IDX:
            raise ValueError(f"non-primary IUPAC char {ch!r} in {seq!r}")
    return s


def score_next_base(
    seq: str,
    log_prob_next_base: Callable[[str, str], float],
) -> ScoreSums:
    try:
        canon = canonicalize_seq(seq)
    except ValueError:
        out = ScoreSums()
        out.invalid_count = 1
        out.sequence_count = 1
        return out
    nats_total = 0.0
    for i in range(len(canon)):
        prefix = canon[:i]
        nxt = canon[i]
        nats_total += log_prob_next_base(prefix, nxt)
    out = ScoreSums()
    out.nll_nats_sum = -nats_total
    out.nll_bits_sum = _nats_to_bits(-nats_total)
    out.valid_nt_count = len(canon)
    out.sequence_count = 1
    return out


def score_canonical_path(
    token_ids: Sequence[int],
    log_prob_token: Callable[[Sequence[int], int], float],
) -> ScoreSums:
    nats_total = 0.0
    for i, tid in enumerate(token_ids):
        ctx = token_ids[:i]
        nats_total += log_prob_token(ctx, tid)
    out = ScoreSums()
    out.nll_nats_sum = -nats_total
    out.nll_bits_sum = _nats_to_bits(-nats_total)
    out.sequence_count = 1
    return out


def score_overlap_path(
    seq: str,
    k: int,
    log_prob_kmer: Callable[[str, str], float],
) -> tuple[ScoreSums, float]:
    try:
        canon = canonicalize_seq(seq)
    except ValueError:
        out = ScoreSums()
        out.invalid_count = 1
        out.sequence_count = 1
        return out, 0.0
    nats_total = 0.0
    illegal_mass = 0.0
    for i in range(len(canon)):
        prefix = canon[max(0, i - k + 1):i]
        nxt = canon[i]
        nats_total += log_prob_kmer(prefix, nxt)
    out = ScoreSums()
    out.nll_nats_sum = -nats_total
    out.nll_bits_sum = _nats_to_bits(-nats_total)
    out.valid_nt_count = len(canon)
    out.sequence_count = 1
    return out, illegal_mass


def aggregate_bpn(list_of_sums: Sequence[ScoreSums]) -> ScoreSums:
    agg = ScoreSums()
    for s in list_of_sums:
        agg.add(s)
    return agg
