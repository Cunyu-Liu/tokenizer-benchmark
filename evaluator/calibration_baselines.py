"""Compression calibration baselines (contract 3.6).

Non-neural reference codecs used to interpret the actual scale of '1% BPN'
differences. They are NOT tokenizer causal attributions and do not enter the
canonical_code_length_BPN headline, only the calibration table next to it.

  - uniform: theoretical reference 2 bits/nt (-log2(1/4)); EOS reported
    separately by the caller (not folded into the 2.0 reference).
  - order-k Markov (k=0..3): conditional probability tables fit on the train
    split with add-alpha smoothing; order/smoothing frozen before validation.
  - PPM (prediction by partial matching, order D, escape method C): classic
    context compressor with a frozen window/alphabet.

All CPU-safe and deterministic given train data + frozen parameters. The
train fit uses a frozen subsample budget (budget_nt) so large releases do not
require a full pass; the same budget is recorded with the artifact.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

ALPHABET = "ACGU"
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}
N_SYM = len(ALPHABET)


def canonicalize_seq(seq: str) -> str:
    return seq.upper().replace("T", "U")


def uniform_bits_per_nt() -> float:
    """Uniform A/C/G/U reference: -log2(1/4) == 2 bits/nt (exact)."""
    return 2.0


class MarkovModel:
    """Order-k Markov compressor with add-alpha smoothing (contract 3.6)."""

    def __init__(self, order: int = 0, alpha: float = 1.0):
        if not 0 <= order <= 3:
            raise ValueError("order must be in 0..3")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.order = order
        self.alpha = alpha
        self.counts: dict[str, list[int]] = {}

    def fit(self, sequences: Iterable[str], budget_nt: int | None = 2_000_000):
        seen = 0
        for raw in sequences:
            seq = canonicalize_seq(raw)
            if budget_nt is not None and seen >= budget_nt:
                break
            for i in range(self.order, len(seq)):
                ctx = seq[i - self.order:i]
                sym = BASE_TO_IDX.get(seq[i])
                if sym is None:
                    continue
                c = self.counts.setdefault(ctx, [0] * N_SYM)
                c[sym] += 1
                seen += 1
        return self

    def bits_per_nt(self, sequences: Iterable[str]) -> float:
        total_bits = 0.0
        n = 0
        for raw in sequences:
            seq = canonicalize_seq(raw)
            for i in range(self.order, len(seq)):
                sym = BASE_TO_IDX.get(seq[i])
                if sym is None:
                    continue
                ctx = seq[i - self.order:i]
                c = self.counts.get(ctx, [0] * N_SYM)
                total = sum(c) + self.alpha * N_SYM
                p = (c[sym] + self.alpha) / total
                total_bits += -math.log2(p)
                n += 1
        return total_bits / n if n else float("nan")


class PPMModel:
    """Prediction by Partial Matching, order D, escape method C (contract 3.6).

    Predictive -log2 P / nt (an ideal code length proxy). Each position is
    predicted from the longest matching context (order D .. 0); unseen symbols
    at an order trigger an escape (PPM-C: P(s)=n_s/(N+t), P(esc)=t/(N+t) with
    t = number of distinct symbols in that context). Order 0 is seeded with
    uniform fallback if a symbol was never observed (bounded by fit data).
    """

    def __init__(self, order: int = 5):
        if order < 1:
            raise ValueError("PPM order must be >= 1")
        self.order = order
        self.counts: dict[str, list[int]] = {}
        self._order0_total = 0

    def fit(self, sequences: Iterable[str], budget_nt: int | None = 2_000_000):
        seen = 0
        for raw in sequences:
            seq = canonicalize_seq(raw)
            if budget_nt is not None and seen >= budget_nt:
                break
            for i in range(len(seq)):
                sym = BASE_TO_IDX.get(seq[i])
                if sym is None:
                    continue
                # update contexts of every length 0..order ending just before i
                for d in range(0, self.order + 1):
                    if i - d < 0:
                        break
                    ctx = seq[i - d:i]
                    c = self.counts.setdefault(ctx, [0] * N_SYM)
                    c[sym] += 1
                seen += 1
        oc = self.counts.setdefault("", [0] * N_SYM)
        self._order0_total = sum(oc)
        return self

    def bits_per_nt(self, sequences: Iterable[str]) -> float:
        total_bits = 0.0
        n = 0
        for raw in sequences:
            seq = canonicalize_seq(raw)
            for i in range(len(seq)):
                sym = BASE_TO_IDX.get(seq[i])
                if sym is None:
                    continue
                p, ok = self._predict(seq, i, sym)
                if not ok:
                    p = 1.0 / N_SYM  # never-seen: uniform fallback
                total_bits += -math.log2(max(p, 1e-300))
                n += 1
        return total_bits / n if n else float("nan")

    def _predict(self, seq: str, i: int, sym: int) -> tuple[float, bool]:
        p = 1.0
        for d in range(self.order, -1, -1):
            ctx = seq[max(0, i - d):i]
            c = self.counts.get(ctx)
            if c is None:
                continue
            N = sum(c)
            if N == 0:
                continue
            t = sum(1 for x in c if x > 0)
            # PPM-C symbol and escape probabilities
            p_sym = c[sym] / (N + t)
            p_esc = t / (N + t)
            if c[sym] > 0:
                return p * p_sym, True
            p *= p_esc
            if d == 0:
                # order-0 did not contain the symbol
                return p, False
        return p, False


def markov_bpn(order: int, train: Iterable[str], holdout: Iterable[str],
               alpha: float = 1.0, budget_nt: int | None = 2_000_000) -> float:
    """Convenience: fit order-k Markov on train, score on holdout."""
    return MarkovModel(order=order, alpha=alpha).fit(train, budget_nt).bits_per_nt(holdout)


def ppm_bpn(order: int, train: Iterable[str], holdout: Iterable[str],
            budget_nt: int | None = 2_000_000) -> float:
    """Convenience: fit PPM on train, score on holdout."""
    return PPMModel(order=order).fit(train, budget_nt).bits_per_nt(holdout)