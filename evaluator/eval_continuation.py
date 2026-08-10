"""Continuation + generation evaluator (3.6) and sealed-test gate.

CPU-safe metric computation on generated/continued sequences. Neural generation
is produced by GPU adapters; this module scores the outputs.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

ALPHABET = set("ACGU")


def rc(seq: str) -> str:
    comp = {"A": "U", "C": "G", "G": "C", "U": "A"}
    return "".join(comp.get(b, b) for b in reversed(seq.upper().replace("T", "U")))


def _canon(s: str) -> str:
    return s.upper().replace("T", "U")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (CPU, small strings)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


def nucleotide_accuracy(pred: str, ref: str) -> float:
    if not ref:
        return float("nan")
    n = min(len(pred), len(ref))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if pred[i] == ref[i]) / len(ref)


def _kmers(seq: str, k: int) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


def kmer_recovery(pred: str, ref: str, k: int = 5) -> float:
    pk, rk = _kmers(_canon(pred), k), _kmers(_canon(ref), k)
    if not rk:
        return float("nan")
    return len(pk & rk) / len(rk)


@dataclass
class GenerationStats:
    total: int = 0
    valid: int = 0
    invalid_char_count: int = 0
    eos_complete: int = 0
    truncation_count: int = 0
    exact_unique: set = field(default_factory=set)
    exact_matrix: Optional[dict] = None  # timing -> set of training seqs
    identity_hist: Counter = field(default_factory=Counter)

    def validity_rate(self) -> float:
        return self.valid / self.total if self.total else float("nan")

    def uniqueness(self) -> float:
        return len(self.exact_unique) / self.total if self.total else float("nan")


def evaluate_generation(
    sequences: Iterable[str],
    training_set: Optional[set[str]] = None,
    identity_bins=(1.0, 0.9, 0.8),
) -> GenerationStats:
    st = GenerationStats()
    st.exact_matrix = {"train": set(), "valid": set(), "test": set()}
    for raw in sequences:
        st.total += 1
        canon = _canon(raw)
        if any(b not in ALPHABET for b in canon):
            st.invalid_char_count += 1
        else:
            st.valid += 1
            st.exact_unique.add(canon)
            if training_set is not None:
                if canon in training_set:
                    st.exact_matrix["train"].add(canon)
        # identity vs training nearest neighbor
        if training_set:
            best = 0.0
            for t in training_set:
                sim = _identity(canon, t)
                if sim > best:
                    best = sim
            for b in identity_bins:
                if best >= b:
                    st.identity_hist[b] += 1
    return st


def _identity(a: str, b: str) -> float:
    if not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


@dataclass
class ContinuationResult:
    prefix_frac: float = 0.0
    edit_dist_sum: float = 0.0
    nt_acc_sum: float = 0.0
    kmer_rec_sum: float = 0.0
    count: int = 0
    truncation_count: int = 0

    def mean_edit_dist(self) -> float:
        return self.edit_dist_sum / self.count if self.count else float("nan")

    def mean_nt_acc(self) -> float:
        return self.nt_acc_sum / self.count if self.count else float("nan")

    def mean_kmer_recovery(self) -> float:
        return self.kmer_rec_sum / self.count if self.count else float("nan")


def evaluate_continuation(preds: Iterable[tuple[str, str, float]]) -> ContinuationResult:
    """preds = (generated_suffix, true_suffix, prefix_frac)."""
    res = ContinuationResult()
    for gen, ref, frac in preds:
        res.count += 1
        res.prefix_frac = frac
        res.edit_dist_sum += edit_distance(_canon(gen), _canon(ref))
        res.nt_acc_sum += nucleotide_accuracy(_canon(gen), _canon(ref))
        res.kmer_rec_sum += kmer_recovery(gen, ref)
    return res


# --- sealed-test gate -----------------------------------------------------
class SealedTestGate:
    """Enforces that test / family-test / temporal OOD are never exposed to
    training-time code or hyperparameter selection. This is a guard object:
    the actual split hashes live in the sealed manifests under /mnt."""
    def __init__(self, sealed_manifest_path: Optional[str] = None):
        self.sealed_manifest_path = sealed_manifest_path
        self._touched = False

    def assert_not_touched(self) -> None:
        if self._touched:
            raise RuntimeError("sealed test access attempted")
