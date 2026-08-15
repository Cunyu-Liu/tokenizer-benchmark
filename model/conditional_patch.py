"""P2 supported-strata conditional random patch (contract 3.2).

Implements the matched-random control for P3 (causal entropy patch) that the
contract requires:

  - `q(boundary | causal prefix-length stratum, causal-entropy stratum)` is
    fitted on train-only data using P3's causal boundary labels.
  - Runtime uses ONLY the current position, the observed prefix, and causal
    entropy (no final sequence length, no future bases).
  - Supported strata (0.05 <= q <= 0.95) sample the boundary with probability q
    (deterministic per (sequence_id, position, patch_randomization_seed) to
    satisfy prefix causality across seeds).
  - Non-supported strata (q < 0.05 or q > 0.95) exactly replay P3's
    deterministic causal boundary (no approximate randomization).
  - A coverage report verifies supported strata cover >= 80% of training
    positions and >= 80% of P3 boundaries (contract 3.2/5.4).

The random draw at position i is a function of
`sequence_or_generation_id`, `i`, and `patch_randomization_seed=20260815` so
the three model seeds share the same frozen rule.

GPU efficiency: the entropy predictor is a causal GRU, so one forward over the
whole sequence yields per-position causal entropy for every position at once.
We expose both the offline per-string `boundary()` and the batched GPU
`boundaries_batch()` interface used by the training loop.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

PATCH_RANDOMIZATION_SEED = 20260815  # contract 3.2
SUPPORT_LO = 0.05
SUPPORT_HI = 0.95
MIN_SUPPORT_COVERAGE = 0.80  # contract 3.2/5.4: >=80% positions + boundaries

# Number of bins for each stratum dimension (frozen preregistration).
N_PREFIX_LEN_BINS = 8     # log-spaced prefix-length bins
N_ENTROPY_BINS = 8        # quantile-spaced causal-entropy bins

_ALPH = {"A": 0, "C": 1, "G": 2, "U": 3}


def _seq_id(canon: str) -> int:
    """Stable 32-bit sequence id (same rule as model/dataset._seq_id)."""
    return zlib.crc32(canon.encode("utf-8"))


@dataclass
class ConditionalRandomPatchPolicy:
    """PatchPolicy-compatible provider for P2 (contract 3.2)."""

    kind: str = "conditional_random"
    seed: int = PATCH_RANDOMIZATION_SEED
    q_table: list[list[float]] | None = None
    n_prefix_bins: int = N_PREFIX_LEN_BINS
    n_entropy_bins: int = N_ENTROPY_BINS
    entropy_predictor = None          # frozen train-only GRU predictor
    device: str = "cuda:0"
    ent_edges: list[float] | None = None
    prefix_edges: list[int] | None = None
    p3_boundary: Callable[[str, int], list[int]] | None = None  # exact P3 replay
    coverage_report: dict | None = None

    # -- binning ------------------------------------------------------------
    def _prefix_bin(self, i: int) -> int:
        edges = self.prefix_edges
        assert edges is not None, "fit_q() must be called first"
        # clamp i into [edges[0], edges[-1])
        i = max(edges[0], min(i, edges[-1] - 1))
        for b in range(len(edges) - 1):
            if edges[b] <= i < edges[b + 1]:
                return b
        return len(edges) - 2

    def _entropy_bin(self, ent: float) -> int:
        edges = self.ent_edges
        assert edges is not None, "fit_q() must be called first"
        # clamp ent into [edges[0], edges[-1])
        ent = max(edges[0], min(ent, edges[-1] - 1e-12))
        for b in range(len(edges) - 1):
            if edges[b] <= ent < edges[b + 1]:
                return b
        return len(edges) - 2

    # -- batched causal entropy (one GRU forward per sequence) ---------------
    def _batch_entropy(self, canon: str) -> np.ndarray:
        """Per-position causal entropy of a canonical sequence (one forward)."""
        import torch
        if not canon:
            return np.zeros(0, dtype=np.float32)
        ids = torch.tensor([[_ALPH[ch] for ch in canon]],
                           dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.entropy_predictor.eval()
            ent = self.entropy_predictor.entropy(ids)[0].cpu().numpy()
        return ent.astype(np.float64)

    # -- fitting ------------------------------------------------------------
    def fit_q(self, seqs: Sequence[str], p3_boundaries: Sequence[Sequence[int]],
              entropy_predictor=None, device: str | None = None) -> dict:
        """Fit q(boundary | prefix-len bin, entropy bin) on train-only data.

        `seqs` are canonical train sequences; `p3_boundaries` is the per-sequence
        deterministic P3 causal boundary (0/1 per position). Returns the
        coverage report (supported-strata position/boundary coverage).
        """
        if entropy_predictor is not None:
            self.entropy_predictor = entropy_predictor
        if device is not None:
            self.device = device
        if self.entropy_predictor is None:
            raise RuntimeError("entropy_predictor required for fit_q")

        prefix_lens: list[int] = []
        ents: list[float] = []
        bounds: list[int] = []
        for seq, bnd in zip(seqs, p3_boundaries):
            c = seq.upper().replace("T", "U")
            n = len(c)
            if n == 0:
                continue
            ent = self._batch_entropy(c)
            for i in range(n):
                prefix_lens.append(i)
                ents.append(float(ent[i]))
                bounds.append(int(bnd[i]) if i < len(bnd) else 0)

        if not prefix_lens:
            raise ValueError("no positions collected for q fit")

        max_len = max(prefix_lens) + 1
        # log-spaced prefix-length edges covering [0, max_len]
        if max_len <= N_PREFIX_LEN_BINS:
            self.prefix_edges = list(range(0, max_len + 1))
        else:
            log_edges = np.unique(np.logspace(
                0, np.log10(max(2, max_len)), N_PREFIX_LEN_BINS + 1).astype(int))
            edges = sorted({0} | {int(x) for x in log_edges if 0 < x <= max_len})
            self.prefix_edges = edges
            if self.prefix_edges[-1] < max_len:
                self.prefix_edges.append(max_len)
        # quantile-spaced entropy edges
        ent_arr = np.asarray(ents)
        qs = np.linspace(0, 100, N_ENTROPY_BINS + 1)
        self.ent_edges = [float(np.percentile(ent_arr, q)) for q in qs]
        self.ent_edges[-1] += 1e-9

        # Tally (boundary_count, position_count) per stratum.
        nb, ne = self.n_prefix_bins, self.n_entropy_bins
        counts = np.zeros((nb, ne), dtype=np.int64)
        pos = np.zeros((nb, ne), dtype=np.int64)
        for pl, e, b in zip(prefix_lens, ents, bounds):
            bp = self._prefix_bin(pl)
            be = self._entropy_bin(e)
            pos[bp, be] += 1
            counts[bp, be] += b

        self.q_table = np.divide(counts, np.maximum(1, pos)).tolist()

        # Coverage: supported strata must cover >= 80% positions + boundaries.
        qt = np.asarray(self.q_table)
        supported = (qt >= SUPPORT_LO) & (qt <= SUPPORT_HI)
        supported_pos = int(np.sum(pos * supported))
        supported_bound = int(np.sum(counts * supported))
        total_pos = int(np.sum(pos))
        total_bound = int(np.sum(counts))
        self.coverage_report = {
            "supported_strata": supported.tolist(),
            "supported_positions": supported_pos,
            "total_positions": total_pos,
            "position_coverage": supported_pos / max(1, total_pos),
            "supported_p3_boundaries": supported_bound,
            "total_p3_boundaries": total_bound,
            "boundary_coverage": supported_bound / max(1, total_bound),
            "passes_support_coverage": (
                supported_pos / max(1, total_pos) >= MIN_SUPPORT_COVERAGE and
                supported_bound / max(1, total_bound) >= MIN_SUPPORT_COVERAGE),
        }
        return dict(self.coverage_report)

    # -- runtime ------------------------------------------------------------
    def _random(self, seq_id: int, i: int) -> float:
        """Deterministic per-(seq_id, position) random draw in [0,1)."""
        return random.Random(
            (self.seed << 32) ^ ((seq_id << 16) ^ i)).random()

    def _boundaries_from_entropy(self, canon: str, ent: np.ndarray,
                                 seq_id: int, p3_bnd: Sequence[int] | None
                                 ) -> list[int]:
        """Boundary indicators from precomputed causal entropy (per position).

        Runtime only reads current position i, observed prefix (implied by i)
        and causal entropy ent[i]; never final length or future bases.
        """
        n = len(canon)
        out = [0] * n
        if n == 0:
            return out
        out[0] = 1
        for i in range(1, n):
            bp = self._prefix_bin(i)
            be = self._entropy_bin(float(ent[i]))
            q = self.q_table[bp][be]
            if SUPPORT_LO <= q <= SUPPORT_HI:
                out[i] = 1 if self._random(seq_id, i) < q else 0
            else:
                # non-supported: exact replay of P3's deterministic boundary
                if p3_bnd is not None and i < len(p3_bnd):
                    out[i] = int(p3_bnd[i])
                else:
                    out[i] = 0
        return out

    def boundary(self, seq: str, length: int, seq_id: int = 0) -> list[int]:
        """Offline per-string boundary (dataset provider interface)."""
        if self.q_table is None:
            raise RuntimeError("fit_q() must be called before boundary()")
        canon = seq.upper().replace("T", "U")[:length]
        sid = seq_id if seq_id else _seq_id(canon)
        ent = self._batch_entropy(canon)
        p3_bnd = self.p3_boundary(canon, length) if self.p3_boundary else None
        return self._boundaries_from_entropy(canon, ent, sid, p3_bnd)

    def boundaries_batch(self, nt_ids) -> "torch.Tensor":
        """Batched GPU interface used by the training loop.

        `nt_ids` is an (B, T) LongTensor of nt ids (0..3). Per-sequence entropy
        comes from one GRU forward; boundaries use only causal prefix + entropy.
        """
        import torch
        if self.q_table is None:
            raise RuntimeError("fit_q() must be called before boundaries_batch()")
        self.entropy_predictor.eval()
        with torch.no_grad():
            ent = self.entropy_predictor.entropy(nt_ids)  # (B, T)
        B, T = ent.shape
        out = torch.zeros_like(ent, dtype=torch.float32)
        ent_np = ent.cpu().numpy()
        ids_np = nt_ids.cpu().numpy()
        for b in range(B):
            n = int((ids_np[b] >= 0).sum())
            if n <= 0:
                continue
            canon = "".join(_ALPH_ID_TO_BASE if False else _base_from_id(ids_np[b][i])
                            for i in range(n))
            sid = _seq_id(canon)
            # P3 boundary for exact replay in non-supported strata
            p3_bnd = self.p3_boundary(canon, n) if self.p3_boundary else None
            bnd = self._boundaries_from_entropy(canon, ent_np[b][:n], sid, p3_bnd)
            for i, v in enumerate(bnd):
                out[b, i] = v
        return out


_ALPH_ID_TO_BASE = ["A", "C", "G", "U"]


def _base_from_id(i: int) -> str:
    return _ALPH_ID_TO_BASE[int(i)]


def build_conditional_random_policy(calib, device: str = "cuda:0"):
    """Build a ConditionalRandomPatchPolicy from an EntropyCalib.

    The calib holds the frozen train-only entropy predictor used both for the
    q-fit (via p3_boundaries) and for runtime causal-entropy binning. P3's
    exact boundary function is wired for non-supported-strata replay.
    """
    from .entropy_predictor import EntropyPatchPolicy
    p3 = EntropyPatchPolicy(calib.predictor, calib.gate, device=device)
    pol = ConditionalRandomPatchPolicy(
        seed=PATCH_RANDOMIZATION_SEED,
        entropy_predictor=calib.predictor,
        device=device,
        p3_boundary=lambda s, l: p3.boundary(s, l))
    return pol, p3


def fit_p2_q(calib, data_path: str, device: str = "cuda:0",
             n_seqs: int = 2000, seed: int = 17) -> dict:
    """Fit P2's q(boundary | prefix-len, entropy) on train-only data.

    Computes P3's deterministic causal boundaries on a train-only sample, then
    fits q from (prefix-length, causal-entropy) strata. Returns the coverage
    report (contract 3.2/5.4: supported strata >= 80% positions + boundaries).
    """
    from .dataset import sample_train_sequences
    from .entropy_predictor import EntropyPatchPolicy

    p3 = EntropyPatchPolicy(calib.predictor, calib.gate, device=device)
    seqs = sample_train_sequences(data_path, n=n_seqs, seed=seed, split="train")
    seqs = [s[:4096] for s in seqs]
    # P3 deterministic causal boundaries for the sample
    p3_bnds = []
    for s in seqs:
        b = p3.boundary(s, len(s))
        p3_bnds.append(b)

    pol = ConditionalRandomPatchPolicy(
        seed=PATCH_RANDOMIZATION_SEED,
        entropy_predictor=calib.predictor,
        device=device,
        p3_boundary=lambda s, l: p3.boundary(s, l))
    report = pol.fit_q(seqs, p3_bnds, device=device)
    return pol, report
