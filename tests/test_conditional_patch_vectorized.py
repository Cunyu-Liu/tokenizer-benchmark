"""Unit fixtures for the vectorized ConditionalRandomPatchPolicy.boundaries_batch.

Validates that the vectorized batched boundary computation:
  - matches an independent scalar oracle implementing the frozen P2 rule exactly
    (supported-strata random draw, non-supported P3 replay, pos0=1, pad=0),
  - is deterministic,
  - produces q-matched boundary rates in supported strata.

Requires CUDA (the training path runs boundaries_batch on GPU).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from model.conditional_patch import (
    ConditionalRandomPatchPolicy, _seq_id, PATCH_RANDOMIZATION_SEED,
    SUPPORT_LO, SUPPORT_HI,
)
from model.entropy_predictor import boundaries_from_entropy

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MAP = {"A": 0, "C": 1, "G": 2, "U": 3}


class _StubEnt:
    def eval(self):
        return None

    def entropy(self, ids):
        B, T = ids.shape
        base = (torch.arange(T, dtype=torch.float32, device=ids.device) % 8) / 4.0
        return base.expand(B, T).clone()


class _P3Stub:
    gate = 0.5


def _build_policy(device):
    pol = ConditionalRandomPatchPolicy(
        seed=PATCH_RANDOMIZATION_SEED, entropy_predictor=_StubEnt(),
        device=device, p3_boundary=None)
    pol.p3_policy = _P3Stub()
    pol.prefix_edges = [0, 10, 100, 1000, 4096]
    pol.ent_edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0000001]
    pol.n_prefix_bins = 4
    pol.n_entropy_bins = 8
    pol.q_table = [[1.0] * 8, [0.3] * 8, [0.3] * 8, [0.3] * 8]  # bp0 non-supported
    return pol


def _oracle(pol, ids_np, ent_t):
    pe = np.asarray(pol.prefix_edges, dtype=np.int64)
    ee = np.asarray(pol.ent_edges, dtype=np.float64)
    qt = np.asarray(pol.q_table, dtype=np.float64)
    ent_np = ent_t.detach().cpu().numpy()
    B, T = ids_np.shape
    out = np.zeros((B, T), dtype=np.int64)
    for b in range(B):
        n = int((ids_np[b] >= 0).sum())
        if n <= 0:
            continue
        canon = "".join("ACGU"[int(ids_np[b][j])] for j in range(n))
        sid = _seq_id(canon)
        p3_row = boundaries_from_entropy(ent_t[b][None, :n], pol.p3_policy.gate)[0].cpu().numpy()
        for i in range(n):
            ic = max(int(pe[0]), min(i, int(pe[-1]) - 1))
            bp = min(max(int(np.searchsorted(pe, ic, side="right")) - 1, 0),
                     pol.n_prefix_bins - 1)
            ec = max(float(ee[0]), min(float(ent_np[b, i]), float(ee[-1]) - 1e-12))
            be = min(max(int(np.searchsorted(ee, ec, side="right")) - 1, 0),
                     pol.n_entropy_bins - 1)
            q = qt[bp, be]
            if SUPPORT_LO <= q <= SUPPORT_HI:
                out[b, i] = 1 if pol._random(sid, i) < q else 0
            else:
                out[b, i] = int(round(p3_row[i])) if i < len(p3_row) else 0
        out[b, 0] = 1
    return out


@cuda
def test_vectorized_matches_scalar_oracle():
    device = "cuda:0"
    pol = _build_policy(device)
    rng = np.random.default_rng(7)
    seqs = ["".join(rng.choice(list("ACGU"), size=int(rng.integers(32, 256))))
            for _ in range(64)]
    maxT = 300
    ids = np.full((len(seqs), maxT), -1, dtype=np.int64)
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = [MAP[c] for c in s]
    nt_ids = torch.tensor(ids, dtype=torch.long, device=device)
    ent_t = _StubEnt().entropy(nt_ids)
    out = pol.boundaries_batch(nt_ids).detach().cpu().numpy()
    oracle = _oracle(pol, ids, ent_t)
    assert np.array_equal(np.round(out).astype(int), oracle), "vectorized != oracle"


@cuda
def test_vectorized_deterministic():
    pol = _build_policy("cuda:0")
    ids = torch.tensor([[0, 1, 2, 3, -1, -1]], dtype=torch.long, device="cuda:0")
    a = pol.boundaries_batch(ids).cpu().numpy()
    b = pol.boundaries_batch(ids).cpu().numpy()
    assert np.array_equal(a, b)


@cuda
def test_pad_zero_and_pos0():
    pol = _build_policy("cuda:0")
    ids = torch.tensor([[0, 1, 2, 3, 0, -1, -1]], dtype=torch.long, device="cuda:0")
    out = pol.boundaries_batch(ids).cpu().numpy()
    assert out[0, 0] == 1.0
    assert out[0, 5] == 0.0 and out[0, 6] == 0.0  # padded


@cuda
def test_supported_q_match():
    pol = _build_policy("cuda:0")
    rng = np.random.default_rng(11)
    seqs = ["".join(rng.choice(list("ACGU"), size=200)) for _ in range(64)]
    maxT = 200
    ids = np.full((len(seqs), maxT), -1, dtype=np.int64)
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = [MAP[c] for c in s]
    nt_ids = torch.tensor(ids, dtype=torch.long, device="cuda:0")
    out = pol.boundaries_batch(nt_ids).cpu().numpy()
    pe = np.asarray(pol.prefix_edges, dtype=np.int64)
    ic = np.clip(np.arange(maxT, dtype=np.int64)[None, :], pe[0], pe[-1] - 1)
    bp = np.clip(np.searchsorted(pe, ic, side="right") - 1, 0, pol.n_prefix_bins - 1)
    sel = (ids >= 0) & (bp >= 1)
    rate = out[sel].mean()
    assert 0.24 < rate < 0.36, f"boundary rate {rate} not matching q~0.30"