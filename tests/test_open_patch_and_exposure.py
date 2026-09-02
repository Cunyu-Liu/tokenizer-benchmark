"""Amendment 2026-09-02 fixtures: open-patch forward, leak-free causality,
nt-based exposure, and the B1==Flat bridge equivalence.

Pure CPU (tiny models): these are mathematical properties of the forward and
the data pipeline, independent of any trained checkpoint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.backbone import (
    FlatCausalLM, BLTCausalLM, PatchInputFlatCausalLM, open_patch_running_mean,
)
from evaluator.tokenizer import KmerTokenizer
from model.dataset import IGNORE, count_valid_nt


def _flat():
    torch.manual_seed(0)
    return FlatCausalLM(vocab_size=4, d_model=32, n_layers=2, n_heads=2,
                        max_len=128, use_checkpoint=False)


def _blt():
    torch.manual_seed(0)
    return BLTCausalLM(vocab_size=4, d_model=32, n_layers=2, n_heads=2,
                       max_len=128, use_checkpoint=False)


def _ids(seq):
    return torch.tensor([["ACGU".index(b) for b in seq]])


def test_b1_patch1_bit_identical_to_flat():
    """Contract 3.2.1 bridge: BLT with patch-size=1 computes the SAME forward
    as the Flat backbone (same trunk, same params). Any trained-B1 vs trained-F1
    difference can therefore only come from training conditions (lr), not
    architecture."""
    flat, blt = _flat(), _blt()
    blt.load_state_dict(flat.state_dict())  # identical params, identical keys
    ids = _ids("ACGUACGUACGUACGU" * 4)
    bnd = torch.ones_like(ids, dtype=torch.float)  # patch=1: every nt a patch
    with torch.no_grad():
        lf, _ = flat(ids)
        lb, _ = blt(ids, bnd)
    assert lf.shape == lb.shape
    assert torch.equal(lf, lb), "B1 (patch=1) must be bit-identical to Flat"


def test_open_patch_suffix_perturbation_invariance():
    """Contract 5.2: logits at prefix positions must not change when an
    unobserved suffix changes (no future-nt leak within patches)."""
    blt = _blt()
    blt.eval()
    p1 = "ACGUACGUACGUACGUACGU"     # shared prefix (spans several 6-patches)
    s1 = p1 + "AAAAAA"
    s2 = p1 + "GGGGGGGG"
    bnd = None
    def fwd(seq):
        ids = _ids(seq)
        b = torch.zeros_like(ids, dtype=torch.float)
        b[:, ::6] = 1.0  # fixed-6 boundaries
        with torch.no_grad():
            lo, _ = blt(ids, b)
        return lo
    l1, l2 = fwd(s1), fwd(s2)
    n = len(p1)
    # cross-length forwards differ only in float reduction order
    assert torch.allclose(l1[0, :n], l2[0, :n], atol=1e-5), \
        "suffix changed prefix logits (leak)"


def test_open_patch_no_target_leak_within_patch():
    """Sharper no-leak: mutating the base at position i+1 (inside the open
    patch) must not change the conditional logits emitted at position i,
    which are used to predict x_{i+1}."""
    blt = _blt()
    blt.eval()
    base = list("ACGUACGUACGUACGU")
    def logits_at(seq, pos):
        ids = _ids(seq)
        b = torch.zeros_like(ids, dtype=torch.float)
        b[:, ::6] = 1.0
        with torch.no_grad():
            lo, _ = blt(ids, b)
        return lo[0, pos]
    a = "".join(base)
    b = list(base); b[4] = "G" if b[4] != "G" else "C"  # mutate mid-patch nt
    b = "".join(b)
    # position 3 predicts x_4; mutating x_4 must not change logits at pos 3
    assert torch.equal(logits_at(a, 3), logits_at(b, 3))
    # and the batched all-positions view equals prefix-recompute semantics:
    # logits at pos k equal logits of the truncated sequence at pos k
    ids_a = _ids(a)
    b_a = torch.zeros_like(ids_a, dtype=torch.float); b_a[:, ::6] = 1.0
    with torch.no_grad():
        full, _ = blt(ids_a, b_a)
    trunc = _ids(a[:10])
    b_t = torch.zeros_like(trunc, dtype=torch.float); b_t[:, ::6] = 1.0
    with torch.no_grad():
        pref, _ = blt(trunc, b_t)
    # cross-length forwards differ only in float reduction order
    assert torch.allclose(full[0, :10], pref[0, :10], atol=1e-5)


def test_open_patch_patch_final_equals_closed_patch_mean():
    """At a patch-final position the open running mean equals the closed patch
    mean (consistency with the previous fold semantics at scored boundaries)."""
    torch.manual_seed(1)
    emb = torch.randn(1, 12, 8)
    bnd = torch.zeros(1, 12); bnd[:, ::6] = 1.0
    pooled = open_patch_running_mean(emb, bnd)
    # patch 0 = positions 0..5: closed mean
    closed = emb[0, :6].mean(0)
    assert torch.allclose(pooled[0, 5], closed, atol=1e-6)
    # mid-patch position 2: open mean of 0..2 only
    assert torch.allclose(pooled[0, 2], emb[0, :3].mean(0), atol=1e-6)


def test_running_means_invertible_within_patch():
    """The open-patch running means are an invertible reparameterisation of
    the raw embeddings within a patch (no information is discarded; the P
    arms study input parameterisation, not a bottleneck)."""
    torch.manual_seed(2)
    emb = torch.randn(2, 24, 8)
    bnd = torch.zeros(2, 24); bnd[:, ::6] = 1.0
    pooled = open_patch_running_mean(emb, bnd)
    # recover emb[k] = m_k*(k-s+1) - m_{k-1}*(k-s)
    for b in range(2):
        for k in range(1, 24):
            if bnd[b, k] > 0.5:
                continue  # patch start: m_k == emb[k]
                # (k is its own patch start)
            s_cnt = k % 6 if k % 6 else 6  # position within fixed-6 patch
            n_new = k - (k - s_cnt) + 1
            n_prev = n_new - 1
            rec = pooled[b, k] * n_new - pooled[b, k - 1] * n_prev
            assert torch.allclose(rec, emb[b, k], atol=1e-4)


def test_count_valid_nt_uses_nt_weights():
    """Exposure counts NUCLEOTIDES, not tokens: a 6-mer token contributes 6."""
    tok = KmerTokenizer(k=6, overlapping=False)
    seq = "ACGUACGUACGUACGUACGUAC"  # 23 nt -> 3 full 6-mers + tail 5
    ids = tok.encode(seq)
    w = tok.token_nt_counts(ids[1:])  # targets = ids[1:]
    batch = {
        "token_ids": [ids[:-1]],
        "targets": [ids[1:]],
        "nt_weights": [w],
    }
    # every real nt attributed exactly once; the FIRST token is pure
    # context (its nt are not scored), so target nt = len(seq) - nt_counts[0]
    first_cov = tok.token_nt_counts(ids)[0]
    assert count_valid_nt(batch) == len(seq) - first_cov
    assert sum(w) == len(seq) - first_cov
    # legacy batches (no weights) still count positions (backward compat)
    legacy = {"token_ids": [ids[:-1]], "targets": [ids[1:]]}
    assert count_valid_nt(legacy) == len(ids) - 1


def test_l2_patched_input_matches_blt_forward():
    """Track L2 shares the Flat trunk and computes the same open-patch forward
    as the P arms (documented redundancy, amendment 2026-09-02)."""
    flat_trunk = _flat()
    l2 = PatchInputFlatCausalLM(vocab_size=4, d_model=32, n_layers=2, n_heads=2,
                                max_len=128, use_checkpoint=False)
    blt = _blt()
    l2.load_state_dict(flat_trunk.state_dict())
    blt.load_state_dict(flat_trunk.state_dict())
    ids = _ids("ACGUACGUACGUACGUACGUAC")
    b = torch.zeros_like(ids, dtype=torch.float); b[:, ::6] = 1.0
    with torch.no_grad():
        ll, _ = l2(ids, b)
        lb, _ = blt(ids, b)
    assert torch.equal(ll, lb)