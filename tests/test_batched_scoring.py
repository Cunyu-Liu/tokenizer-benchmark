"""Tests: batched (single-forward) flat scoring equals the per-position path.

The batched all_log_probs_* methods must be provably identical to calling
log_prob_next_base / log_prob_token per position. GPU-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from evaluator.internal_adapter import InternalFlatAdapter  # noqa: E402
from model.backbone import FlatCausalLM  # noqa: E402
from model.train_config import RunConfig, ArchConfig, EmbedConfig, OptimConfig, arm_  # noqa: E402


def _tiny_cfg():
    arm = arm_("F1")
    return RunConfig(
        run_id="tiny_F1", scale="100M", arm=arm, seed=17,
        arch=ArchConfig(d_model=64, n_layers=2, n_heads=1, max_len=256),
        embed=EmbedConfig(embed_dim=64, factorized=False),
        optim=OptimConfig(lr=3e-4),
        budget_nt=2_000_000_000, batch_nt=16384, warmup_nt=1000, context_nt=256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_next_base_matches_per_position(tmp_path):
    dev = "cuda:0"
    cfg = _tiny_cfg()
    model = FlatCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=1, max_len=256)
    ckpt = str(tmp_path / "tiny.pt")
    torch.save({"model": model.state_dict()}, ckpt)
    a = InternalFlatAdapter("F1", 17, ckpt, device=dev, cfg=cfg)

    seq = "ACGUACGUACGU"
    batch = a.all_log_probs_next_base(seq)
    perpos = []
    for i, nxt in enumerate(seq):
        perpos.append(a.log_prob_next_base(seq[:i], nxt))
    assert len(batch) == len(seq) == len(perpos)
    for b, p in zip(batch, perpos):
        # batched path = single forward with causal masking; per-position path =
        # forward over each prefix. Under bf16 autocast these differ by ~1e-4
        # (bf16 has ~3 decimal digits), so use a bf16-appropriate tolerance.
        assert abs(b - p) < 1e-3, (b, p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_token_matches_per_position(tmp_path):
    dev = "cuda:0"
    cfg = _tiny_cfg()
    model = FlatCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=1, max_len=256)
    ckpt = str(tmp_path / "tiny.pt")
    torch.save({"model": model.state_dict()}, ckpt)
    a = InternalFlatAdapter("F1", 17, ckpt, device=dev, cfg=cfg)

    ids = [0, 1, 2, 3, 0, 1, 2, 3]
    batch = a.all_log_probs_token(ids)
    perpos = [a.log_prob_token(ids[:i], ids[i]) for i in range(len(ids))]
    assert len(batch) == len(perpos)
    for b, p in zip(batch, perpos):
        # bf16 tolerance (see next_base test above)
        assert abs(b - p) < 1e-3, (b, p)
