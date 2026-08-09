"""Tests for the Phase 4 BLT inference adapter (contract 3.2, 3.5, 5.4).

Verifies exact replay of the persisted entropy calibration into patch
boundaries (P1 fixed / P2 random / P3 entropy) and causal per-base likelihood.
GPU-only (adapter is GPU-only by contract); skipped when CUDA is unavailable.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from evaluator.blt_adapter import InternalBLTAdapter, _ReconCalib  # noqa: E402
from model.backbone import BLTCausalLM  # noqa: E402
from model.entropy_predictor import EntropyPatchPolicy, EntropyPredictor  # noqa: E402
from model.train_config import RunConfig, ArchConfig, EmbedConfig, OptimConfig, arm_  # noqa: E402


def _tiny_blt_cfg(arm_id: str):
    arm = arm_(arm_id)
    assert arm.backbone == "blt"
    return RunConfig(
        run_id="tiny_%s" % arm_id, scale="100M", arm=arm, seed=17,
        arch=ArchConfig(d_model=64, n_layers=2, n_heads=1, max_len=256),
        embed=EmbedConfig(embed_dim=64, factorized=False),
        optim=OptimConfig(lr=3e-4),
        budget_nt=2_000_000_000, batch_nt=16384, warmup_nt=1000,
        context_nt=256)


def _make_calib_payload(arm_id: str):
    """Deterministic synthetic persisted-calib payload matching p4_train."""
    payload = {
        "mean_patch_len": 4.0,
        "gate": 1.0,
        "length_dist": [0.25, 0.25, 0.25, 0.25],  # pmf over lengths 1..4
        "checkpoint_hash": "abc123",
    }
    if arm_id == "P3":
        pred = EntropyPredictor()  # default dims, matching calibrate_entropy / _ReconCalib
        payload["predictor_state"] = pred.state_dict()
    else:
        payload["predictor_state"] = None
    return payload


def _make_ckpt(tmp_path, arm_id: str, with_calib: bool = True) -> str:
    cfg = _tiny_blt_cfg(arm_id)
    model = BLTCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=1, max_len=256)
    ckpt = str(tmp_path / ("ckpt_%s.pt" % arm_id))
    payload = {"model": model.state_dict()}
    if with_calib:
        payload["calib"] = _make_calib_payload(arm_id)
    torch.save(payload, ckpt)
    return ckpt


def test_missing_calib_raises(tmp_path):
    ckpt = _make_ckpt(tmp_path, "P3", with_calib=False)
    with pytest.raises(ValueError):
        InternalBLTAdapter("P3", 17, ckpt, device="cuda:0", cfg=_tiny_blt_cfg("P3"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_blt_fixed_boundary(tmp_path):
    ckpt = _make_ckpt(tmp_path, "P1")
    a = InternalBLTAdapter("P1", 17, ckpt, device="cuda:0", cfg=_tiny_blt_cfg("P1"))
    bnd = a._boundary("ACGUACGUACGU", 0)
    # mean_patch_len=4 -> boundary at positions 0,4,8
    assert bnd == [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    lp = a.log_prob_next_base("ACG", "U")
    assert torch.isfinite(torch.tensor(lp)) and lp <= 0.0
    # leading base uniform prior
    assert abs(a.log_prob_next_base("", "A") - (-math.log(4.0))) < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_blt_random_prefix_consistent(tmp_path):
    ckpt = _make_ckpt(tmp_path, "P2")
    a = InternalBLTAdapter("P2", 17, ckpt, device="cuda:0", cfg=_tiny_blt_cfg("P2"))
    full = a._boundary("ACGUACGUACGUACGU", 0)
    # prefix consistency: boundary on a shorter prefix equals the full prefix
    assert a._boundary("ACGUACGU", 0) == full[:8]
    # deterministic for the same sequence id
    assert a._boundary("ACGUACGUACGUACGU", 0) == full
    lp = a.log_prob_next_base("ACG", "U")
    assert torch.isfinite(torch.tensor(lp)) and lp <= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_blt_entropy_forward(tmp_path):
    ckpt = _make_ckpt(tmp_path, "P3")
    a = InternalBLTAdapter("P3", 17, ckpt, device="cuda:0", cfg=_tiny_blt_cfg("P3"))
    # boundary replay matches the EntropyPatchPolicy built from the same calib
    payload = _make_calib_payload("P3")
    pred = EntropyPredictor()  # default dims
    pred.load_state_dict(payload["predictor_state"])
    pred.to("cuda:0").eval()
    policy = EntropyPatchPolicy(pred, payload["gate"], device="cuda:0")
    seq = "ACGUACGUACGUACGU"
    assert a._boundary(seq, 0) == policy.boundary(seq, len(seq))
    lp = a.log_prob_next_base("ACG", "U")
    assert torch.isfinite(torch.tensor(lp)) and lp <= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_score_arm_blt(tmp_path):
    from p4_eval import score_arm
    ckpt = _make_ckpt(tmp_path, "P3")
    a = InternalBLTAdapter("P3", 17, ckpt, device="cuda:0", cfg=_tiny_blt_cfg("P3"))
    res = score_arm(a, ["ACGUACGU", "ACGU"], "test")
    assert res["metric"] == "next_base_BPN"
    assert res["valid_nt_count"] == 8 + 4
    assert torch.isfinite(torch.tensor(res["bpn"]))
    assert res["cpu_fallback_count"] == 0
