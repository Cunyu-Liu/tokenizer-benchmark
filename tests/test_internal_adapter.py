"""Tests for the Phase 4 internal inference adapter (contract 3.5, 5.4).

Pure/CPU-safe tests run always; the GPU smoke test builds a tiny NUC (F1)
checkpoint and verifies forward scoring / generation / round-trip. It is
skipped when CUDA is unavailable (the adapter is GPU-only by contract).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from evaluator.internal_adapter import InternalFlatAdapter, next_base_ctx_length  # noqa: E402
from model.train_config import RunConfig, ArchConfig, EmbedConfig, OptimConfig, arm_  # noqa: E402
from model.backbone import FlatCausalLM  # noqa: E402


def _tiny_cfg():
    arm = arm_("F1")  # NUC
    return RunConfig(
        run_id="tiny_F1", scale="100M", arm=arm, seed=17,
        arch=ArchConfig(d_model=64, n_layers=2, n_heads=1, max_len=256),
        embed=EmbedConfig(embed_dim=64, factorized=False),
        optim=OptimConfig(lr=3e-4),
        budget_nt=2_000_000_000, batch_nt=16384, warmup_nt=1000,
        context_nt=256)


def test_next_base_ctx_length():
    assert next_base_ctx_length("NUC", 0) == 1
    assert next_base_ctx_length("overlap_mer", 3) == 2
    assert next_base_ctx_length("overlap_mer", 6) == 5
    with pytest.raises(ValueError):
        next_base_ctx_length("BPE", 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_adapter_forward(tmp_path):
    dev = "cuda:0"
    cfg = _tiny_cfg()
    # build tiny model, save a synthetic "checkpoint"
    model = FlatCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=1, max_len=256)
    ckpt = str(tmp_path / "tiny.pt")
    torch.save({"model": model.state_dict()}, ckpt)
    adapter = InternalFlatAdapter("F1", 17, ckpt, device=dev, cfg=cfg)

    assert adapter.canonicalize("acguACGU") == "ACGUACGU"
    assert adapter.encode("ACGU") == [0, 1, 2, 3]
    assert adapter.decode([0, 1, 2, 3]) == "ACGU"

    lp = adapter.log_prob_next_base("ACG", "U")
    assert torch.isfinite(torch.tensor(lp)) and lp <= 0.0
    lpt = adapter.log_prob_token([0, 1, 2], 3)
    assert torch.isfinite(torch.tensor(lpt)) and lpt <= 0.0
    # leading position (no context) -> uniform prior -log(4)
    assert abs(adapter.log_prob_next_base("", "A") - (-math.log(4.0))) < 1e-9

    gen = adapter.generate("ACG", 4, temperature=1.0, top_p=1.0)
    assert len(gen) == 4 and all(c in "ACGU" for c in gen)

    # unconditional generation (empty prefix) must not crash (contract 3.6)
    gen0 = adapter.generate("", 4, temperature=1.0, top_p=1.0)
    assert len(gen0) == 4 and all(c in "ACGU" for c in gen0)

    assert adapter.fallback("ACGU") == 0.0
    assert adapter.guard.cpu_fallback_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_score_arm(tmp_path):
    from p4_eval import score_arm
    dev = "cuda:0"
    cfg = _tiny_cfg()
    model = FlatCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=1, max_len=256)
    ckpt = str(tmp_path / "tiny.pt")
    torch.save({"model": model.state_dict()}, ckpt)
    adapter = InternalFlatAdapter("F1", 17, ckpt, device=dev, cfg=cfg)
    res = score_arm(adapter, ["ACGUACGU", "ACGU"], "test")
    assert res["arm"] == "F1"
    assert res["metric"] == "next_base_BPN"
    assert res["valid_nt_count"] == 8 + 4
    assert torch.isfinite(torch.tensor(res["bpn"]))
    assert res["cpu_fallback_count"] == 0
