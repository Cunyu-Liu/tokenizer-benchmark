"""Phase 3 census + GPU-guard tests. Neural smoke requires CUDA torch."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.census import Census, GPUGuard, count_params
from model.backbone import FlatCausalLM, BLTCausalLM, EntropyPatcher


def test_gpu_guard_blocks_cpu():
    g = GPUGuard("cpu")
    with pytest.raises(RuntimeError):
        g.check()
    assert g.cpu_fallback_count == 1


def test_gpu_guard_allows_cuda():
    g = GPUGuard("cuda:0")
    assert g.check() == "cuda:0"
    assert g.cpu_fallback_count == 0


def test_census_matched_within():
    a = Census(total_params=100_000_000, non_embedding_params=90_000_000)
    b = Census(total_params=101_500_000, non_embedding_params=90_000_000)
    assert a.matched_within(b, tol=0.02)
    c = Census(total_params=110_000_000, non_embedding_params=90_000_000)
    assert not a.matched_within(c, tol=0.02)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(),
                    reason="requires CUDA")
def test_flat_cuda_smoke():
    import torch
    dev = "cuda:0"
    m = FlatCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=4).to(dev)
    ids = torch.randint(0, 4, (2, 16)).to(dev)
    tgt = torch.randint(0, 4, (2, 16)).to(dev)
    logits, loss = m(ids, tgt)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert not torch.isnan(loss).item()


@pytest.mark.skipif(not __import__("torch").cuda.is_available(),
                    reason="requires CUDA")
def test_blt_cuda_smoke():
    import torch
    dev = "cuda:0"
    m = BLTCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=4).to(dev)
    ids = torch.randint(0, 4, (2, 16)).to(dev)
    boundary = torch.zeros(2, 16).to(dev)
    boundary[:, ::8] = 1.0  # fixed patch len 8
    tgt = torch.randint(0, 4, (2, 16)).to(dev)
    out, loss = m(ids, boundary, tgt)
    loss.backward()
    assert torch.isfinite(loss).item()


def test_count_params():
    m = FlatCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=4)
    c = count_params(m)
    assert c.total_params > 0
    assert c.non_embedding_params > 0
    assert c.embedding_params == c.total_params - c.non_embedding_params
