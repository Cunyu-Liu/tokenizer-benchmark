"""Track L2 (approved amendment 2026-08-30): same-backbone dynamic-input pilot.

CPU-only unit tests (the GPU training loop is covered by p4_train smoke / core
tests). Verifies: (1) PatchInputFlatCausalLM has parameters bit-identical to the
F1 FlatCausalLM arm, (2) forward shapes / loss are finite, (3) boundary policies
build for L2_FIXED / L2_RANDOM / L2_ENTROPY via the shared policy builder.
"""
import pytest
import torch

from model import train_config as tc
from model.backbone import PatchInputFlatCausalLM
from model.train import _patch_policy_for_arm


def _cfg(arm_id, seed=101):
    return tc.resolved_config(arm_id, seed)


def test_l2_model_params_identical_to_flat():
    mf = tc.build_model(_cfg("F1", 101))
    ml = tc.build_model(_cfg("L2_ENTROPY", 101))
    nf = sum(p.numel() for p in mf.parameters())
    nl = sum(p.numel() for p in ml.parameters())
    assert nf == nl, (nf, nl)
    # same named parameter keys: no added modules, trunk/embedding identical
    assert set(dict(ml.named_parameters()).keys()) == set(dict(mf.named_parameters()).keys())


def test_l2_forward_shapes_and_loss_cpu():
    torch.manual_seed(0)
    ml = tc.build_model(_cfg("L2_ENTROPY", 101)).eval()
    nt = torch.randint(0, 4, (2, 32))
    tgt = torch.randint(0, 4, (2, 32)).to(nt.dtype)
    tgt[0, 3] = -100  # ignore token path
    bnd = torch.zeros(2, 32)
    bnd[:, 0] = 1.0
    bnd[:, ::6] = 1.0  # deterministic patching for shape test
    logits, loss = ml(nt, bnd, targets=tgt)
    assert logits.shape == (2, 32, 4)
    assert torch.isfinite(loss)


def test_l2_policies_build_cpu():
    assert _patch_policy_for_arm(_cfg("L2_FIXED", 101), calib=None, device="cpu").kind == "fixed"
    pd = _patch_policy_for_arm(_cfg("L2_ENTROPY", 101), calib=None, device="cpu")
    assert hasattr(pd, "boundaries_batch")  # EntropyPatchPolicy (no .kind) -> GPU path
    bnd = pd.boundaries_batch(torch.randint(0, 4, (1, 48)))
    assert bnd.shape == (1, 48) and (bnd[0, 0] == 1.0)
    pr = _patch_policy_for_arm(_cfg("L2_RANDOM", 101), calib=None, device="cpu")
    assert pr.kind == "random"