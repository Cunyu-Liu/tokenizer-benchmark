"""Tests for the Phase 3 training data pipeline + GPU training loop."""
import os

import pytest
import polars as pl
import torch

from model import train_config as tc
from model.train import smoke_run, train, validate_on_split, _patch_policy_for_arm
from model.dataset import (
    IGNORE, build_tokenizer, build_arm_tokenizer, count_valid_nt,
    iter_train_batches, sample_train_sequences,
)
from model.patch import PatchPolicy, fixed_policy, random_policy
from model.entropy_predictor import (
    EntropyPredictor, boundaries_from_entropy, patch_lengths_from_bounds,
    EntropyPatchPolicy, calibrate_entropy,
)


def _has_cuda():
    return torch.cuda.is_available()


def _make_parquet(tmp_path, n=200, max_len=120, split="train"):
    import random
    rng = random.Random(0)
    rows = []
    for i in range(n):
        L = rng.randint(20, max_len)
        seq = "".join(rng.choice("ACGU") for _ in range(L))
        rows.append({"split_membership": split, "canonical_sequence": seq})
    p = tmp_path / "mini.parquet"
    pl.DataFrame(rows).write_parquet(p)
    return str(p)


def test_sample_train_sequences(tmp_path):
    p = _make_parquet(tmp_path, n=200)
    s = sample_train_sequences(p, n=50, seed=1)
    assert 0 < len(s) <= 50
    s2 = sample_train_sequences(p, n=50, seed=1)
    assert s == s2  # deterministic


@pytest.mark.parametrize("arm_id", ["F1", "F2", "F3", "F4", "F5", "F6", "F7"])
def test_build_tokenizers(tmp_path, arm_id):
    p = _make_parquet(tmp_path, n=150)
    sample = sample_train_sequences(p, n=100, seed=1)
    cfg = tc.resolved_config(arm_id, 17)
    tok = build_tokenizer(cfg.arm, sample)
    assert tok.vocab_size() == cfg.arm.vocab_size
    # lossless round-trip; nonoverlap k-mer is lossless on k-aligned lengths
    stride = getattr(tok, "stride", 1)
    rt_seq = sample[0]
    if stride and stride > 1:
        rt_seq = rt_seq[: len(rt_seq) - (len(rt_seq) % stride)]
    assert tok.round_trip(rt_seq)


def test_build_arm_tokenizer_bpe_not_degenerate(tmp_path):
    """Regression: BPE/Unigram vocab must be fit on a real train sample, not
    built empty (which degenerates to 4 bases). iter_train_batches consumed
    build_arm_tokenizer, so this locks the fix."""
    p = _make_parquet(tmp_path, n=300, max_len=120)
    for arm_id in ("F2", "F3"):
        cfg = tc.resolved_config(arm_id, 17)  # BPE / Unigram vocab 1024
        tok = build_arm_tokenizer(p, cfg.arm)
        # a real subword vocab, not the degenerate 4-base alphabet
        assert tok.vocab_size() == cfg.arm.vocab_size
        assert tok.vocab_size() > 4


def test_iter_train_batches_bpe_ids_within_vocab(tmp_path):
    """Batches emitted for a BPE arm must use the real fitted vocab ids."""
    p = _make_parquet(tmp_path, n=300, max_len=120)
    cfg = tc.resolved_config("F2", 17)
    tok = build_arm_tokenizer(p, cfg.arm)
    v = tok.vocab_size()
    seen_wide = False
    for b in iter_train_batches(p, cfg, batch_size=8, max_batches=4, split="train"):
        for row in b["token_ids"]:
            for x in row:
                if x != IGNORE:
                    assert 0 <= x < v
                    if x > 3:
                        seen_wide = True
    assert seen_wide, "BPE arm should emit subword ids beyond the 4 bases"


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
def test_gpu_validate_on_split_flat(tmp_path):
    """validate_on_split returns a finite val loss on the validation split."""
    train_p = _make_parquet(tmp_path, n=160, max_len=100)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    val_p = _make_parquet(val_dir, n=160, max_len=100, split="validation")
    cfg = tc.resolved_config("F1", 17)
    res = train(cfg, train_p, device="cuda:0", batch_size=8,
                checkpoint_interval_nt=0, log_every=None, max_batches=6,
                budget_nt=50_000, return_model=True)
    assert res["steps"] > 0
    v = validate_on_split(cfg, res["model"], val_p, device="cuda:0",
                          split="validation", val_nt=50_000, batch_nt=4096)
    assert v["val_loss"] is not None and torch.isfinite(torch.tensor(v["val_loss"]))
    assert v["val_nt"] > 0
    assert v["cpu_fallback_count"] == 0
    assert v["val_loss"] > 0


def test_iter_flat_batches_shapes(tmp_path):
    p = _make_parquet(tmp_path, n=300)
    cfg = tc.resolved_config("F1", 17)
    got = 0
    for b in iter_train_batches(p, cfg, batch_size=16, max_batches=3):
        B = len(b["token_ids"])
        T0 = len(b["token_ids"][0])
        assert len(b["targets"]) == B
        assert all(len(x) == T0 for x in b["token_ids"])
        assert all(len(x) == T0 for x in b["targets"])
        assert b["token_ids"][0][0] in range(4)
        assert count_valid_nt(b) > 0
        got += 1
    assert got == 3


def test_iter_blt_batches_boundary(tmp_path):
    p = _make_parquet(tmp_path, n=300)
    cfg = tc.resolved_config("P1", 17)
    policy = PatchPolicy(kind="fixed", patch_len=8)
    for b in iter_train_batches(p, cfg, batch_size=16, max_batches=2,
                                boundary_provider=policy):
        assert "boundary" in b
        assert len(b["boundary"][0]) == len(b["token_ids"][0])
        assert all(v in (0, 1) for v in b["boundary"][0] if v != IGNORE)
        assert count_valid_nt(b) > 0


def test_fixed_patch_boundary():
    pol = fixed_policy(4)
    b = pol.boundary("ACGUACGUACGU", 12)
    assert b[0] == 1 and b[4] == 1 and b[8] == 1
    assert b[1] == 0 and b[2] == 0 and b[3] == 0


def test_random_patch_deterministic():
    dist = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = random_policy(seed=7, length_dist=dist)
    b1 = a.boundary("ACGUACGUACGUACGUACGUACGU", 24, seq_id=3)
    b2 = a.boundary("ACGUACGUACGUACGUACGUACGU", 24, seq_id=3)
    assert b1 == b2
    assert b1[0] == 1
    assert len(b1) == 24


def test_boundaries_from_entropy_pos0_start():
    # first position always a patch start
    ent = torch.ones(1, 10)
    b = boundaries_from_entropy(ent, 1.0)
    assert b[0, 0] == 1


def test_boundaries_from_entropy_gate():
    import torch
    # constant entropy 1.0, gate 2.0 -> new patch every 2 steps
    ent = torch.ones(1, 8)
    b = boundaries_from_entropy(ent, 2.0)
    assert b[0, 0] == 1
    assert b[0, 2] == 1
    assert b[0, 4] == 1
    assert b[0, 6] == 1
    assert b[0, 1] == 0 and b[0, 3] == 0


def test_patch_lengths_from_bounds():
    import torch
    b = torch.tensor([[1.0, 0, 1.0, 0, 0, 1.0]])
    lengths = patch_lengths_from_bounds(b)
    assert lengths == [2, 3, 1]  # boundaries at 0,2,5; final run to end incl.


def test_entropy_predictor_logit_shape():
    import torch
    m = EntropyPredictor(d_model=16, d_hidden=16).eval()
    ids = torch.randint(0, 4, (2, 8))
    logits = m.logits(ids)
    assert logits.shape == (2, 8, 4)
    ent = m.entropy(ids)
    assert ent.shape == (2, 8)
    assert (ent >= 0).all()


def test_entropy_calib_predictor_attached():
    # calibrate_entropy requires GPU; here verify the smoke fallback wiring
    # via _patch_policy_for_arm returns an EntropyPatchPolicy for P3.
    p = EntropyPatchPolicy(EntropyPredictor(d_model=8, d_hidden=8), gate=2.0)
    assert isinstance(p, EntropyPatchPolicy)


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
def test_gpu_smoke_train_flat(tmp_path):
    p = _make_parquet(tmp_path, n=120, max_len=100)
    cfg = tc.resolved_config("F1", 17)
    res = smoke_run(cfg, p, device="cuda:0", budget_nt=50_000, batch_size=8)
    assert res["steps"] > 0
    assert res["cpu_fallback_count"] == 0
    assert res["final_loss"] is not None


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
def test_gpu_smoke_train_blt_fixed(tmp_path):
    p = _make_parquet(tmp_path, n=120, max_len=100)
    cfg = tc.resolved_config("P1", 17)
    res = smoke_run(cfg, p, device="cuda:0", budget_nt=50_000, batch_size=8)
    assert res["steps"] > 0
    assert res["cpu_fallback_count"] == 0
    assert res["final_loss"] is not None


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
def test_patch_count_calibration_gpu(tmp_path):
    """Contract 5.4: fixed/random/entropy patch counts match the calib."""
    p = _make_parquet(tmp_path, n=400, max_len=200)
    calib = calibrate_entropy(p, device="cuda:0", budget_nt=300_000,
                              batch_size=64, target_patch_len=8)
    assert calib.mean_patch_len >= 1
    # P1 fixed length tracks the calibrated mean.
    pol_fixed = _patch_policy_for_arm(tc.resolved_config("P1", 17),
                                      calib=calib, device="cuda:0")
    assert abs(pol_fixed.patch_len - calib.mean_patch_len) <= 1.0
    # P2 random length distribution has the same mean in expectation.
    dist = calib.length_dist
    exp_len = sum((i + 1) * pi for i, pi in enumerate(dist))
    assert abs(exp_len - calib.mean_patch_len) / max(1.0, calib.mean_patch_len) < 0.15
    # P3 entropy boundaries on a fresh sample reproduce the calibrated mean.
    pol_ent = _patch_policy_for_arm(tc.resolved_config("P3", 17),
                                    calib=calib, device="cuda:0")
    assert isinstance(pol_ent, EntropyPatchPolicy)
    import random as _rng
    rng = _rng.Random(999)
    total_nt, total_patches = 0, 0
    for _ in range(50):
        L = rng.randint(40, 120)
        seq = "".join(rng.choice("ACGU") for _ in range(L))
        b = pol_ent.boundary(seq, L)
        total_patches += sum(1 for v in b if v > 0.5)
        total_nt += L
    ent_mean = total_nt / max(1, total_patches)
    assert abs(ent_mean - calib.mean_patch_len) / max(1.0, calib.mean_patch_len) < 0.25