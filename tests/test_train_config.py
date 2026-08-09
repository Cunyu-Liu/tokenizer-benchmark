"""Tests for the training config generator (contract 3.2-3.4)."""
import pytest

from model import train_config as tc
from model.arms import ARMS_100M, ARMS_350M, SEEDS
from model.census import count_params


def test_targets_and_budgets():
    assert tc.target_for_scale("100M") == tc.TARGET_100M
    assert tc.target_for_scale("350M") == tc.TARGET_350M
    assert tc.budget_for_scale("100M") == tc.BUDGET_100M_NT
    assert tc.budget_for_scale("350M") == tc.BUDGET_350M_NT
    assert tc.base_lr_for_scale("100M") == 3e-4
    assert tc.base_lr_for_scale("350M") == 2e-4


def test_arch_solver_in_tolerance():
    for scale in ("100M", "350M"):
        target = tc.target_for_scale(scale)
        arch = tc.solve_arch(target)
        assert arch.d_model % 64 == 0
        assert arch.n_heads == arch.d_model // 64
        assert arch.max_len == tc.CONTEXT_NT


def test_all_100M_arms_in_target_range():
    for a in ARMS_100M:
        cfg = tc.resolved_config(a.id, SEEDS[0])
        c = count_params(tc.build_model(cfg))
        lo = tc.TARGET_100M * (1 - tc.TOL)
        hi = tc.TARGET_100M * (1 + tc.TOL)
        assert lo <= c.total_params <= hi, "%s total=%d" % (a.id, c.total_params)


def test_all_350M_arms_in_target_range():
    for a in ARMS_350M:
        cfg = tc.resolved_config(a.id, SEEDS[0], scale="350M", batch_nt=65536)
        c = count_params(tc.build_model(cfg))
        lo = tc.TARGET_350M * (1 - tc.TOL)
        hi = tc.TARGET_350M * (1 + tc.TOL)
        assert lo <= c.total_params <= hi, "%s total=%d" % (a.id, c.total_params)


def test_non_embedding_identical_within_scale():
    nonemb_100 = {count_params(tc.build_model(tc.resolved_config(a.id, 17))).non_embedding_params
                  for a in ARMS_100M}
    assert len(nonemb_100) == 1, "100M non-emb differs across arms: %s" % nonemb_100
    nonemb_350 = {count_params(tc.build_model(tc.resolved_config(a.id, 17, scale="350M", batch_nt=65536))).non_embedding_params
                  for a in ARMS_350M}
    assert len(nonemb_350) == 1, "350M non-emb differs across arms: %s" % nonemb_350


def test_run_counts_and_seeds():
    assert len(tc.all_100M()) == len(ARMS_100M) * len(SEEDS) == 30
    assert len(tc.all_350M()) == len(ARMS_350M) * len(SEEDS) == 12
    assert SEEDS == [17, 29, 43]


def test_lr_tuning_candidates_seed17_only():
    cfg = tc.resolved_config("F1", 17)
    cands = tc.lr_tuning_candidates(cfg)
    assert cands == [3e-4 * 0.5, 3e-4 * 1.0, 3e-4 * 2.0]
    with pytest.raises(ValueError):
        tc.lr_tuning_candidates(tc.resolved_config("F1", 29))


def test_frozen_lr_selection_applied():
    # Contract 3.4: LR selected by validation metric, then frozen. Phase 3 pilot
    # selected 2.0x for every 100M arm except P3 (1.0x); untuned arms (350M C*)
    # stay at base 1.0x. Every 100M tuned arm must carry its frozen factor.
    for arm_id, factor in tc.LR_FACTOR_SELECTED.items():
        cfg = tc.resolved_config(arm_id, tc.TUNE_SEED)
        assert abs(cfg.optim.lr - tc.base_lr_for_scale("100M") * factor) < 1e-12, \
            "%s lr %r != base*%s" % (arm_id, cfg.optim.lr, factor)
    # P3 (entropy-patch) is the single arm frozen at 1.0x.
    assert tc.LR_FACTOR_SELECTED["P3"] == 1.0
    assert tc.resolved_config("F1", 17).optim.lr == 3e-4 * 2.0


def test_frozen_lr_untuned_arms_default_base():
    # 350M arms (C1-C4) have no 100M pilot selection yet: they must stay at base.
    for a in ARMS_350M:
        cfg = tc.resolved_config(a.id, tc.TUNE_SEED, scale="350M", batch_nt=65536)
        assert abs(cfg.optim.lr - tc.base_lr_for_scale("350M")) < 1e-12, \
            "%s unexpectedly tuned lr=%r" % (a.id, cfg.optim.lr)


def test_batch_nt_100M_fastest_default():
    # 100M science runs default to the empirically fastest batch_nt (16384)
    # measured on this cohort; 350M keeps 65536. Batch_nt is an efficiency
    # constant (contract mandates budget/context/optimizer/LR + a consistent
    # effective batch, not a fixed number), so both scale defaults are valid.
    assert tc.BATCH_NT_100M == 16384
    assert tc.resolved_config("F1", 17).batch_nt == 16384
    assert tc.resolved_config("C1", 17, scale="350M").batch_nt == 65536


def test_build_model_output_vocab():
    cfg = tc.resolved_config("F2", 17)  # BPE vocab 1024
    m = tc.build_model(cfg)
    import torch
    ids = torch.randint(0, 1024, (2, 8))
    logits, loss = m(ids, ids)
    assert logits.shape == (2, 8, 1024)


def test_factorized_embedding_path():
    # Real 100M arch, but a huge vocab (16k) pushes plain embedding out of the
    # 2% tolerance, so the solver must fall back to a factorized embedding.
    arch = tc.solve_arch(tc.TARGET_100M)
    vocab = 16384
    emb = tc.solve_embed(vocab, arch, tc.TARGET_100M, arch.max_len)
    assert emb.factorized is True
    assert emb.embed_dim < arch.d_model
    non_emb = tc._non_emb_params(arch.d_model, arch.n_layers, arch.d_ff)
    tot = non_emb + tc._embedding_params_fact(vocab, arch.d_model, emb.embed_dim, arch.max_len)
    assert abs(tot - tc.TARGET_100M) / tc.TARGET_100M <= tc.TOL
    # Plain embedding for this vocab must indeed be out of tolerance.
    plain = non_emb + tc._embedding_params_plain(vocab, arch.d_model, arch.max_len)
    assert abs(plain - tc.TARGET_100M) / tc.TARGET_100M > tc.TOL


def test_verify_arm_ok():
    assert tc.verify_arm(tc.resolved_config("P3", 17))
    assert tc.verify_arm(tc.resolved_config("C4", 17, scale="350M", batch_nt=65536))


def test_run_config_dict_roundtrip():
    cfg = tc.all_100M()[0]
    d = cfg.to_dict()
    assert d["run_id"] == cfg.run_id
    assert d["budget_nt"] == tc.BUDGET_100M_NT
    assert d["context_nt"] == tc.CONTEXT_NT
    assert d["batch_size_seq"] == cfg.batch_size_seq