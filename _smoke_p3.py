"""GPU smoke test for the Phase 3 training pipeline (synthetic mini-parquet)."""
import os
import random
import sys

sys.path.insert(0, ".")

import torch
import polars as pl

from model import train_config as tc
from model.train import smoke_run, build_model_for_cfg, train, _patch_policy_for_arm
from model.dataset import iter_train_batches, sample_train_sequences, build_tokenizer, count_valid_nt
from model.patch import PatchPolicy
from model.entropy_predictor import calibrate_entropy, EntropyPatchPolicy

TMP = "/mnt/cunyuliu/tokenizer-benchmark/tmp"
os.makedirs(TMP, exist_ok=True)
PARQUET = os.path.join(TMP, "smoke_mini.parquet")

rng = random.Random(0)


def synth_train_parquet(n=2000, max_len=200):
    rows = []
    for i in range(n):
        L = rng.randint(20, max_len)
        seq = "".join(rng.choice("ACGU") for _ in range(L))
        rows.append({"split_membership": "train", "canonical_sequence": seq})
    df = pl.DataFrame(rows)
    df.write_parquet(PARQUET)
    print("wrote synthetic parquet:", PARQUET)


def main():
    synth_train_parquet()
    dev = "cuda:0"
    assert torch.cuda.is_available(), "CUDA required"
    print("GPU:", torch.cuda.get_device_name(0))

    # --- flat arm F1 smoke ---
    cfg = tc.resolved_config("F1", 17)
    res = smoke_run(cfg, PARQUET, device=dev, budget_nt=200_000, batch_size=16)
    print("F1 smoke:", res)
    assert res["steps"] > 0
    assert res["cpu_fallback_count"] == 0
    assert res["final_loss"] is not None and torch.isfinite(torch.tensor(res["final_loss"]))

    # --- BLT arm P1 (fixed patch) smoke ---
    cfgp = tc.resolved_config("P1", 17)
    policy = PatchPolicy(kind="fixed", patch_len=8)
    res = smoke_run(cfgp, PARQUET, device=dev, budget_nt=200_000, batch_size=16)
    print("P1 smoke:", res)
    assert res["steps"] > 0 and res["cpu_fallback_count"] == 0

    # --- P3 entropy calibration + entropy-patch training smoke ---
    calib = calibrate_entropy(PARQUET, device=dev, target_patch_len=6, seed=17,
                              budget_nt=200_000, batch_size=64)
    print("CALIB gate=%.4f mean_patch_len=%.3f pred_params=%d hash=%s" % (
        calib.gate, calib.mean_patch_len, calib.predictor_params,
        calib.checkpoint_hash[:12]))
    assert calib.mean_patch_len > 0 and calib.predictor_params > 0
    p1 = _patch_policy_for_arm(tc.resolved_config("P1", 17), calib=calib, device=dev)
    p3 = _patch_policy_for_arm(tc.resolved_config("P3", 17), calib=calib, device=dev)
    print("P1 fixed len =", p1.patch_len)
    assert isinstance(p3, EntropyPatchPolicy)
    import copy
    ecfg = copy.copy(tc.resolved_config("P3", 17))
    object.__setattr__(ecfg, "budget_nt", 200_000)
    res = train(ecfg, PARQUET, device=dev, batch_size=8,
                checkpoint_interval_nt=0, log_every=None, calib=calib)
    print("P3 smoke:", res)
    assert res["steps"] > 0 and res["cpu_fallback_count"] == 0
    assert res["final_loss"] is not None and torch.isfinite(torch.tensor(res["final_loss"]))

    # --- tokenizer build on sample ---
    sample = sample_train_sequences(PARQUET, n=200, seed=1)
    assert 0 < len(sample) <= 200
    of = tc.resolved_config("F4", 17)  # overlap 3-mer
    tok = build_tokenizer(of.arm, sample)
    assert tok.vocab_size() == 4 ** 3
    for b in iter_train_batches(PARQUET, of, batch_size=8, max_batches=3):
        assert b["token_ids"][0][0] in range(tok.vocab_size())
        n = count_valid_nt(b)
        assert n > 0
    print("OK: all smoke checks passed")


if __name__ == "__main__":
    main()