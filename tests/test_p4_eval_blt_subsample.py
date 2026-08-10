"""Integration test: p4_eval.py BLT branch on a real p4_train-format checkpoint.

De-risks the actual sealed-test command path before real checkpoints exist:
  - builds a tiny BLT model + entropy calibration,
  - saves a checkpoint in the EXACT p4_train.py payload format
    (payload keys: model / calib{...}),
  - loads a few rows from the real homology-stratified subsample parquet,
  - runs p4_eval.score_arm (BLT next_base_BPN, exact causal per-base)
    on GPU and asserts finite BPN + zero CPU fallback.

GPU-only; skipped when CUDA is unavailable.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

BASE = "/mnt/cunyuliu/tokenizer-benchmark"
SUBSAMPLE = os.path.join(BASE, "runs/subsample/blt_test_subsample.parquet")


def _tiny_blt_cfg():
    from model.train_config import RunConfig, ArchConfig, EmbedConfig, OptimConfig, arm_
    arm = arm_("P3")
    return RunConfig(
        run_id="tiny_P3", scale="100M", arm=arm, seed=17,
        arch=ArchConfig(d_model=64, n_layers=2, n_heads=1, max_len=256),
        embed=EmbedConfig(embed_dim=64, factorized=False),
        optim=OptimConfig(lr=3e-4),
        budget_nt=2_000_000_000, batch_nt=16384, warmup_nt=1000, context_nt=256)


def _make_p4_ckpt(path: str):
    """Write a checkpoint identical in structure to p4_train.py's save."""
    from model.backbone import BLTCausalLM
    from model.entropy_predictor import EntropyPredictor
    model = BLTCausalLM(vocab_size=4, d_model=64, n_layers=2, n_heads=1, max_len=256)
    pred = EntropyPredictor()
    payload = {
        "model": model.state_dict(),
        "opt": {},
        "nt": 2_000_000_000, "step": 1000,
        "val_loss": 1.0, "cfg_run_id": "tiny_P3", "arm": "P3", "seed": 17,
        "calib": {
            "predictor_state": pred.state_dict(),
            "gate": 1.0,
            "mean_patch_len": 4.0,
            "length_dist": [0.25, 0.25, 0.25, 0.25],
            "checkpoint_hash": "smoke",
        },
    }
    torch.save(payload, path)


def _subsample_rows(n: int) -> list[str]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(SUBSAMPLE)
    out = []
    for batch in pf.iter_batches(batch_size=100, columns=["canonical_sequence"]):
        for seq in batch.column("canonical_sequence").to_pylist():
            out.append(seq)
            if len(out) >= n:
                return out
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_p4_eval_blt_on_subsample(tmp_path):
    import pyarrow.parquet as pq
    assert os.path.exists(SUBSAMPLE), "subsample parquet missing; run p4_subsample.py first"
    ckpt = str(tmp_path / "p4_blt.pt")
    _make_p4_ckpt(ckpt)

    from p4_eval import score_arm
    from evaluator.blt_adapter import InternalBLTAdapter

    adapter = InternalBLTAdapter("P3", 17, ckpt, device="cuda:0", cfg=_tiny_blt_cfg())
    seqs = _subsample_rows(4)
    assert len(seqs) == 4
    res = score_arm(adapter, seqs, "test")
    assert res["metric"] == "next_base_BPN"
    assert res["n_sequences"] == 4
    assert res["valid_nt_count"] > 0
    assert torch.isfinite(torch.tensor(res["bpn"]))
    assert res["cpu_fallback_count"] == 0

    # the subsample rows are real canonical nucleotides; score_arm truncates
    # to max_len=4096 (contract 3.1: 16-4096 is the main length bin; longer
    # sequences are length-OOD and scored on the 4096-nt prefix)
    n_nt = sum(min(len(s), 4096) for s in seqs)
    assert res["valid_nt_count"] == n_nt