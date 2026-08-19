"""Unit tests for canonical codec scoring + calibration baselines (contract 3.6)."""
import math

from evaluator.codec_scoring import canonical_codec_score, calibration_bpns
from evaluator.continuation_code_length import BASE_TO_IDX


class _UniformAdapter:
    vocab = 4

    def encode(self, seq):
        return [BASE_TO_IDX[c] for c in seq]

    def all_log_probs_full(self, ids):
        return [[-math.log(4.0)] * 4 for _ in ids]


class _DeterministicAdapter:
    vocab = 4

    def encode(self, seq):
        return [BASE_TO_IDX[c] for c in seq]

    def all_log_probs_full(self, ids):
        # deterministic: assign the observed token a large log-prob so its
        # normalized probability is ~1 (natural-log softmax).
        rows = []
        for tid in ids:
            lp = [1e-9, 1e-9, 1e-9, 1e-9]
            lp[tid] = 10.0
            rows.append(lp)
        return rows


def test_uniform_codec_is_2_bits():
    # long enough that the range-coder final-flush overhead is negligible
    seqs = ["ACGU" * 2000]
    r = canonical_codec_score(_UniformAdapter(), seqs)
    assert r["coder_consistency_ok"] is True
    assert abs(r["canonical_code_nll_BPN"] - 2.0) < 1e-6, r["canonical_code_nll_BPN"]
    # ACTUAL codec length includes the frozen coder's flush overhead
    assert abs(r["canonical_code_length_BPN"] - 2.0) < 0.02, r["canonical_code_length_BPN"]
    assert r["valid_nt"] == 8000


def test_deterministic_codec_near_zero():
    r = canonical_codec_score(_DeterministicAdapter(), ["ACGUACGUACGU"])
    assert r["canonical_code_nll_BPN"] < 0.01, r["canonical_code_nll_BPN"]


def test_coder_consistency_gate():
    r = canonical_codec_score(_UniformAdapter(), ["ACGU" * 50])
    assert r["coder_consistency_ok"] is True
    assert abs(r["coded_bits"] - r["quantized_cdf_nll_bits"]) <= 64


def test_calibration_baselines_reasonable():
    train = ["ACGU" * 1000]
    hold = ["ACGU" * 1000]
    b = calibration_bpns(train, hold, markov_order=1)
    assert b["uniform_BPN"] == 2.0
    # "ACGU" repeated: order-1 Markov is near-deterministic
    assert b["markov_order1_BPN"] < 0.1, b["markov_order1_BPN"]
    assert b["ppm_BPN"] < b["markov_order1_BPN"] + 0.5


def test_calibration_uniform_holdout_2_bits():
    train = ["ACGU" * 500, "UACG" * 500]
    hold = ["GUAC" * 500]
    b = calibration_bpns(train, hold, markov_order=0)
    assert abs(b["markov_order0_BPN"] - 2.0) < 0.05, b["markov_order0_BPN"]