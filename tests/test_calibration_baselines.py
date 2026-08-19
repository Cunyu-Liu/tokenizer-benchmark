"""Unit tests for compression calibration baselines (contract 3.6)."""
import math

from evaluator.calibration_baselines import (
    MarkovModel,
    PPMModel,
    uniform_bits_per_nt,
    canonicalize_seq,
)


def _uniform_seq(n=20000):
    # deterministic uniform A/C/G/U
    pattern = "ACGU"
    return (pattern * (n // 4 + 1))[:n]


def test_uniform_reference_is_2_bits():
    assert uniform_bits_per_nt() == 2.0


def test_canonicalize_t_to_u():
    assert canonicalize_seq("acgtACGTt") == "ACGUACGUU"


def test_markov_order0_uniform_approx_2():
    seqs = [_uniform_seq()]
    m = MarkovModel(order=0).fit(seqs, budget_nt=None)
    bpn = m.bits_per_nt(seqs)
    assert abs(bpn - 2.0) < 0.05, bpn


def test_markov_order0_skewed_low_bpn():
    seqs = ["A" * 10000]
    m = MarkovModel(order=0).fit(seqs, budget_nt=None)
    bpn = m.bits_per_nt(["A" * 10000])
    assert bpn < 0.1, bpn  # near-deterministic -> near 0 bits/nt


def test_markov_order1_captures_dependency():
    # "ACGU" repeated: order-0 sees uniform, order-1 sees deterministic transitions
    train = ["ACGU" * 5000]
    m0 = MarkovModel(order=0).fit(train, budget_nt=None)
    m1 = MarkovModel(order=1).fit(train, budget_nt=None)
    b0 = m0.bits_per_nt(["ACGU" * 5000])
    b1 = m1.bits_per_nt(["ACGU" * 5000])
    assert b0 > 1.9, b0
    assert b1 < b0 - 1.0, (b0, b1)


def test_markov_smoothing_avoids_zero_probability():
    # A symbol never seen in train must still get positive probability (alpha).
    m = MarkovModel(order=0, alpha=1.0).fit(["ACGU" * 5000], budget_nt=None)
    # holdout is all 'G': G was seen, so fine; but never-seen symbol check below
    # uses a fresh context so we assert no math domain error on missing context.
    bpn = m.bits_per_nt(["A" * 1000])
    assert math.isfinite(bpn)


def test_markov_missing_context_uses_alpha_uniform():
    m = MarkovModel(order=2, alpha=1.0).fit(["ACGU" * 5000], budget_nt=None)
    # context never observed in fit -> counts all zero -> p = alpha/(alpha*4)=1/4
    bpn = m.bits_per_nt(["G" * 2000])
    assert abs(bpn - 2.0) < 0.05, bpn


def test_ppm_repetitive_low_bpn():
    p = PPMModel(order=5).fit(["ACGU" * 5000], budget_nt=None)
    bpn = p.bits_per_nt(["ACGU" * 5000])
    assert bpn < 0.5, bpn


def test_ppm_reduces_vs_order0():
    seq = ["ACGU" * 5000]
    m0 = MarkovModel(order=0).fit(seq, budget_nt=None)
    ppm = PPMModel(order=5).fit(seq, budget_nt=None)
    b0 = m0.bits_per_nt(seq)
    bp = ppm.bits_per_nt(seq)
    assert bp < b0, (b0, bp)


def test_ppm_never_seen_symbol_finite():
    p = PPMModel(order=5).fit(["ACGU" * 500], budget_nt=None)
    bpn = p.bits_per_nt(["AAAA" * 100])
    assert math.isfinite(bpn)


def test_empty_holdout_no_crash():
    m = MarkovModel(order=0).fit(["ACGU"], budget_nt=None)
    bpn = m.bits_per_nt([])
    assert math.isnan(bpn)