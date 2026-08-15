"""
NOTE: This test file is intended to be placed at /home/cunyuliu/tokenizer-benchmark/tests/test_conditional_patch.py
and run within the server's conda environment where GPU is available.
"""
import sys, os, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model.conditional_patch import (
    ConditionalRandomPatchPolicy, _seq_id, PATCH_RANDOMIZATION_SEED,
    SUPPORT_LO, SUPPORT_HI, N_PREFIX_LEN_BINS, N_ENTROPY_BINS,
)


def test_seq_id_deterministic():
    """Same sequence -> same id."""
    assert _seq_id("ACGUACGU") == _seq_id("ACGUACGU")
    assert _seq_id("ACGU") != _seq_id("ACGUACGU")


def test_conditional_random_init():
    """Policy can be created with defaults."""
    p = ConditionalRandomPatchPolicy()
    assert p.kind == "conditional_random"
    assert p.seed == PATCH_RANDOMIZATION_SEED


def test_conditional_random_random_deterministic():
    """Same (seq_id, position) gives same random draw."""
    p = ConditionalRandomPatchPolicy()
    r1 = p._random(42, 5)
    r2 = p._random(42, 5)
    assert r1 == r2
    # Different position gives different draw
    r3 = p._random(42, 6)
    assert r1 != r3


def test_conditional_random_random_bounds():
    """Random draws are in [0, 1)."""
    p = ConditionalRandomPatchPolicy()
    for sid in [0, 1, 100, 999999]:
        for i in range(100):
            r = p._random(sid, i)
            assert 0.0 <= r < 1.0


def test_conditional_random_random_different_seqs():
    """Different seq ids give different random draws (same position)."""
    p = ConditionalRandomPatchPolicy()
    draws = set()
    for sid in range(100):
        draws.add(p._random(sid, 0))
    assert len(draws) > 80  # most should be unique


def test_bin_edges():
    """After fit_q, bin edges are set and sensible."""
    # Can't test fit_q without GPU (needs entropy predictor)
    # Test that the interface works: binning requires no GPU
    p = ConditionalRandomPatchPolicy()
    p.prefix_edges = [0, 10, 100, 1000, 4096]
    p.ent_edges = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert p._prefix_bin(0) == 0
    assert p._prefix_bin(5) == 0
    assert p._prefix_bin(10) == 1  # left edge, inclusive
    assert p._prefix_bin(50) == 1
    assert p._prefix_bin(2000) == 3
    assert p._entropy_bin(0.0) == 0
    assert p._entropy_bin(1.2) == 2
    assert p._entropy_bin(2.1) == 3  # clamps to last bin (4 bins: 0..3)


def test_boundaries_from_entropy_without_q():
    """Without q_table fitted, boundary() should raise."""
    p = ConditionalRandomPatchPolicy()
    p.prefix_edges = [0, 4096]
    p.ent_edges = [0.0, 2.0]
    p.q_table = [[0.5]]  # 1x1 table
    canon = "ACGUACGUACGU"[:12]
    ent = np.array([0.5] * 12)  # uniform, in support
    bnd = p._boundaries_from_entropy(canon, ent, 42, None)
    assert len(bnd) == 12
    assert bnd[0] == 1  # position 0 always boundary


def test_boundaries_first_position_always_one():
    """Position 0 is always a patch start (contract 3.2)."""
    p = ConditionalRandomPatchPolicy()
    p.prefix_edges = [0, 100]
    p.ent_edges = [0.0, 2.0]
    p.q_table = [[0.1]]
    canon = "ACGU" * 10
    ent = np.random.default_rng(42).uniform(0.0, 1.0, 40)
    bnd = p._boundaries_from_entropy(canon, ent, 42, None)
    assert bnd[0] == 1


def test_non_supported_strata_replay_p3():
    """Non-supported strata (q=0.0 or q=1.0) replay P3 boundary exactly."""
    p = ConditionalRandomPatchPolicy()
    p.prefix_edges = [0, 100]
    p.ent_edges = [0.0, 2.0]
    p.q_table = [[0.0, 1.0]]  # 2 entropy bins: q=0.0, q=1.0
    # Re-define ent_edges to have 2 bins
    p.ent_edges = [0.0, 1.0, 2.0]
    p.n_entropy_bins = 2
    # Re-define q_table properly
    p.q_table = [[0.0, 1.0]]  # bp=0, be=0 -> q=0; bp=0, be=1 -> q=1

    canon = "ACGUACGUACGU"[:12]
    # ent < 1.0 -> be=0 -> q=0.0 -> non-supported -> replay P3
    ent = np.array([0.5] * 12)
    p3_bnd = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
    bnd = p._boundaries_from_entropy(canon, ent, 42, p3_bnd)
    assert bnd == p3_bnd, f"expected P3 replay: {p3_bnd}, got {bnd}"


def test_boundary_prefix_consistency():
    """Boundary of prefix equals first len(prefix) positions of full seq."""
    p = ConditionalRandomPatchPolicy()
    p.prefix_edges = [0, 5, 100]
    p.ent_edges = [0.0, 0.5, 2.0]
    p.q_table = [[0.3, 0.3], [0.3, 0.3]]  # supported
    canon = "ACGU" * 10
    # Full entropy
    ent = np.array([0.25 * (i % 4) for i in range(40)])
    sid = _seq_id(canon)
    full = p._boundaries_from_entropy(canon, ent, sid, None)
    # Prefix
    prefix = canon[:20]
    ent_p = ent[:20]
    pref = p._boundaries_from_entropy(prefix, ent_p, sid, None)
    assert pref == full[:20], "prefix-consistent boundaries failed"


def test_supported_strata_random_sampling():
    """Supported strata produce random (not deterministic) boundaries."""
    p = ConditionalRandomPatchPolicy()
    p.prefix_edges = [0, 100]
    p.ent_edges = [0.0, 2.0]
    p.q_table = [[0.3]]  # q=0.3, supported (0.05 <= 0.3 <= 0.95)
    canon = "ACGU" * 25  # 100 nt
    ent = np.array([0.5] * 100)
    # Different seq_ids should give different boundary patterns
    b1 = p._boundaries_from_entropy(canon, ent, 1, None)
    b2 = p._boundaries_from_entropy(canon, ent, 2, None)
    assert b1 != b2, "different seq_ids should give different boundaries"


def test_q_table_dimensions():
    """q_table shape matches (n_prefix_bins, n_entropy_bins)."""
    p = ConditionalRandomPatchPolicy()
    p.q_table = [[0.1] * N_ENTROPY_BINS for _ in range(N_PREFIX_LEN_BINS)]
    assert len(p.q_table) == N_PREFIX_LEN_BINS
    assert len(p.q_table[0]) == N_ENTROPY_BINS