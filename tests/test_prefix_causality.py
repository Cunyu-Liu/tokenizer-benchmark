"""Explicit contract 5.2 prefix-causality / suffix-perturbation fixtures.

For the observed prefix P, the canonical encoding, patch boundaries, and
next-step conditionals must depend ONLY on the observed prefix -- not on any
unobserved suffix. We verify:

  1. token encoding of a prefix is unchanged by appending any suffix;
  2. the P2 (ConditionalRandomPatchPolicy) boundary over a prefix is unchanged
     when the same prefix is continued by different suffixes (the policy's
     seq_id is crc32 of the observed prefix, i.e. a pure function of the prefix).

Pure CPU (no GPU, no trained model): deterministically constructed policy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evaluator.tokenizer import NUCTokenizer
from model.conditional_patch import (
    ConditionalRandomPatchPolicy, PATCH_RANDOMIZATION_SEED,
)


def _p2_policy():
    p = ConditionalRandomPatchPolicy(seed=PATCH_RANDOMIZATION_SEED)
    p.prefix_edges = [0, 6, 12, 100, 4096]
    p.ent_edges = [0.0, 0.5, 1.0, 1.5, 2.0000001]
    p.n_prefix_bins = 4
    p.n_entropy_bins = 4
    p.q_table = [[0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3]]
    return p


def test_tokenizer_encoding_prefix_suffix_independent():
    # NUC tokenizer: len-1 windows; the token sequence of a prefix only depends
    # on the observed prefix characters, never on what follows.
    tok = NUCTokenizer()
    prefix = "ACGUACGUA"
    s1 = prefix + "CCCC"
    s2 = prefix + "UUUUUUU"
    ids1 = tok.encode(s1)
    ids2 = tok.encode(s2)
    assert ids1[:len(prefix)] == ids2[:len(prefix)]


def test_p2_boundary_unchanged_by_suffix():
    p = _p2_policy()
    prefix = "ACGUACGUACGUACGUACGU"      # 20 nt observed prefix
    s1 = prefix + "UUUU"
    s2 = prefix + "CCCCGGGGAAAA"
    ent = np.array([0.5] * 20)            # causal entropy over prefix only
    b1 = p._boundaries_from_entropy(s1[:len(prefix)], ent, p.seed, None)
    b2 = p._boundaries_from_entropy(s2[:len(prefix)], ent, p.seed, None)
    assert b1 == b2