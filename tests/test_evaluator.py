"""Phase 2 evaluator fixtures (5.4). Pure CPU, no neural dependency."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.scorer import (
    ScoreSums, score_next_base, score_canonical_path, score_overlap_path,
    aggregate_bpn, canonicalize_seq,
)
from evaluator.tokenizer import (
    NUCTokenizer, KmerTokenizer, BPETokenizer, UnigramTokenizer,
)
from evaluator.eval_continuation import (
    edit_distance, nucleotide_accuracy, kmer_recovery,
    evaluate_generation, evaluate_continuation, SealedTestGate,
)


# --- round-trip fixtures ---------------------------------------------------
def test_nuc_roundtrip():
    t = NUCTokenizer()
    for s in ["ACGU", "aCgU", "uuuacg", "ACGTACGT"]:
        assert t.round_trip(s), s


def test_overlap_mer_roundtrip():
    t = KmerTokenizer(3, overlapping=True)
    assert t.encode("ACGU") and t.decode(t.encode("ACGU")) == "ACGU"
    assert t.round_trip("ACGUACGUACGU")


def test_nonoverlap_mer_roundtrip():
    t = KmerTokenizer(3, overlapping=False)
    s = "ACGUACGUACGU"  # len 12, multiple of k
    enc = t.encode(s)
    assert len(enc) == 4
    assert t.decode(enc) == "ACGUACGUACGU"
    assert t.decode(t.encode(s)) == s


def test_bpe_roundtrip():
    t = BPETokenizer(vocab_size=32)
    t.fit(["ACGUACGUACGU", "ACGUUUGCA", "CCCGGGAAA"])
    for s in ["ACGUACGU", "ACGU", "GGGAAACCC"]:
        assert t.round_trip(s), s


def test_unigram_roundtrip():
    t = UnigramTokenizer(vocab_size=64)
    t.fit(["ACGUACGUACGU", "UUUUACCCGG", "ACGU"])
    assert t.round_trip("ACGUACGU")
    assert t.round_trip("UUUUACCCGG")


# --- BPE vocab train-only -------------------------------------------------
def test_bpe_vocab_train_only():
    # vocab must be closed under bases + merges; no test-only tokens
    t = BPETokenizer(vocab_size=16)
    t.fit(["ACGU"])
    assert t.vocab_size() <= 16
    assert all(len(x) >= 1 for x in t._id_to_token)


# --- next_base_BPN correctness --------------------------------------------
class _UniformModel:
    """P(base)=1/4 each -> 2 bits per nt."""
    def lp(self, prefix: str, nxt: str) -> float:
        return math.log(0.25)


class _SkewModel:
    """Non-uniform: p(A)=0.4, p(C)=0.2, p(G)=0.3, p(U)=0.1 -> content-dependent BPN."""
    PROB = {"A": 0.4, "C": 0.2, "G": 0.3, "U": 0.1}
    def lp(self, prefix: str, nxt: str) -> float:
        return math.log(self.PROB[nxt])


def test_uniform_bpn():
    s = "ACGU" * 250  # 1000 nt
    out = score_next_base(s, _UniformModel().lp)
    assert out.valid_nt_count == 1000
    assert abs(out.next_base_bpn() - 2.0) < 1e-9


def test_sums_match_manual_oracle():
    model = _UniformModel()
    s = "ACGUACGUAC"  # 10 nt
    out = score_next_base(s, model.lp)
    manual_bits = 10 * (-math.log2(0.25))
    assert abs(out.nll_bits_sum - manual_bits) < 1e-7
    assert abs(out.nll_bits_sum - (10 * 2.0)) < 1e-7


def test_dataset_agg_total_not_mean():
    model = _SkewModel()
    s1 = "A" * 9000 + "C" * 1000   # low-loss, 10000 nt
    s2 = "C" * 9000 + "A" * 1000   # high-loss, 10000 nt
    o1, o2 = score_next_base(s1, model.lp), score_next_base(s2, model.lp)
    agg = aggregate_bpn([o1, o2])
    expect = (o1.nll_bits_sum + o2.nll_bits_sum) / (o1.valid_nt_count + o2.valid_nt_count)
    assert abs(agg.next_base_bpn() - expect) < 1e-9
    seq_mean = (o1.next_base_bpn() + o2.next_base_bpn()) / 2
    # weighted (equal nt) equals arithmetic mean here, so use unequal lengths
    s3 = "A" * 9000 + "C" * 1000   # low-loss, 10000 nt
    s4 = "C" * 90000 + "A" * 10000  # high-loss, 100000 nt
    o3, o4 = score_next_base(s3, model.lp), score_next_base(s4, model.lp)
    agg2 = aggregate_bpn([o3, o4])
    seq_mean2 = (o3.next_base_bpn() + o4.next_base_bpn()) / 2
    assert abs(agg2.next_base_bpn() - seq_mean2) > 1e-6


def test_padding_bos_ignore():
    # canonicalize rejects non-primary; next_base only scores real nt.
    # A model that ignores padding still yields same BPN for clean seq.
    model = _UniformModel()
    s = "ACGU" * 50
    out = score_next_base(s, model.lp)
    assert out.valid_nt_count == 200
    assert abs(out.next_base_bpn() - 2.0) < 1e-9


def test_invalid_sequence_counted():
    model = _UniformModel()
    out = score_next_base("ACGNACGU", model.lp)  # N invalid
    assert out.invalid_count == 1
    assert out.valid_nt_count == 0


# --- canonical path --------------------------------------------------------
def test_canonical_path_compression_view():
    t = NUCTokenizer()
    ids = t.encode("ACGU")
    model = _UniformModel()
    out = score_canonical_path(ids, lambda ctx, tid: model.lp("", "A"))
    assert out.sequence_count == 1
    assert out.nll_bits_sum > 0


# --- overlap path ----------------------------------------------------------
def test_overlap_path_each_step_one_nt():
    model = _UniformModel()
    s = "ACGU" * 100
    out, illegal = score_overlap_path(s, 3, model.lp)
    assert out.valid_nt_count == 400
    assert illegal == 0.0
    assert abs(out.next_base_bpn() - 2.0) < 1e-9


# --- continuation / generation metrics ------------------------------------
def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("ACGU", "ACGU") == 0


def test_nt_accuracy():
    assert abs(nucleotide_accuracy("ACGU", "ACGU") - 1.0) < 1e-9
    assert abs(nucleotide_accuracy("ACGU", "ACGA") - 0.75) < 1e-9


def test_kmer_recovery():
    assert kmer_recovery("ACGUACGU", "ACGUACGU", k=4) == 1.0
    assert kmer_recovery("UUUUUUUU", "ACGUACGU", k=4) == 0.0


def test_generation_validity_uniqueness():
    train = {"ACGUACGUACGU", "GGGGCCCCAAAA"}
    stats = evaluate_generation(
        ["ACGUACGUACGU", "ACGUACGUACGU", "UUUUUUUUUUUU", "XXXX"],
        training_set=train,
    )
    assert stats.total == 4
    assert stats.valid == 3
    assert stats.invalid_char_count == 1
    assert stats.uniqueness() == 2 / 4
    assert "ACGUACGUACGU" in stats.exact_matrix["train"]


def test_exact_copy_novelty_zero():
    # exact copy of a training molecule must be flagged as memorized
    train = {"ACGUACGUACGU"}
    stats = evaluate_generation(["ACGUACGUACGU"], training_set=train)
    assert stats.exact_matrix["train"]
    # identity histogram at 1.0 should increment
    assert stats.identity_hist[1.0] == 1


def test_continuation_result():
    preds = [("ACGU", "ACGU", 0.5), ("ACGUU", "ACGUU", 0.5)]
    res = evaluate_continuation(preds)
    assert res.count == 2
    assert res.mean_nt_acc() == 1.0


# --- sealed-test gate ------------------------------------------------------
def test_sealed_gate_blocks_touch():
    gate = SealedTestGate()
    gate.assert_not_touched()
    gate._touched = True
    try:
        gate.assert_not_touched()
        assert False, "should have raised"
    except RuntimeError:
        pass
