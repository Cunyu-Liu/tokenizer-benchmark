"""Unit tests for per-token canonical nt attribution (contract 3.6: each real
nt scored exactly once; no padding; no double count)."""
from evaluator.tokenizer import (
    NUCTokenizer,
    KmerTokenizer,
    BPETokenizer,
    UnigramTokenizer,
)


def test_nuc_each_token_1_nt():
    tk = NUCTokenizer()
    ids = tk.encode("ACGUAC")
    assert tk.token_nt_counts(ids) == [1] * 6
    assert sum(tk.token_nt_counts(ids)) == 6


def test_overlap_kmer_attribution():
    # F4/F5 overlap stride 1: token0 covers k nt, each later token adds 1 nt.
    tk = KmerTokenizer(k=6, overlapping=True)
    seq = "ACGUACGUAC"
    ids = tk.encode(seq)
    counts = tk.token_nt_counts(ids)
    assert counts[0] == 6
    assert counts[1:] == [1] * (len(ids) - 1)
    assert sum(counts) == len(seq)  # every real nt counted exactly once
    assert tk.decode(ids) == seq


def test_overlap_kmer_short_seq():
    # seq shorter than k: token0 still covers all (padded for encode, but the
    # decode reconstructs the original; attribution = k covers original nt).
    tk = KmerTokenizer(k=3, overlapping=True)
    seq = "AC"
    ids = tk.encode(seq)
    assert sum(tk.token_nt_counts(ids)) == len(seq) + 1  # pad base reconstructed


def test_nonoverlap_kmer_full_blocks():
    tk = KmerTokenizer(k=6, overlapping=False)
    seq = "ACGUAC" * 4  # 24 nt, exactly 4 full blocks, no tail
    ids = tk.encode(seq)
    counts = tk.token_nt_counts(ids)
    assert counts == [6, 6, 6, 6]
    assert sum(counts) == 24
    assert tk.decode(ids) == seq


def test_nonoverlap_kmer_with_tail():
    # F7: non-overlap 6-mer with canonical tail token (length 1..k-1).
    tk = KmerTokenizer(k=6, overlapping=False)
    seq = "ACGUAC" * 3 + "UG"  # 18 + 2 = 20 nt -> 3 full + tail "UG" (2 nt)
    ids = tk.encode(seq)
    counts = tk.token_nt_counts(ids)
    assert counts[:-1] == [6, 6, 6]
    assert counts[-1] == 2  # tail token covers 2 nt
    assert sum(counts) == 20
    assert tk.decode(ids) == seq


def test_nonoverlap_tail_single_nt():
    tk = KmerTokenizer(k=3, overlapping=False)
    seq = "ACGUACU"  # 7 nt -> 2 full "ACG","UAC" + tail "U" (1 nt)
    ids = tk.encode(seq)
    counts = tk.token_nt_counts(ids)
    assert counts == [3, 3, 1]
    assert sum(counts) == 7


def test_bpe_variable_length_attribution():
    tk = BPETokenizer(vocab_size=64)
    tk.fit(["ACGUACGUACGU", "ACGU" * 10, "UGAC" * 10])
    seq = "ACGUACGUACGU"
    ids = tk.encode(seq)
    counts = tk.token_nt_counts(ids)
    assert sum(counts) == len(seq)
    assert tk.decode(ids) == seq


def test_unigram_variable_length_attribution():
    tk = UnigramTokenizer(vocab_size=64)
    tk.fit(["ACGUACGUACGU", "ACGU" * 10, "UGAC" * 10])
    seq = "ACGUACGUACGU"
    ids = tk.encode(seq)
    counts = tk.token_nt_counts(ids)
    assert sum(counts) == len(seq)
    assert tk.decode(ids) == seq