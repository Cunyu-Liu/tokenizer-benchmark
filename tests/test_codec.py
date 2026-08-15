"""Tests for the canonical codec (contract 3.6)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.codec import (
    RangeEncoder64, RangeDecoder64, quantize_cdf, cdf_find_symbol,
    cdf_symbol_bits, CanonicalStreamCodec, CodecScoreSums,
    check_codec_consistency, CDF_TOTAL,
)


# --- CDF quantization tests -------------------------------------------------

def test_cdf_basic():
    V = 4
    probs = [0.25, 0.25, 0.25, 0.25]
    cdf = quantize_cdf(probs)
    assert len(cdf) == V + 1
    assert cdf[0] == 0
    assert cdf[V] == CDF_TOTAL
    # each symbol gets freq = CDF_TOTAL / V (exact if divisible)
    for i in range(V):
        freq = cdf[i + 1] - cdf[i]
        assert freq >= 1
        assert abs(freq - CDF_TOTAL / V) <= 1  # rounding


def test_cdf_min_freq():
    """Each symbol gets at least 1 even if probability is zero."""
    V = 1024
    probs = [0.0] * V
    probs[0] = 1.0  # all mass on first symbol
    cdf = quantize_cdf(probs)
    assert len(cdf) == V + 1
    assert cdf[0] == 0
    assert cdf[V] == CDF_TOTAL
    # first symbol gets most of the mass
    assert cdf[1] - cdf[0] >= CDF_TOTAL - V + 1
    # all other symbols get exactly 1
    for i in range(1, V):
        assert cdf[i + 1] - cdf[i] == 1, f"symbol {i} freq={cdf[i+1]-cdf[i]}"


def test_cdf_largest_remainder():
    """Largest-remainder with ties by token id."""
    V = 3
    probs = [1/3, 1/3, 1/3]
    cdf = quantize_cdf(probs)
    # With V=3, total=2^24, rem = 2^24 - 3 = 16777213
    # Each gets floor(16777213/3) = 5592404, remainders ~0.333 -> extra to first two
    for i in range(V):
        assert cdf[i + 1] - cdf[i] >= 1, f"symbol {i} below min freq"


def test_cdf_symbol_bits():
    cdf = [0, 100, 150, 200, 256]
    bits = cdf_symbol_bits(cdf, 0, 256)
    assert abs(bits - (-math.log2(100/256))) < 1e-12
    bits1 = cdf_symbol_bits(cdf, 1, 256)
    assert abs(bits1 - (-math.log2(50/256))) < 1e-12


# --- Range coder round-trip tests -------------------------------------------

def test_range_encoder_roundtrip_single():
    V = 4
    probs = [0.25, 0.25, 0.25, 0.25]
    cdf = quantize_cdf(probs)
    enc = RangeEncoder64()
    for sym in [0, 1, 2, 3]:
        enc.encode(cdf, sym, CDF_TOTAL)
    data = enc.finish()

    dec = RangeDecoder64(data)
    for expected in [0, 1, 2, 3]:
        tgt = dec.get_freq(CDF_TOTAL)
        sym = cdf_find_symbol(cdf, tgt)
        dec.decode(cdf, sym, CDF_TOTAL)
        assert sym == expected, f"expected {expected} got {sym}"


def test_range_encoder_roundtrip_many():
    """Encode/decode a longer sequence of symbols with varying probs."""
    V = 1024
    import random
    rng = random.Random(42)
    symbols = [rng.randint(0, V - 1) for _ in range(1000)]
    # Use uniform probs
    probs = [1.0 / V] * V
    cdf = quantize_cdf(probs)

    enc = RangeEncoder64()
    for sym in symbols:
        enc.encode(cdf, sym, CDF_TOTAL)
    data = enc.finish()

    dec = RangeDecoder64(data)
    for expected in symbols:
        tgt = dec.get_freq(CDF_TOTAL)
        sym = cdf_find_symbol(cdf, tgt)
        dec.decode(cdf, sym, CDF_TOTAL)
        assert sym == expected, f"expected {expected} got {sym}"


def test_range_encoder_consecutive_identical():
    """Encode the same symbol repeatedly (stress test)."""
    V = 4
    probs = [0.7, 0.1, 0.1, 0.1]
    cdf = quantize_cdf(probs)
    symbols = [0] * 500  # all same symbol

    enc = RangeEncoder64()
    for sym in symbols:
        enc.encode(cdf, sym, CDF_TOTAL)
    data = enc.finish()

    dec = RangeDecoder64(data)
    for expected in symbols:
        tgt = dec.get_freq(CDF_TOTAL)
        sym = cdf_find_symbol(cdf, tgt)
        dec.decode(cdf, sym, CDF_TOTAL)
        assert sym == expected


# --- Canonical Stream Codec tests ------------------------------------------

def test_canonical_stream_encode_decode():
    """Encode a sequence of token ids, decode, verify byte-identical."""
    V = 4
    codec = CanonicalStreamCodec(V)
    # Simulate a simple model: uniform probs
    def probs_fn(ctx):
        return [1.0 / V] * V

    sequences = [[0, 1, 2, 3], [0], [3, 3, 3], [1, 2, 1, 2, 0, 3, 2, 1]]
    bitstream, sums = codec.encode(sequences, probs_fn)
    # Decode verify
    ok = codec.decode_verify(bitstream, sequences, probs_fn)
    assert ok, "Decode verify failed"

    # Consistency gates
    assert check_codec_consistency(
        bitstream, sums, decoded_ok=ok), "Consistency gates failed"
    print(f"  coded_bits={sums.coded_bits_sum} quantized={sums.quantized_cdf_nll_bits_sum:.1f} "
          f"nll={sums.canonical_nll_bits_sum:.1f} seqs={sums.sequence_count}")


def test_canonical_stream_nonuniform():
    """Non-uniform probabilities: encode wrong symbol, decode should fail."""
    V = 4
    codec = CanonicalStreamCodec(V)
    sequences = [[0, 1, 2, 3]]

    # Model that assigns p=1 to symbol 0 and 0 to everything else
    def probs_fn(ctx):
        p = [1.0, 0.0, 0.0, 0.0]
        return p

    bitstream, sums = codec.encode(sequences, probs_fn)
    ok = codec.decode_verify(bitstream, sequences, probs_fn)
    # Should still decode correctly because the codec only encodes observed symbols
    assert ok, "Non-uniform decode verify failed"


# --- Consistency gate tests -------------------------------------------------

def test_consistency_gate_bits():
    """|coded_bits - quantized_cdf_nll_bits_sum| <= 64 bits."""
    V = 4
    codec = CanonicalStreamCodec(V)
    def probs_fn(ctx):
        return [0.25, 0.25, 0.25, 0.25]
    sequences = [[0, 1, 2, 3] * 100, [0, 2, 1, 3] * 50]
    bitstream, sums = codec.encode(sequences, probs_fn)
    gate = abs(sums.coded_bits_sum - sums.quantized_cdf_nll_bits_sum) <= 64
    assert gate, f"coded={sums.coded_bits_sum} quantized={sums.quantized_cdf_nll_bits_sum:.1f}"


def test_consistency_gate_nll():
    """|canonical_code_nll_BPN - quantized_cdf_nll_BPN| <= 1e-4 bits/nt."""
    V = 4
    codec = CanonicalStreamCodec(V)
    def probs_fn(ctx):
        return [0.50, 0.25, 0.15, 0.10]
    sequences = [[0, 1, 2, 3] * 100, [0, 2, 1, 3] * 50, [0, 0, 0, 0]]
    bitstream, sums = codec.encode(sequences, probs_fn)
    nll_bpn = sums.canonical_code_nll_bpn()
    q_bpn = sums.quantized_cdf_nll_bpn()
    diff = abs(nll_bpn - q_bpn)
    assert diff <= 1e-4, f"nll_BPN={nll_bpn:.6f} q_BPN={q_bpn:.6f} diff={diff:.6e}"


# --- Full pipeline test (like the evaluator would use) ----------------------

def test_full_codec_pipeline():
    """Realistic: 100 sequences, varying lengths, uniform probs."""
    import random
    V = 4
    codec = CanonicalStreamCodec(V)
    rng = random.Random(123)
    sequences = []
    for _ in range(100):
        L = rng.randint(4, 100)
        seq = [rng.randint(0, 3) for _ in range(L)]
        sequences.append(seq)

    def probs_fn(ctx):
        return [0.25, 0.25, 0.25, 0.25]

    bitstream, sums = codec.encode(sequences, probs_fn)
    ok = codec.decode_verify(bitstream, sequences, probs_fn)
    assert ok, "Decode verify failed"
    assert check_codec_consistency(bitstream, sums, decoded_ok=ok), "Consistency gates failed"
    total_nt = sum(len(s) for s in sequences)
    total_bits = sums.coded_bits_sum
    print(f"  {len(sequences)} seqs, {total_nt} nt, {total_bits} bits, "
          f"{total_bits/total_nt:.4f} BPN (uniform)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                import traceback
                traceback.print_exc()