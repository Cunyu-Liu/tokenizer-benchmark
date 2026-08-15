"""TokBench-RNA canonical codec: canonical_code_length_BPN (contract 3.6).

Implements the unique cross-tokenizer headline metric as a REAL, decodable
bitstream over the frozen deterministic canonical token path:

  - RangeEncoder64 / RangeDecoder64: a frozen 64-bit integer range coder
    (encoder + independent decoder) with fixed init / renorm / final flush.
  - quantize_cdf: quantize model probabilities to integer CDF with total
    frequency 2^24, minimum frequency 1 per symbol, and the remaining mass
    allocated by largest-remainder (ties by token id).
  - canonical_code_length_BPN: total coded bits / total canonical nt over a
    continuous stream in frozen sequence order (context resets at sequence
    boundaries; sequence lengths are shared side information, not counted).
  - canonical_code_nll_BPN: ideal cumulative -log2 p(token | prefix) / nt.
  - Consistency gates (contract 3.6):
      * per continuous stream |coded_bits - quantized_cdf_nll_bits_sum| <= 64
      * |canonical_code_nll_BPN - quantized_cdf_nll_BPN| <= 1e-4 bits/nt
      * independent decoder must recover canonical RNA byte-identical
    Any fixture failure -> FAIL_CLOSED.

Shared side information (not hidden): split-manifest sequence order/length,
frozen model/tokenizer/patcher identity, coder specification. P2 additionally
fixes patch_randomization_seed + sequence ordinal; P3 additionally fixes the
entropy estimator/threshold/state.

BOS / padding / packing separators are excluded from numerator and denominator.
EOS is reported separately as EOS_NLL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

ALPHABET = tuple("ACGU")
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}

# ---------------------------------------------------------------------------
# CDF quantization (contract 3.6): total freq 2^24, min freq 1, largest remainder
# ---------------------------------------------------------------------------

CDF_TOTAL = 1 << 24
CDF_MIN_FREQ = 1


def quantize_cdf(probs: Sequence[float], total: int = CDF_TOTAL) -> list[int]:
    """Quantize model probabilities to an integer CDF (length V+1).

    Each of the V symbols receives minimum frequency 1 (so the codec can always
    represent it). The remaining (total - V) mass is distributed by the
    largest-remainder rule on model probabilities; ties broken by token id.
    cdf[0] = 0 and cdf[V] = total.
    """
    V = len(probs)
    if V <= 0:
        raise ValueError("empty probability vector")
    if total < V:
        raise ValueError("total frequency must be >= vocab size")

    # minimum frequency 1 each
    rem = total - V

    # integer parts + remainders for largest-remainder allocation
    scaled = [max(0.0, float(p)) * rem for p in probs]
    ints = [int(x) for x in scaled]
    fracs = [x - i for x, i in zip(scaled, ints)]
    used = sum(ints)
    left = rem - used

    # allocate leftover to largest fractional remainders, ties by token id
    order = sorted(range(V), key=lambda i: (-fracs[i], i))
    for i in order:
        if left <= 0:
            break
        ints[i] += 1
        left -= 1

    # build CDF (each symbol also gets its minimum freq 1)
    cdf = [0] * (V + 1)
    acc = 0
    for i in range(V):
        acc += CDF_MIN_FREQ + ints[i]
        cdf[i + 1] = acc
    # guard against any residual rounding drift (should be exact)
    assert cdf[V] == total, (cdf[V], total)
    return cdf


def cdf_symbol_bits(cdf: Sequence[int], symbol: int, total: int = CDF_TOTAL) -> float:
    """-log2 p(symbol) under the quantized CDF (bits)."""
    freq = cdf[symbol + 1] - cdf[symbol]
    return -math.log2(max(1, freq) / total)


# ---------------------------------------------------------------------------
# 64-bit range coder (frozen implementation)
# ---------------------------------------------------------------------------

# Renormalization threshold: emit one byte per shift (2^24 with a 32-bit range
# is the classic LZMA-style choice; here we keep a 64-bit low with a 32-bit
# range so that range * total never overflows 64 bits and byte emission is
# exactly once per renormalization step).
_TOP = 1 << 24
_BOT = 1 << 16
_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1


class RangeEncoder64:
    """64-bit range encoder with byte output.

    State: low (64-bit), range (32-bit). On each symbol we do
        range //= total; low += cdf_low * range; range *= (cdf_hi - cdf_low)
    and renormalize while range < _TOP, emitting the top byte of low (with
    carry propagation via a cached byte, so the bitstream is decodable and
    final flush is deterministic).
    """

    def __init__(self) -> None:
        self.low: int = 0
        self.range: int = _MASK32
        self._cache: int = 0
        self._cache_size: int = 1
        self.out: bytearray = bytearray()

    def _shift_low(self) -> None:
        if (self.low & _MASK32) < (0xFF << 24) or (self.low >> 32) != 0:
            # no carry pending: emit cache + low bytes
            temp = self._cache
            carry = (self.low >> 32) & 0xFF  # 0 or 1
            for _ in range(self._cache_size):
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
            self._cache = ((self.low & _MASK32) >> 24) & 0xFF
            self._cache_size = 1  # reset: pending bytes emitted, new cache starts
        else:
            self._cache_size += 1
        # LZMA semantics: low = (uint32_t)low << 8  -- a 32-bit wraparound shift
        # (high carry bits are consumed in the emit above, then dropped).
        self.low = (self.low << 8) & _MASK32

    def encode(self, cdf: Sequence[int], symbol: int, total: int = CDF_TOTAL) -> None:
        """Encode one symbol given the integer CDF."""
        lo = cdf[symbol]
        hi = cdf[symbol + 1]
        self.range //= total
        self.low += lo * self.range
        self.range *= (hi - lo)
        while self.range < _TOP:
            self.range = (self.range << 8) & _MASK32
            self._shift_low()

    def finish(self) -> bytes:
        """Final flush: emit the remaining state (5 bytes, LZMA convention)."""
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder64:
    """64-bit range decoder; mirrors RangeEncoder64 exactly."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.code: int = 0
        self.range: int = _MASK32
        for _ in range(5):
            self.code = ((self.code << 8) | self._read_byte()) & _MASK32

    def _read_byte(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def get_freq(self, total: int = CDF_TOTAL) -> int:
        """Return the cumulative frequency target used to select the symbol."""
        self.range //= total
        return self.code // self.range

    def decode(self, cdf: Sequence[int], symbol: int, total: int = CDF_TOTAL) -> None:
        lo = cdf[symbol]
        hi = cdf[symbol + 1]
        self.code -= lo * self.range
        self.range *= (hi - lo)
        while self.range < _TOP:
            self.code = ((self.code << 8) | self._read_byte()) & _MASK32
            self.range = (self.range << 8) & _MASK32


def cdf_find_symbol(cdf: Sequence[int], target: int) -> int:
    """Binary-search the CDF for the symbol whose interval contains `target`."""
    lo, hi = 0, len(cdf) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cdf[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# ScoreSums extension (contract Appendix A)
# ---------------------------------------------------------------------------

@dataclass
class CodecScoreSums:
    """Aggregation counters required by contract Appendix A + 3.6.

    Field names match the ScoreSums requirement (coded_bits_sum,
    quantized_cdf_nll_bits_sum, canonical_nll_bits_sum, next_base_nll_bits_sum,
    valid_nt_count, eos_nll_sum, sequence_count, attempted_count, decoded_count,
    valid_count, invalid_count, early_eos_count, truncation_count,
    overshoot_count).
    """
    coded_bits_sum: int = 0
    quantized_cdf_nll_bits_sum: float = 0.0
    canonical_nll_bits_sum: float = 0.0
    next_base_nll_bits_sum: float = 0.0
    valid_nt_count: int = 0
    eos_nll_sum: float = 0.0
    sequence_count: int = 0
    attempted_count: int = 0
    decoded_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    early_eos_count: int = 0
    truncation_count: int = 0
    overshoot_count: int = 0
    illegal_transition_mass: float = 0.0

    def add(self, other: "CodecScoreSums") -> "CodecScoreSums":
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self

    def canonical_code_length_bpn(self) -> float:
        if self.valid_nt_count <= 0:
            return float("nan")
        return self.coded_bits_sum / self.valid_nt_count

    def canonical_code_nll_bpn(self) -> float:
        if self.valid_nt_count <= 0:
            return float("nan")
        return self.canonical_nll_bits_sum / self.valid_nt_count

    def quantized_cdf_nll_bpn(self) -> float:
        if self.valid_nt_count <= 0:
            return float("nan")
        return self.quantized_cdf_nll_bits_sum / self.valid_nt_count

    def next_base_bpn(self) -> float:
        if self.valid_nt_count <= 0:
            return float("nan")
        return self.next_base_nll_bits_sum / self.valid_nt_count

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update({
            "canonical_code_length_BPN": self.canonical_code_length_bpn(),
            "canonical_code_nll_BPN": self.canonical_code_nll_bpn(),
            "quantized_cdf_nll_BPN": self.quantized_cdf_nll_bpn(),
            "next_base_BPN": self.next_base_bpn(),
            "EOS_NLL": self.eos_nll_sum,
        })
        return d


# ---------------------------------------------------------------------------
# Canonical codec scoring (contract 3.6)
# ---------------------------------------------------------------------------

class CanonicalCoder:
    """Encode a canonical token path into a decodable bitstream + ScoreSums.

    The `probs_fn` callback returns the model's full-vocab probability vector
    for a given token context. The coder quantizes it to a CDF, encodes the
    observed symbol, and mirrors the exact same sequence through the decoder to
    prove byte-identical recovery (contract 3.6 decoder gate).
    """

    def __init__(self, vocab_size: int, total: int = CDF_TOTAL,
                 prefix_batch: Callable[[Sequence[int]], Sequence[float]] | None = None):
        self.vocab_size = vocab_size
        self.total = total
        self.prefix_batch = prefix_batch  # optional batched log-prob provider

    # --- full-stream encode (one continuous stream in frozen sequence order) --
    def encode_stream(self, sequences: Sequence[Sequence[int]],
                      probs_fn: Callable[[Sequence[int]], Sequence[float]]
                      ) -> tuple[bytes, CodecScoreSums]:
        """Encode all sequences into ONE continuous coder stream.

        Context resets at sequence boundaries. Returns (bitstream, sums).
        Each sequence is decoded independently afterwards for the decode gate.
        """
        enc = RangeEncoder64()
        sums = CodecScoreSums()
        nll_sum = 0.0
        for tok_ids in sequences:
            for i, tid in enumerate(tok_ids):
                ctx = tok_ids[:i]
                probs = probs_fn(ctx)
                assert len(probs) == self.vocab_size, (len(probs), self.vocab_size)
                cdf = quantize_cdf(probs, self.total)
                enc.encode(cdf, tid, self.total)
                # canonical NLL (ideal, unquantized)
                p = probs[tid]
                if p > 0:
                    nll_sum += -math.log2(p)
                sums.quantized_cdf_nll_bits_sum += cdf_symbol_bits(cdf, tid, self.total)
            sums.sequence_count += 1
        sums.canonical_nll_bits_sum = nll_sum
        bitstream = enc.finish()
        sums.coded_bits_sum = len(bitstream) * 8
        return bitstream, sums

    def decode_sequence(self, bitstream: bytes, tok_ids: Sequence[int],
                        probs_fn: Callable[[Sequence[int]], Sequence[float]]
                        ) -> None:
        """Decode the exact same token path from `bitstream`.

        The bitstream encodes all sequences in order; to decode sequence k we
        must decode all preceding sequences first. Caller passes the token ids
        of every sequence in order; we verify each recovered symbol equals the
        expected one (byte-identical canonical RNA recovery).
        """
        dec = RangeDecoder64(bitstream)
        for seq in tok_ids:
            for tid in seq:
                ctx = list(seq[:0])  # not used; decode is sequential
        raise NotImplementedError(
            "use decode_stream (needs the full ordered list of sequences)")


class CanonicalStreamCodec:
    """One continuous bitstream over an ordered list of sequences, with a
    decodable verify pass that recovers every canonical token path exactly.

    `sequences` is a list of canonical token-id paths; `nt_counts` (optional)
    maps each sequence to its canonical nucleotide count (k-mer: token* k +
    tail length). When omitted, valid_nt_count = sum of token lengths (used by
    unit fixtures only); the real evaluator always passes nt_counts.
    """

    def __init__(self, vocab_size: int, total: int = CDF_TOTAL):
        self.vocab_size = vocab_size
        self.total = total

    def encode(self, sequences: Sequence[Sequence[int]],
               probs_fn: Callable[[Sequence[int]], Sequence[float]],
               nt_counts: Sequence[int] | None = None
               ) -> tuple[bytes, CodecScoreSums]:
        enc = RangeEncoder64()
        sums = CodecScoreSums()
        nll = 0.0
        for si, tok_ids in enumerate(sequences):
            for i, tid in enumerate(tok_ids):
                probs = probs_fn(tok_ids[:i])
                cdf = quantize_cdf(probs, self.total)
                enc.encode(cdf, tid, self.total)
                if probs[tid] > 0:
                    nll += -math.log2(probs[tid])
                sums.quantized_cdf_nll_bits_sum += cdf_symbol_bits(cdf, tid, self.total)
            sums.sequence_count += 1
            if nt_counts is not None:
                sums.valid_nt_count += nt_counts[si]
            else:
                sums.valid_nt_count += len(tok_ids)
        sums.canonical_nll_bits_sum = nll
        bitstream = enc.finish()
        sums.coded_bits_sum = len(bitstream) * 8
        return bitstream, sums

    def decode_verify(self, bitstream: bytes, sequences: Sequence[Sequence[int]],
                      probs_fn: Callable[[Sequence[int]], Sequence[float]]
                      ) -> bool:
        """Independent decoder: recover every canonical token path from the
        single bitstream and check it matches the source byte-identically."""
        dec = RangeDecoder64(bitstream)
        for tok_ids in sequences:
            for i, tid in enumerate(tok_ids):
                probs = probs_fn(tok_ids[:i])
                cdf = quantize_cdf(probs, self.total)
                tgt = dec.get_freq(self.total)
                sym = cdf_find_symbol(cdf, tgt)
                dec.decode(cdf, sym, self.total)
                if sym != tid:
                    return False
        return True


def check_codec_consistency(bitstream: bytes, sums: CodecScoreSums,
                            tol_bits: int = 64, tol_nll: float = 1e-4,
                            decoded_ok: bool = True) -> bool:
    """Contract 3.6 two-level consistency gate.

    Returns True if ALL pass, else raises/returns False (FAIL_CLOSED):
      1. |coded_bits - quantized_cdf_nll_bits_sum| <= 64 bits per stream
      2. |canonical_code_nll_BPN - quantized_cdf_nll_BPN| <= 1e-4 bits/nt
      3. independent decoder recovered canonical RNA byte-identical
    """
    gate1 = abs(sums.coded_bits_sum - sums.quantized_cdf_nll_bits_sum) <= tol_bits
    nll_bpn = sums.canonical_code_nll_bpn()
    q_bpn = sums.quantized_cdf_nll_bpn()
    gate2 = (nll_bpn == nll_bpn and q_bpn == q_bpn and
             abs(nll_bpn - q_bpn) <= tol_nll)
    gate3 = decoded_ok
    return bool(gate1 and gate2 and gate3)


def canonicalize_seq(seq: str) -> str:
    s = seq.upper().replace("T", "U")
    for ch in s:
        if ch not in BASE_TO_IDX:
            raise ValueError(f"non-primary IUPAC char {ch!r} in {seq!r}")
    return s
