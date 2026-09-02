"""Per-arm training data pipeline (contract 3.1, 3.2, 3.4).

- Builds the train-only tokenizer for each arm (NUC/BPE/Unigram/k-mer).
- Streams train sequences from the frozen split parquet via pyarrow (row
  groups), so the 29M-sequence table is never fully materialized.
- Tracks effective nucleotide exposure in cumulative valid target nt.
- BLT arms run at nt level (vocab 4) with a boundary provider (patch.py).

Exposure rule: padding/BOS/separator/overlap windows do NOT add nt exposure.
Only real target positions count toward budget_nt.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from .arms import ArmSpec

ALPHABET = "ACGU"
IGNORE = -100  # PyTorch cross_entropy ignore_index


@dataclass
class Exposure:
    cumulative_valid_target_nt: int = 0
    batches: int = 0
    sequences: int = 0

    def add_batch(self, valid_nt: int, n_seq: int):
        self.cumulative_valid_target_nt += valid_nt
        self.batches += 1
        self.sequences += n_seq


def _canon(s: str) -> str:
    return s.upper().replace("T", "U")


def _seq_id(canon: str) -> int:
    """Stable per-sequence id for sequence-specific random patch (contract 3.2).

    A deterministic 32-bit hash of the canonical sequence. Using content (not a
    running index) guarantees:
      - train / eval see the identical random boundary pattern for the same
        sequence (no train/eval mismatch),
      - prefix consistency (the boundary sampled from position 0 is unchanged
        when the sequence is truncated to any prefix).
    """
    return zlib.crc32(canon.encode("utf-8"))


def sample_train_sequences(path: str, n: int, seed: int, split: str = "train"):
    """Deterministic sample of train sequences for tokenizer vocab building."""
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    pf = pq.ParquetFile(path)
    collected: list[str] = []
    for batch in pf.iter_batches(batch_size=50_000, columns=["split_membership", "canonical_sequence"]):
        d = batch.to_pydict()
        for split_v, seq in zip(d["split_membership"], d["canonical_sequence"]):
            if split_v == split:
                collected.append(seq)
                if len(collected) >= n:
                    rng.shuffle(collected)
                    return collected
    rng.shuffle(collected)
    return collected


def build_tokenizer(arm: ArmSpec, train_seqs: list[str]):
    """Construct + fit the per-arm tokenizer on train sequences only."""
    from evaluator.tokenizer import (
        NUCTokenizer, KmerTokenizer, BPETokenizer, UnigramTokenizer,
    )

    t = arm.tokenizer_type
    if t == "NUC":
        return NUCTokenizer()
    if t == "BPE":
        tok = BPETokenizer(vocab_size=arm.vocab_size)
        tok.fit(train_seqs)
        return tok
    if t == "Unigram":
        tok = UnigramTokenizer(vocab_size=arm.vocab_size)
        tok.fit(train_seqs)
        return tok
    if t == "overlap_mer":
        return KmerTokenizer(k=arm.k, overlapping=True)
    if t == "nonoverlap_mer":
        return KmerTokenizer(k=arm.k, overlapping=False)
    if t in ("fixed_patch", "random_patch", "entropy_patch"):
        # BLT arms run at nt level; patch boundary handled separately.
        return NUCTokenizer()
    raise ValueError("unknown tokenizer_type %r" % t)


# Deterministic per-arm tokenizer, keyed by arm id. BPE/Unigram vocab is fit on
# a fixed train-only sample (same seed) so it is identical across train/val and
# across seeds. Cached because the training loop is entered once per arm.
_arm_tok_cache: dict = {}
# Deterministic train-only sample for BPE/Unigram vocab building. 20k sequences
# is ample for a 1024-vocab tokenizer on a 4-letter alphabet and keeps the fit
# fast; the exact count is recorded in the tokenizer_spec.
_VOCAB_TRAIN_N = 5_000
_VOCAB_TRAIN_SEED = 17
# Cap per-sequence length for BPE/Unigram vocab building. A 4-letter alphabet
# saturates k-mer statistics quickly; 128 nt/sample retains ample coverage for a
# 1024-vocab tokenizer while keeping the fit fast and deterministic. The cap is
# recorded in the tokenizer_spec.
_VOCAB_MAX_LEN = 128


def build_arm_tokenizer(path: str, arm: ArmSpec):
    """Build the frozen, train-only tokenizer for an arm (deterministic).

    Deterministic (NUC / k-mer / BLT) tokenizers need no fit. BPE/Unigram are
    fit on a fixed sample of train sequences only (contract 3.1: vocab built
    from train split only). Result is cached by arm id so train and validation
    iterators share the identical tokenizer.
    """
    cached = _arm_tok_cache.get(arm.id)
    if cached is not None:
        return cached
    if arm.tokenizer_type in ("BPE", "Unigram"):
        sample = sample_train_sequences(path, n=_VOCAB_TRAIN_N,
                                        seed=_VOCAB_TRAIN_SEED, split="train")
        sample = [seq[:_VOCAB_MAX_LEN] for seq in sample]
        tok = build_tokenizer(arm, sample)
    else:
        tok = build_tokenizer(arm, [])
    _arm_tok_cache[arm.id] = tok
    return tok


def _window(seq: str, context_nt: int, tok) -> tuple[list[int], list[int], list[int]]:
    """Return (input_ids, targets) for one context window of a sequence."""
    ids = tok.encode(seq)
    if len(ids) <= 1:
        return [], [], []
    ctx = ids[:-1]
    tgt = ids[1:]
    if len(ctx) > context_nt:
        ctx = ctx[-context_nt:]
        tgt = tgt[-context_nt:]
    w = tok.token_nt_counts(tgt)
    return ctx, tgt, w


def iter_train_batches(path: str, cfg, boundary_provider=None,
                       batch_size: int = 32, max_batches: int | None = None,
                       seed: int = 0, split: str = "train", batch_nt: int | None = None,
                       tok=None):
    """Stream train batches via pyarrow row groups. Yields dict batches.

    flat: {"token_ids": (B,T), "targets": (B,T)}
    blt : {"token_ids": (B,T) nt, "boundary": (B,T), "targets": (B,T)}

    Batching is bounded in effective nt so a long (4096-nt) sequence does not
    pad an entire fixed-size batch and OOM the quadratic self-attention:
      - `batch_nt` set  -> adaptive nt-budgeted batching (n * max_len <= batch_nt)
      - `batch_nt` None -> fixed `batch_size` sequences per batch

    `tok` is the per-arm train-only tokenizer (see build_arm_tokenizer). When
    None it is built deterministically from a train-only sample.
    """
    import pyarrow.parquet as pq

    tok = tok if tok is not None else build_arm_tokenizer(path, cfg.arm)
    context_nt = cfg.context_nt
    pf = pq.ParquetFile(path)

    batch_tok, batch_ign, batch_bndry, batch_seqcnt, batch_w = [], [], [], [], []
    cur_max = 0
    batches = 0

    def flush():
        if not batch_tok:
            return None
        T = max(len(x) for x in batch_tok)
        tok_ids = [_pad(x, T) for x in batch_tok]
        ign = [_pad(x, T, ignore=True) for x in batch_ign]
        out = {"token_ids": tok_ids, "targets": ign,
               "nt_weights": [_pad(x, T) for x in batch_w],
               "_seq_count": list(batch_seqcnt)}
        if boundary_provider is not None:
            out["boundary"] = [_pad(b, T) for b in batch_bndry]
        return out

    def would_overflow(L: int) -> bool:
        if batch_nt is None:
            return len(batch_tok) >= batch_size
        if not batch_tok:
            return False
        return (len(batch_tok) + 1) * max(cur_max, L) > batch_nt

    for rb in pf.iter_batches(batch_size=50_000, columns=["split_membership", "canonical_sequence"]):
        d = rb.to_pydict()
        for split_v, seq in zip(d["split_membership"], d["canonical_sequence"]):
            if split_v != split:
                continue
            if cfg.arm.backbone == "blt":
                c = _canon(seq)
                if len(c) <= 1:
                    continue
                c = c[-context_nt:]
                nt_ids = [ALPHABET.index(b) for b in c]
                ctx = nt_ids[:-1]
                tgt = nt_ids[1:]
                L = len(ctx)
                if would_overflow(L):
                    b = flush()
                    if b is not None:
                        yield b
                        batches += 1
                        if max_batches is not None and batches >= max_batches:
                            return
                    batch_tok, batch_ign, batch_bndry, batch_seqcnt, batch_w = (
                        [], [], [], [], [])
                    cur_max = 0
                batch_tok.append(ctx)
                batch_ign.append(tgt)
                batch_w.append([1] * len(tgt))
                # P3 entropy computes boundaries on GPU from the nt batch, so
                # the dataset emits no per-sequence boundary when provider is
                # None (training loop derives them online).
                if boundary_provider is not None:
                    # Contract 3.2: P2 random patch boundaries must be decided by
                    # (sequence_id + seed) with prefix consistency. Use a stable
                    # per-sequence content hash so train and eval (and any prefix)
                    # reproduce the identical boundary pattern. Fixed/entropy
                    # policies ignore seq_id.
                    batch_bndry.append(
                        boundary_provider.boundary(c, len(c), _seq_id(c))[:-1])
            else:
                cc, tt, ww = _window(seq, context_nt, tok)
                if not cc:
                    continue
                L = len(cc)
                if would_overflow(L):
                    b = flush()
                    if b is not None:
                        yield b
                        batches += 1
                        if max_batches is not None and batches >= max_batches:
                            return
                    batch_tok, batch_ign, batch_bndry, batch_seqcnt, batch_w = (
                        [], [], [], [], [])
                    cur_max = 0
                batch_tok.append(cc)
                batch_ign.append(tt)
                batch_w.append(ww)
            cur_max = max(cur_max, L)
            batch_seqcnt.append(1)
    b = flush()
    if b is not None:
        yield b


def _pad(ids: list[int], length: int, ignore: bool = False) -> list[int]:
    pad = IGNORE if ignore else 0
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad] * (length - len(ids))


def count_valid_nt(batch) -> int:
    """Real (non-ignore) target NUCLEOTIDES in a batch (amendment 2026-09-02).

    Uses per-target nt weights when present (each real nt counted exactly
    once, regardless of how many nt a token covers); falls back to counting
    target positions for legacy batches without weights.
    """
    if "nt_weights" in batch:
        return int(sum(sum(w for w, v in zip(rw, rt) if v != IGNORE)
                       for rw, rt in zip(batch["nt_weights"], batch["targets"])))
    return sum(1 for row in batch["targets"] for v in row if v != IGNORE)
