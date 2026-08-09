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


def _window(seq: str, context_nt: int, tok) -> tuple[list[int], list[int]]:
    """Return (input_ids, targets) for one context window of a sequence."""
    ids = tok.encode(seq)
    if len(ids) <= 1:
        return [], []
    ctx = ids[:-1]
    tgt = ids[1:]
    if len(ctx) > context_nt:
        ctx = ctx[-context_nt:]
        tgt = tgt[-context_nt:]
    return ctx, tgt


def iter_train_batches(path: str, cfg, boundary_provider=None,
                       batch_size: int = 32, max_batches: int | None = None,
                       seed: int = 0, split: str = "train"):
    """Stream train batches via pyarrow row groups. Yields dict batches.

    flat: {"token_ids": (B,T), "targets": (B,T)}
    blt : {"token_ids": (B,T) nt, "boundary": (B,T), "targets": (B,T)}
    """
    import pyarrow.parquet as pq

    tok = build_tokenizer(cfg.arm, [])
    context_nt = cfg.context_nt
    pf = pq.ParquetFile(path)

    batch_tok, batch_ign, batch_bndry, batch_seqcnt = [], [], [], []
    batches = 0

    def flush():
        if not batch_tok:
            return None
        T = max(len(x) for x in batch_tok)
        tok_ids = [_pad(x, T) for x in batch_tok]
        ign = [_pad(x, T, ignore=True) for x in batch_ign]
        out = {"token_ids": tok_ids, "targets": ign, "_seq_count": list(batch_seqcnt)}
        if boundary_provider is not None:
            out["boundary"] = [_pad(b, T) for b in batch_bndry]
        return out

    for rb in pf.iter_batches(batch_size=batch_size, columns=["split_membership", "canonical_sequence"]):
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
                batch_tok.append(ctx)
                batch_ign.append(tgt)
                # P3 entropy computes boundaries on GPU from the nt batch, so
                # the dataset emits no per-sequence boundary when provider is
                # None (training loop derives them online).
                if boundary_provider is not None:
                    batch_bndry.append(boundary_provider.boundary(c, len(c))[:-1])
            else:
                cc, tt = _window(seq, context_nt, tok)
                if not cc:
                    continue
                batch_tok.append(cc)
                batch_ign.append(tt)
            batch_seqcnt.append(1)
            if len(batch_tok) >= batch_size:
                b = flush()
                if b is not None:
                    yield b
                    batches += 1
                    if max_batches is not None and batches >= max_batches:
                        return
                batch_tok, batch_ign, batch_bndry, batch_seqcnt = [], [], [], []
    b = flush()
    if b is not None:
        yield b


def _pad(ids: list[int], length: int, ignore: bool = False) -> list[int]:
    pad = IGNORE if ignore else 0
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad] * (length - len(ids))


def count_valid_nt(batch) -> int:
    """Real (non-ignore) target positions in a batch."""
    return sum(1 for row in batch["targets"] for v in row if v != IGNORE)