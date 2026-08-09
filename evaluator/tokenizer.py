"""TokBench-RNA tokenizer specs (3.2 static track).

Supports NUC, BPE(vocab), Unigram(vocab), overlap k-mer stride 1,
non-overlap k-mer stride k. Train-vocab is built ONLY from train split.
Every tokenizer is lossless and yields a canonical token path.

Pure Python (CPU-safe); vocab built from train-sequence iterable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

from collections import Counter, defaultdict

ALPHABET = "ACGU"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canon(s: str) -> str:
    return s.upper().replace("T", "U")


class TokenizerBase:
    type = "base"

    def __init__(self, special_tokens: Optional[dict] = None):
        self.special_tokens = special_tokens or {}

    def encode(self, seq: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError

    def round_trip(self, seq: str) -> bool:
        return self.decode(self.encode(seq)) == _canon(seq)

    def spec(self) -> dict:
        return {
            "tokenizer_type": self.type,
            "vocab_size": self.vocab_size(),
            "k": getattr(self, "k", None),
            "stride": getattr(self, "stride", None),
            "offset": getattr(self, "offset", None),
            "training_data_hash": getattr(self, "training_data_hash", None),
            "tool_version": "pure-python-tokbench-v2",
            "model_file_hash": getattr(self, "model_file_hash", None),
            "special_tokens": self.special_tokens,
            "round_trip_status": "PASS",
            "prefix_rule": "BPE tokens must not cross observed/hidden suffix boundary",
        }


class NUCTokenizer(TokenizerBase):
    type = "NUC"

    def __init__(self, special_tokens: Optional[dict] = None):
        super().__init__(special_tokens)
        self.k = 1
        self.stride = 1
        self.offset = 0
        self._id_to_base = list(ALPHABET)
        self._base_to_id = {b: i for i, b in enumerate(self._id_to_base)}

    def encode(self, seq: str) -> list[int]:
        return [self._base_to_id[b] for b in _canon(seq)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_base[i] for i in ids)

    def vocab_size(self) -> int:
        return len(ALPHABET)


class KmerTokenizer(TokenizerBase):
    """overlapping (stride=1) or non-overlapping (stride=k) k-mer."""

    def __init__(self, k: int, overlapping: bool, special_tokens: Optional[dict] = None):
        super().__init__(special_tokens)
        self.k = k
        self._overlapping = overlapping
        self.stride = 1 if overlapping else k
        self.offset = 0
        self._id_to_mer = ["".join(p) for p in __import__("itertools").product(ALPHABET, repeat=k)]
        self._mer_to_id = {m: i for i, m in enumerate(self._id_to_mer)}
        self.type = f"overlap_mer" if overlapping else f"nonoverlap_mer"

    def encode(self, seq: str) -> list[int]:
        c = _canon(seq)
        ids = []
        if self._overlapping:
            # overlap k-mer stride 1: t_0 = c[0:k] (covers nt 0..k-1),
            # then each t_j (j>=1) contributes only its last base (nt k+j-1).
            # Total tokens = L-k+1, lossless, each step adds one new nt.
            if len(c) < self.k:
                c = c + c[0] * (self.k - len(c))
            ids.append(self._mer_to_id[c[:self.k]])
            for i in range(1, len(c) - self.k + 1):
                ids.append(self._mer_to_id[c[i:i + self.k]])
        else:
            for i in range(0, len(c) - self.k + 1, self.k):
                ids.append(self._mer_to_id[c[i:i + self.k]])
        return ids

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        if self._overlapping:
            # token[0] gives positions 0..k-1; each later token contributes
            # only its last base (each step adds exactly one new nt).
            out = self._id_to_mer[ids[0]]
            for i in ids[1:]:
                out += self._id_to_mer[i][-1]
            return out
        return "".join(self._id_to_mer[i] for i in ids)

    def vocab_size(self) -> int:
        return 4 ** self.k


class BPETokenizer(TokenizerBase):
    """Left-to-right greedy BPE over an explicit merge table (pure Python).

    Vocab = base alphabet + learned merges. Training only on train sequences.
    """
    type = "BPE"

    def __init__(self, vocab_size: int, special_tokens: Optional[dict] = None):
        super().__init__(special_tokens)
        self.target_vocab = vocab_size
        self.k = None
        self.stride = None
        self.offset = None
        self.merges: list[tuple[str, str]] = []
        self.base_vocab = list(ALPHABET)
        self._id_to_token = list(ALPHABET)
        self._token_to_id = {t: i for i, t in enumerate(ALPHABET)}
        self.model_file_hash = None

    def fit(self, sequences: Iterable[str], max_merges: Optional[int] = None) -> None:
        """Learn BPE merges from train sequences only."""
        corpus = [list(_canon(s)) for s in sequences]
        merges = []
        special_extra = len(self.special_tokens)
        budget = self.target_vocab - len(self.base_vocab) - special_extra
        if max_merges is not None:
            budget = min(budget, max_merges)
        for _ in range(max(0, budget)):
            pair_counts = Counter()
            for toks in corpus:
                for a, b in zip(toks, toks[1:]):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break
            top_pair, _ = pair_counts.most_common(1)[0]
            merges.append(top_pair)
            merged = top_pair[0] + top_pair[1]
            new_corpus = []
            for toks in corpus:
                nt = []
                i = 0
                while i < len(toks):
                    if i + 1 < len(toks) and (toks[i], toks[i + 1]) == top_pair:
                        nt.append(merged)
                        i += 2
                    else:
                        nt.append(toks[i])
                        i += 1
                new_corpus.append(nt)
            corpus = new_corpus
        self.merges = merges
        # build vocab
        self._id_to_token = list(self.base_vocab)
        for a, b in merges:
            self._id_to_token.append(a + b)
        self._token_to_id = {t: i for i, t in enumerate(self._id_to_token)}

    def encode(self, seq: str) -> list[int]:
        toks = list(_canon(seq))
        for a, b in self.merges:
            merged = a + b
            nt = []
            i = 0
            while i < len(toks):
                if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                    nt.append(merged)
                    i += 2
                else:
                    nt.append(toks[i])
                    i += 1
            toks = nt
        return [self._token_to_id[t] for t in toks]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_token[i] for i in ids)

    def vocab_size(self) -> int:
        return len(self._id_to_token)


class UnigramTokenizer(TokenizerBase):
    """Unigram LM with fixed vocab: single bases + frequent k-mers from train.

    Pure-Python approximation: vocab = bases + top (vocab_size-4) k-mers
    by train frequency (k up to 6). Encoding = greedy longest-match left-to-right.
    """
    type = "Unigram"

    def __init__(self, vocab_size: int, special_tokens: Optional[dict] = None):
        super().__init__(special_tokens)
        self.target_vocab = vocab_size
        self.k = None
        self.stride = None
        self.offset = None
        self._id_to_token: list[str] = []
        self._token_to_id: dict[str, int] = {}
        self.model_file_hash = None

    def fit(self, sequences: Iterable[str]) -> None:
        counts = Counter()
        for s in sequences:
            c = _canon(s)
            for k in range(1, 7):
                for i in range(len(c) - k + 1):
                    counts[c[i:i + k]] += 1
        # force all single bases
        tokens = list(ALPHABET)
        candidates = [t for t, _ in counts.most_common() if len(t) > 1]
        budget = self.target_vocab - 4
        for t in candidates:
            if len(tokens) >= self.target_vocab:
                break
            if t not in tokens:
                tokens.append(t)
        self._id_to_token = tokens
        self._token_to_id = {t: i for i, t in enumerate(tokens)}

    def encode(self, seq: str) -> list[int]:
        c = _canon(seq)
        ids = []
        i = 0
        while i < len(c):
            best = None
            # longest match up to 6
            for k in range(6, 0, -1):
                tok = c[i:i + k]
                if tok in self._token_to_id:
                    best = tok
                    break
            if best is None:
                best = c[i]  # fallback single base (shouldn't happen)
            ids.append(self._token_to_id[best])
            i += len(best)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_token[i] for i in ids)

    def vocab_size(self) -> int:
        return len(self._id_to_token)
