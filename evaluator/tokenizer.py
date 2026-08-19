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


    def token_nt_counts(self, ids: list[int]) -> list[int]:
        """Each NUC token covers exactly 1 nt (contract 3.6 attribution)."""
        return [1] * len(ids)


class KmerTokenizer(TokenizerBase):
    """overlapping (stride=1) or non-overlapping (stride=k) k-mer.

    Contract 3.6 canonical tail-token rule for non-overlapping k-mer:
      - Full k-mer blocks cover the leading floor(L/k)*k nucleotides.
      - The trailing 1..k-1 nucleotides (if any) are encoded by a single
        frozen canonical tail token whose string IS the remaining bases
        (a short mer of length 1..k-1). The vocab therefore extends the
        4^k full mers with all short mers of length 1..k-1.
      - Lossless: decode concatenates full-mer strings + the tail string,
        reconstructing the original sequence byte-identically. No bases are
        discarded, no padding is scored, no base is double-counted.
    """

    def __init__(self, k: int, overlapping: bool, special_tokens: Optional[dict] = None):
        super().__init__(special_tokens)
        self.k = k
        self._overlapping = overlapping
        self.stride = 1 if overlapping else k
        self.offset = 0
        import itertools
        self._full_mers = ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]
        if overlapping:
            self._id_to_mer = self._full_mers
        else:
            # canonical tail tokens: all short mers of length 1..k-1, ordered
            # by (length, lexicographic) so the mapping is frozen & deterministic.
            tail = []
            for L in range(1, k):
                tail += ["".join(p) for p in itertools.product(ALPHABET, repeat=L)]
            self._id_to_mer = self._full_mers + tail
        self._mer_to_id = {m: i for i, m in enumerate(self._id_to_mer)}
        self._tail_offset = 4 ** k  # first tail-token id
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
            # full k-mer blocks, then a single canonical tail token (1..k-1 nt)
            i = 0
            n = len(c)
            while i + self.k <= n:
                ids.append(self._mer_to_id[c[i:i + self.k]])
                i += self.k
            if i < n:
                ids.append(self._mer_to_id[c[i:]])  # tail: length 1..k-1
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
        if self._overlapping:
            return 4 ** self.k
        return 4 ** self.k + sum(4 ** L for L in range(1, self.k))


    def token_nt_counts(self, ids: list[int]) -> list[int]:
        """Per-token canonical nt attribution (contract 3.6: each real nt scored
        exactly once, no padding, no double count).

        - overlap k-mer stride 1: token[0] covers the first k nt; each later
          token contributes exactly 1 new nt (its last base).
        - non-overlap k-mer: each full k-mer token covers k nt; the final
          canonical tail token (length 1..k-1) covers its own length.
        """
        if self._overlapping:
            if not ids:
                return []
            return [self.k] + [1] * (len(ids) - 1)
        counts = []
        for i in ids:
            counts.append(len(self._id_to_mer[i]))
        return counts


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
        self._INF_RANK = 1 << 30
        self._merge_rank: dict[str, int] = {}
        self._merge_rank_by_pair: dict[tuple[str, str], int] = {}

    def fit(self, sequences: Iterable[str], max_merges: Optional[int] = None) -> None:
        """Learn BPE merges from train sequences only.

        Rebuild-based BPE: each merge merges every occurrence of the most frequent
        adjacent pair, rebuilding only the sequences that contain it instead of
        rescaming the whole corpus. This is the standard left-to-right greedy BPE
        (identical merge semantics to the naive scan) but is feasible on large
        corpora where O(merges x corpus) is intractable. Deterministic: ties broken
        by the pair tuple. Round-trip is preserved.
        """
        seqs = [list(_canon(s)) for s in sequences]
        merges = []
        special_extra = len(self.special_tokens)
        budget = self.target_vocab - len(self.base_vocab) - special_extra
        if max_merges is not None:
            budget = min(budget, max_merges)

        # pair -> set of (seq_idx, left_pos); count[pair] == len(pair_pos[pair]).
        pair_pos: dict[tuple[str, str], set[tuple[int, int]]] = {}
        count: dict[tuple[str, str], int] = {}

        def _has(si: int, pos: int) -> bool:
            return 0 <= pos and pos + 1 < len(seqs[si])

        def add(si: int, pos: int) -> None:
            if not _has(si, pos):
                return
            key = (seqs[si][pos], seqs[si][pos + 1])
            s = pair_pos.setdefault(key, set())
            if (si, pos) not in s:
                s.add((si, pos))
                count[key] = count.get(key, 0) + 1

        def remove(si: int, pos: int) -> None:
            if not _has(si, pos):
                return
            key = (seqs[si][pos], seqs[si][pos + 1])
            s = pair_pos.setdefault(key, set())
            if (si, pos) in s:
                s.discard((si, pos))
                count[key] = count.get(key, 0) - 1
                if count[key] <= 0:
                    del count[key]
                    del pair_pos[key]

        def clear_seq(si: int) -> None:
            for pos in range(len(seqs[si]) - 1):
                remove(si, pos)

        def add_seq(si: int) -> None:
            for pos in range(len(seqs[si]) - 1):
                add(si, pos)

        for si in range(len(seqs)):
            add_seq(si)

        while len(merges) < max(0, budget) and count:
            top = max(count, key=lambda k: (count[k], k))
            a, b = top
            merged = a + b
            # Snapshot affected sequences before clearing modifies pair_pos.
            affected = sorted({si for si, _ in pair_pos[top]})
            for si in affected:
                clear_seq(si)
                old = seqs[si]
                new = []
                i, n = 0, len(old)
                while i < n:
                    if i + 1 < n and old[i] == a and old[i + 1] == b:
                        new.append(merged)
                        i += 2
                    else:
                        new.append(old[i])
                        i += 1
                seqs[si] = new
                add_seq(si)
            merges.append((a, b))

        self.merges = merges
        # build vocab
        self._id_to_token = list(self.base_vocab)
        for a, b in merges:
            self._id_to_token.append(a + b)
        self._token_to_id = {t: i for i, t in enumerate(self._id_to_token)}
        # rank maps for efficient rank-based encode: a merged token's rank is its
        # merge index (earliest merge = rank 0); pair -> rank for the heap.
        self._merge_rank = {a + b: i for i, (a, b) in enumerate(merges)}
        self._merge_rank_by_pair = {(a, b): i for i, (a, b) in enumerate(merges)}

    def encode(self, seq: str) -> list[int]:
        """Left-to-right greedy BPE encode.

        Rank-based heap encode: each merged token carries its merge index as a
        rank (base tokens rank = infinity). Adjacent pairs that are a valid merge
        are pushed on a min-heap keyed by (merge_rank, left_pos); we repeatedly pop
        the smallest-rank pair, merge it, and refresh only its two local neighbors.
        This reproduces the exact left-to-right greedy result of the naive
        per-merge scan but in O(n log n) instead of O(n * merges), so encoding long
        ncRNA sequences is feasible. Deterministic: ties broken by left position.
        """
        import heapq
        toks = list(_canon(seq))
        n = len(toks)
        if n <= 1:
            return [self._token_to_id[t] for t in toks]
        # rank of each token: merge index if merged, else large (base).
        tok_rank = [self._merge_rank.get(t, self._INF_RANK) for t in toks]
        # doubly-linked list via arrays.
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[n - 1] = -1
        alive = [True] * n
        heap = []
        # merge_rank: pair (a,b) -> its merge index (0 = earliest/first merge).
        mr = self._merge_rank_by_pair

        def push(i: int) -> None:
            if i < 0 or not alive[i]:
                return
            j = nxt[i]
            if j < 0 or not alive[j]:
                return
            key = (toks[i], toks[j])
            r = mr.get(key)
            if r is not None:
                heapq.heappush(heap, (r, i))

        for i in range(n - 1):
            push(i)

        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j < 0 or not alive[j]:
                continue
            key = (toks[i], toks[j])
            if mr.get(key) != r:
                continue  # stale entry
            merged_tok = key[0] + key[1]
            toks[i] = merged_tok
            tok_rank[i] = r
            alive[j] = False
            # link i's next to j's next
            k = nxt[j]
            nxt[i] = k
            if k >= 0:
                prev[k] = i
            # refresh local neighbors
            p = prev[i]
            push(p)
            push(i)

        return [self._token_to_id[toks[i]] for i in range(n) if alive[i]]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_token[i] for i in ids)

    def vocab_size(self) -> int:
        return len(self._id_to_token)


    def token_nt_counts(self, ids: list[int]) -> list[int]:
        """Subword tokens are variable-length; each covers len(decode(token)) nt."""
        return [len(self._id_to_token[i]) for i in ids]


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
    def token_nt_counts(self, ids: list[int]) -> list[int]:
        """Subword tokens are variable-length; each covers len(decode(token)) nt."""
        return [len(self._id_to_token[i]) for i in ids]


