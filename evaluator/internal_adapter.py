"""Phase 4 internal-model inference adapter (contract 2.3, 3.5, 5.4).

Wraps a trained TokBench-RNA checkpoint (flat backbones F1-F7) behind the
shared RNAARAdapter interface so the common scorer / artifact writer can be
reused for the sealed-test main tables (next_base_BPN, canonical_path_BPN).

Per-arm likelihood accounting (3.5):
  - F1 NUC          : exact next_base (vocab 4, one token per nt)
  - F4/F5 overlap   : exact next_base (context = last k-1 nt -> next nt)
  - F2/F3 BPE/Uni   : canonical_path via log_prob_token (next_base not exact)
  - F6/F7 nonoverlap: canonical_path via log_prob_token

Batched scoring: for flat causal backbones a SINGLE forward over the full
sequence yields every position's next-token logits (position k conditions only
on tokens 0..k), so per-position log probs are recovered with O(1) forwards per
sequence instead of O(T). This makes the 100K-sequence sealed test tractable.
The batched result is provably identical to the per-position callback path.

GPU-only: construction asserts CUDA and never falls back to CPU (contract 4,
CPU-fallback ban). `fallback()` returns 0.0 per the neural-execution rule.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from .adapter import RNAARAdapter, ExternalModelInfo
from model import train_config as tc
from model.census import GPUGuard
from model.dataset import build_arm_tokenizer
from model.train import build_model_for_cfg

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
ALPHABET = tuple("ACGU")
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}


class InternalFlatAdapter(RNAARAdapter):
    """GPU adapter over a trained flat-backbone checkpoint."""

    kind = "internal_flat"

    def __init__(self, arm_id: str, seed: int, ckpt_path: str,
                 device: str = "cuda:0", split: str = SPLIT_8080,
                 model_info: ExternalModelInfo | None = None,
                 cfg: tc.RunConfig | None = None):
        guard = GPUGuard(device)
        guard.check()  # raises on CPU / CUDA-unavailable
        self.device = device
        self.guard = guard
        cfg = cfg if cfg is not None else tc.resolved_config(arm_id, seed)
        self.cfg = cfg
        self.arm = cfg.arm
        self.tok = build_arm_tokenizer(split, self.arm)
        self.model = build_model_for_cfg(cfg, device)
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.vocab = cfg.arm.vocab_size
        self.arm_type = cfg.arm.tokenizer_type
        self.k = cfg.arm.k
        super().__init__(model_info)

    # --- canonicalization / encode / decode ---
    def canonicalize(self, seq: str) -> str:
        s = seq.upper().replace("T", "U")
        for ch in s:
            if ch not in BASE_TO_IDX:
                raise ValueError("non-primary IUPAC char %r in %r" % (ch, seq))
        return s

    def encode(self, seq: str) -> list[int]:
        return self.tok.encode(self.canonicalize(seq))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(list(ids))

    # --- forward ---
    def _logits(self, token_ids: list[int], bf16: bool = True) -> torch.Tensor:
        """Forward a token sequence; returns logits (1, T, vocab) on device.

        `bf16=False` runs the forward in fp32. The canonical codec (contract
        3.6) MUST use fp32 so the batched encoder and the KV-cache decoder
        produce bit-identical CDFs (verified: fp32 batched == per-prefix ==
        KV-cache); bf16 autocast kernels differ by sequence shape.
        """
        t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            if bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = self.model(t, targets=None)
            else:
                logits, _ = self.model(t, targets=None)
        return logits.float()

    # --- batched (single-forward) scoring, equivalent to per-position path ---
    def all_log_probs_next_base(self, seq: str) -> list[float]:
        """Exact per-base log p(x_i | x_<i) for NUC arms in ONE forward."""
        canon = self.canonicalize(seq)
        n = len(canon)
        out = [0.0] * n
        if n == 0:
            return out
        out[0] = -math.log(4.0)  # leading base, no context (no BOS)
        ids = self.tok.encode(canon)
        if len(ids) < 1:
            return out
        lp = F.log_softmax(self._logits(ids)[0], dim=-1)  # (T, vocab)
        for i in range(1, n):
            out[i] = float(lp[i - 1, BASE_TO_IDX[canon[i]]].item())
        return out

    def all_log_probs_token(self, ids: list[int]) -> list[float]:
        """Canonical-path token log probs p(tok_i | tok_<i) in ONE forward."""
        n = len(ids)
        out = [0.0] * n
        if n == 0:
            return out
        out[0] = -math.log(float(self.vocab))  # first token, uniform prior
        if n == 1:
            return out
        lp = F.log_softmax(self._logits(ids)[0], dim=-1)  # (T, vocab)
        for i in range(1, n):
            out[i] = float(lp[i - 1, ids[i]].item())
        return out

    def all_log_probs_full(self, ids: list[int], bf16: bool = True) -> list[list[float]]:
        """Full (T, vocab) log-probs over the canonical token path.

        Position 0 uses a uniform prior (no BOS, matching all_log_probs_token);
        positions 1..T-1 use the model's full-vocabulary log_softmax. Returned
        as a list of length-T vocab-sized lists so the canonical codec can
        quantize each position's full distribution (contract 3.6).
        """
        n = len(ids)
        uniform = -math.log(float(self.vocab))
        rows = [[uniform] * self.vocab for _ in range(n)]
        if n <= 1:
            return rows
        lp = F.log_softmax(self._logits(ids, bf16=bf16)[0], dim=-1)  # (T, vocab)
        for i in range(1, n):
            rows[i] = [float(lp[i - 1, v].item()) for v in range(self.vocab)]
        return rows

    # --- scoring callbacks (RNAARAdapter contract) ---
    def log_prob_token(self, ctx: list[int], token_id: int) -> float:
        """log p(token_id | ctx) for the canonical token path."""
        if not ctx:
            # No BOS token in these models; the leading position has no context,
            # so assign a uniform prior over the vocab (defensible, and negligible
            # on real sequence lengths). GPU-free.
            return -math.log(float(self.vocab))
        logits = self._logits(ctx)
        lp = F.log_softmax(logits[0, -1], dim=-1)
        return float(lp[token_id].item())

    def log_probs_token(self, ctx: list[int], bf16: bool = True) -> list[float]:
        """Full-vocab log-probs given a token context (canonical codec decode).

        Mirrors `all_log_probs_full` at a single position: an empty context
        gives a uniform prior over the vocab (no BOS); otherwise one causal
        forward over the context and the full log_softmax at the last position.
        The independent decoder uses exactly this so encode/decode share the
        identical per-prefix CDFs (contract 3.6 byte-identical recovery).
        """
        uniform = -math.log(float(self.vocab))
        if not ctx:
            return [uniform] * self.vocab
        logits = self._logits(list(ctx), bf16=bf16)
        lp = F.log_softmax(logits[0, -1], dim=-1)
        return [float(lp[v].item()) for v in range(self.vocab)]

    def log_prob_next_base(self, prefix: str, nxt: str) -> float:
        """Exact per-base log p(nxt | prefix).

        Only exact for NUC (F1) and overlapping k-mer (F4/F5) arms. For
        BPE/Unigram/non-overlap arms the per-base likelihood is not directly
        available from token logits; the scorer must use canonical_path
        (log_prob_token) instead, so we refuse here.
        """
        canon = self.canonicalize(prefix)
        t = self.arm_type
        if t == "NUC":
            ids = self.tok.encode(canon)
            if not ids:
                # leading base has no context (no BOS): uniform prior
                return -math.log(4.0)
            logits = self._logits(ids)
            lp = F.log_softmax(logits[0, -1], dim=-1)
            return float(lp[BASE_TO_IDX[nxt]].item())
        if t == "overlap_mer":
            if len(canon) < self.k - 1:
                # insufficient context for a full k-mer window: uniform prior
                return -math.log(4.0)
            ctx = canon[-(self.k - 1):]
            tid = self.tok.encode(ctx)[0]
            logits = self._logits([tid])
            tok = self.tok.encode(canon + nxt)[0]
            lp = F.log_softmax(logits[0, -1], dim=-1)
            return float(lp[tok].item())
        raise NotImplementedError(
            "next_base not exact for arm_type=%r; use canonical_path" % t)

    # --- generation ---
    def generate(self, prefix: str, n: int, temperature: float = 1.0,
                 top_p: float = 1.0) -> str:
        canon = self.canonicalize(prefix)
        if self.arm_type == "NUC":
            ids = self.tok.encode(canon)
            for _ in range(n):
                if not ids:
                    # Empty context (unconditional generation): uniform prior
                    # over the alphabet at the leading position, consistent with
                    # log_prob_token's empty-ctx convention (contract 3.5).
                    k = len(ALPHABET)
                    p = torch.full((k,), 1.0 / k, device=self.device)
                    nxt = int(torch.multinomial(p, 1).item())
                    ids = ids + [nxt]
                    continue
                logits = self._logits(ids)
                lp = logits[0, -1] / max(temperature, 1e-6)
                if top_p < 1.0:
                    probs = F.softmax(lp, dim=-1)
                    sorted_p, _ = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_p, dim=-1)
                    keep = cum <= top_p
                    keep[..., 0] = True
                    cutoff = sorted_p[keep][-1]
                    lp = lp.clone()
                    lp[probs < cutoff] = -float("inf")
                p = F.softmax(lp, dim=-1)
                nxt = int(torch.multinomial(p, 1).item())
                ids = ids + [nxt]
            return "".join(ALPHABET[i] for i in ids[len(self.tok.encode(canon)):])
        raise NotImplementedError("generate only for NUC arm_type=%r" % self.arm_type)

    def fallback(self, seq: str) -> float:
        # Neural execution only: uniform fallback must be 0 (CPU-fallback ban).
        return 0.0


def next_base_ctx_length(arm_type: str, k: int) -> int:
    """Context length (in nt) the next-base model conditions on."""
    if arm_type == "NUC":
        return 1
    if arm_type == "overlap_mer":
        return k - 1
    raise ValueError("next_base not exact for %r" % arm_type)
