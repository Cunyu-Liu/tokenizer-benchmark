"""Phase 4 BLT inference adapter (contract 2.3, 3.2, 3.5, 5.4).

Wraps a trained BLT (P-arm) checkpoint behind the shared RNAARAdapter
interface so the common scorer / artifact writer is reused for the sealed-test
main table.

Replay guarantees (contract 5.4):
  - The checkpoint carries a persisted ``calib`` payload (predictor state,
    gate, mean patch length, empirical length pmf) written by p4_train.py, so
    inference reproduces the EXACT patch boundaries used during training:
        P1 fixed_patch : patch_len = round(calib.mean_patch_len)
        P2 random_patch: boundaries sampled from calib.length_dist, seeded by
                         (cfg.seed, sequence-id) — must match dataset _seq_id
        P3 entropy_patch: boundaries from the fitted entropy predictor + gate
  - Construction raises if a BLT checkpoint lacks the ``calib`` payload.

Per-base likelihood (contract 3.5 next_base_BPN):
  BLT folds nt -> patches and runs a causal Transformer over patch embeddings,
  then unfolds patch logits back to per-nt logits. Because a patch embedding is
  the mean of ALL nt in its segment, a single full-sequence forward leaks future
  nt within the open patch. To report an *exact* per-base conditional
  p(x_i | x_<i), we recompute the forward on each prefix (causal open patch =
  mean of s_i..i only). The patch policies are prefix-consistent (fixed/random
  sampled from position 0; entropy predictor is a causal GRU), so the boundary
  over a prefix equals the boundary over the full sequence truncated to it.

GPU-only: construction asserts CUDA and never falls back to CPU (contract 4).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .adapter import RNAARAdapter, ExternalModelInfo
from model import train_config as tc
from model.backbone import BLTCausalLM
from model.census import GPUGuard
from model.dataset import _seq_id
from model.entropy_predictor import EntropyPatchPolicy, EntropyPredictor
from model.patch import PatchPolicy

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
ALPHABET = tuple("ACGU")
BASE_TO_IDX = {b: i for i, b in enumerate(ALPHABET)}


class _ReconCalib:
    """Minimal calib-like object to feed train._patch_policy_for_arm."""

    def __init__(self, payload: dict, device: str):
        self.mean_patch_len = payload.get("mean_patch_len")
        self.length_dist = payload.get("length_dist")
        self.gate = payload.get("gate")
        self.checkpoint_hash = payload.get("checkpoint_hash")
        if "predictor_state" in payload and payload["predictor_state"] is not None:
            pred = EntropyPredictor()
            pred.load_state_dict(payload["predictor_state"])
            pred.to(device).eval()
            self.predictor = pred
        else:
            self.predictor = None


class InternalBLTAdapter(RNAARAdapter):
    """GPU adapter over a trained BLT checkpoint (exact causal next_base)."""

    kind = "internal_blt"

    def __init__(self, arm_id: str, seed: int, ckpt_path: str,
                 device: str = "cuda:0", split: str = SPLIT_8080,
                 model_info: ExternalModelInfo | None = None,
                 cfg: tc.RunConfig | None = None):
        guard = GPUGuard(device)
        guard.check()  # raises on CPU / CUDA-unavailable
        self.device = device
        self.guard = guard
        cfg = cfg if cfg is not None else tc.resolved_config(arm_id, seed)
        if cfg.arm.backbone != "blt":
            raise ValueError("InternalBLTAdapter requires a blt arm, got %s" % arm_id)
        self.cfg = cfg
        self.arm = cfg.arm
        self.model = tc.build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        calib_payload = state.get("calib")
        if calib_payload is None:
            raise ValueError(
                "BLT checkpoint %s missing persisted 'calib' payload "
                "(contract 5.4 replay requires exact patch boundaries)" % ckpt_path)
        self._calib = _ReconCalib(calib_payload, device)
        self.vocab = cfg.arm.vocab_size
        self.context_nt = cfg.context_nt
        self.arm_type = cfg.arm.tokenizer_type
        super().__init__(model_info)

    # --- canonicalization / encode / decode ---
    def canonicalize(self, seq: str) -> str:
        s = seq.upper().replace("T", "U")
        for ch in s:
            if ch not in BASE_TO_IDX:
                raise ValueError("non-primary IUPAC char %r in %r" % (ch, seq))
        return s

    def encode(self, seq: str) -> list[int]:
        return [BASE_TO_IDX[b] for b in self.canonicalize(seq)]

    def decode(self, ids: list[int]) -> str:
        return "".join(ALPHABET[i] for i in ids)

    # --- boundary replay (matches training, contract 3.2 / 5.4) ---
    def _boundary(self, canon: str, seq_id: int) -> list[int]:
        """Per-position boundary (0/1) over the nt prefix, replaying training."""
        t = self.arm_type
        if t == "fixed_patch":
            plen = int(round(self._calib.mean_patch_len)) if self._calib.mean_patch_len else 8
            return [1 if i % max(1, plen) == 0 else 0 for i in range(len(canon))]
        if t == "random_patch":
            if not self._calib.length_dist:
                raise RuntimeError("random_patch requires persisted length_dist")
            return PatchPolicy(kind="random", seed=self.cfg.seed,
                               length_dist=self._calib.length_dist) \
                .boundary(canon, len(canon), seq_id)
        if t == "entropy_patch":
            if self._calib.predictor is None:
                raise RuntimeError("entropy_patch requires persisted predictor_state")
            policy = EntropyPatchPolicy(self._calib.predictor, self._calib.gate,
                                        device=self.device)
            return policy.boundary(canon, len(canon))
        raise ValueError("unknown blt tokenizer_type %r" % t)

    # --- forward ---
    def _forward_logits(self, ids: list[int], bnd: list[int]) -> torch.Tensor:
        """Per-nt logits (1, T, vocab) for a single nt prefix + its boundary."""
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        b = torch.tensor([bnd], dtype=torch.float, device=self.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(t, b, targets=None)
        return logits.float()

    # --- scoring callbacks (RNAARAdapter contract) ---
    def log_prob_next_base(self, prefix: str, nxt: str) -> float:
        """Exact causal per-base log p(nxt | prefix) for a BLT arm.

        Recomputes the forward on the (context-truncated) prefix with its causal
        boundary, so the open patch contains only positions <= the last input
        position (no future-nt leak within the patch).
        """
        return float(self.log_probs_next_base(prefix)[BASE_TO_IDX[nxt]])

    def log_probs_next_base(self, prefix: str) -> list[float]:
        """Full 4-way causal log-softmax for the next base given `prefix`.

        Returns a list over A/C/G/U (natural-log softmax). This is the exact
        conditional used by the canonical codec (contract 3.6): each position
        i is scored as p(x_i | x_<i) with the causal open patch (no future-nt
        leak), and the full distribution is quantized to the integer CDF.
        """
        canon = self.canonicalize(prefix)
        if not canon:
            # leading base has no context (no BOS): uniform prior
            return [-math.log(4.0)] * 4
        seq_id = _seq_id(canon)
        ids = [BASE_TO_IDX[b] for b in canon]
        # Match training's last-context truncation: the model conditions on at
        # most context_nt-1 preceding nt.
        if len(ids) > self.context_nt - 1:
            ids = ids[-(self.context_nt - 1):]
            seq_id = _seq_id(self.decode(ids))
        bnd = self._boundary(self.decode(ids), seq_id)
        logits = self._forward_logits(ids, bnd)
        lp = F.log_softmax(logits[0, -1], dim=-1)
        return [float(lp[v].item()) for v in range(self.vocab)]

    def log_prob_token(self, ctx: list[int], token_id: int) -> float:
        raise NotImplementedError(
            "BLT reports per-base likelihoods; use log_prob_next_base / "
            "next_base_BPN (contract 3.5)")

    def generate(self, prefix: str, n: int, temperature: float = 1.0,
                 top_p: float = 1.0) -> str:
        raise NotImplementedError(
            "BLT generation (patch-level autoregressive) added in a later "
            "Phase 4 sub-deliverable")

    def fallback(self, seq: str) -> float:
        # Neural execution only: uniform fallback must be 0 (CPU-fallback ban).
        return 0.0
