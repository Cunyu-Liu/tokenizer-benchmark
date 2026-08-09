"""External model adapter schema (3.7) and RNAARAdapter base (2.3).

All models implement canonicalization, encode/decode, prefix preparation,
forward/scoring and generation. Shared scorer + artifact writer are used,
not per-baseline evaluators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExternalModelInfo:
    model_id: str
    revision: str
    paper_status: str            # e.g. "published" / "preprint"
    checkpoint_sha256: Optional[str]
    code_commit: Optional[str]
    tokenizer: str
    training_data_notes: str
    license: str
    gpu_environment: str
    comparability: str           # STRICTLY_COMPARABLE / REFERENCE_ONLY / UNAVAILABLE_WITH_EVIDENCE
    decoder: str = ""
    adapter_module: str = ""


class RNAARAdapter:
    """Contract every internal + external model must implement."""
    kind = "base"

    def __init__(self, model_info: Optional[ExternalModelInfo] = None):
        self.model_info = model_info

    def canonicalize(self, seq: str) -> str:
        raise NotImplementedError

    def encode(self, seq: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def prepare_prefix(self, prefix: str, target_len: int) -> dict:
        """Return context for continuation; prefix defined in raw nt."""
        raise NotImplementedError

    def log_prob_next_base(self, prefix: str, nxt: str) -> float:
        """log p(nxt | prefix), exact per-base. GPU adaptation layer."""
        raise NotImplementedError

    def log_prob_token(self, ctx: list[int], token_id: int) -> float:
        raise NotImplementedError

    def generate(self, prefix: str, n: int, temperature: float, top_p: float) -> str:
        raise NotImplementedError

    def fallback(self, seq: str) -> float:
        """Uniform fallback; must be 0 for neural execution (CPU fallback ban)."""
        return 0.0
