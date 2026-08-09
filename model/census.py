"""Parameter / FLOP / effective-nucleotide-exposure census and GPU guard.

Confirms contract 3.4: matched arm params within 2%, exposure counted in
cumulative valid target nt (not optimizer steps), and neural execution is
GPU-only with cpu_fallback_count == 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Census:
    total_params: int = 0
    non_embedding_params: int = 0
    embedding_params: int = 0
    flops: float = 0.0
    rng_seed: int = 0
    context_nt: int = 0
    effective_batch_nt: int = 0
    cumulative_valid_target_nt: int = 0

    def __post_init__(self):
        self.embedding_params = self.total_params - self.non_embedding_params

    def matched_within(self, other: "Census", tol: float = 0.02) -> bool:
        """Total trainable params within tol (2% default)."""
        if self.total_params == 0:
            return False
        return abs(self.total_params - other.total_params) / self.total_params <= tol


def count_params(model) -> Census:
    total = 0
    non_emb = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if "tok_emb" not in name and "lm_head" not in name and "pos_emb" not in name:
            non_emb += n
    return Census(total_params=total, non_embedding_params=non_emb)


def estimate_attention_flops(q, k, v) -> float:
    """Rough FLOPs for one attention head over sequence len L, d_model D."""
    B, L, D = q.shape
    return 2 * B * L * L * D * 3  # q@k^T + att@v + projection ~ 3 matrix products


def estimate_mlp_flops(d_model: int, d_ff: int, n_layers: int, L: int) -> float:
    """Per-token FLOPs for MLP: 2 * d_model * d_ff per layer * 3 (2 linears + gelu)."""
    return n_layers * L * 2 * d_model * d_ff * 3


class GPUGuard:
    """Enforces neural forward/backward on CUDA; cpu_fallback_count must be 0."""
    def __init__(self, device: str):
        self.device = device
        self.cpu_fallback_count = 0

    def check(self) -> str:
        if self.device.startswith("cpu"):
            self.cpu_fallback_count += 1
            raise RuntimeError(
                "CPU fallback denied: neural execution requires CUDA. fallback_count=%d"
                % self.cpu_fallback_count)
        return self.device
