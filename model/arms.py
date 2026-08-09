"""Ten-arm 100M registry (3.2). Maps arm ID to (backbone, representation)."

F1-F7 static track on FlatCausalLM; P1-P3 patch track on BLTCausalLM.

Contract constraints:
  - F1-F7 share depth/width/attn/pos-enc/context and sequence order.
  - Larger vocabularies use factorized/tied embeddings so total params within 2%.
  - P1-P3 share identical BLT params/modules; BLT explicit n-gram/hash OFF.
  - P1 fixed patch length from train-only entropy calibration mean.
  - P2 random patch from train-only entropy length distribution (seed-based).
  - P3 entropy predictor trained on train split only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmSpec:
    id: str
    backbone: str          # "flat" | "blt"
    tokenizer_type: str    # NUC/BPE/Unigram/overlap_mer/nonoverlap_mer/fixed_patch/random_patch/entropy_patch
    vocab_size: int
    k: int | None
    stride: int | None
    note: str


ARMS_100M: list[ArmSpec] = [
    ArmSpec("F1", "flat", "NUC", 4, 1, 1, "static single-nt baseline"),
    ArmSpec("F2", "flat", "BPE", 1024, None, None, "GenerRNA-aligned subword"),
    ArmSpec("F3", "flat", "Unigram", 1024, None, None, "exclude BPE-merge-only gain"),
    ArmSpec("F4", "flat", "overlap_mer", 4 ** 3, 3, 1, "GARNET-style local"),
    ArmSpec("F5", "flat", "overlap_mer", 4 ** 6, 6, 1, "BEACON 6-mer"),
    ArmSpec("F6", "flat", "nonoverlap_mer", 4 ** 3, 3, 3, "overlap vs block"),
    ArmSpec("F7", "flat", "nonoverlap_mer", 4 ** 6, 6, 6, "k/stride sensitivity"),
    ArmSpec("P1", "blt", "fixed_patch", 4, None, None, "fixed-length patch"),
    ArmSpec("P2", "blt", "random_patch", 4, None, None, "length-matched random patch"),
    ArmSpec("P3", "blt", "entropy_patch", 4, None, None, "causal entropy patch"),
]

ARMS_350M: list[ArmSpec] = [
    ArmSpec("C1", "flat", "NUC", 4, 1, 1, "NUC flat"),
    ArmSpec("C2", "flat", "BPE", 1024, None, None, "BPE-1024 flat"),
    ArmSpec("C3", "blt", "fixed_patch", 4, None, None, "fixed-patch BLT"),
    ArmSpec("C4", "blt", "entropy_patch", 4, None, None, "entropy-patch BLT"),
]

SEEDS = [17, 29, 43]


def arm(id_: str) -> ArmSpec:
    for a in ARMS_100M + ARMS_350M:
        if a.id == id_:
            return a
    raise KeyError(id_)
