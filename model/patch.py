"""BLT patch boundary providers (contract 3.2 P1-P3).

P1 fixed_patch : boundaries every `patch_len` nt (mean of train-only entropy
                 calibration is used to set patch_len).
P2 random_patch: boundaries sampled from the train-only entropy patch-length
                 empirical distribution; deterministic per (sequence_id, seed)
                 to satisfy prefix consistency.
P3 entropy_patch: boundaries predicted by the entropy predictor (separate
                 module trained on train split only). Plugged in via the
                 predictor's boundary() once calibrated.

Fixed/random policies are deterministic and do NOT add trainable params.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class PatchPolicy:
    kind: str                    # fixed | random | entropy
    seed: int = 0
    patch_len: int | None = None         # fixed
    length_dist: list[float] | None = None  # random: pmf over patch lengths
    predictor = None                      # entropy: callable -> boundary vec

    def boundary(self, seq: str, length: int, seq_id: int = 0) -> list[int]:
        """Return per-position boundary indicator (0/1) for a sequence."""
        if self.kind == "fixed":
            plen = self.patch_len or 1
            return [1 if i % plen == 0 else 0 for i in range(length)]
        if self.kind == "random":
            return self._random_boundary(seq, length, seq_id)
        if self.kind == "entropy":
            if self.predictor is None:
                raise RuntimeError("entropy policy requires a predictor")
            return self.predictor.boundary(seq, length)
        raise ValueError("unknown patch kind %r" % self.kind)

    def _random_boundary(self, seq: str, length: int, seq_id: int) -> list[int]:
        if not self.length_dist:
            raise RuntimeError("random policy requires length_dist")
        rng = random.Random((self.seed << 32) ^ seq_id)
        dist = self.length_dist
        boundary = [1]
        i = 1
        while i < length:
            # sample next patch length from pmf
            r = rng.random()
            acc = 0.0
            plen = 1
            for plen, p in enumerate(dist, start=1):
                acc += p
                if r <= acc:
                    break
            for _ in range(plen - 1):
                if i < length:
                    boundary.append(0)
                    i += 1
            if i < length:
                boundary.append(1)
                i += 1
        return boundary[:length]


def fixed_policy(patch_len: int) -> PatchPolicy:
    return PatchPolicy(kind="fixed", patch_len=int(patch_len))


def random_policy(seed: int, length_dist: list[float]) -> PatchPolicy:
    return PatchPolicy(kind="random", seed=seed, length_dist=list(length_dist))