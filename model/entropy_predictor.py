"""Causal entropy patch predictor (contract 3.2 P3).

A small next-base predictor (GRU over one-hot nucleotides) trained on the
train split only. It produces a per-position next-base distribution; patch
boundaries are placed where cumulative entropy since the last boundary
reaches a calibrated gate.

P1/P2 derive their fixed length / random length distribution from the same
train-only entropy calibration, so fixed / random / entropy patch policies
are matched on mean patch count (contract 3.2, 5.4).

The predictor keeps an independent parameter count, train-only budget, and a
checkpoint hash; it is NOT part of the main BLT backbone parameters
(contract 3.2: "记录独立参数量、训练 FLOPs、checkpoint hash").

Performance notes (smoke/calib speed):
  - The entropy collection pass uses nt-budgeted batches so the sequential
    GRU never has to run over a long padded batch (keeps it fast on 4096-nt
    contexts).
  - ``boundaries_from_entropy`` is computed on CPU with a vectorized-over-
    batch loop instead of one GPU kernel launch per position, which was the
    O(T) CUDA-launch bottleneck during gate calibration.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import ALPHABET

# Calibration collection: keep the (sequential) GRU cheap and collect a fixed
# number of positions for a stable gate estimate.
CALIB_COLLECT_NT = 32768      # per-batch effective nt budget during collection
CALIB_POSITIONS = 200_000     # total positions collected for gate calibration


@dataclass
class EntropyCalib:
    gate: float                 # cumulative-entropy boundary threshold (nats)
    mean_patch_len: float       # empirical mean patch length at `gate` (train-only)
    length_dist: list[float]    # empirical patch-length pmf (train-only)
    predictor_params: int       # independent parameter count
    train_nt: int               # train-only nt used to fit the predictor
    checkpoint_hash: str        # sha256 of predictor state_dict
    target_patch_len: int       # requested target before entropy calibration
    predictor: object = None    # fitted EntropyPredictor (held for P3 online use)


class EntropyPredictor(nn.Module):
    """GRU next-base predictor over 4 nt; entropy = H(next-base | context)."""

    def __init__(self, d_model: int = 64, d_hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.embed = nn.Embedding(4, d_model)
        self.gru = nn.GRU(d_model, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, 4)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def logits(self, nt_ids: torch.Tensor) -> torch.Tensor:
        """(B, T, 4) next-base logits for each position given context < i."""
        return self.head(self.gru(self.embed(nt_ids))[0])

    def entropy(self, nt_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) per-position next-base entropy H(x_{i+1} | x_<i), nats."""
        logits = self.logits(nt_ids)
        p = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        return -(p * logp).sum(-1)


def boundaries_from_entropy(ent: torch.Tensor, gate: float) -> torch.Tensor:
    """Hard 0/1 per-position boundaries from per-position entropy (nats).

    Position 0 is always a patch start. A later position i starts a new patch
    when cumulative entropy since the last start reaches `gate`. The resetting
    sum is sequential over positions, so it is computed on CPU with a fast
    vectorized-over-batch loop (one numpy op per position instead of a CUDA
    kernel launch per position). Returns same shape as `ent` (B, T).
    """
    B, T = ent.shape
    device = ent.device
    e = ent.detach().cpu().numpy()
    starts = np.zeros((B, T), dtype=np.float32)
    starts[:, 0] = 1.0
    cum = np.zeros((B,), dtype=np.float32)
    g = float(gate)
    for i in range(1, T):
        cum += e[:, i - 1]
        new_start = cum >= g
        starts[:, i] = new_start.astype(np.float32)
        if new_start.any():
            cum = np.where(new_start, 0.0, cum)
    return torch.from_numpy(starts).to(device)


def patch_lengths_from_bounds(bounds: torch.Tensor) -> list[int]:
    """Empirical per-sequence patch lengths from a (B, T) boundary tensor."""
    lengths = []
    B, T = bounds.shape
    for b in range(B):
        idxs = [i for i in range(T) if bounds[b, i] > 0.5]
        idxs.append(T)
        for a, z in zip(idxs[:-1], idxs[1:]):
            lengths.append(z - a)
    return lengths


def _sha256_state_dict(sd) -> str:
    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(-1).numpy().tobytes())
    return h.hexdigest()


def train_entropy_predictor(
    data_path: str,
    device: str = "cuda:0",
    budget_nt: int = 2_000_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_batches: int | None = None,
    seed: int = 0,
    d_model: int = 64,
    d_hidden: int = 128,
) -> tuple[EntropyPredictor, dict]:
    """Train the next-base entropy predictor on the train split only (GPU)."""
    from .dataset import Exposure, count_valid_nt, iter_train_batches
    from .census import GPUGuard

    guard = GPUGuard(device)
    guard.check()
    # The predictor fits on raw nt ids; use the NUC (F1) data path so the
    # iterator emits nt-level token_ids (each nt -> 0..3).
    from .train_config import resolved_config
    cfg = resolved_config("F1", seed)
    model = EntropyPredictor(d_model=d_model, d_hidden=d_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    expo = Exposure()
    step = 0
    total_loss = 0.0
    gen = iter_train_batches(
        data_path, cfg, batch_size=batch_size, max_batches=max_batches,
        seed=seed, batch_nt=CALIB_COLLECT_NT)  # nt-budgeted keeps GRU cheap
    for batch in gen:
        if expo.cumulative_valid_target_nt >= budget_nt:
            break
        tok = torch.tensor(batch["token_ids"], dtype=torch.long, device=device)
        tgt = torch.tensor(batch["targets"], dtype=torch.long, device=device)
        valid_nt = count_valid_nt(batch)
        model.train()
        logits = model.logits(tok)
        loss = F.cross_entropy(
            logits.view(-1, 4), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        expo.add_batch(valid_nt, len(batch["token_ids"]))
        total_loss += loss.item()
        step += 1
    return model, {
        "steps": step,
        "train_nt": expo.cumulative_valid_target_nt,
        "final_loss": total_loss / max(1, step),
        "params": sum(p.numel() for p in model.parameters()),
        "device": device,
        "cpu_fallback_count": guard.cpu_fallback_count,
    }


def calibrate_entropy(
    data_path: str,
    device: str = "cuda:0",
    target_patch_len: int = 8,
    seed: int = 0,
    budget_nt: int = 2_000_000,
    batch_size: int = 256,
    predictor: Optional[EntropyPredictor] = None,
) -> EntropyCalib:
    """Train the entropy predictor and calibrate the gate on train split only.

    Returns an EntropyCalib usable by all three patch policies:
      - P1 fixed:    patch_len = round(mean_patch_len)
      - P2 random:   length_dist = length_dist (empirical pmf)
      - P3 entropy:  gate = gate with the fitted predictor
    """
    from .dataset import iter_train_batches

    if predictor is None:
        predictor, _ = train_entropy_predictor(
            data_path, device=device, budget_nt=budget_nt,
            batch_size=batch_size, seed=seed)
    predictor.eval()
    from .train_config import resolved_config
    cfg = resolved_config("F1", seed)

    # Accumulate per-position entropy over a train sample. Use nt-budgeted
    # batches so the sequential GRU stays cheap even on 4096-nt contexts, and
    # cap by total collected positions (CALIB_POSITIONS).
    all_ent = []
    n_pos = 0
    with torch.no_grad():
        for batch in iter_train_batches(
                data_path, cfg, seed=seed, batch_nt=CALIB_COLLECT_NT):
            tok = torch.tensor(batch["token_ids"], dtype=torch.long, device=device)
            ent = predictor.entropy(tok)
            all_ent.append(ent.reshape(-1))
            n_pos += ent.numel()
            if n_pos >= CALIB_POSITIONS:
                break
    ent = torch.cat(all_ent, dim=0)[:CALIB_POSITIONS]

    # Choose gate so mean patch length ~ target_patch_len.
    hbar = ent.mean().item()
    gate = target_patch_len * max(hbar, 1e-9)
    bounds = boundaries_from_entropy(ent.unsqueeze(0), gate)
    lengths = patch_lengths_from_bounds(bounds)
    mean_plen = sum(lengths) / len(lengths) if lengths else float(target_patch_len)

    # Empirical patch-length pmf (cap at max observable length).
    max_len = max(lengths) if lengths else target_patch_len
    dist = [0.0] * max_len
    for L in lengths:
        dist[min(L, max_len) - 1] += 1.0
    total = sum(dist)
    if total > 0:
        dist = [v / total for v in dist]

    ckpt_hash = _sha256_state_dict(predictor.state_dict())
    return EntropyCalib(
        gate=gate,
        mean_patch_len=mean_plen,
        length_dist=dist,
        predictor_params=sum(p.numel() for p in predictor.parameters()),
        train_nt=budget_nt,
        checkpoint_hash=ckpt_hash,
        target_patch_len=int(target_patch_len),
        predictor=predictor,
    )


class EntropyPatchPolicy:
    """PatchPolicy-compatible entropy boundary provider (P3).

    Holds the trained predictor + calibrated gate. Supports both the offline
    per-string interface (boundary) and the batched GPU interface
    (boundaries_batch) that the training loop uses for online entropy.
    """
    def __init__(self, predictor: EntropyPredictor, gate: float, device: str = "cuda:0"):
        self.predictor = predictor
        self.gate = gate
        self.device = device

    def boundaries_batch(self, nt_ids: torch.Tensor) -> torch.Tensor:
        self.predictor.eval()
        with torch.no_grad():
            ent = self.predictor.entropy(nt_ids)
        return boundaries_from_entropy(ent, self.gate)

    def boundary(self, seq: str, length: int, seq_id: int = 0) -> list[int]:
        ids = [ALPHABET.index(b) for b in seq[:length]]
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        b = self.boundaries_batch(t)[0][:length].cpu().tolist()
        return [int(round(v)) for v in b]