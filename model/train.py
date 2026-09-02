"""GPU-only training loop consuming RunConfig (contract 3.4).

- Encodes/streams train batches from the frozen split parquet.
- Applies the per-arm tokenizer / patch boundary provider.
- Optimizer: AdamW (0.9,0.95), weight_decay 0.1, bf16 autocast.
- Stop / warmup / checkpoint counted in cumulative valid target nt.
- LR: linear warmup -> cosine decay over budget_nt (no repeated tuning).
- GPU-only: device must be cuda; cpu_fallback_count stays 0.
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arms import ArmSpec
from .backbone import FlatCausalLM, BLTCausalLM, PatchInputFlatCausalLM
from .census import GPUGuard, count_params
from .dataset import (
    IGNORE, Exposure, build_tokenizer, count_valid_nt, iter_train_batches,
)
from .entropy_predictor import (
    EntropyCalib, EntropyPatchPolicy, EntropyPredictor, calibrate_entropy,
)
from .conditional_patch import ConditionalRandomPatchPolicy
from .patch import PatchPolicy


def build_model_for_cfg(cfg, device: str):
    guard = GPUGuard(device)
    guard.check()  # raises on cpu
    if cfg.arm.backbone == "flat":
        m = FlatCausalLM(
            vocab_size=cfg.arm.vocab_size, d_model=cfg.arch.d_model,
            n_layers=cfg.arch.n_layers, n_heads=cfg.arch.n_heads,
            max_len=cfg.arch.max_len, embed_dim=cfg.embed.embed_dim,
            tied_embed=cfg.embed.tied)
    elif cfg.arm.backbone == "l2":
        m = PatchInputFlatCausalLM(
            vocab_size=cfg.arm.vocab_size, d_model=cfg.arch.d_model,
            n_layers=cfg.arch.n_layers, n_heads=cfg.arch.n_heads,
            max_len=cfg.arch.max_len, embed_dim=cfg.embed.embed_dim,
            tied_embed=cfg.embed.tied)
    else:
        m = BLTCausalLM(
            vocab_size=cfg.arm.vocab_size, d_model=cfg.arch.d_model,
            n_layers=cfg.arch.n_layers, n_heads=cfg.arch.n_heads,
            max_len=cfg.arch.max_len, embed_dim=cfg.embed.embed_dim,
            tied_embed=cfg.embed.tied, patcher=None,
            default_patch_len=cfg.arch.d_model and 6)
    return m.to(device)


def _patch_policy_for_arm(cfg, calib: EntropyCalib | None = None,
                          device: str = "cuda:0",
                          p2_policy=None) -> PatchPolicy | EntropyPatchPolicy | None:
    """Return the boundary policy for a BLT arm.

    P1 derives fixed length from the train-only entropy calibration; P2 uses
    the fitted supported-strata conditional random patch (contract 3.2) when a
    fitted policy is supplied, falling back to the distribution-matched random
    for dev-only smoke tests; P3 uses the fitted entropy predictor + gate.
    """
    if cfg.arm.backbone not in ("blt", "l2"):
        return None
    t = cfg.arm.tokenizer_type
    if t == "fixed_patch":
        if cfg.arm.id == "B1":
            # Bridge B1 (contract 3.2.1): BLT with patch-size=1 (every nt is
            # its own patch), not the entropy-calibrated mean used by P1.
            return PatchPolicy(kind="fixed", patch_len=1)
        plen = 6  # contract 3.2 fixed-6 (owner 2026-08-31)
        return PatchPolicy(kind="fixed", patch_len=max(1, plen))
    if t == "random_patch":
        if p2_policy is not None:
            # Contract 3.2: supported-strata conditional random patch.
            return p2_policy
        dist = list(calib.length_dist) if calib else [0.125] * 8
        return PatchPolicy(kind="random", seed=cfg.seed, length_dist=dist)
    if t == "entropy_patch":
        if calib is None:
            # smoke-only fallback: a fresh untrained predictor + default gate
            pred = EntropyPredictor().to(device)
            return EntropyPatchPolicy(pred, gate=8.0, device=device)
        return EntropyPatchPolicy(calib.predictor, calib.gate, device=device)
    raise ValueError(t)


def _tensor(batch, device, key):
    return torch.tensor(batch[key], dtype=torch.long, device=device)


def train(cfg, data_path, device="cuda:0", batch_size=32,
          max_batches=None, log_every=50, checkpoint_interval_nt=2_000_000,
          checkpoint_dir=None, start_seed=0, calib: EntropyCalib | None = None,
          budget_nt: int | None = None, lr: float | None = None,
          peak_mem: bool = False, batch_nt: int | None = None,
          return_model: bool = False, p2_policy=None):
    """Run one full training budget. Checkpoints by valid target nt.

    `budget_nt`/`lr` optionally override the frozen config (used by smoke,
    calibration and the LR pilot; the science runs use the config values).
    """
    guard = GPUGuard(device)
    guard.check()
    model = build_model_for_cfg(cfg, device)
    policy = _patch_policy_for_arm(cfg, calib=calib, device=device,
                                   p2_policy=p2_policy)
    param_count = count_params(model)
    # P3 entropy AND P2 conditional-random compute boundaries on GPU from the
    # nt batch (they need per-position causal entropy); the dataset must not
    # emit a per-sequence boundary for these arms.
    gpu_policy = policy if isinstance(
        policy, (EntropyPatchPolicy, ConditionalRandomPatchPolicy)) else None
    entropy_policy = policy if isinstance(policy, EntropyPatchPolicy) else None

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr if lr is not None else cfg.optim.lr,
        betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay)

    budget = budget_nt if budget_nt is not None else cfg.budget_nt
    base_lr = lr if lr is not None else cfg.optim.lr
    warmup = cfg.warmup_nt
    expo = Exposure()
    step = 0
    last_ck = 0
    t0 = time.time()
    performed_nt = 0
    if peak_mem:
        torch.cuda.reset_peak_memory_stats(device)

    def lr_at(nt):
        if nt < warmup:
            return base_lr * (nt / max(1, warmup))
        progress = min(1.0, (nt - warmup) / max(1, budget - warmup))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # For P3 entropy and P2 conditional-random, boundaries are computed on GPU
    # from the nt batch; the dataset does not emit a per-sequence boundary.
    dataset_policy = None if gpu_policy is not None else policy
    batch_nt = batch_nt if batch_nt is not None else (
        getattr(cfg, "batch_nt", None) or (batch_size * cfg.context_nt))
    gen = iter_train_batches(
        data_path, cfg, boundary_provider=dataset_policy,
        batch_size=batch_size, max_batches=max_batches, seed=start_seed,
        batch_nt=batch_nt)

    for batch in gen:
        if expo.cumulative_valid_target_nt >= budget:
            break
        valid_nt = count_valid_nt(batch)
        tok = _tensor(batch, device, "token_ids")
        tgt = _tensor(batch, device, "targets")
        for g in opt.param_groups:
            g["lr"] = lr_at(expo.cumulative_valid_target_nt)
        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if cfg.arm.backbone in ("blt", "l2"):
                if gpu_policy is not None:
                    bnd = gpu_policy.boundaries_batch(tok)
                else:
                    bnd = _tensor(batch, device, "boundary").float()
                logits, _ = model(tok, bnd, targets=None)
            else:
                logits, _ = model(tok, targets=None)
            # nt-weighted CE (amendment 2026-09-02): a multi-nt token
            # contributes proportionally to the nt it covers, so the per-nt
            # objective is identical across tokenizers.
            w = torch.tensor(batch.get("nt_weights") or batch["targets"],
                             dtype=torch.float32, device=device).view(-1)
            tv = tgt.view(-1)
            m = (tv != IGNORE)
            w = torch.where(m, w, torch.zeros_like(w))
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), tv,
                ignore_index=IGNORE, reduction="none")
            loss = (ce * w).sum() / w.sum().clamp(min=1.0)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
        opt.step()

        expo.add_batch(valid_nt, len(batch["token_ids"]))
        performed_nt += valid_nt
        step += 1

        if checkpoint_interval_nt and expo.cumulative_valid_target_nt - last_ck >= checkpoint_interval_nt:
            last_ck = expo.cumulative_valid_target_nt
            if checkpoint_dir:
                ck = os.path.join(checkpoint_dir, "step_%06d.pt" % step)
                torch.save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "nt": expo.cumulative_valid_target_nt,
                    "loss": loss.item(),
                    "cfg_run_id": cfg.run_id,
                }, ck)

        if log_every and step % log_every == 0:
            print("[%s] step=%d nt=%d loss=%.4f lr=%.2e %.1f nt/s" % (
                cfg.run_id, step, expo.cumulative_valid_target_nt,
                loss.item(), opt.param_groups[0]["lr"],
                performed_nt / max(1e-9, time.time() - t0)))

    ret = {
        "run_id": cfg.run_id,
        "steps": step,
        "cumulative_valid_target_nt": expo.cumulative_valid_target_nt,
        "final_loss": loss.item() if loss is not None else None,
        "params": param_count.total_params,
        "device": device,
        "cpu_fallback_count": guard.cpu_fallback_count,
        "wall_seconds": time.time() - t0,
    }
    if peak_mem:
        ret["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if return_model:
        ret["model"] = model
    return ret


def validate_on_split(cfg, model, data_path, device="cuda:0",
                      calib: EntropyCalib | None = None,
                      split: str = "validation", val_nt: int = 2_000_000,
                      batch_nt: int = 8192, seed: int = 0,
                      p2_policy=None) -> dict:
    """Evaluation of a trained model on a held-out split (contract 3.4).

    Streams the given split (default validation) once, computes a cumulative
    nt-weighted loss (same objective as training: mean CE over valid targets),
    and returns the scalar. Used to select hyper-parameters (LR) on validation
    only, never on test. GPU-only; cpu_fallback_count must stay 0.
    """
    guard = GPUGuard(device)
    guard.check()
    policy = _patch_policy_for_arm(cfg, calib=calib, device=device,
                                   p2_policy=p2_policy)
    gpu_policy = policy if isinstance(
        policy, (EntropyPatchPolicy, ConditionalRandomPatchPolicy)) else None
    dataset_policy = None if gpu_policy is not None else policy
    model.eval()

    gen = iter_train_batches(
        data_path, cfg, boundary_provider=dataset_policy,
        batch_size=8, max_batches=None, seed=seed, split=split, batch_nt=batch_nt)

    total_nll = 0.0
    total_nt = 0
    n_batches = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in gen:
            tok = _tensor(batch, device, "token_ids")
            tgt = _tensor(batch, device, "targets")
            if cfg.arm.backbone in ("blt", "l2"):
                if gpu_policy is not None:
                    bnd = gpu_policy.boundaries_batch(tok)
                else:
                    bnd = _tensor(batch, device, "boundary").float()
                logits, _ = model(tok, bnd, targets=None)
            else:
                logits, _ = model(tok, targets=None)
            w = torch.tensor(batch.get("nt_weights") or batch["targets"],
                             dtype=torch.float32, device=device).view(-1)
            tv = tgt.view(-1)
            m = (tv != IGNORE)
            w = torch.where(m, w, torch.zeros_like(w))
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), tv,
                ignore_index=IGNORE, reduction="none")
            vnt = count_valid_nt(batch)
            total_nll += float((ce * w).sum().item())
            total_nt += vnt
            n_batches += 1
            if total_nt >= val_nt:
                break
    return {
        "val_loss": total_nll / max(1, total_nt),
        "val_nt": total_nt,
        "val_batches": n_batches,
        "cpu_fallback_count": guard.cpu_fallback_count,
    }


def smoke_run(cfg, data_path, device="cuda:0", budget_nt=100_000, batch_size=16):
    """Small GPU smoke: verify forward/backward/opt over a tiny budget."""
    import copy
    small = copy.copy(cfg)
    object.__setattr__(small, "budget_nt", budget_nt)
    return train(small, data_path, device=device, batch_size=batch_size,
                 checkpoint_interval_nt=0, log_every=None, max_batches=None)