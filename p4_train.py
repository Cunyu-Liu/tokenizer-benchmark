"""Phase 4: 100M full-matrix training runner (contract Phase 4 gate).

Runs ONE arm/seed to the FULL frozen budget (2.0B cumulative valid target nt)
on GPU, with periodic validation-based checkpoint selection (best = lowest
validation loss on the held-out validation split; never test). Emits a run
bundle (checkpoints + manifest) under /mnt/cunyuliu.

Frozen hyperparameters come from the Phase 3 acceptance (contract 3.4):
  - LR per arm from LR_FACTOR_SELECTED (train_config)
  - AdamW (0.9,0.95), wd 0.1, bf16, context 4096, batch_nt 32768
  - budget = 2.0B valid nt; warmup = 0.5% of budget; cosine decay on nt

GPU-only: device must be cuda; cpu_fallback_count must stay 0 everywhere.

Usage:
  python p4_train.py --arm F1 --seed 17 --device 0 --out-dir /mnt/.../P4_F1_s17
  python p4_train.py --arm P3 --seed 17 --device 2 --out-dir /mnt/.../P4_P3_s17
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from model.train_config import resolved_config
from model.train import (
    build_model_for_cfg, _patch_policy_for_arm, validate_on_split,
)
from model.census import GPUGuard, count_params
from model.dataset import count_valid_nt, iter_train_batches
from model.entropy_predictor import EntropyCalib, calibrate_entropy
from model.conditional_patch import fit_p2_q, ConditionalRandomPatchPolicy

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
# Phase 4 defaults
VALIDATION_NT = 4_000_000      # nt budget for each validation eval
VAL_INTERVAL_NT = 100_000_000  # validate every 100M valid nt (20/run at 2B)
ENTROPY_BUDGET_NT = 2_000_000  # train-only entropy calibration budget (BLT arms)
# P2 supported-strata conditional-random q-fit sample size (train-only, frozen).
P2_QFIT_N_SEQS = 2000


def _device(dev) -> str:
    d = "cuda:%d" % dev if isinstance(dev, int) else dev
    assert torch.cuda.is_available(), "CUDA required"
    return d


def entropy_calib_for_arm(arm_id: str, device: str) -> EntropyCalib | None:
    """Train-only entropy calibration for BLT arms P1/P2/P3 (deterministic)."""
    if arm_id not in ("P1", "P2", "P3"):
        return None
    return calibrate_entropy(SPLIT_8080, device=device, target_patch_len=8,
                             seed=17, budget_nt=ENTROPY_BUDGET_NT, batch_size=256)


def _tensor(batch, device, key):
    return torch.tensor(batch[key], dtype=torch.long, device=device)


def run_single(arm_id: str, seed: int, device: str, out_dir: str,
               val_interval_nt: int = VAL_INTERVAL_NT,
               val_nt: int = VALIDATION_NT,
               smoke_nt: int | None = None):
    guard = GPUGuard(device)
    guard.check()
    cfg = resolved_config(arm_id, seed)  # frozen Phase 3 config (budget 2.0B, batch_nt 32768)
    if smoke_nt is not None:
        cfg = cfg.__class__(
            run_id=cfg.run_id, scale=cfg.scale, arm=cfg.arm, seed=cfg.seed,
            arch=cfg.arch, embed=cfg.embed, optim=cfg.optim,
            budget_nt=smoke_nt, batch_nt=cfg.batch_nt, warmup_nt=cfg.warmup_nt,
            context_nt=cfg.context_nt, entropy_predictor_params=cfg.entropy_predictor_params)
    os.makedirs(out_dir, exist_ok=True)

    calib = entropy_calib_for_arm(arm_id, device)
    # P2: fit the supported-strata conditional-random q on train-only data
    # (contract 3.2); the fitted policy is shared by all three P2 model seeds
    # via the frozen fit rule, and is persisted with each checkpoint.
    p2_policy = None
    p2_coverage = None
    if arm_id == "P2" and calib is not None:
        p2_policy, p2_coverage = fit_p2_q(
            calib, SPLIT_8080, device=device, n_seqs=P2_QFIT_N_SEQS, seed=17)
        print("[P2 qfit] supported coverage: pos=%.3f bnd=%.3f pass=%s" % (
            p2_coverage.get("position_coverage", float("nan")),
            p2_coverage.get("boundary_coverage", float("nan")),
            p2_coverage.get("passes_support_coverage", False)), flush=True)
    model = build_model_for_cfg(cfg, device)
    policy = _patch_policy_for_arm(cfg, calib=calib, device=device,
                                   p2_policy=p2_policy)
    # P3 entropy AND P2 conditional-random compute boundaries on GPU from the
    # nt batch (they need per-position causal entropy).
    gpu_policy = policy if isinstance(
        policy, (ConditionalRandomPatchPolicy,
                 __import__("model.entropy_predictor", fromlist=["EntropyPatchPolicy"]).EntropyPatchPolicy)
    ) else None
    dataset_policy = None if gpu_policy is not None else policy
    param_count = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay)
    budget = cfg.budget_nt
    base_lr = cfg.optim.lr
    warmup = cfg.warmup_nt
    batch_nt = cfg.batch_nt

    def lr_at(nt):
        if nt < warmup:
            return base_lr * (nt / max(1, warmup))
        progress = min(1.0, (nt - warmup) / max(1, budget - warmup))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    gen = iter_train_batches(SPLIT_8080, cfg, boundary_provider=dataset_policy,
                             batch_size=cfg.batch_size_seq, max_batches=None,
                             seed=0, batch_nt=batch_nt)

    cumulative_nt = 0
    step = 0
    last_val_nt = 0
    best_val = float("inf")
    best_ck = None
    t0 = time.time()
    perf_nt = 0
    torch.cuda.reset_peak_memory_stats(device)

    manifest = {
        "phase": 4, "run_id": cfg.run_id, "arm": arm_id, "seed": seed,
        "device": device, "scale": cfg.scale,
        "config": cfg.to_dict(),
        "checkpoints": [], "validations": [],
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for batch in gen:
        if cumulative_nt >= budget:
            break
        valid_nt = count_valid_nt(batch)
        tok = _tensor(batch, device, "token_ids")
        tgt = _tensor(batch, device, "targets")
        for g in opt.param_groups:
            g["lr"] = lr_at(cumulative_nt)
        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if cfg.arm.backbone == "blt":
                if gpu_policy is not None:
                    bnd = gpu_policy.boundaries_batch(tok)
                else:
                    bnd = _tensor(batch, device, "boundary").float()
                logits, loss = model(tok, bnd, targets=tgt)
            else:
                logits, loss = model(tok, targets=tgt)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
        opt.step()

        cumulative_nt += valid_nt
        perf_nt += valid_nt
        step += 1

        # Validation-based checkpoint selection (best = lowest validation loss).
        if val_interval_nt and cumulative_nt - last_val_nt >= val_interval_nt:
            last_val_nt = cumulative_nt
            v = validate_on_split(cfg, model, SPLIT_8080, device=device,
                                  calib=calib, split="validation",
                                  val_nt=VALIDATION_NT, batch_nt=batch_nt,
                                  p2_policy=p2_policy)
            assert v["cpu_fallback_count"] == 0, "cpu fallback in val %s" % arm_id
            val_entry = {"nt": cumulative_nt, "step": step, "val_loss": v["val_loss"],
                         "val_nt": v["val_nt"], "val_batches": v["val_batches"]}
            manifest["validations"].append(val_entry)
            ck_path = os.path.join(out_dir, "ckpt_nt%09d_step%06d.pt" % (cumulative_nt, step))
            payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                       "nt": cumulative_nt, "step": step,
                       "val_loss": v["val_loss"], "cfg_run_id": cfg.run_id,
                       "arm": arm_id, "seed": seed}
            if calib is not None:
                # Persist the train-only entropy calibration with the checkpoint so
                # P1/P2/P3 inference reproduces the exact patch boundaries (the
                # fitted predictor + calibrated gate + empirical length stats).
                payload["calib"] = {
                    "predictor_state": calib.predictor.state_dict(),
                    "gate": calib.gate,
                    "mean_patch_len": calib.mean_patch_len,
                    "length_dist": calib.length_dist,
                    "checkpoint_hash": calib.checkpoint_hash,
                }
            if p2_policy is not None:
                # Persist P2's frozen conditional-random q + bin edges so eval
                # replays the exact same supported-strata random boundaries.
                payload["p2_q"] = {
                    "q_table": p2_policy.q_table,
                    "prefix_edges": p2_policy.prefix_edges,
                    "ent_edges": p2_policy.ent_edges,
                    "seed": p2_policy.seed,
                    "coverage_report": p2_policy.coverage_report,
                }
            torch.save(payload, ck_path)
            manifest["checkpoints"].append({"nt": cumulative_nt, "step": step,
                                            "val_loss": v["val_loss"], "path": ck_path,
                                            "calib_persisted": calib is not None})
            if v["val_loss"] < best_val:
                best_val = v["val_loss"]
                best_ck = ck_path
            print("[%s] nt=%d step=%d val=%.4f (best=%.4f %s)" % (
                cfg.run_id, cumulative_nt, step, v["val_loss"], best_val,
                os.path.basename(best_ck)))
            # Write manifest incrementally so partial progress survives a crash.
            with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=2, default=str)

    final_loss = loss.item() if loss is not None else None
    guard.check()
    manifest.update({
        "final_nt": cumulative_nt,
        "final_step": step,
        "final_loss": final_loss,
        "best_val_loss": best_val if best_val != float("inf") else None,
        "best_checkpoint": best_ck,
        "params": param_count.total_params,
        "non_embedding_params": param_count.non_embedding_params,
        "throughput_nt_s": perf_nt / max(1e-9, time.time() - t0),
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "wall_seconds": time.time() - t0,
        "cpu_fallback_count": guard.cpu_fallback_count,
        "calib_persisted": calib is not None,
        "p2_qfit_coverage": p2_coverage,
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DONE",
    })
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print("DONE %s | nt=%d steps=%d final_loss=%.4f best_val=%.4f | %.0f nt/s peak=%.0fMB fallback=%d" % (
        cfg.run_id, cumulative_nt, step, final_loss if final_loss is not None else -1,
        best_val if best_val != float("inf") else -1,
        manifest["throughput_nt_s"], manifest["peak_vram_mb"], guard.cpu_fallback_count))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-interval-nt", type=int, default=VAL_INTERVAL_NT)
    ap.add_argument("--val-nt", type=int, default=VALIDATION_NT)
    ap.add_argument("--smoke-nt", type=int, default=None,
                    help="override budget to a small nt for verification (dev only)")
    args = ap.parse_args()

    dev = _device(args.device)
    manifest = run_single(args.arm, args.seed, dev, args.out_dir,
                          val_interval_nt=args.val_interval_nt,
                          val_nt=args.val_nt, smoke_nt=args.smoke_nt)
    print("MANIFEST", os.path.join(args.out_dir, "manifest.json"))


if __name__ == "__main__":
    main()