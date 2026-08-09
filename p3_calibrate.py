"""Phase 3 per-arm GPU smoke + calibration + LR pilot runner (contract 4).

Runs on the real frozen train split. For each arm in the 100M ten-arm matrix:
  - resolves the frozen seed-17 config,
  - (BLT P1/P2/P3) derives the train-only entropy calibration ONCE and shares it
    across the three BLT arms (they use the same seed/data/target_patch_len),
  - runs a short GPU smoke then a calibration trajectory of `steps` optimizer
    steps (bounded by max_batches, so smoke=20 steps stays genuinely 20 steps),
  - records params, throughput (nt/s), peak VRAM, final loss, cpu_fallback_count.

Per-arm results are written incrementally so partial progress survives a crash
or timeout; output is flushed for real-time monitoring.

smoke / calib / lr are dev-only trajectories governed by an explicit optimizer-
step count. The science runs use the frozen cfg.budget_nt and are launched
separately. cpu_fallback_count must stay 0 on every neural arm.

Usages:
  python p3_calibrate.py --mode smoke                 # 20-step smoke per arm
  python p3_calibrate.py --mode calib --steps 200     # 200-step calibration
  python p3_calibrate.py --mode lr --lr-factor 0.5    # LR pilot candidate
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, ".")

import torch

from model import train_config as tc
from model.train import train, validate_on_split
from model.entropy_predictor import calibrate_entropy, EntropyCalib

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
ARMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "P1", "P2", "P3"]
# Contract 3.4: LR candidates are selected by a validation metric. val metrics
# are computed on a fixed validation-nt budget, never on test.
VAL_NT = 2_000_000


def _device(dev: str) -> str:
    d = "cuda:%d" % dev if isinstance(dev, int) else dev
    assert torch.cuda.is_available(), "CUDA required"
    return d


def entropy_calib(dev: str, budget_nt: int, seed: int = 17) -> EntropyCalib:
    return calibrate_entropy(SPLIT_8080, device=dev, target_patch_len=8,
                             seed=seed, budget_nt=budget_nt, batch_size=256)


class EntropyCache:
    """Compute the train-only entropy calibration once, share across BLT arms."""

    def __init__(self, dev: str, budget_nt: int, seed: int = 17):
        self.dev, self.budget_nt, self.seed = dev, budget_nt, seed
        self._c: EntropyCalib | None = None

    def get(self) -> EntropyCalib:
        if self._c is None:
            t0 = time.time()
            self._c = entropy_calib(self.dev, self.budget_nt, self.seed)
            print("  [calib] gate=%.3f mean_plen=%.2f pred_params=%d (%.1fs)" % (
                self._c.gate, self._c.mean_patch_len, self._c.predictor_params,
                time.time() - t0), flush=True)
        return self._c


def run_arm(arm_id: str, dev: str, mode: str, steps: int, lr_factor: float,
            ent_cache: EntropyCache, out_dir: str, batch_nt: int) -> dict:
    cfg = tc.resolved_config(arm_id, 17)
    batch_size = cfg.batch_size_seq
    calib = None
    if cfg.arm.backbone == "blt":
        calib = ent_cache.get()

    # smoke/calib/lr are dev-only and bounded by an explicit optimizer-step
    # count (max_batches). A large budget upper-bound ensures the step count
    # governs and the run stops after `steps` optimizer steps.
    small = copy.copy(cfg)
    object.__setattr__(small, "budget_nt", steps * batch_nt * 100)
    base_lr = cfg.optim.lr * lr_factor

    res = train(small, SPLIT_8080, device=dev, batch_size=batch_size,
                max_batches=steps, checkpoint_interval_nt=0, log_every=None,
                calib=calib, budget_nt=small.budget_nt, lr=base_lr,
                peak_mem=True, batch_nt=batch_nt, return_model=(mode == "lr"))
    nt = res["cumulative_valid_target_nt"]
    if mode == "lr":
        # Contract 3.4: select LR by a validation metric, never train loss.
        model = res.pop("model")
        v = validate_on_split(small, model, SPLIT_8080, device=dev, calib=calib,
                              split="validation", val_nt=VAL_NT, batch_nt=batch_nt)
        res["val_loss"] = v["val_loss"]
        res["val_nt"] = v["val_nt"]
        res["val_batches"] = v["val_batches"]
        assert v["cpu_fallback_count"] == 0, "cpu fallback in val %s" % arm_id
        del model
        torch.cuda.empty_cache()
    res["arm"] = arm_id
    res["mode"] = mode
    res["batch_size_seq"] = batch_size
    res["batch_nt"] = batch_nt
    res["throughput_nt_s"] = nt / max(1e-9, res["wall_seconds"])
    res["lr"] = base_lr
    res["lr_factor"] = lr_factor
    res["non_embedding_params"] = cfg.arch.n_layers * (
        4 * cfg.arch.d_model ** 2 + 2 * cfg.arch.d_model * cfg.arch.d_ff + 4 * cfg.arch.d_model
    ) + 2 * cfg.arch.d_model
    res["entropy_predictor_params"] = cfg.entropy_predictor_params
    res["d_model"] = cfg.arch.d_model
    res["n_layers"] = cfg.arch.n_layers
    assert res["cpu_fallback_count"] == 0, "cpu fallback in %s" % arm_id
    assert res["final_loss"] is not None and torch.isfinite(torch.tensor(res["final_loss"])), \
        "NaN/Inf in %s" % arm_id
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "calib", "lr"], default="smoke")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--lr-factor", type=float, default=1.0)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--entropy-budget", type=int, default=2_000_000)
    ap.add_argument("--batch-nt", type=int, default=8192,
                    help="per-step effective nt budget (adaptive batching)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    dev = _device(args.device)
    arms = ARMS if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    out_dir = args.out_dir or os.path.join(
        "/mnt/cunyuliu/tokenizer-benchmark/runs",
        "phase3_%s_%s" % (args.mode, time.strftime("%Y%m%dT%H%M%S")))
    os.makedirs(out_dir, exist_ok=True)

    print("Phase3 [%s] arms=%s device=%s steps=%d lr_factor=%s" % (
        args.mode, arms, dev, args.steps, args.lr_factor), flush=True)

    ent_cache = EntropyCache(dev, args.entropy_budget)
    results = {}
    for arm in arms:
        print("=== arm %s (%s) ===" % (arm, args.mode), flush=True)
        r = run_arm(arm, dev, args.mode, args.steps, args.lr_factor,
                    ent_cache, out_dir, args.batch_nt)
        results[arm] = r
        partial = {
            "phase": 3, "mode": args.mode, "device": dev,
            "split": SPLIT_8080, "steps": args.steps, "lr_factor": args.lr_factor,
            "arms": results,
            "all_cpu_fallback_zero": all(x["cpu_fallback_count"] == 0 for x in results.values()),
            "all_finite": all(x["final_loss"] is not None for x in results.values()),
        }
        with open(os.path.join(out_dir, "phase3_%s_partial.json" % args.mode), "w") as f:
            json.dump(partial, f, indent=2, default=str)
        print("  -> steps=%d nt=%d loss=%.4f val=%.4f %.2f nt/s peak_vram=%.0fMB fallback=%d" % (
            r["steps"], r["cumulative_valid_target_nt"], r["final_loss"],
            r.get("val_loss", float("nan")),
            r["throughput_nt_s"], r.get("peak_vram_mb", -1), r["cpu_fallback_count"]), flush=True)

    manifest = {
        "phase": 3,
        "mode": args.mode,
        "device": dev,
        "gpu_uuid": _gpu_uuid(dev),
        "split": SPLIT_8080,
        "steps": args.steps,
        "lr_factor": args.lr_factor,
        "arms": results,
        "all_cpu_fallback_zero": all(r["cpu_fallback_count"] == 0 for r in results.values()),
        "all_finite": all(r["final_loss"] is not None for r in results.values()),
    }
    with open(os.path.join(out_dir, "phase3_%s_report.json" % args.mode), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("WROTE report:", os.path.join(out_dir, "phase3_%s_report.json" % args.mode), flush=True)
    assert manifest["all_cpu_fallback_zero"], "cpu fallback detected"
    assert manifest["all_finite"], "non-finite loss detected"
    print("PHASE3_%s_OK" % args.mode.upper(), flush=True)


def _gpu_uuid(dev: str) -> str:
    try:
        idx = int(dev.split(":")[-1])
        return torch.cuda.get_device_properties(idx).uuid if hasattr(
            torch.cuda.get_device_properties(idx), "uuid") else str(idx)
    except Exception:
        return dev


if __name__ == "__main__":
    main()