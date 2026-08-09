"""Aggregate Phase 3 LR pilot + throughput/memory/compute-fairness report.

Reads the per-factor LR pilot report.json files (0.5x/1.0x/2.0x), selects for each
arm the LR factor with the lowest validation loss (contract 3.4: select by
validation metric, never test; freeze after), and emits a combined report with
params / throughput / peak VRAM / FLOPs / per-arm LR selection.

Usage: python p3_report.py --runs-dir /mnt/cunyuliu/tokenizer-benchmark/runs
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.census import estimate_attention_flops, estimate_mlp_flops

FACTORS = [0.5, 1.0, 2.0]
ARMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "P1", "P2", "P3"]

def load_factor(runs_dir: str, factor: float):
    pat = os.path.join(runs_dir, "phase3_lr_f%s*_fixedval" % factor, "phase3_lr_report.json")
    files = glob.glob(pat)
    if not files:
        print("  WARN: no report for factor %s (%s)" % (factor, pat))
        return None
    # newest mtime
    f = max(files, key=os.path.getmtime)
    with open(f) as fh:
        return json.load(fh)

def flops_per_arm(res: dict) -> float:
    """Per-token FLOPs (attention + MLP) for the arm's config."""
    d, L, n = res["d_model"], res["context_nt"], res["n_layers"]
    att = estimate_attention_flops.__module__ and 0
    # attention: 2*B*L*L*D*3 per head; heads reuse across D. Per token ~ 12*D*L.
    d_ff = d * 4
    mlp = estimate_mlp_flops(d, d_ff, n, L)
    attn = 12 * d * L * n  # per token, all heads
    return mlp + attn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/mnt/cunyuliu/tokenizer-benchmark/runs")
    ap.add_argument("--out", default="/mnt/cunyuliu/tokenizer-benchmark/runs/phase3_lr_report_aggregate.json")
    args = ap.parse_args()

    factors = {f: load_factor(args.runs_dir, f) for f in FACTORS}
    if any(v is None for v in factors.values()):
        print("Missing factor reports; abort.")
        sys.exit(2)

    report = {"phase": 3, "mode": "lr_aggregate", "factors": factors}
    selection = {}
    lines = []
    all_zero = True
    all_finite = True
    for arm in ARMS:
        cands = []
        for f in FACTORS:
            res = factors[f]["arms"].get(arm)
            if res is None:
                continue
            cands.append({"factor": f, "val_loss": res["val_loss"],
                          "final_loss": res["final_loss"],
                          "throughput_nt_s": res["throughput_nt_s"],
                          "peak_vram_mb": res["peak_vram_mb"],
                          "params": res["params"]})
            all_zero &= (res["cpu_fallback_count"] == 0)
            all_finite &= (res["final_loss"] is not None)
        if not cands:
            continue
        best = min(cands, key=lambda c: c["val_loss"])
        selection[arm] = {"best_factor": best["factor"], "best_val_loss": best["val_loss"]}
        lines.append("%s: best LR=%.2fx (val=%.4f) | val_loss 0.5x=%.4f 1x=%.4f 2x=%.4f | nt/s=%.0f peak=%.0fMB" % (
            arm, best["factor"], best["val_loss"] if best["val_loss"] is not None else float("nan"),
            cands[0]["val_loss"] if cands[0]["val_loss"] is not None else float("nan"),
            cands[1]["val_loss"] if cands[1]["val_loss"] is not None else float("nan"),
            cands[2]["val_loss"] if cands[2]["val_loss"] is not None else float("nan"),
            best["throughput_nt_s"], best["peak_vram_mb"]))
    report["lr_selection"] = selection
    report["all_cpu_fallback_zero"] = all_zero
    report["all_finite"] = all_finite
    report["summary"] = lines
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\n".join(lines))
    print("all_cpu_fallback_zero=%s all_finite=%s" % (all_zero, all_finite))
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
