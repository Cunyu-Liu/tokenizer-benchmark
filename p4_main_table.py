"""Phase 4 sealed-test main-table aggregator (contract 3.7, 5.4).

Synthesizes the three main tables from per-arm eval artifacts + run manifests:

  A. Internal controlled causal table (F1-F7 static, P1-P3 patch), 100M/350M separate.
     Columns: arm, params, non_embedding_params, budget_nt, best_val_loss,
              sealed likelihood (next_base_BPN / canonical_path_BPN /
              overlap_path_BPN), uniqueness, validity, throughput_nt_s,
              peak_vram_mb, cpu_fallback_count.
  B. Public autoregressive/generative model reference table (external models;
     filled when external adapters are run).
  C. family/structure-conditioned table (external, filled separately).

Reads:
  - run manifest(s): per (arm, seed) run dir under --runs, selecting the best
    validation checkpoint's manifest entry (best_val_loss, best_checkpoint,
    params, throughput, peak_vram).
  - sealed-test likelihood results: expected at
        {--results}/{arm}_{seed}_{split}.json   (from p4_eval.py)
  - generation results: expected at
        {--results}/gen_{arm}_{seed}.json       (from p4_generate.py)

Produces a JSON registry + a human-readable markdown table. Missing cells are
reported (NOT silently dropped). All numbers are means over the 3 seeds of a
cell; per-seed rows are retained in the JSON.

Read-only over experiment artifacts; writes only the output table.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time

ARMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "P1", "P2", "P3"]
SEEDS = [17, 29, 43]
RUN_RE = re.compile(r"^phase4_(F[1-7]|P[1-3])_s(\d+)_\d{8}T\d{6}$")

# Which likelihood metric each arm reports (contract 3.5).
METRIC_BY_ARM = {
    "F1": "next_base_BPN", "F4": "overlap_path_BPN", "F5": "overlap_path_BPN",
    "F2": "canonical_path_BPN", "F3": "canonical_path_BPN",
    "F6": "canonical_path_BPN", "F7": "canonical_path_BPN",
    "P1": "next_base_BPN", "P2": "next_base_BPN", "P3": "next_base_BPN",
}


def load_run_manifest(run_dir: str) -> dict | None:
    mf = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mf):
        return None
    with open(mf) as fh:
        return json.load(fh)


def best_checkpoint_entry(m: dict) -> dict | None:
    """The validation-selected best checkpoint entry + params/throughput."""
    if not m:
        return None
    ck = m.get("best_checkpoint")
    if not ck:
        return None
    return {
        "best_checkpoint": ck,
        "best_val_loss": m.get("best_val_loss"),
        "final_nt": m.get("final_nt"),
        "params": m.get("params"),
        "non_embedding_params": m.get("non_embedding_params"),
        "throughput_nt_s": m.get("throughput_nt_s"),
        "peak_vram_mb": m.get("peak_vram_mb"),
        "cpu_fallback_count": m.get("cpu_fallback_count"),
        "status": m.get("status"),
    }


def load_likelihood(results_dir: str, arm: str, seed: int, split: str) -> dict | None:
    p = os.path.join(results_dir, "%s_%d_%s.json" % (arm, seed, split))
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def load_generation(results_dir: str, arm: str, seed: int) -> dict | None:
    p = os.path.join(results_dir, "gen_%s_%d.json" % (arm, seed))
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_table_a(runs_dir: str, results_dir: str, split: str) -> dict:
    """Aggregate A (internal controlled) across the 3 seeds of each arm."""
    cells = []
    for arm in ARMS:
        seed_rows = []
        for seed in SEEDS:
            # find the (arm, seed) run dir
            dirs = [d for d in os.listdir(runs_dir)
                    if RUN_RE.match(d) and RUN_RE.match(d).group(1) == arm
                    and int(RUN_RE.match(d).group(2)) == seed]
            if not dirs:
                seed_rows.append({"seed": seed, "status": "MISSING_RUN_DIR"})
                continue
            d = os.path.join(runs_dir, sorted(dirs)[-1])
            m = load_run_manifest(d)
            bce = best_checkpoint_entry(m)
            lik = load_likelihood(results_dir, arm, seed, split)
            gen = load_generation(results_dir, arm, seed)
            row = {"seed": seed, "run_dir": d}
            if bce is None:
                row["status"] = "NO_BEST_CKPT"
            else:
                row.update(bce)
                row["status"] = bce["status"]
            if lik is not None:
                row["likelihood_metric"] = lik.get("metric")
                row["likelihood_bpn"] = lik.get("bpn")
                row["likelihood_n_sequences"] = lik.get("n_sequences")
                row["likelihood_valid_nt"] = lik.get("valid_nt_count")
            if gen is not None:
                row["gen_unique"] = gen.get("uniqueness")
                row["gen_validity"] = gen.get("validity_rate")
            seed_rows.append(row)
        # mean over completed seeds
        bpn = _mean([r.get("likelihood_bpn") for r in seed_rows])
        val = _mean([r.get("best_val_loss") for r in seed_rows])
        params = _mean([r.get("params") for r in seed_rows])
        nonemb = _mean([r.get("non_embedding_params") for r in seed_rows])
        thr = _mean([r.get("throughput_nt_s") for r in seed_rows])
        vram = _mean([r.get("peak_vram_mb") for r in seed_rows])
        uniq = _mean([r.get("gen_unique") for r in seed_rows])
        cells.append({
            "arm": arm, "metric": METRIC_BY_ARM[arm],
            "mean_likelihood_bpn": bpn, "mean_best_val_loss": val,
            "mean_params": params, "mean_non_embedding": nonemb,
            "mean_throughput_nt_s": thr, "mean_peak_vram_mb": vram,
            "mean_uniqueness": uniq,
            "seeds": seed_rows,
        })
    return {"table": "A", "split": split, "arms": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/mnt/cunyuliu/tokenizer-benchmark/runs")
    ap.add_argument("--results", required=True,
                    help="dir with per-arm sealed-test likelihood + generation JSONs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    table_a = build_table_a(args.runs, args.results, args.split)
    out = {
        "phase": 4,
        "split": args.split,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "table_A": table_a,
        "table_B": {"table": "B", "note": "public external models; populated when external adapters run"},
        "table_C": {"table": "C", "note": "family/structure-conditioned; populated separately"},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    # human-readable markdown
    md = ["# Phase 4 main table A (internal controlled), split=%s\n" % args.split,
          "| arm | metric | bpn | val_loss | params | non-emb | nt/s | vram(MB) | uniq |",
          "|-----|--------|-----|----------|--------|---------|------|----------|------|"]
    for c in table_a["arms"]:
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c["arm"], c["metric"],
            _fmt(c["mean_likelihood_bpn"]), _fmt(c["mean_best_val_loss"]),
            _fmt(c["mean_params"], 0), _fmt(c["mean_non_embedding"], 0),
            _fmt(c["mean_throughput_nt_s"], 0), _fmt(c["mean_peak_vram_mb"], 0),
            _fmt(c["mean_uniqueness"])))
    md_path = args.out.rsplit(".", 1)[0] + ".md"
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("WROTE", args.out)
    print("WROTE", md_path)


if __name__ == "__main__":
    main()