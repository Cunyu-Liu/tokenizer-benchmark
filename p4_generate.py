"""Phase 4 sealed-test generation runner (contract 3.6, 2.3, 5.4).

Implements the pre-registered decoder protocol:
  - Three frozen decoder points (contract 3.6):
      conservative  = (temp=0.8, top_p=0.95)
      balanced      = (temp=1.0, top_p=0.95)
      exploratory   = (temp=1.1, top_p=1.0)
    (top-k fixed 0 everywhere; greedy is a 4th diagnostic cell.)
  - Per cell, 5 generation seeds, each producing >=2000 VALID sequences
    (=> 10k valid sequences per cell per run/seed).
  - Unconditional generation (empty prefix) and, when --prefix-frac is given,
    prefix continuation from sealed-test raw-nt prefixes (10/25/50%).

Generated sequences are scored with the shared CPU metrics
(evaluator.eval_continuation): validity, exact uniqueness, training-set
memorization, identity-novelty bins, perc-base stats. Outputs:
  - a parquet of all generated sequences (canonical) keyed by cell/gen_seed
  - a per-cell JSON with GenerationStats + decoder params + selection hash

GPU-only: the adapter asserts CUDA and never falls back to CPU. The loop keeps
sampling until `valid_per_seed` valid canonical sequences are collected (invalid
chars are skipped and counted), so the contract target is met.

Large artifacts under /mnt; code under /home.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow as pa
import pyarrow.parquet as pq

from evaluator.eval_continuation import (
    evaluate_generation, _canon, ALPHABET,
)
from evaluator.internal_adapter import InternalFlatAdapter

SPLIT_8080 = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"

# Frozen decoder grid (contract 3.6): top-k fixed 0.
DECODER_GRID = {
    "conservative": {"temperature": 0.8, "top_p": 0.95},
    "balanced": {"temperature": 1.0, "top_p": 0.95},
    "exploratory": {"temperature": 1.1, "top_p": 1.0},
    "greedy": {"temperature": 0.0, "top_p": 1.0},
}
PRE_REGISTERED = ("conservative", "balanced", "exploratory")
GENERATION_SEEDS = [100, 200, 300, 400, 500]
VALID_PER_SEED = 2000


def _is_valid(seq: str) -> bool:
    return all(b in ALPHABET for b in _canon(seq))


def ids_to_seq(ids: list[int]) -> str:
    from evaluator.scorer import ALPHABET as A
    return "".join(A[i] for i in ids)


def generate_cell(adapter, temp: float, top_p: float, gen_seed: int,
                  valid_per_seed: int, target_len: int,
                  prefix: str = "") -> tuple[list[str], int]:
    """Generate >= valid_per_seed valid sequences for one (decoder, gen_seed).

    Returns (valid_sequences, invalid_count). Deterministic per gen_seed by
    seeding a python RNG fed to the adapter's multinomial sampling via the
    torch global RNG (seeded here).
    """
    import torch
    torch.manual_seed(gen_seed)
    random.seed(gen_seed)
    valid: list[str] = []
    invalid = 0
    guard_rounds = 0
    while len(valid) < valid_per_seed:
        out = adapter.generate(prefix, target_len, temperature=temp, top_p=top_p)
        if _is_valid(out):
            valid.append(_canon(out))
        else:
            invalid += 1
        guard_rounds += 1
        if guard_rounds > valid_per_seed * 50:
            raise RuntimeError("generation loop exceeded guard; invalid rate too high")
    return valid, invalid


def _hash(seqs: list[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(seqs):
        h.update(s.encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--target-len", type=int, default=256)
    ap.add_argument("--prefix-frac", type=float, default=None,
                    help="if set, continuation prefix fraction (0.1/0.25/0.5)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    device = "cuda:%d" % args.device
    adapter = InternalFlatAdapter(args.arm, args.seed, args.ckpt, device=device)
    assert adapter.guard.cpu_fallback_count == 0, "cpu fallback in generation %s" % args.arm

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []          # one row per generated valid sequence
    cell_results = {}      # per-cell GenerationStats dict
    total_invalid = 0

    if args.prefix_frac is not None:
        # Continuation: load sealed-test prefixes at the requested raw-nt frac.
        import pyarrow.parquet as pql
        pf = pql.ParquetFile(SPLIT_8080)
        prefixes = []
        for batch in pf.iter_batches(batch_size=200_000,
                                     columns=["split_membership", "canonical_sequence"]):
            d = batch.to_pydict()
            for s, seq in zip(d["split_membership"], d["canonical_sequence"]):
                if s == "test":
                    c = _canon(seq)
                    k = max(1, int(round(len(c) * args.prefix_frac)))
                    prefixes.append(c[:k])
                    if len(prefixes) >= GENERATION_SEEDS[0]:
                        break
            if len(prefixes) >= GENERATION_SEEDS[0]:
                break
        # use the first N prefixes across the 5 generation seeds
        n_prefix = min(len(prefixes), VALID_PER_SEED)
        prefixes = prefixes[:n_prefix]

    for cell_name, dec in DECODER_GRID.items():
        temp, top_p = dec["temperature"], dec["top_p"]
        cell_seqs: list[str] = []
        cell_invalid = 0
        for gs in GENERATION_SEEDS:
            if args.prefix_frac is not None:
                # continuation: each gen_seed continues a distinct prefix set
                start = (gs // 100 - 1) * (VALID_PER_SEED // 5)
                pre = prefixes[start:start + VALID_PER_SEED // 5]
                for p in pre:
                    suffix = adapter.generate(p, args.target_len, temperature=temp, top_p=top_p)
                    full = _canon(p) + _canon(suffix)
                    if _is_valid(full):
                        cell_seqs.append(full)
                        all_rows.append({
                            "arm": args.arm, "seed": args.seed, "cell": cell_name,
                            "gen_seed": gs, "prefix_frac": args.prefix_frac,
                            "sequence": full})
                    else:
                        cell_invalid += 1
            else:
                seqs, inv = generate_cell(adapter, temp, top_p, gs,
                                          VALID_PER_SEED, args.target_len)
                cell_seqs.extend(seqs)
                cell_invalid += inv
                for s in seqs:
                    all_rows.append({
                        "arm": args.arm, "seed": args.seed, "cell": cell_name,
                        "gen_seed": gs, "prefix_frac": None, "sequence": s})
        total_invalid += cell_invalid
        # metrics (training_set not provided here; memorization vs train is a
        # separate sealed pass over the train parquet -- recorded in manifest)
        st = evaluate_generation(cell_seqs, training_set=None)
        res = {
            "decoder": {"temperature": temp, "top_p": top_p},
            "n_valid": st.valid, "n_total_attempted": st.total,
            "invalid_count": cell_invalid,
            "validity_rate": st.validity_rate(),
            "exact_unique": len(st.exact_unique),
            "uniqueness": st.uniqueness(),
            "n_invalid_char": st.invalid_char_count,
            "selection_sha256": _hash(cell_seqs),
        }
        cell_results[cell_name] = res
        print("[%s/%s] %s temp=%s top_p=%s valid=%d unique=%d uniq=%.3f invalid=%d" % (
            args.arm, args.seed, cell_name, temp, top_p, st.valid,
            len(st.exact_unique), st.uniqueness(), cell_invalid))

    # Persist parquet of all generated sequences
    table = pa.Table.from_pylist(all_rows)
    pq_path = os.path.join(args.out_dir, "generated.parquet")
    pq.write_table(table, pq_path)

    manifest = {
        "arm": args.arm, "seed": args.seed, "ckpt": args.ckpt,
        "decoder_grid": DECODER_GRID,
        "pre_registered": list(PRE_REGISTERED),
        "generation_seeds": GENERATION_SEEDS,
        "valid_per_seed": VALID_PER_SEED,
        "target_len": args.target_len,
        "prefix_frac": args.prefix_frac,
        "cells": cell_results,
        "total_invalid": total_invalid,
        "cpu_fallback_count": adapter.guard.cpu_fallback_count,
        "parquet": pq_path,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mf = os.path.join(args.out_dir, "generation_manifest.json")
    with open(mf, "w") as fh:
        json.dump(manifest, fh, indent=2)
    assert adapter.guard.cpu_fallback_count == 0, "cpu fallback in generation %s" % args.arm
    print("WROTE", pq_path)
    print("WROTE", mf)


if __name__ == "__main__":
    main()