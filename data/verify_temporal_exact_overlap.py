"""Verify temporal_ood_clean has ZERO exact overlap with the release-22 train.

Phase 1 / contract 3.1 (line 200) and 5.2: every release-shift entity must be
removed of exact AND 80/80 overlap against the release-22 train. The 80/80 arm
is guaranteed by construction (survivors have no qualifying hit to any train
cluster representative); this script independently proves the exact=0 arm by
intersecting canonical_sequence_hash sets. Needs >= several GB RAM (parquet sets).
"""
from __future__ import annotations

import argparse
import json
import time

import polars as pl

DEFAULT_SPLIT = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/"
                 "release22_split_8080.parquet")
DEFAULT_CLEAN = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/time_ood_rel27/"
                 "temporal_ood_clean.parquet")


def exact_overlap(split: str, clean: str) -> dict:
    train = (pl.read_parquet(split, columns=["split_membership", "canonical_sequence_hash"])
                .filter(pl.col("split_membership") == "train"))
    train_hashes = set(train["canonical_sequence_hash"].to_list())
    clean_hashes = set(pl.read_parquet(clean, columns=["canonical_sequence_hash"])
                         ["canonical_sequence_hash"].to_list())
    overlap = train_hashes & clean_hashes
    return {
        "check": "temporal_ood_clean exact overlap vs release-22 train",
        "train_entities": len(train_hashes),
        "clean_entities": len(clean_hashes),
        "exact_overlap_count": len(overlap),
        "contract_gate": "exact=0",
        "pass": len(overlap) == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--clean", default=DEFAULT_CLEAN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = exact_overlap(args.split, args.clean)
    report["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("EXACT_OVERLAP>0 -> gate FAIL")


if __name__ == "__main__":
    main()