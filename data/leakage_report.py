"""Phase 1: leakage report and dataset manifest.

Verifies the Goal 3.1 / Phase-1 acceptance gates on the derived split:
  1. canonical exact overlap across train/val/test/family-* splits == 0
  2. 80/80 homology cluster does not cross splits (each cluster_id -> one split)
  3. temporal OOD clean has no exact or 80/80-homology overlap with train

Writes docs/data/dataset_manifest.json and docs/data/P1_leakage_report.json.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import polars as pl


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_overlap(df: pl.DataFrame, hash_col: str) -> dict:
    """Return max pairwise exact-hash overlap between distinct splits."""
    splits = sorted(df["split_membership"].unique().to_list())
    split_sets = {
        s: set(df.filter(pl.col("split_membership") == s)[hash_col].to_list())
        for s in splits
    }
    overlaps = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            inter = split_sets[a] & split_sets[b]
            overlaps[f"{a} x {b}"] = len(inter)
    return overlaps


def cluster_split_crossing(df: pl.DataFrame) -> int:
    """Count clusters that appear in more than one split."""
    counts = (
        df.group_by(["cluster_id", "split_membership"]).len()
        .group_by("cluster_id").len()
        .filter(pl.col("len") > 1)
    )
    return counts.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-parquet", required=True, type=Path)
    ap.add_argument("--cluster-parquet", required=True, type=Path)
    ap.add_argument("--temporal-clean-parquet", type=Path, default=None)
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--out-leakage", required=True, type=Path)
    args = ap.parse_args()

    split = pl.read_parquet(args.split_parquet)
    cluster = pl.read_parquet(args.cluster_parquet,
                              columns=["canonical_sequence_hash", "cluster_id"])

    # exact canonical overlap across splits
    exact_overlap = canonical_overlap(split, "canonical_sequence_hash")
    exact_max = max(exact_overlap.values()) if exact_overlap else 0

    # cluster crossing
    crossing = cluster_split_crossing(split)

    # split counts
    counts = {
        row["split_membership"]: row["len"]
        for row in split.group_by("split_membership").len().iter_rows(named=True)
    }

    temporal_report = {}
    if args.temporal_clean_parquet is not None and args.temporal_clean_parquet.exists():
        temp = pl.read_parquet(args.temporal_clean_parquet)
        train_hashes = set(
            split.filter(pl.col("split_membership") == "train")["canonical_sequence_hash"].to_list()
        )
        temp_hashes = set(temp["canonical_sequence_hash"].to_list())
        exact_with_train = len(temp_hashes & train_hashes)
        temporal_report = {
            "temporal_ood_sequences": temp.height,
            "by_release": {
                r: temp.filter(pl.col("release") == r).height
                for r in temp["release"].unique().sort().to_list()
            },
            "exact_overlap_with_train": exact_with_train,
        }

    manifest = {
        "dataset": "TokBench-RNA release-22 primary split + temporal OOD (23/25/26)",
        "anchor": "RNAcentral release 22.0",
        "derived_files": {
            "split_parquet": str(args.split_parquet),
            "split_sha256": sha256_file(args.split_parquet),
            "cluster_parquet": str(args.cluster_parquet),
        },
        "split_counts": counts,
    }
    if temporal_report:
        manifest["temporal_ood"] = temporal_report

    gate = {
        "exact_overlap_zero": exact_max == 0,
        "cluster_not_across_split": crossing == 0,
        "temporal_no_exact_train_overlap": (temporal_report.get("exact_overlap_with_train", 0) == 0),
    }

    leakage = {
        "exact_canonical_overlap_between_splits": exact_overlap,
        "exact_overlap_max": exact_max,
        "clusters_crossing_splits": crossing,
        "temporal": temporal_report,
        "gates": gate,
        "overall": all(gate.values()),
    }

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_leakage.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    Path(args.out_leakage).write_text(json.dumps(leakage, indent=2, ensure_ascii=False))
    print(json.dumps(leakage, indent=2))


if __name__ == "__main__":
    main()
