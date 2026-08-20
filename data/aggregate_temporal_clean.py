"""Merge per-chunk 80/80 removal lists into the final release-27 shift clean set.

Inputs:
  - candidates parquet (time_ood_27_0_candidates.parquet): all release-27 new
    accessions (relative to release 22) with metadata.
  - search dir with removed_<i>.txt: global candidate row indices that got a
    qualifying 80% identity / 80% bidirectional coverage hit against the
    release-22 train clusters (produced by data/temporal_chunk_search.py).

Output: temporal_ood_clean.parquet (canonical entities surviving the 80/80 gate).

Contract 3.1 (line 200): each release-shift entity must remove exact + 80/80
overlap with the release-22 train. Exact dedup against train was already done
when the candidates file was built; this script applies the 80/80 removal and
collapses intra-candidate exact duplicates into canonical entities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

CAND = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/time_ood_rel27/"
        "time_ood_27_0_candidates.parquet")
DEFAULT_SEARCH = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/"
                  "time_ood_rel27/search")
OUT_RAW = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/time_ood_rel27/"
           "temporal_ood_clean.parquet")
OUT_SCHEMA_COLS = ["accession", "release", "source_database", "rna_type",
                   "canonical_sequence", "canonical_sequence_hash", "length",
                   "length_bin"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=CAND, type=Path)
    ap.add_argument("--search-dir", default=DEFAULT_SEARCH, type=Path)
    ap.add_argument("--n-chunks", type=int, default=48)
    ap.add_argument("--out", default=OUT_RAW, type=Path)
    args = ap.parse_args()

    # 1. collect removed raw row indices across all chunks
    removed: set[int] = set()
    missing = []
    for i in range(args.n_chunks):
        p = args.search_dir / f"removed_{i}.txt"
        if not p.exists():
            missing.append(i)
            continue
        with open(p) as f:
            removed.update(int(line) for line in f if line.strip())
    if missing:
        print(f"WARNING: {len(missing)} chunks have no removed file yet: "
              f"{missing[:10]}...; continuing with partial result")
    print(f"removed indices from {args.n_chunks - len(missing)}/{args.n_chunks} chunks: {len(removed):,}")

    # 2. read candidates with metadata
    df = pl.read_parquet(args.candidates)
    df = df.with_row_index("_raw_idx")
    if "source_database" not in df.columns:
        df = df.with_columns(pl.lit("").alias("source_database"))
    print(f"candidates rows: {df.height:,}; cols: {df.columns}")

    # 3. mark removed, then collapse duplicates into canonical entities
    kept = df.filter(~pl.col("_raw_idx").is_in(sorted(removed)))
    survivors = kept.unique(subset=["canonical_sequence_hash"], keep="first")
    # final cols matching OUT_SCHEMA
    out = survivors.select(OUT_SCHEMA_COLS)
    out.write_parquet(args.out)
    print(f"after 80/80: kept {kept.height:,} (removed {df.height - kept.height:,})")
    print(f"after intra-candidate exact dedup: {survivors.height:,}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
