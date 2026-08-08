"""Phase 1: exact canonical deduplication (polars-based, memory-efficient).

Canonical exact duplicates merge into one training entity; full accession &
metadata mapping is preserved. train/validation/test canonical exact overlap
must be zero (enforced at split time).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def dedup(df: pl.DataFrame) -> pl.DataFrame:
    """Group by canonical_sequence_hash; keep first representative, list accessions.

    Returns one row per unique canonical sequence:
      canonical_sequence_hash, canonical_sequence, rna_type, length, length_bin,
      num_accessions, accessions (tab-joined).
    """
    out = (
        df.sort("canonical_sequence_hash")
        .group_by("canonical_sequence_hash")
        .agg(
            pl.col("canonical_sequence").first().alias("canonical_sequence"),
            pl.col("rna_type").first().alias("rna_type"),
            pl.col("length").first().alias("length"),
            pl.col("length_bin").first().alias("length_bin"),
            pl.col("accession").count().alias("num_accessions"),
            pl.col("accession").str.join("\t").alias("accessions"),
        )
        .sort("canonical_sequence_hash")
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-parquet", required=True, type=Path)
    ap.add_argument("--out-parquet", required=True, type=Path)
    args = ap.parse_args()
    df = pl.read_parquet(args.canonical_parquet)
    ded = dedup(df)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    ded.write_parquet(args.out_parquet)
    print(f"input={len(df):,} unique_canonical={len(ded):,}")
    print(f"wrote {args.out_parquet}")


if __name__ == "__main__":
    main()