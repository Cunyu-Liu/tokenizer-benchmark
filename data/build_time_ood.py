"""Phase 1: temporal OOD candidate extraction (releases 23-26).

For each temporal release, keep only accessions NEW relative to release 22,
canonicalize them (primary alphabet), and remove exact canonical overlap with
the release-22 train split. Output is a combined per-release parquet of temporal
OOD *candidates*; homology removal (80/80 vs release-22 train) is a separate step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.canonicalize import iter_fasta, parse_header, canonicalize_one


def load_release22_accessions(fasta: Path) -> set[str]:
    acc = set()
    for header, _seq in iter_fasta(fasta):
        a, _ = parse_header(header)
        acc.add(a)
    return acc


def classify_temporal(acc: str, seq: str, rna_type: str,
                      r22_acc: set[str], train_hashes: set[str]):
    """Return (keep, reason) for a temporal candidate sequence.

    Reasons: 'new', 'in_release22', 'ambiguous', 'exact_overlap_train'.
    """
    if acc in r22_acc:
        return False, "in_release22"
    res = canonicalize_one(acc, seq, rna_type)
    if res.alphabet_status != "primary":
        return False, "ambiguous"
    if res.canonical_hash in train_hashes:
        return False, "exact_overlap_train"
    return True, "new"


def build_release22_train_hashes(split_parquet) -> set[str]:
    import polars as pl
    df = pl.read_parquet(split_parquet, columns=["canonical_sequence_hash", "split_membership"])
    return set(df.filter(pl.col("split_membership") == "train")["canonical_sequence_hash"].to_list())


_SCHEMA = pa.table({
    "accession": pa.array([], pa.string()),
    "release": pa.array([], pa.string()),
    "rna_type": pa.array([], pa.string()),
    "canonical_sequence": pa.array([], pa.string()),
    "canonical_sequence_hash": pa.array([], pa.string()),
    "length": pa.array([], pa.int64()),
    "length_bin": pa.array([], pa.string()),
}).schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release22-fasta", required=True, type=Path)
    ap.add_argument("--split-parquet", required=True, type=Path)
    ap.add_argument("--release24", action="store_true",
                    help="include release 24 (default false; 24 content is a subset of 22)")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading release-22 accession set...", flush=True)
    r22_acc = load_release22_accessions(args.release22_fasta)
    print(f"release-22 accessions: {len(r22_acc):,}", flush=True)

    print("loading release-22 train canonical hashes...", flush=True)
    train_hashes = build_release22_train_hashes(args.split_parquet)
    print(f"release-22 train hashes: {len(train_hashes):,}", flush=True)

    releases = ["23.0", "24.0", "25.0", "26.0"] if args.release24 else ["23.0", "25.0", "26.0"]
    base = Path("/mnt/cunyuliu/tokenizer-benchmark/data/raw")

    for rel in releases:
        fasta = base / f"release_{rel}" / "rnacentral_active.fasta.gz"
        out = args.out_dir / f"time_ood_{rel.replace('.', '_')}_candidates.parquet"
        print(f"processing release {rel} from {fasta}", flush=True)
        new_count = 0
        exact_removed = 0
        kept = []
        for header, seq in iter_fasta(fasta):
            acc, rna_type = parse_header(header)
            if acc in r22_acc:
                continue
            res = canonicalize_one(acc, seq, rna_type)
            if res.alphabet_status != "primary":
                continue
            if res.canonical_hash in train_hashes:
                exact_removed += 1
                continue
            kept.append({
                "accession": acc, "release": rel, "rna_type": rna_type,
                "canonical_sequence": res.canonical_seq,
                "canonical_sequence_hash": res.canonical_hash,
                "length": res.length, "length_bin": res.length_bin,
            })
            new_count += 1
            if new_count % 2_000_000 == 0:
                print(f"  release {rel}: kept {new_count:,} (exact_removed={exact_removed:,})",
                      flush=True)
        if kept:
            tbl = pa.table({k: [r[k] for k in kept] for k in kept[0].keys()})
            tbl = tbl.select(_SCHEMA.names).cast(_SCHEMA)
            pq.write_table(tbl, out)
        print(f"release {rel}: total_new={new_count:,} exact_overlap_removed={exact_removed:,} -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
