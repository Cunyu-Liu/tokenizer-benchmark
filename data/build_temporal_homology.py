"""Phase 1: remove temporal OOD candidates with 80/80 homology overlap to release-22 train.

For each temporal candidate (releases 23/25/26, already exact-deduped against train),
cluster the temporal candidates TOGETHER WITH the release-22 train sequences at
80% identity / 80% coverage. Any temporal candidate whose 80/80 cluster also
contains a release-22 train sequence is removed (marked homology_overlap_train=True).

This enforces Goal 3.1: "release 23-26 temporal OOD ... 再次移除对 release 22 train
的 exact 和 80/80 homology overlap".
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl


def load_train_sequences(split_parquet: Path) -> pl.DataFrame:
    """Return canonical_sequence for all release-22 train rows (hash -> seq)."""
    df = pl.read_parquet(split_parquet, columns=[
        "canonical_sequence_hash", "canonical_sequence", "split_membership"])
    return df.filter(pl.col("split_membership") == "train")


def load_temporal_candidates(time_ood_dir: Path) -> pl.DataFrame:
    """Concatenate release candidate parquets and dedup by canonical hash."""
    files = sorted(time_ood_dir.glob("time_ood_*_candidates.parquet"))
    frames = [pl.read_parquet(f) for f in files]
    df = pl.concat(frames)
    df = df.unique(subset=["canonical_sequence_hash"], keep="first")
    return df


def write_fasta(seqs: list[str], ids: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for sid, s in zip(ids, seqs):
            f.write(f">{sid}\n{s}\n")


def run_mmseqs_cluster(fasta: Path, out_prefix: Path, threads: int, tmp_dir: Path) -> Path:
    cmd = [
        "mmseqs", "easy-cluster", str(fasta), str(out_prefix),
        str(tmp_dir / "mmseqs_temporal_tmp"),
        "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "1",
        "--threads", str(threads),
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return Path(str(out_prefix) + "_cluster.tsv")


def parse_cluster_tsv(tsv: Path) -> dict[int, int]:
    """member id -> representative id."""
    member_to_rep = {}
    with open(tsv) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            rep, member = fields[0], fields[1]
            member_to_rep[int(member)] = int(rep)
    return member_to_rep



def kept_temporal_indices(n_train: int, n_temp: int,
                          member_to_rep: dict[int, int]) -> list[int]:
    """Return 0-based temporal indices (among temp_ids) whose 80/80 cluster
    does NOT contain any release-22 train sequence."""
    train_ids = set(range(n_train))
    cluster_has_train = set()
    for member, rep in member_to_rep.items():
        if member in train_ids:
            cluster_has_train.add(rep)
    kept = []
    for tid in range(n_train, n_train + n_temp):
        rep = member_to_rep.get(tid, tid)
        if rep in cluster_has_train:
            continue
        kept.append(tid - n_train)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-ood-dir", required=True, type=Path)
    ap.add_argument("--split-parquet", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tmp-dir", type=Path, default=Path("/mnt/cunyuliu/tokenizer-benchmark/tmp"))
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = load_train_sequences(args.split_parquet)
    train_seq = train["canonical_sequence"].to_list()
    print(f"train sequences: {len(train_seq):,}", flush=True)

    temp = load_temporal_candidates(args.time_ood_dir)
    temp_seq = temp["canonical_sequence"].to_list()
    print(f"temporal candidates (dedup): {len(temp_seq):,}", flush=True)

    n_train = len(train_seq)
    n_temp = len(temp_seq)
    # ids: train = 0..n_train-1, temporal = n_train..n_train+n_temp-1
    train_ids = list(range(n_train))
    temp_ids = list(range(n_train, n_train + n_temp))
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.tmp_dir / "combo_temporal.fa"
    write_fasta(train_seq + temp_seq, [str(i) for i in train_ids + temp_ids], fasta)

    out_prefix = args.tmp_dir / "combo_temporal_cluster"
    tsv = run_mmseqs_cluster(fasta, out_prefix, args.threads, args.tmp_dir)
    member_to_rep = parse_cluster_tsv(tsv)

    keep = kept_temporal_indices(n_train, n_temp, member_to_rep)
    kept_idx = set(keep)
    out_rows = []
    for i, row in enumerate(temp.iter_rows(named=True)):
        if i in kept_idx:
            out_rows.append({
                "accession": row["accession"], "release": row["release"],
                "rna_type": row["rna_type"],
                "canonical_sequence": row["canonical_sequence"],
                "canonical_sequence_hash": row["canonical_sequence_hash"],
                "length": row["length"], "length_bin": row["length_bin"],
            })

    schema = pa.schema([
        ("accession", pa.string()), ("release", pa.string()),
        ("rna_type", pa.string()), ("canonical_sequence", pa.string()),
        ("canonical_sequence_hash", pa.string()), ("length", pa.int64()),
        ("length_bin", pa.string()),
    ])
    if out_rows:
        tbl = pa.table(out_rows)
        tbl = tbl.select([f.name for f in schema]).cast(schema)
    else:
        tbl = pa.table({
            "accession": pa.array([], pa.string()), "release": pa.array([], pa.string()),
            "rna_type": pa.array([], pa.string()),
            "canonical_sequence": pa.array([], pa.string()),
            "canonical_sequence_hash": pa.array([], pa.string()),
            "length": pa.array([], pa.int64()), "length_bin": pa.array([], pa.string()),
        })
    out_path = args.out_dir / "temporal_ood_clean.parquet"
    pq.write_table(tbl, out_path)
    removed = n_temp - len(out_rows)
    print(f"temporal candidates kept: {len(out_rows):,} removed(80/80 to train): {removed:,}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
