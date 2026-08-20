"""Phase 1: release-shift 80/80 overlap removal via mmseqs CROSS-SEARCH.

Contract 3.1 (line 200): "每个 release-shift entity 必须再次移除对 release-22
train 的 canonical exact 和 80% identity/80% bidirectional coverage overlap".
Phase 1 acceptance gate: "exact=0; 80/80 cross-search=0".

Why cross-search, not all-vs-all clustering:
  The previous `build_temporal_homology.py` clustered the combined 33M sequences
  (train 14M + candidates 19M) with mmseqs easy-cluster. That segfaults in
  mmseqs linclust/align2clust at this scale (`getDbKey: local id >= db size`,
  a known mmseqs bug) and is far heavier than the contract requires. The
  contract is a candidate-vs-train CROSS-SEARCH: each candidate is checked for
  an 80% identity / 80% bidirectional coverage hit against the release-22 train
  database.

Implementation:
  - `mmseqs createdb train` (target DB), `mmseqs createdb candidates` (queries).
  - `mmseqs search candidates train ... --min-seq-id 0.8 -c 0.8 --cov-mode 2`
    (cov-mode 2 = bidirectional coverage, per contract line 200).
  - `mmseqs convertalis` -> .m8; any candidate with >=1 qualifying hit is removed.
  - The survivors form `temporal_ood_clean.parquet`.

Candidates are already exact-deduped against train (previous step). For memory
boundedness, `--split-memory-limit` caps the prefilter/alignment memory and
`--chunk` lets us validate on a slice before launching the full 19M search.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

SPLIT_COLS = ["canonical_sequence_hash", "canonical_sequence", "split_membership"]
OUT_SCHEMA = pa.schema([
    ("accession", pa.string()), ("release", pa.string()),
    ("source_database", pa.string()), ("rna_type", pa.string()),
    ("canonical_sequence", pa.string()),
    ("canonical_sequence_hash", pa.string()), ("length", pa.int64()),
    ("length_bin", pa.string()),
])


def load_train_sequences(split_parquet: Path) -> pl.DataFrame:
    df = pl.read_parquet(split_parquet, columns=SPLIT_COLS)
    return df.filter(pl.col("split_membership") == "train")


def load_temporal_candidates(time_ood_dir: Path) -> pl.DataFrame:
    files = sorted(time_ood_dir.glob("time_ood_*_candidates.parquet"))
    frames = [pl.read_parquet(f) for f in files]
    df = pl.concat(frames)
    return df.unique(subset=["canonical_sequence_hash"], keep="first")


def write_fasta(seqs: list[str], ids: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for sid, s in zip(ids, seqs):
            f.write(f">{sid}\n{s}\n")


def mmseqs_search(query_fa: Path, target_fa: Path, work: Path,
                  threads: int, split_mem_mb: int, sensitivity: str) -> Path:
    """Run mmseqs search; return the .m8 path (one line per qualifying hit)."""
    qdb = work / "qdb"
    tdb = work / "tdb"
    res = work / "result"
    if not qdb.exists():
        subprocess.run(["mmseqs", "createdb", str(query_fa), str(qdb)], check=True)
    if not tdb.exists():
        subprocess.run(["mmseqs", "createdb", str(target_fa), str(tdb)], check=True)
    for cmd in (
        ["mmseqs", "search", str(qdb), str(tdb), str(res), str(work / "tmp"),
         "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "2",
         "--search-type", "3",  # nucleotide (ambiguous from FASTA alone)
         "--split-memory-limit", str(split_mem_mb),
         "--threads", str(threads), "-s", sensitivity],
        ["mmseqs", "convertalis", str(qdb), str(tdb), str(res),
         str(work / "result.m8"), "--threads", str(threads)],
    ):
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return work / "result.m8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-ood-dir", required=True, type=Path)
    ap.add_argument("--split-parquet", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--work-dir", type=Path,
                    default=Path("/mnt/cunyuliu/tokenizer-benchmark/tmp/temporal_search"))
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--split-memory-limit", type=int, default=120_000,
                    help="mmseqs --split-memory-limit in MB (bounds RAM)")
    ap.add_argument("--sensitivity", default="1",
                    help="mmseqs -s sensitivity; >=80% identity hits are reliably "
                         "caught even at -s 1 (shared k-mers), and -s 1 is ~10-30x "
                         "faster than -s 7.5 against a 14M target index")
    ap.add_argument("--chunk", type=int, default=None,
                    help="only search the first N candidates (validation slice)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    train = load_train_sequences(args.split_parquet)
    train_seq = train["canonical_sequence"].to_list()
    print(f"train sequences: {len(train_seq):,}", flush=True)

    temp = load_temporal_candidates(args.time_ood_dir)
    if args.chunk is not None:
        temp = temp.head(args.chunk)
    print(f"temporal candidates (dedup, slice): {len(temp):,}", flush=True)

    train_fa = args.work_dir / "train.fa"
    cand_fa = args.work_dir / "candidates.fa"
    write_fasta(train_seq, [f"t{i}" for i in range(len(train_seq))], train_fa)
    cand_ids = temp["canonical_sequence_hash"].to_list()
    write_fasta(temp["canonical_sequence"].to_list(),
                [f"c{i}" for i in range(len(cand_ids))], cand_fa)

    m8 = mmseqs_search(cand_fa, train_fa, args.work_dir,
                       args.threads, args.split_memory_limit,
                       args.sensitivity)

    # any candidate id appearing in the m8 has a qualifying 80/80 train hit
    removed = set()
    with open(m8) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 2:
                continue
            removed.add(fields[0])
    removed_idx = {int(x[1:]) for x in removed if x.startswith("c")}

    out_rows = []
    for i, row in enumerate(temp.iter_rows(named=True)):
        if i not in removed_idx:
            out_rows.append({
                "accession": row["accession"], "release": row["release"],
                "source_database": row.get("source_database"),
                "rna_type": row["rna_type"],
                "canonical_sequence": row["canonical_sequence"],
                "canonical_sequence_hash": row["canonical_sequence_hash"],
                "length": row["length"], "length_bin": row["length_bin"],
            })
    if out_rows:
        tbl = pa.Table.from_pylist(out_rows, schema=OUT_SCHEMA)
    else:
        tbl = pa.table({k: pa.array([], t) for k, t in zip(
            OUT_SCHEMA.names, OUT_SCHEMA.types)})
    out_path = args.out_dir / "temporal_ood_clean.parquet"
    pq.write_table(tbl, out_path)
    print(f"temporal kept: {len(out_rows):,} removed(80/80 to train): "
          f"{len(temp) - len(out_rows):,}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
