"""Parallel release-release 80/80 cross-search vs train cluster reps.

For one chunk of candidate sequences, search against the SHARED train
representative index (in /dev/shm tmpfs to avoid NFS IO bottleneck), and emit
the chunk's m8 hits. The parallel driver merges chunks.

Key design choices (validated empirically):
  - Target DB = train CLUSTER REPRESENTATIVES (~3.0M), not all 14.1M train rows:
    Phase-1 cluster members are pairwise 80/80, so any candidate with an 80/80
    hit to a member also hits its representative; missing a representative
    means no member can be hit. This shrinks the index 114GB->40GB and the
    search cost ~4.7x with no loss of 80/80 coverage.
  - Index + db live in /dev/shm (tmpfs): mmseqs spent ~700s reading the NFS-hosted
    index for 1K queries; in tmpfs the same work is <10s. Input/output remain on
    NFS. 43GB of 352GB tmpfs is used.

Each chunk: createdb -> search (--min-seq-id 0.8 -c 0.8 --cov-mode 2 -s 1)
-> convertalis to a per-chunk .m8 on NFS. Also writes
<chunk>_queries.txt = one int per line = the GLOBAL candidate row index that
WAS removed (has a qualifying 80/80 train hit).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

IDX = Path("/dev/shm/tindex")
CAND = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/time_ood_rel27/"
        "time_ood_27_0_candidates.parquet")


def write_fasta(seqs, ids, path: Path) -> None:
    with open(path, "w") as f:
        for sid, s in zip(ids, seqs):
            f.write(f">{sid}\n{s}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-i", type=int, required=True)
    ap.add_argument("--n-chunks", type=int, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="NFS dir for per-chunk m8 + removed lists")
    args = ap.parse_args()

    tdb = IDX / "trep_dbs"
    assert (IDX / "trep_dbs.idx").exists(), "train rep index missing in /dev/shm"

    # stable global index = row in the dedup candidates parquet
    df = pl.read_parquet(CAND, columns=["canonical_sequence"])
    n_total = df.height
    size = (n_total + args.n_chunks - 1) // args.n_chunks
    lo = args.chunk_i * size
    hi = min(n_total, lo + size)
    if lo >= n_total:
        print(f"chunk {args.chunk_i} out of range (n_total={n_total})")
        return
    sub_df = df.slice(lo, hi - lo)
    print(f"chunk {args.chunk_i}: rows {lo}..{hi} ({hi-lo} queries)", flush=True)

    work = args.out_dir / f"chunk_{args.chunk_i}"
    work.mkdir(parents=True, exist_ok=True)
    fa = work / "queries.fa"
    write_fasta(sub_df["canonical_sequence"].to_list(),
                [str(lo + i) for i in range(hi - lo)], fa)

    env = dict(os.environ) if (os := __import__("os")) else None
    qdb = work / "qdb"
    res = work / "result"
    tmp = work / "tmp"

    def run(cmd, **kw):
        print("RUN:", " ".join(map(str, cmd)), flush=True)
        subprocess.run([str(c) for c in cmd], check=True, **kw)

    mmseqs = "mmseqs"
    run([mmseqs, "createdb", str(fa), str(qdb)])
    run([mmseqs, "search", str(qdb), str(tdb), str(res), str(tmp),
         "--min-seq-id", "0.8", "-c", "0.8", "--cov-mode", "2",
         "--search-type", "3", "--split-memory-limit", "40000",
         "--threads", str(args.threads), "-s", "1"])
    m8 = work / "result.m8"
    run([mmseqs, "convertalis", str(qdb), str(tdb), str(res), str(m8),
         "--threads", str(args.threads)])

    removed = set()
    with open(m8) as f:
        for line in f:
            fields = line.strip().split("\t")
            if fields:
                removed.add(int(fields[0]))
    # keep only global indices in this chunk's range
    removed = {i for i in removed if lo <= i < hi}
    with open(args.out_dir / f"removed_{args.chunk_i}.txt", "w") as f:
        for i in sorted(removed):
            f.write(f"{i}\n")
    print(f"chunk {args.chunk_i}: removed {len(removed)}/{hi-lo} ({removed})",
          flush=True)


if __name__ == "__main__":
    main()