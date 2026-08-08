"""Phase 1: MMseqs2 homology clustering (80/80 and 90/90 sensitivity).

Wraps `mmseqs easy-cluster` on the deduplicated canonical nucleotide sequences.
Clustering is a CPU task (allowed). Cluster membership is written back per
canonical sequence for split stratification.

Note: MMseqs clustering of nucleotide sequences is run in nucleotide mode so
that identity and coverage are computed on the nucleotide alignment.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_fasta(table: pa.Table, path: Path) -> None:
    """Write canonical sequences as FASTA with stable ids <n>."""
    seq = table.column("canonical_sequence").to_pylist()
    with open(path, "w") as f:
        for i, s in enumerate(seq):
            f.write(f">{i}\n{s}\n")


def run_mmseqs_cluster(
    fasta: Path,
    out_prefix: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int = 1,
    threads: int = 32,
) -> Path:
    """Run mmseqs easy-cluster; returns path to cluster TSV (representative<TAB>member)."""
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta),
        str(out_prefix),
        str(tempfile.mkdtemp(prefix="mmseqs_easy_")),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--threads", str(threads),
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return Path(str(out_prefix) + "_cluster.tsv")


def parse_cluster_tsv(tsv: Path, n_sequences: int) -> pa.Table:
    """Map each sequence id -> cluster representative id.

    easy-cluster output: `rep<TAB>member` one line per member.
    """
    member_to_rep = {}
    with open(tsv) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            rep, member = fields[0], fields[1]
            member_to_rep[int(member)] = int(rep)
    reps = []
    clusters = []
    for i in range(n_sequences):
        rep = member_to_rep.get(i, i)
        reps.append(str(rep))
        clusters.append(str(rep))  # cluster id = representative id
    return pa.table({"sequence_id": list(range(n_sequences)), "cluster_id": clusters})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-parquet", required=True, type=Path)
    ap.add_argument("--out-cluster-parquet", required=True, type=Path)
    ap.add_argument("--out-fasta", required=True, type=Path)
    ap.add_argument("--min-seq-id", type=float, default=0.8)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--cov-mode", type=int, default=1)
    ap.add_argument("--keep-cluster-files", action="store_true")
    args = ap.parse_args()

    table = pq.read_table(args.dedup_parquet)
    tmpdir = tempfile.mkdtemp(prefix="tokbench_cluster_")
    fasta = Path(tmpdir) / "seqs.fa"
    write_fasta(table, fasta)
    tsv = run_mmseqs_cluster(
        fasta, Path(tmpdir) / "cluster", args.min_seq_id, args.coverage, args.cov_mode
    )
    cluster_tab = parse_cluster_tsv(tsv, table.num_rows)
    # attach cluster_id to dedup table
    out = table.append_column("cluster_id", cluster_tab.column("cluster_id"))
    args.out_cluster_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out_cluster_parquet)
    n_clusters = len(set(out.column("cluster_id").to_pylist()))
    print(f"sequences={out.num_rows:,} clusters={n_clusters:,}")
    print(f"wrote {args.out_cluster_parquet}")
    if args.keep_cluster_files:
        args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(fasta, args.out_fasta)
        shutil.copy(tsv, Path(str(args.out_fasta) + ".tsv"))


if __name__ == "__main__":
    main()