"""Phase 1: exact canonical deduplication.

Canonical exact duplicates merge into one training entity; full accession &
metadata mapping is preserved. train/validation/test canonical exact overlap
must be zero (enforced at split time).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def dedup(table: pa.Table) -> pa.Table:
    """Group by canonical_sequence_hash; keep first representative, list accessions.

    Returns a table with one row per unique canonical sequence:
      canonical_sequence_hash, canonical_sequence, rna_type, length, length_bin,
      num_accessions, accessions (list, as string join), alphabet_status.
    """
    # Sort by hash for stable grouping
    idx = table.column("canonical_sequence_hash").to_pylist()
    order = sorted(range(len(idx)), key=lambda i: idx[i])
    acc = table.column("accession").to_pylist()
    seq = table.column("canonical_sequence").to_pylist()
    rna = table.column("rna_type").to_pylist()
    length = table.column("length").to_pylist()
    length_bin = table.column("length_bin").to_pylist()

    rows = {
        "canonical_sequence_hash": [],
        "canonical_sequence": [],
        "rna_type": [],
        "length": [],
        "length_bin": [],
        "num_accessions": [],
        "accessions": [],
    }
    i = 0
    n = len(order)
    while i < n:
        h = idx[order[i]]
        j = i
        acc_list = []
        while j < n and idx[order[j]] == h:
            acc_list.append(acc[order[j]])
            j += 1
        rep = order[i]
        rows["canonical_sequence_hash"].append(h)
        rows["canonical_sequence"].append(seq[rep])
        rows["rna_type"].append(rna[rep])
        rows["length"].append(length[rep])
        rows["length_bin"].append(length_bin[rep])
        rows["num_accessions"].append(j - i)
        rows["accessions"].append("\t".join(acc_list))
        i = j
    return pa.table(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-parquet", required=True, type=Path)
    ap.add_argument("--out-parquet", required=True, type=Path)
    args = ap.parse_args()
    table = pq.read_table(args.canonical_parquet)
    ded = dedup(table)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(ded, args.out_parquet)
    print(f"input={table.num_rows:,} unique_canonical={ded.num_rows:,}")
    print(f"wrote {args.out_parquet}")


if __name__ == "__main__":
    main()