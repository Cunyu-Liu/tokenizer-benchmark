"""Phase 1: canonicalization of RNAcentral release active FASTA (streaming).

Turns a release FASTA (headers: `>URS0000XXXX <RNA type> from <N> species`)
into a sequence-level Parquet with canonicalization, QC status, and length bin.

Canonicalization rules (Goal §5):
- uppercase
- T -> U
- primary alphabet A/C/G/U
- sequences with other IUPAC chars go to QC ledger (ambiguity), excluded from primary
- length bins: 16-4096 primary; 4097-16384 length-OOD; >16384 descriptive only

Streams batches to disk to bound memory use at ~40M-sequence scale.
"""
from __future__ import annotations

import gzip
import hashlib
import argparse
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PRIMARY_ALPHA = frozenset("ACGU")


@dataclass(frozen=True)
class CanonicalResult:
    accession: str
    canonical_seq: str
    canonical_hash: str
    raw_hash: str
    rna_type: str
    alphabet_status: str  # "primary" | "ambiguous"
    length: int
    length_bin: str  # "16-4096" | "4097-16384" | ">16384" | "<16"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def canonicalize_one(accession: str, raw_seq: str, rna_type: str) -> CanonicalResult:
    raw_hash = sha256_hex(raw_seq)
    canon = raw_seq.upper().replace("T", "U")
    chars = set(canon)
    alphabet_status = "primary" if chars <= PRIMARY_ALPHA else "ambiguous"
    length = len(canon)
    if length < 16:
        length_bin = "<16"
    elif length <= 4096:
        length_bin = "16-4096"
    elif length <= 16384:
        length_bin = "4097-16384"
    else:
        length_bin = ">16384"
    return CanonicalResult(
        accession=accession,
        canonical_seq=canon,
        canonical_hash=sha256_hex(canon),
        raw_hash=raw_hash,
        rna_type=rna_type,
        alphabet_status=alphabet_status,
        length=length,
        length_bin=length_bin,
    )


def parse_header(header: str) -> tuple[str, str]:
    """Parse FASTA header `>URS0000XXXX <RNA type> from <N> species`.

    Returns (accession, rna_type). RNA type defaults to 'unknown' if absent.
    """
    parts = header.split()
    accession = parts[0] if parts else ""
    rna_type = parts[1] if len(parts) >= 2 else "unknown"
    return accession, rna_type


def iter_fasta(path: Path):
    """Yield (header, sequence) records from a (possibly gzipped) FASTA."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        header = None
        chunks = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None and chunks:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            elif line:
                chunks.append(line)
        if header is not None and chunks:
            yield header, "".join(chunks)


def records_to_table(records: list[CanonicalResult]) -> pa.Table:
    return pa.table(
        {
            "accession": [r.accession for r in records],
            "rna_type": [r.rna_type for r in records],
            "raw_sequence_hash": [r.raw_hash for r in records],
            "canonical_sequence_hash": [r.canonical_hash for r in records],
            "canonical_sequence": [r.canonical_seq for r in records],
            "alphabet_status": [r.alphabet_status for r in records],
            "length": [r.length for r in records],
            "length_bin": [r.length_bin for r in records],
        }
    )


_SCHEMA = pa.table(
    {
        "accession": pa.array([], pa.string()),
        "rna_type": pa.array([], pa.string()),
        "raw_sequence_hash": pa.array([], pa.string()),
        "canonical_sequence_hash": pa.array([], pa.string()),
        "canonical_sequence": pa.array([], pa.string()),
        "alphabet_status": pa.array([], pa.string()),
        "length": pa.array([], pa.int64()),
        "length_bin": pa.array([], pa.string()),
    }
).schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-fasta", required=True, type=Path)
    ap.add_argument("--out-parquet", required=True, type=Path)
    ap.add_argument("--out-qc", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=500_000)
    args = ap.parse_args()

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_qc.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    primary_count = 0
    qc_count = 0
    primary_buf: list[CanonicalResult] = []
    qc_buf: list[CanonicalResult] = []

    with pq.ParquetWriter(args.out_parquet, _SCHEMA, compression="snappy") as pw, \
         pq.ParquetWriter(args.out_qc, _SCHEMA, compression="snappy") as qw:
        def flush():
            nonlocal primary_buf, qc_buf
            if primary_buf:
                pw.write_table(records_to_table(primary_buf))
                primary_buf = []
            if qc_buf:
                qw.write_table(records_to_table(qc_buf))
                qc_buf = []

        for header, seq in iter_fasta(args.release_fasta):
            accession, rna_type = parse_header(header)
            res = canonicalize_one(accession, seq, rna_type)
            total += 1
            if res.alphabet_status == "primary":
                primary_buf.append(res)
                primary_count += 1
            else:
                qc_buf.append(res)
                qc_count += 1
            if total % args.batch == 0:
                flush()
                print(f"  processed {total:,} (primary={primary_count:,} qc={qc_count:,})", flush=True)
        flush()

    print(f"TOTAL={total:,} PRIMARY={primary_count:,} QC={qc_count:,}")
    print(f"wrote {args.out_parquet}")
    print(f"wrote {args.out_qc}")


if __name__ == "__main__":
    main()