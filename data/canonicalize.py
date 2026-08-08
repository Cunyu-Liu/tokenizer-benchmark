"""Phase 1: canonicalization of RNAcentral release active FASTA.

Turns a release FASTA (headers: `>URS0000XXXX <RNA type> from <N> species`)
into a sequence-level Parquet with canonicalization, QC status, and length bin.

Canonicalization rules (Goal §5):
- uppercase
- T -> U
- primary alphabet A/C/G/U
- sequences with other IUPAC chars go to QC ledger (ambiguity), excluded from primary
- length bins: 16-4096 primary; 4097-16384 length-OOD; >16384 descriptive only
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
    raw_seq: str
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
        raw_seq=raw_seq,
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
    # RNA type is the token(s) between accession and 'from'. Take the first token.
    rna_type = "unknown"
    if len(parts) >= 2:
        rna_type = parts[1]
    return accession, rna_type


def iter_fasta(path: Path):
    """Yield (header, sequence) records from a (possibly gzipped) FASTA."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        header = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None and seq_chunks:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            elif line:
                seq_chunks.append(line)
        if header is not None and seq_chunks:
            yield header, "".join(seq_chunks)


def build_arrow_table(records: list[CanonicalResult]) -> pa.Table:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-fasta", required=True, type=Path)
    ap.add_argument("--out-parquet", required=True, type=Path)
    ap.add_argument("--out-qc", required=True, type=Path)
    args = ap.parse_args()

    primary_records: list[CanonicalResult] = []
    qc_records: list[CanonicalResult] = []
    total = 0
    for header, seq in iter_fasta(args.release_fasta):
        accession, rna_type = parse_header(header)
        res = canonicalize_one(accession, seq, rna_type)
        total += 1
        if res.alphabet_status == "primary":
            primary_records.append(res)
        else:
            qc_records.append(res)
        if total % 1_000_000 == 0:
            print(f"  processed {total:,} sequences", flush=True)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_qc.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(build_arrow_table(primary_records), args.out_parquet)
    pq.write_table(build_arrow_table(qc_records), args.out_qc)

    print(f"TOTAL={total:,} PRIMARY={len(primary_records):,} QC={len(qc_records):,}")
    print(f"wrote {args.out_parquet}")
    print(f"wrote {args.out_qc}")


if __name__ == "__main__":
    main()