"""Phase 1 (release 27): source_database x RNA_type x length composition audit.

Goal V3 3.1: release 27 adds ~10M sequences incl. circRNA/mirtron sources, so
the release-shift sensitivity must report `release_id x source_database x
RNA_type x length` composition, and REMOVE canonical exact + 80/80 overlap with
the release-22 train.

source_database comes from RNAcentral's `id_mapping/database_mappings/*.tsv`
(column 2), which is more complete than the FASTA header RNA type. Each row is:
  accession \t database \t external_id \t taxon_id \t rna_type \t gene_name

This script:
  1. loads all database_mappings TSVs -> accession -> (source_database, rna_type)
  2. streams the release-27 active FASTA; keeps only accessions NOT in release 22
  3. canonicalizes + drops non-primary alphabet + exact-overlap vs release-22 train
  4. writes release27_composition.json (counts by source db x rna_type x length bin)
     and time_ood_27_0_candidates.parquet (for the later 80/80 homology step).

Note: the FASTA header RNA type is used only as a fallback when an accession is
absent from database_mappings (those rows are flagged `source_database=UNKNOWN_MAPPING`).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.canonicalize import iter_fasta, parse_header, canonicalize_one


def parse_db_mapping_line(line: str) -> tuple[str, str, str] | None:
    """Parse one database_mappings row -> (accession, source_database, rna_type).

    Columns: accession, database, external_id, taxon_id, rna_type, gene_name.
    Returns None on malformed rows.
    """
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 2 or not fields[0].startswith("URS"):
        return None
    accession = fields[0].strip()
    source_db = fields[1].strip() if len(fields) >= 2 else ""
    rna_type = fields[4].strip() if len(fields) >= 5 else ""
    if not source_db:
        return None
    if not rna_type:
        rna_type = "unknown"
    return accession, source_db, rna_type


def length_bin(length: int) -> str:
    if length < 16:
        return "<16"
    if length <= 4096:
        return "16-4096"
    if length <= 16384:
        return "4097-16384"
    return ">16384"


def composition_counts(rows: list[dict]) -> dict:
    """Count kept candidates by (source_database, rna_type, length_bin)."""
    c = Counter()
    for r in rows:
        c[(r["source_database"], r["rna_type"], length_bin(r["length"]))] += 1
    # Also report marginal length-bin counts for row sanity.
    return {"cells": {f"{db}|{rtype}|{lb}": n for (db, rtype, lb), n in sorted(c.items())}}


def load_db_mappings(db_dir: Path) -> dict[str, tuple[str, str]]:
    """accession -> (source_database, rna_type) across all *.tsv in db_dir."""
    mapping: dict[str, tuple[str, str]] = {}
    files = sorted(db_dir.glob("*.tsv")) if db_dir.exists() else []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parsed = parse_db_mapping_line(line)
                if parsed is None:
                    continue
                acc, db, rtype = parsed
                # keep first seen (files are per-database; a URS may appear in
                # several -> that is the multi-database membership).
                if acc not in mapping:
                    mapping[acc] = (db, rtype)
    return mapping


def load_release22_accessions(fasta: Path) -> set[str]:
    acc = set()
    for header, _seq in iter_fasta(fasta):
        a, _ = parse_header(header)
        acc.add(a)
    return acc


def load_train_hashes(split_parquet: Path) -> set[str]:
    import polars as pl
    df = pl.read_parquet(split_parquet, columns=["canonical_sequence_hash", "split_membership"])
    return set(df.filter(pl.col("split_membership") == "train")["canonical_sequence_hash"].to_list())


_CAND_SCHEMA = pa.schema([
    ("accession", pa.string()), ("release", pa.string()), ("source_database", pa.string()),
    ("rna_type", pa.string()), ("canonical_sequence", pa.string()),
    ("canonical_sequence_hash", pa.string()), ("length", pa.int64()), ("length_bin", pa.string()),
])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release27-fasta", required=True, type=Path,
                    default=Path("/mnt/cunyuliu/tokenizer-benchmark/data/raw/release_27.0/rnacentral_active.fasta.gz"))
    ap.add_argument("--db-mappings-dir", required=True, type=Path,
                    default=Path("/mnt/cunyuliu/tokenizer-benchmark/data/raw/release_27.0/database_mappings"))
    ap.add_argument("--release22-fasta", required=True, type=Path,
                    default=Path("/mnt/cunyuliu/tokenizer-benchmark/data/raw/release_22.0/rnacentral_active.fasta.gz"))
    ap.add_argument("--split-parquet", required=True, type=Path,
                    default=Path("/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"))
    ap.add_argument("--out-dir", required=True, type=Path,
                    default=Path("/mnt/cunyuliu/tokenizer-benchmark/data/derived/time_ood"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading database_mappings...", flush=True)
    db_map = load_db_mappings(args.db_mappings_dir)
    print(f"database_mappings accessions: {len(db_map):,}", flush=True)

    print("loading release-22 accessions...", flush=True)
    r22_acc = load_release22_accessions(args.release22_fasta)
    print(f"release-22 accessions: {len(r22_acc):,}", flush=True)

    print("loading release-22 train hashes...", flush=True)
    train_hashes = load_train_hashes(args.split_parquet)
    print(f"release-22 train hashes: {len(train_hashes):,}", flush=True)

    kept: list[dict] = []
    n_old = 0
    n_exact = 0
    n_ambiguous = 0
    n_unknown_map = 0
    for header, seq in iter_fasta(args.release27_fasta):
        acc, header_rtype = parse_header(header)
        if acc in r22_acc:
            n_old += 1
            continue
        res = canonicalize_one(acc, seq, header_rtype)
        if res.alphabet_status != "primary":
            n_ambiguous += 1
            continue
        if res.canonical_hash in train_hashes:
            n_exact += 1
            continue
        src, rtype = db_map.get(acc, ("UNKNOWN_MAPPING", header_rtype))
        if src == "UNKNOWN_MAPPING":
            n_unknown_map += 1
        kept.append({
            "accession": acc, "release": "27.0", "source_database": src,
            "rna_type": rtype, "canonical_sequence": res.canonical_seq,
            "canonical_sequence_hash": res.canonical_hash, "length": res.length,
            "length_bin": res.length_bin,
        })
        if len(kept) % 2_000_000 == 0:
            print(f"  kept={len(kept):,} old={n_old:,} exact={n_exact:,} amb={n_ambiguous:,}", flush=True)

    summary = {
        "release": "27.0",
        "kept_candidates": len(kept),
        "in_release22": n_old,
        "exact_overlap_train": n_exact,
        "ambiguous_alphabet": n_ambiguous,
        "unknown_db_mapping": n_unknown_map,
        "composition": composition_counts(kept),
    }
    comp_path = args.out_dir / "release27_composition.json"
    with open(comp_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"wrote {comp_path}", flush=True)

    if kept:
        tbl = pa.table({k: [r[k] for r in kept] for k in kept[0].keys()}).cast(_CAND_SCHEMA)
    else:
        tbl = pa.table({c: pa.array([], _CAND_SCHEMA.field(c).type) for c in _CAND_SCHEMA.names})
    cand_path = args.out_dir / "time_ood_27_0_candidates.parquet"
    pq.write_table(tbl, cand_path)
    print(f"wrote {cand_path} ({len(kept):,} rows)", flush=True)


if __name__ == "__main__":
    main()