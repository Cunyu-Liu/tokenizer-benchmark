"""Verify release 22 active FASTA integrity against RNAcentral md5.tsv.

Cross-checks a sample of sequences: parse URS + sequence from the FASTA,
compute md5, and compare to the authoritative per-sequence md5 in md5.tsv.gz.
"""
from __future__ import annotations

import gzip
import hashlib
import argparse
import random
from pathlib import Path


def load_md5_index(md5_tsv: Path) -> dict[str, str]:
    """Return {urs_lower: md5} from md5.tsv.gz (RNAcentral format: md5<TAB>urs)."""
    idx = {}
    opener = gzip.open if str(md5_tsv).endswith(".gz") else open
    with opener(md5_tsv, "rt") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) >= 2:
                md5, urs = fields[0], fields[1]
                idx[urs.lower()] = md5
    return idx


def iter_fasta(path: Path):
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


def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--md5-tsv", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("loading md5 index...", flush=True)
    idx = load_md5_index(args.md5_tsv)
    print(f"md5 index size: {len(idx):,}", flush=True)

    # reservoir sampling over all sequences
    sample = []
    seen = 0
    for header, seq in iter_fasta(args.fasta):
        urs = header.split()[0]
        if len(sample) < args.n:
            sample.append((urs, seq))
        else:
            j = rng.randint(0, seen)
            if j < args.n:
                sample[j] = (urs, seq)
        seen += 1
        if seen % 2_000_000 == 0:
            print(f"  scanned {seen:,}", flush=True)

    print(f"scanned {seen:,} sequences; verifying {len(sample)} sample", flush=True)
    matched = 0
    mismatched = 0
    unmatched_lookup = 0
    for urs, seq in sample:
        expect = idx.get(urs.lower())
        if expect is None:
            unmatched_lookup += 1
            continue
        got = md5_hex(seq)
        if got == expect:
            matched += 1
        else:
            mismatched += 1
            print(f"MISMATCH {urs}: local={got} expected={expect}")
    print(f"RESULT sampled={len(sample)} matched={matched} mismatched={mismatched} no_md5_lookup={unmatched_lookup}")
    ok = mismatched == 0 and matched > 0
    print("INTEGRITY_PASS" if ok else "INTEGRITY_FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()