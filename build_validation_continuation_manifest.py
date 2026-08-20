"""Build the validation continuation query manifest (contract 3.7).

The decoder-validation grid needs a FIXED raw-prefix query/prompt manifest in
advance: for a frozen `validation_query_seed` (contract 3.7, default 1201) it
selects a deterministic set of validation sequences and, per sequence, the
10% / 25% / 50% raw-nt continuation prefixes (each cut at
`continuation_cut = 6*floor(ratio*len/6)`, keeping only samples with >=1 real-nt
suffix). The same (non-final-test) algorithm is reused at Phase-8 for the final
query manifest with `final_query_seed=1202` by swapping the split/seed.

The selection is deterministic: candidates ordered by (canonical_sequence_hash)
so no dependence on retraining or row order, then consumed up to --n. The same
sequence may appear under each prefix fraction but each (hash, frac) entry is a
distinct scoring query.

Pure CPU (parquet only). GPU-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import pyarrow.parquet as pq

from p4_generate import continuation_cut

DEFAULT_SPLIT = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/"
                 "release22_split_8080.parquet")
DEFAULT_SEED = 1201  # contract 3.7 validation_query_seed
FRACS = [0.10, 0.25, 0.50]


def _sha(d: bytes) -> str:
    return hashlib.sha256(d).hexdigest()


def build(split: str, split_name: str, seed: int, n: int, fracs) -> dict:
    pf = pq.ParquetFile(split)
    rows = []
    col = ["split_membership", "canonical_sequence_hash", "canonical_sequence",
           "length", "rna_type", "source_database"]
    cols_present = [c for c in col if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=200_000, columns=cols_present):
        d = batch.to_pydict()
        for i in range(len(d["split_membership"])):
            if d["split_membership"][i] == split_name:
                rows.append({c: d[c][i] for c in cols_present})
    # Deterministic order keyed by hash-of(seed, hash): stable within a seed yet
    # different across seeds, all retraining-independent.
    def _order_key(r):
        s = r["canonical_sequence_hash"]
        return hashlib.sha256(("%d:%s" % (seed, s)).encode()).hexdigest()

    rows.sort(key=_order_key)
    entries = []
    counts = {f: 0 for f in fracs}
    for r in rows:
        seq = r["canonical_sequence"]
        L = len(seq)
        h = r["canonical_sequence_hash"]
        # seed-based stable shuffle of the entry order within hash ties
        for frac in fracs:
            k = continuation_cut(L, frac)
            if k is None:
                continue
            entries.append({
                "canonical_sequence_hash": h, "rna_type": r.get("rna_type"),
                "source_database": r.get("source_database"), "length": L,
                "prefix_len": k, "prefix": seq[:k], "ratio": frac,
                "actual_ratio": round(k / L, 6),
            })
            counts[frac] += 1
        if min(counts.values()) >= n:
            break
    # keep exactly n per fraction
    keep = []
    for frac in fracs:
        keep.extend([e for e in entries if e["ratio"] == frac][:n])
    keep.sort(key=lambda e: (e["ratio"], e["canonical_sequence_hash"]))
    payload = "\n".join("|".join(str(keep[i][k]) for k in
                                 ("canonical_sequence_hash", "length", "prefix_len"))
                        for i in range(len(keep))).encode()
    manifest = {
        "kind": "validation_continuation_query_manifest",
        "split": split_name,
        "validation_query_seed": seed,
        "fracs": sorted(fracs),
        "n_per_frac": n,
        "counts": {str(frac): c for frac, c in counts.items()},
        "n_entries": len(keep),
        "selection_sha256": _sha(payload),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return manifest, keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--split-name", default="validation")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n", type=int, default=1000, help="entries per fraction")
    ap.add_argument("--out", required=True, help="JSON manifest")
    ap.add_argument("--out-entries", default=None, help="optional JSONL of entries")
    args = ap.parse_args()
    manifest, keep = build(args.split, args.split_name, args.seed, args.n, FRACS)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    if args.out_entries:
        with open(args.out_entries, "w") as fh:
            for e in keep:
                fh.write(json.dumps(e) + "\n")
    print(json.dumps(manifest, indent=2))
    print("WROTE", args.out, "entries", len(keep))


if __name__ == "__main__":
    main()