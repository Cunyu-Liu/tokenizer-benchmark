"""Phase 4 sealed-test subsample for BLT arms (contract 3.5, 5.4, user decision).

BLT arms (P1/P2/P3) report an *exact causal* per-base next_base_BPN, which is
O(L^2) per sequence (one forward per position over the causal prefix). Running
the full 108,979-sequence test split is prohibitive, so the sealed-test BLT
likelihood is computed on a fixed, deterministic, homology-stratified subsample:

  - Stratify by homology cluster (cluster_id): no cluster contributes more than
    `--n-per-cluster` sequences, so high-abundance families cannot dominate the
    subsample (contract 3.1 homogeneity / family-balanced accounting).
  - Within each cluster, sequences are selected deterministically by a stable
    sort of their canonical hash (no randomness, no retraining dependence).
  - Clusters are ordered deterministically and consumed until the target total
    `--n` is reached.
  - The exact selection (seed, cluster set, per-strata composition, selection
    hash) is recorded in an auditable JSON manifest so the same subsample is
    reused at final table time and is reproducible.

Outputs (large artifacts under /mnt):
  - a Parquet of the selected test rows (split_membership, cluster_id,
    canonical_sequence, canonical_sequence_hash, rna_type, length_bin,
    family_annotation, clan_annotation, eligible_family)
  - a JSON manifest with selection parameters + strata composition + hash

Pure CPU (parquet only). No GPU, no training dependence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SPLIT = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
COLUMNS = ["canonical_sequence_hash", "canonical_sequence", "rna_type", "length",
           "length_bin", "cluster_id", "family_annotation", "clan_annotation",
           "eligible_family", "split_membership"]


def load_test_rows(path: str, split: str = "test"):
    """Stream the given split's rows grouped by cluster_id (ordered)."""
    pf = pq.ParquetFile(path)
    by_cluster: dict[str, list[dict]] = OrderedDict()
    for batch in pf.iter_batches(batch_size=200_000, columns=COLUMNS):
        d = batch.to_pydict()
        for i in range(len(d["split_membership"])):
            if d["split_membership"][i] != split:
                continue
            cid = d["cluster_id"][i]
            by_cluster.setdefault(cid, []).append(
                {c: d[c][i] for c in COLUMNS})
    return by_cluster


def pick_subsample(by_cluster: dict[str, list[dict]], n: int,
                   n_per_cluster: int) -> list[dict]:
    """Deterministic cluster-stratified subsample of exactly <= n rows.

    For each cluster, sort its rows by canonical_sequence_hash (stable) and take
    the first `n_per_cluster`. Consume clusters in sorted cluster_id order until
    the target `n` is reached (or clusters exhausted).
    """
    out: list[dict] = []
    for cid in sorted(by_cluster.keys()):
        rows = sorted(by_cluster[cid], key=lambda r: r["canonical_sequence_hash"])
        for r in rows[:n_per_cluster]:
            out.append(r)
            if len(out) >= n:
                return out
    return out


def strata_composition(rows: list[dict]) -> dict:
    comp = {
        "n_clusters": len({r["cluster_id"] for r in rows}),
        "column_counts": {},
    }
    for col in ("rna_type", "length_bin", "family_annotation", "clan_annotation",
                "eligible_family"):
        cnt = defaultdict(int)
        for r in rows:
            cnt[r[col]] += 1
        comp["column_counts"][col] = dict(cnt)
    return comp


def selection_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    # stable across runs: sort by hash then by cluster
    for r in sorted(rows, key=lambda x: (x["canonical_sequence_hash"],
                                         x["cluster_id"])):
        h.update(str(r["canonical_sequence_hash"]).encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", default=DEFAULT_SPLIT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, required=True,
                    help="target total number of sequences in the subsample")
    ap.add_argument("--n-per-cluster", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    by_cluster = load_test_rows(args.split_path, args.split)
    rows = pick_subsample(by_cluster, args.n, args.n_per_cluster)

    # Parquet of selected rows
    table = pa.Table.from_pylist(rows)
    pq_path = os.path.join(args.out_dir, "blt_test_subsample.parquet")
    pq.write_table(table, pq_path)

    manifest = {
        "description": "Deterministic homology-cluster-stratified sealed-test "
                       "subsample for exact causal BLT next_base_BPN (contract 3.5)",
        "split": args.split,
        "target_n": args.n,
        "n_per_cluster": args.n_per_cluster,
        "selected_n": len(rows),
        "n_available": sum(len(v) for v in by_cluster.values()),
        "n_clusters_available": len(by_cluster),
        "composition": strata_composition(rows),
        "selection_sha256": selection_hash(rows),
        "parquet": pq_path,
    }
    mf_path = os.path.join(args.out_dir, "blt_test_subsample_manifest.json")
    with open(mf_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("selected=%d/%d seqs, %d clusters, sha=%s" % (
        manifest["selected_n"], manifest["n_available"],
        manifest["composition"]["n_clusters"], manifest["selection_sha256"][:12]))
    print("WROTE", pq_path)
    print("WROTE", mf_path)


if __name__ == "__main__":
    main()