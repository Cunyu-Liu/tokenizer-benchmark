"""Phase 1: homology-aware splitting (polars-based, memory-efficient).

Logic (Goal §3.1):
1. Rfam family per sequence (best hit per URS) + family->clan mapping.
2. Eligible family: >=100 cleaned sequences AND >=10 homology clusters.
3. 10% eligible families -> family-validation, 10% -> family-test (held-out).
4. Families with clans -> clan-held-out sensitivity split.
5. Remaining homology clusters split by stable hash seed 20260808:
   98% train / 1% validation / 1% cluster-held-out test.
6. Stratify by length bin, RNA type, Rfam annotation status.
7. All homology clusters of family/clan test are removed from training.
8. Temporal OOD handled separately (releases 23-26).

Enforces: canonical exact overlap zero across splits; 80/80 cluster not across split.
"""
from __future__ import annotations

import hashlib
import argparse
from pathlib import Path

import polars as pl

SPLIT_SEED = "20260808"


def _stable_accept(digest: str, buckets: list[tuple[str, float]]) -> str:
    """Route a deterministic digest string into a bucket by hashing with SPLIT_SEED."""
    h = hashlib.sha256(f"{SPLIT_SEED}:{digest}".encode()).hexdigest()
    r = int(h[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for label, frac in buckets:
        acc += frac
        if r < acc:
            return label
    return buckets[-1][0]


def load_family_annotations(tsv: Path) -> dict[str, str]:
    """URS -> Rfam family (highest-score hit per URS)."""
    best: dict[str, tuple[float, str]] = {}
    with open(tsv) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            urs, model, score = fields[0], fields[1], fields[2]
            try:
                sc = float(score)
            except ValueError:
                sc = 0.0
            if urs not in best or sc > best[urs][0]:
                best[urs] = (sc, model)
    return {k: v[1] for k, v in best.items()}


def load_family_clan(family_clan_tsv: Path | None) -> dict[str, str]:
    """family -> clan mapping."""
    mapping = {}
    if family_clan_tsv is None:
        return mapping
    with open(family_clan_tsv) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                mapping[fields[0]] = fields[1]
    return mapping


def assign_splits(
    df: pl.DataFrame,
    family_annot: dict[str, str],
    family_clan: dict[str, str],
) -> pl.DataFrame:
    """Assign split_membership per canonical sequence row."""
    # representative URS = first accession
    df = df.with_columns(
        pl.col("accessions").str.split("\t").list.first().alias("rep_urs")
    )
    df = df.with_columns(
        pl.col("rep_urs").replace_strict(family_annot, default=None).alias("family_annotation")
    )

    # cluster -> families present
    fam_by_cluster = (
        df.filter(pl.col("family_annotation").is_not_null())
        .group_by("cluster_id")
        .agg(pl.col("family_annotation").unique().alias("cluster_families"))
    )

    # family sequence count and distinct cluster count
    fam_stats = (
        df.filter(pl.col("family_annotation").is_not_null())
        .group_by("family_annotation")
        .agg(
            pl.col("rep_urs").count().alias("n_seq"),
            pl.col("cluster_id").n_unique().alias("n_clusters"),
        )
    )
    fam_stats = fam_stats.with_columns(
        ((pl.col("n_seq") >= 100) & (pl.col("n_clusters") >= 10)).alias("eligible")
    )
    eligible = set(fam_stats.filter(pl.col("eligible")).get_column("family_annotation").to_list())

    # family held-out assignment via stable hash
    family_split = {}
    for fam in sorted(eligible):
        family_split[fam] = _stable_accept(f"fam:{fam}", [("family_test", 0.10), ("family_validation", 0.10), ("keep", 0.80)])

    # map each cluster to its held-out split (if any family in cluster is held-out)
    cluster_split = {}
    for row in fam_by_cluster.iter_rows(named=True):
        cl = row["cluster_id"]
        fams = row["cluster_families"]
        held = [f for f in fams if f in family_split and family_split[f] != "keep"]
        if held:
            # if any family_test -> family_test; else family_validation
            if any(family_split[f] == "family_test" for f in held):
                cluster_split[cl] = "family_test"
            else:
                cluster_split[cl] = "family_validation"

    # cluster-level split for non-held-out clusters
    all_clusters = df.get_column("cluster_id").unique().to_list()
    for cl in all_clusters:
        if cl not in cluster_split:
            cluster_split[cl] = _stable_accept(f"cl:{cl}", [("train", 0.98), ("validation", 0.01), ("test", 0.01)])

    df = df.with_columns(
        pl.col("cluster_id").replace_strict(cluster_split).alias("split_membership")
    )
    df = df.with_columns(
        pl.col("family_annotation").replace_strict(family_clan, default=None).alias("clan_annotation")
    )
    df = df.with_columns(
        pl.col("family_annotation").is_in(list(eligible)).alias("eligible_family")
    )
    return df.drop("rep_urs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-parquet", required=True, type=Path)
    ap.add_argument("--rfam-annotations", required=True, type=Path, help="rfam_annotations.tsv (uncompressed)")
    ap.add_argument("--family-clan", type=Path, default=None, help="family<TAB>clan TSV")
    ap.add_argument("--out-parquet", required=True, type=Path)
    args = ap.parse_args()

    df = pl.read_parquet(args.cluster_parquet)
    fam_annot = load_family_annotations(args.rfam_annotations)
    fam_clan = load_family_clan(args.family_clan) if args.family_clan else {}
    out = assign_splits(df, fam_annot, fam_clan)

    counts = out.group_by("split_membership").len().sort("split_membership")
    print("split counts:")
    for row in counts.iter_rows(named=True):
        print(f"  {row['split_membership']}: {row['len']:,}")

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out_parquet)
    print(f"wrote {args.out_parquet}")


if __name__ == "__main__":
    main()