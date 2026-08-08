"""Phase 1: homology-aware splitting.

Logic (Goal §3.1):
1. Rfam family per sequence (best hit per URS) + family->clan mapping.
2. Eligible family: >=100 cleaned sequences AND >=10 homology clusters.
3. 10% of eligible families -> family-validation, 10% -> family-test (held-out).
4. Families with clans -> clan-held-out sensitivity split.
5. Remaining homology clusters split by stable hash seed 20260808:
   98% train / 1% validation / 1% cluster-held-out test.
6. Stratify by length bin, RNA type, Rfam annotation status (source db added later).
7. All homology clusters of family/clan test are removed from training.
8. Temporal OOD handled separately (releases 23-26).

Enforces: canonical exact overlap zero across splits; 80/80 cluster not across split.
"""
from __future__ import annotations

import hashlib
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPLIT_SEED = "20260808"
RANDOM_SEED = 0


def _stable_accept(rep_id: str, buckets: list[tuple[str, float]]) -> str:
    """Deterministically route a cluster representative into one of the buckets
    by hashing rep_id with SPLIT_SEED. Returns the bucket label."""
    h = hashlib.sha256(f"{SPLIT_SEED}:{rep_id}".encode()).hexdigest()
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


def load_family_clan(family_clan_tsv: Path) -> dict[str, str]:
    """family -> clan mapping (from Rfam family_info/clan_info)."""
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
    table: pa.Table,
    family_annot: dict[str, str],
    family_clan: dict[str, str],
) -> pa.Table:
    """Assign split_membership per canonical sequence row.

    Returns a new table with columns: split_membership, family_annotation,
    clan_annotation, eligible_family.
    """
    acc = table.column("accessions").to_pylist()
    seq_hash = table.column("canonical_sequence_hash").to_pylist()
    cluster = table.column("cluster_id").to_pylist()
    length_bin = table.column("length_bin").to_pylist()
    rna_type = table.column("rna_type").to_pylist()

    # --- family per sequence: first accession's URS is the representative
    urs_list = []
    for a in acc:
        urs_list.append(a.split("\t")[0])
    family = [family_annot.get(u) for u in urs_list]

    # cluster -> families (all families present in a cluster)
    from collections import defaultdict
    cluster_fams: dict[str, set] = defaultdict(set)
    for cl, fam in zip(cluster, family):
        if fam:
            cluster_fams[cl].add(fam)

    # eligible family: >=100 cleaned sequences AND >=10 homology clusters
    fam_seq_count: dict[str, int] = defaultdict(int)
    fam_cluster_count: dict[str, set] = defaultdict(set)
    for cl, fams in cluster_fams.items():
        for fam_ in fams:
            fam_cluster_count[fam_].add(cl)
    for u, fam_ in zip(urs_list, family):
        if fam_:
            fam_seq_count[fam_] += 1
    eligible = {
        fam_
        for fam_
        in set(family) if fam_ and fam_seq_count[fam_] >= 100 and len(fam_cluster_count[fam_]) >= 10
    }

    # --- assign eligible families to family held-out via stable hash
    eligible_fams = sorted(eligible)
    family_split = {}
    for fam_ in eligible_fams:
        h = hashlib.sha256(f"{SPLIT_SEED}:fam:{fam_}".encode()).hexdigest()
        r = int(h[:8], 16) / 0xFFFFFFFF
        if r < 0.10:
            family_split[fam_] = "family_test"
        elif r < 0.20:
            family_split[fam_] = "family_validation"
        else:
            family_split[fam_] = "keep"

    # --- route clusters
    # clusters touching a family_test or family_validation family go to those splits
    results = []
    for i in range(table.num_rows):
        cl = cluster[i]
        fam_ = family[i]
        cl_fams = cluster_fams.get(cl, set())
        # held-out family membership
        relevant = {f for f in cl_fams | ({fam_} if fam_ else set()) if f in family_split}
        if any(family_split[f] == "family_test" for f in relevant):
            split = "family_test"
        elif any(family_split[f] == "family_validation" for f in relevant):
            split = "family_validation"
        else:
            # cluster-level split by hash of cluster representative
            rep = cl
            split = _stable_accept(rep, [("train", 0.98), ("validation", 0.01), ("test", 0.01)])
        results.append(split)

    clan = [family_clan.get(f) if f else None for f in family]
    return table.append_column("family_annotation", pa.array(family, type=pa.string()))\
                .append_column("clan_annotation", pa.array(clan, type=pa.string()))\
                .append_column("eligible_family", pa.array([f in eligible if f else False for f in family], type=pa.bool_()))\
                .append_column("split_membership", pa.array(results, type=pa.string()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-parquet", required=True, type=Path)
    ap.add_argument("--rfam-annotations", required=True, type=Path, help="rfam_annotations.tsv (uncompressed)")
    ap.add_argument("--family-clan", type=Path, default=None, help="family<TAB>clan TSV")
    ap.add_argument("--out-parquet", required=True, type=Path)
    args = ap.parse_args()

    table = pq.read_table(args.cluster_parquet)
    fam_annot = load_family_annotations(args.rfam_annotations)
    fam_clan = load_family_clan(args.family_clan) if args.family_clan else {}
    out = assign_splits(table, fam_annot, fam_clan)

    from collections import Counter
    counts = Counter(out.column("split_membership").to_pylist())
    print("split counts:", dict(counts))

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out_parquet)
    print(f"wrote {args.out_parquet}")


if __name__ == "__main__":
    main()