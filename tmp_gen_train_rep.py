"""Build a train cluster-representative FASTA (one longest seq per cluster).

train 14.1M rows collapse to ~3.0M unique cluster_ids; each cluster's longest
sequence is a valid 80/80 representative (members of a cluster are pairwise
80/80-equivalent under the Phase-1 clustering). For release-shift 80/80 removal
against train, if a candidate does not 80/80-hit any representative it cannot
hit any member (homology is anchored by cluster reps).
"""
import polars as pl

SPLIT = "/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/release22_split_8080.parquet"
OUT = "/mnt/cunyuliu/tokenizer-benchmark/tmp/tindex/train_rep.fa"
META = "/mnt/cunyuliu/tokenizer-benchmark/tmp/tindex/train_rep_meta.parquet"

df = pl.read_parquet(SPLIT, columns=[
    "cluster_id", "split_membership", "canonical_sequence", "canonical_sequence_hash"])
tr = df.filter(pl.col("split_membership") == "train")
tr = tr.with_columns(pl.col("canonical_sequence").str.len_chars().alias("_len"))
rep = (tr.sort("_len", descending=True)
         .group_by("cluster_id", maintain_order=True)
         .first())
rep = rep.drop("_len").drop("split_membership")
rep.write_parquet(META)
print("representative rows:", rep.height)
with open(OUT, "w") as f:
    for i, (cid, seq) in enumerate(zip(rep["cluster_id"].to_list(),
                                       rep["canonical_sequence"].to_list())):
        f.write(f">rep_{cid}\n{seq}\n")
print("wrote", OUT)