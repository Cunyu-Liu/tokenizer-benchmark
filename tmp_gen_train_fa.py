"""Generate train.fa for the release-shift 80/80 target index."""
import sys
sys.path.insert(0, "/home/cunyuliu/tokenizer-benchmark")
import polars as pl
from pathlib import Path

SPLIT = "release22_split_8080.parquet"
train = pl.read_parquet(f"/mnt/cunyuliu/tokenizer-benchmark/data/derived/split/{SPLIT}",
                        columns=["canonical_sequence", "split_membership"])
train = train.filter(pl.col("split_membership") == "train")
out = Path("/mnt/cunyuliu/tokenizer-benchmark/tmp/tindex/train.fa")
with open(out, "w") as f:
    for i, s in enumerate(train["canonical_sequence"].to_list()):
        f.write(f">t{i}\n{s}\n")
print(f"wrote {out} with {len(train):,} sequences")