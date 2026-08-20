"""Lay submit for the parallel release-release 80/80 cross-search.

Launches n-run (default 16) chunk jobs concurrently, each with threads such
that n-run*threads ~ 96. Uses GNU parallel-like bounded concurrency via a
simple FIFO. Each chunk runs data/temporal_chunk_search.py.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

OUT = ("/mnt/cunyuliu/tokenizer-benchmark/data/derived/"
       "time_ood_rel27/search")
MOD = "data.temporal_chunk_search"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=48)
    ap.add_argument("--runs", type=int, default=16, help="concurrent jobs")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    Path(OUT).mkdir(parents=True, exist_ok=True)

    procs = {}
    next_i = 0
    done = set()
    while next_i < args.n_chunks or any(p.poll() is None for p in procs):
        # launch until concurrency cap
        while next_i < args.n_chunks and len(procs) < args.runs and len(procs) < args.n_chunks:
            i = next_i; next_i += 1
            cmd = (f"cd /home/cunyuliu/tokenizer-benchmark && "
                   f"PATH=/home/cunyuliu/tools/mmseqs2/bin:$PATH "
                   f"{sys.executable} -m {MOD} --chunk-i {i} "
                   f"--n-chunks {args.n_chunks} --threads {args.threads} "
                   f"--out-dir {OUT}")
            p = subprocess.Popen(cmd, shell=True)
            procs[i] = p
            print(f"[launch] chunk {i} pid={p.pid}", flush=True)
        # reap done
        for i in list(procs.keys()):
            if procs[i].poll() is not None:
                rc = procs[i].returncode
                done.add(i)
                print(f"[done] chunk {i} rc={rc} (completed {len(done)}/{args.n_chunks})",
                      flush=True)
                del procs[i]
        time.sleep(10)
    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()