#!/bin/bash
# Phase 4: launch one 100M full-matrix run (2.0B valid nt) on a GPU.
# Usage: bash p4_launch.sh <arm> <seed> <device> [gpu_uuid_check]
set -u
ARM=$1
SEED=$2
DEV=$3
TS=$(date +%Y%m%dT%H%M%S)
OUTDIR=/mnt/cunyuliu/tokenizer-benchmark/runs/phase4_${ARM}_s${SEED}_${TS}
mkdir -p $OUTDIR
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate toktokenbench
cd /home/cunyuliu/tokenizer-benchmark
# ~7-day safety timeout (2.0B nt at ~4650 nt/s ~= 5 days)
CUDA_VISIBLE_DEVICES=$DEV nohup timeout 650000 python -u p4_train.py \
  --arm $ARM --seed $SEED --device 0 --out-dir $OUTDIR \
  > $OUTDIR/run.log 2>&1 &
echo "P4 $ARM s$SEED -> GPU $DEV -> $OUTDIR (pid=$!)"
echo "P4 $ARM s$SEED -> GPU $DEV -> $OUTDIR (pid=$!)" >> /mnt/cunyuliu/tokenizer-benchmark/runs/phase4_launch.log
