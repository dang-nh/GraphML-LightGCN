#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../third_party/LightGCL"
source ../../.venv/bin/activate
LOGDIR="../../results/logs"
mkdir -p "$LOGDIR"

echo "=== MF-BPR (yelp) ==="
# default lambda2=1e-7 diverges to NaN for plain MF (no graph smoothing to bound embedding scale);
# 1e-4 over-regularizes to a trivial all-equal-score collapse. 1e-6 verified stable+improving.
python main.py --data yelp --gnn_layer 0 --lambda1 0 --lambda2 1e-6 --note mfbpr 2>&1 | tee "$LOGDIR/mfbpr_yelp.log"

echo "=== LightGCN (yelp) ==="
python main.py --data yelp --lambda1 0 --note lightgcn 2>&1 | tee "$LOGDIR/lightgcn_yelp.log"

echo "=== LightGCL full / S0 baseline (yelp) ==="
python main.py --data yelp --note lightgcl_s0 2>&1 | tee "$LOGDIR/lightgcl_s0_yelp.log"

echo "ALL BASELINES DONE"
