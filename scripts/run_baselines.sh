#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../third_party/LightGCL"
source ../../.venv/bin/activate
LOGDIR="../../results/logs"
mkdir -p "$LOGDIR"

echo "=== MF-BPR (yelp) ==="
# Repo has no seed control and no LR decay actually wired into the loop despite the --decay
# flag, so plain MF (no graph smoothing to bound embedding scale) diverges to NaN by ~epoch
# 10-17 with LightGCN-tuned defaults (lr=1e-3, lambda2=1e-7), even after adding global
# grad-norm clipping (max_norm=5, see model.py/main.py) and a numerically-stable logsigmoid
# BPR loss (both now permanent fixes benefiting all models). lr=1e-4 + lambda2=1e-5 verified
# stable for a full 100-epoch run: no NaN, monotonic convergence, Recall@20 0.039->0.030
# plateau (final), Ndcg@20 0.027.
python main.py --data yelp --gnn_layer 0 --lambda1 0 --lambda2 1e-5 --lr 1e-4 --note mfbpr 2>&1 | tee "$LOGDIR/mfbpr_yelp.log"

echo "=== LightGCN (yelp) ==="
python main.py --data yelp --lambda1 0 --note lightgcn 2>&1 | tee "$LOGDIR/lightgcn_yelp.log"

echo "=== LightGCL full / S0 baseline (yelp) ==="
python main.py --data yelp --note lightgcl_s0 2>&1 | tee "$LOGDIR/lightgcl_s0_yelp.log"

echo "ALL BASELINES DONE"
