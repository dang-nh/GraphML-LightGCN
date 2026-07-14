"""Multi-seed final runs, building on the single-seed screening results
(results/screening_summary.csv, seed=0 already done for every config there).

- lightgcl_s0, s2 (best alpha), s3 (best gate) get 2 MORE seeds (1,2) WITH per-user
  metrics saved (for the degree-bucket breakdown table -- the core required result).
- mfbpr, lightgcn, s1 (best band) get 1 more seed (1) for the main comparison table.
- simgcl was not part of screening (separate script/model) -- run fresh at seeds 0,1.

Appends to results/finals_summary.csv and results/peruser/<config>_seed<k>.csv.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LGCL = ROOT / 'third_party' / 'LightGCL'
PPR_DIR = str(ROOT / 'results' / 'ppr' / 'yelp')
LOG_DIR = ROOT / 'results' / 'logs' / 'finals'
LOG_DIR.mkdir(parents=True, exist_ok=True)
PERUSER_DIR = ROOT / 'results' / 'peruser'
PERUSER_DIR.mkdir(parents=True, exist_ok=True)

FINAL_RE = re.compile(
    r'Final test:\s*Recall@20:\s*([\d.eE+-]+)\s*Ndcg@20:\s*([\d.eE+-]+)\s*'
    r'Recall@40:\s*([\d.eE+-]+)\s*Ndcg@40:\s*([\d.eE+-]+)')

BEST_BAND = '0.02,0.10'


def run_one(name, extra_args, seed, epoch=100, peruser=False, script='main.py', cwd=None):
    log_path = LOG_DIR / f'{name}_seed{seed}.log'
    cwd = cwd or LGCL
    cmd = [sys.executable, script]
    if script == 'main.py':
        cmd += ['--data', 'yelp', '--epoch', str(epoch), '--seed', str(seed)] + extra_args
        if peruser:
            cmd += ['--peruser_out', str(PERUSER_DIR / f'{name}_seed{seed}.csv')]
    else:
        cmd += extra_args + ['--seed', str(seed)]
    print(f'>>> {name} seed={seed}: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    text = log_path.read_text()
    m = FINAL_RE.search(text)
    row = {'config': name, 'seed': seed, 'epoch': epoch, 'returncode': proc.returncode,
           'elapsed_sec': round(elapsed, 1), 'nan_detected': 'nan' in text.lower()}
    if m:
        row.update({'recall@20': float(m.group(1)), 'ndcg@20': float(m.group(2)),
                     'recall@40': float(m.group(3)), 'ndcg@40': float(m.group(4))})
        print(f'    -> recall@20={row["recall@20"]:.4f} ndcg@20={row["ndcg@20"]:.4f} ({elapsed:.0f}s)', flush=True)
    else:
        print(f'    -> FAILED to parse final metrics (returncode={proc.returncode}, see {log_path})', flush=True)
    return row


def main():
    results = []
    summary_path = ROOT / 'results' / 'finals_summary.csv'

    def save():
        pd.DataFrame(results).to_csv(summary_path, index=False)

    # core comparison: S0, S2-best, S3-best with 2 extra seeds + per-user metrics
    core_configs = [
        ('lightgcl_s0', ['--neg_mode', 'S0']),
        ('s2_a0.3', ['--neg_mode', 'S2', '--ppr_dir', PPR_DIR, '--band', BEST_BAND, '--alpha_bar', '0.3']),
        ('s3_a1.0_tw10', ['--neg_mode', 'S3', '--ppr_dir', PPR_DIR, '--band', BEST_BAND, '--gate_a', '1.0', '--Tw', '10']),
    ]
    for name, args in core_configs:
        for seed in [1, 2]:
            results.append(run_one(name, args, seed, peruser=True))
            save()

    # supporting baselines/ablation: 1 extra seed each
    support_configs = [
        ('mfbpr', ['--gnn_layer', '0', '--lambda1', '0', '--lr', '1e-4', '--lambda2', '1e-5']),
        ('lightgcn', ['--lambda1', '0']),
        ('s1_band002_010', ['--neg_mode', 'S1', '--ppr_dir', PPR_DIR, '--band', BEST_BAND]),
    ]
    for name, args in support_configs:
        results.append(run_one(name, args, seed=1))
        save()

    # SimGCL: not covered by screening, run fresh at seeds 0,1
    for seed in [0, 1]:
        row = run_one('simgcl', ['--data', str(LGCL / 'data' / 'yelp'), '--epoch', '100',
                                   '--out', str(ROOT / 'results' / 'logs' / 'finals' / f'simgcl_seed{seed}_metrics.csv')],
                       seed=seed, script=str(ROOT / 'scripts' / 'baselines' / 'simgcl.py'), cwd=ROOT)
        results.append(row)
        save()

    save()
    print(f'\nFinals done. Summary at {summary_path}')
    print(pd.DataFrame(results)[['config', 'seed', 'recall@20', 'ndcg@20', 'elapsed_sec', 'nan_detected']].to_string(index=False))


if __name__ == '__main__':
    main()
