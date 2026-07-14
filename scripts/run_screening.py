"""Single-seed screening grid over BPR negative-sampling strategies (S0-S3) on LightGCL,
per docs/design/sampler_design.md Section 5, plus the three external baselines.

Runs each config as a subprocess of third_party/LightGCL/main.py, parses the final test
metrics, and writes results/screening_summary.csv.
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
LOG_DIR = ROOT / 'results' / 'logs' / 'screening'
LOG_DIR.mkdir(parents=True, exist_ok=True)

FINAL_RE = re.compile(
    r'Final test:\s*Recall@20:\s*([\d.eE+-]+)\s*Ndcg@20:\s*([\d.eE+-]+)\s*'
    r'Recall@40:\s*([\d.eE+-]+)\s*Ndcg@40:\s*([\d.eE+-]+)')


def run_one(name, extra_args, epoch=100, seed=0):
    log_path = LOG_DIR / f'{name}.log'
    cmd = [sys.executable, 'main.py', '--data', 'yelp', '--epoch', str(epoch), '--seed', str(seed)] + extra_args
    print(f'>>> {name}: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, cwd=LGCL, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    text = log_path.read_text()
    m = FINAL_RE.search(text)
    has_nan = 'nan' in text.lower().split('Final test:')[0][-2000:].lower() if 'Final test' in text else 'nan' in text.lower()
    row = {'config': name, 'seed': seed, 'epoch': epoch, 'returncode': proc.returncode,
           'elapsed_sec': round(elapsed, 1), 'nan_detected': has_nan}
    if m:
        row.update({'recall@20': float(m.group(1)), 'ndcg@20': float(m.group(2)),
                     'recall@40': float(m.group(3)), 'ndcg@40': float(m.group(4))})
        print(f'    -> recall@20={row["recall@20"]:.4f} ndcg@20={row["ndcg@20"]:.4f} ({elapsed:.0f}s)', flush=True)
    else:
        print(f'    -> FAILED to parse final metrics (returncode={proc.returncode}, see {log_path})', flush=True)
    return row


def main():
    results = []
    summary_path = ROOT / 'results' / 'screening_summary.csv'

    def save():
        pd.DataFrame(results).to_csv(summary_path, index=False)

    # --- external baselines (single seed) ---
    results.append(run_one('mfbpr', ['--gnn_layer', '0', '--lambda1', '0', '--lr', '1e-4', '--lambda2', '1e-5']))
    save()
    results.append(run_one('lightgcn', ['--lambda1', '0']))
    save()

    # --- S0 baseline (LightGCL default, uniform sampler) ---
    results.append(run_one('lightgcl_s0', ['--neg_mode', 'S0']))
    save()

    # --- S1: pure PPR, band sweep ---
    bands = ['0.05,0.15', '0.02,0.10', '0.10,0.20']
    for band in bands:
        tag = band.replace(',', '_').replace('.', '')
        results.append(run_one(f's1_band{tag}', ['--neg_mode', 'S1', '--ppr_dir', PPR_DIR, '--band', band]))
        save()

    # pick best band by recall@20 among the S1 runs just completed
    s1_rows = [r for r in results if r['config'].startswith('s1_band') and 'recall@20' in r]
    best_band = bands[0]
    if s1_rows:
        best_idx = max(range(len(s1_rows)), key=lambda i: s1_rows[i]['recall@20'])
        best_band = bands[best_idx]
    print(f'*** Best S1 band: {best_band} ***', flush=True)

    # --- S2: fixed global mixture at best band ---
    for alpha_bar in [0.3, 0.5]:
        results.append(run_one(f's2_a{alpha_bar}', ['--neg_mode', 'S2', '--ppr_dir', PPR_DIR,
                                                       '--band', best_band, '--alpha_bar', str(alpha_bar)]))
        save()

    # --- S3: degree-gated curriculum (ours) at best band ---
    for gate_a, Tw in [(1.0, 10), (1.5, 10), (1.0, 5)]:
        results.append(run_one(f's3_a{gate_a}_tw{Tw}', ['--neg_mode', 'S3', '--ppr_dir', PPR_DIR,
                                                           '--band', best_band, '--gate_a', str(gate_a), '--Tw', str(Tw)]))
        save()

    save()
    print(f'\nScreening done. Summary at {summary_path}')
    print(pd.DataFrame(results)[['config', 'recall@20', 'ndcg@20', 'elapsed_sec', 'nan_detected']].to_string(index=False))


if __name__ == '__main__':
    main()
