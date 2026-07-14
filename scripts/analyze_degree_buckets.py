"""Aggregate per-user metric CSVs (results/peruser/<config>_seed<k>.csv) into a
head/mid/tail degree-bucket breakdown table, per docs/design/sampler_design.md Section 6:
tertiles by training-interaction count, computed once (from any config's per-user file,
since buckets are fixed by training data and identical across models) and applied to all.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PERUSER_DIR = ROOT / 'results' / 'peruser'


def bucket_of(deg, lo, hi):
    return np.where(deg <= lo, 'tail', np.where(deg <= hi, 'mid', 'head'))


def main():
    files = sorted(glob.glob(str(PERUSER_DIR / '*_seed*.csv')))
    if not files:
        print(f'No per-user CSVs found in {PERUSER_DIR}')
        return

    # compute tertile thresholds once, from the first file (degrees are identical across
    # configs/seeds since they only depend on the fixed training data)
    ref = pd.read_csv(files[0])
    lo, hi = np.percentile(ref['degree'], [33, 67])
    print(f'Degree tertile thresholds: tail <= {lo:.1f} < mid <= {hi:.1f} < head')

    rows = []
    for fp in files:
        fname = Path(fp).stem  # e.g. lightgcl_s0_seed1
        config, _, seedpart = fname.rpartition('_seed')
        seed = int(seedpart)
        df = pd.read_csv(fp)
        df = df.dropna(subset=['recall@20'])
        df['bucket'] = bucket_of(df['degree'].values, lo, hi)
        for bucket in ['tail', 'mid', 'head', 'overall']:
            sub = df if bucket == 'overall' else df[df['bucket'] == bucket]
            rows.append({'config': config, 'seed': seed, 'bucket': bucket, 'n_users': len(sub),
                          'recall@20': sub['recall@20'].mean(), 'ndcg@20': sub['ndcg@20'].mean()})

    long_df = pd.DataFrame(rows)
    long_path = ROOT / 'results' / 'degree_bucket_per_seed.csv'
    long_df.to_csv(long_path, index=False)

    summary = long_df.groupby(['config', 'bucket']).agg(
        n_seeds=('seed', 'nunique'),
        recall20_mean=('recall@20', 'mean'), recall20_std=('recall@20', 'std'),
        ndcg20_mean=('ndcg@20', 'mean'), ndcg20_std=('ndcg@20', 'std'),
    ).reset_index()
    summary_path = ROOT / 'results' / 'degree_bucket_summary.csv'
    summary.to_csv(summary_path, index=False)

    print(f'\nSaved per-seed table to {long_path}')
    print(f'Saved summary (mean+/-std across seeds) to {summary_path}\n')
    pd.set_option('display.width', 160)
    order = {'tail': 0, 'mid': 1, 'head': 2, 'overall': 3}
    summary['ord'] = summary['bucket'].map(order)
    print(summary.sort_values(['config', 'ord']).drop(columns='ord').to_string(index=False))


if __name__ == '__main__':
    main()
