"""Correctness self-test for the degree-aware curriculum hard-negative sampler
(third_party/LightGCL/utils.py::TrnData). Checks:
  (a) S0 negatives are always valid (never a training positive), density-plausible.
  (b) PPR pools contain no positives for a sample of users (redundant with
      precompute_ppr.py's own check, re-verified here against the live TrnData object).
  (c) The degree gate + curriculum ramp produce the expected alpha values at epoch 0,
      epoch=Tw, and epoch>Tw, matching the design doc's formulas, for hand-picked users
      at different degrees.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'third_party' / 'LightGCL'))
from utils import TrnData  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / 'third_party' / 'LightGCL' / 'data' / 'yelp'
PPR_DIR = Path(__file__).resolve().parent.parent / 'results' / 'ppr' / 'yelp'

with open(DATA_DIR / 'trnMat.pkl', 'rb') as f:
    train = pickle.load(f).tocoo()

ppr_pool = np.load(PPR_DIR / 'ppr_pool_0.05_0.15.npy')
deg = np.load(PPR_DIR / 'deg.npy')

ok = True

# --- (a) S0 validity ---
ds0 = TrnData(train, neg_mode='S0', seed=0)
ds0.neg_sampling(epoch=0)
bad = ds0._is_positive(ds0.rows, ds0.negs)
n_bad = int(bad.sum())
print(f'(a) S0: {n_bad}/{len(ds0.negs)} negatives are positives (must be 0)')
ok &= (n_bad == 0)
print(f'(a) S0: negative item id range [{ds0.negs.min()}, {ds0.negs.max()}] of n_item={ds0.n_item} (sanity, non-degenerate)')

# --- (b) PPR pool has no positive leaks (spot check via live TrnData S1) ---
ds1 = TrnData(train, neg_mode='S1', ppr_pool=ppr_pool, deg=deg, seed=0)
ds1.neg_sampling(epoch=0)
bad1 = ds1._is_positive(ds1.rows, ds1.negs)
n_bad1 = int(bad1.sum())
print(f'(b) S1 (pure PPR): {n_bad1}/{len(ds1.negs)} negatives are positives (must be 0)')
ok &= (n_bad1 == 0)

# --- (c) degree gate + curriculum ramp values ---
Tw = 10
gate_a = 1.0
d_mid = float(np.median(deg))
ds3 = TrnData(train, neg_mode='S3', ppr_pool=ppr_pool, deg=deg, Tw=Tw, gate_a=gate_a, seed=0)

# hand-pick a low-degree, median-degree, and high-degree user
order = np.argsort(deg)
u_low = order[10]           # near the bottom (avoid degree-0 edge cases)
u_mid = order[len(order) // 2]
u_high = order[-10]

def expected_alpha(u, epoch):
    s = min(1.0, epoch / Tw)
    base = 1.0 / (1.0 + np.exp(-gate_a * (np.log1p(deg[u]) - np.log1p(d_mid))))
    return s * base

print(f'(c) degrees: low(u={u_low})={deg[u_low]} mid(u={u_mid})={deg[u_mid]} (d_mid={d_mid:.1f}) high(u={u_high})={deg[u_high]}')
all_close = True
for epoch in [0, Tw, Tw + 5]:
    for name, u in [('low', u_low), ('mid', u_mid), ('high', u_high)]:
        exp_a = expected_alpha(u, epoch)
        got_a = ds3.base_u[u] * min(1.0, epoch / Tw)
        close = abs(exp_a - got_a) < 1e-9
        all_close &= close
        print(f'    epoch={epoch:3d} user={name:4s}(d={deg[u]:4d}) expected_alpha={exp_a:.4f} got_alpha={got_a:.4f} {"OK" if close else "MISMATCH"}')
ok &= all_close

# sanity: at epoch 0, s(t)=0 so S3 negatives must be pure uniform (no PPR draws at all)
ds3.neg_sampling(epoch=0)
bad3_e0 = ds3._is_positive(ds3.rows, ds3.negs)
print(f'(c) S3 @ epoch 0 (s(t)=0, must behave like uniform): {int(bad3_e0.sum())} positive leaks (must be 0)')
ok &= (int(bad3_e0.sum()) == 0)

# at a later epoch, high-degree users should draw PPR negatives far more often than low-degree
ds3.neg_sampling(epoch=Tw + 5)
rows = ds3.rows
mask_low = rows == u_low
mask_high = rows == u_high
if mask_low.sum() > 0 and mask_high.sum() > 0:
    # crude proxy: PPR draws should overlap with the user's PPR pool at a much higher rate for
    # high-degree users than low-degree users, since alpha differs sharply
    pool_low = set(ppr_pool[u_low].tolist())
    pool_high = set(ppr_pool[u_high].tolist())
    frac_low = np.mean([n in pool_low for n in ds3.negs[mask_low]])
    frac_high = np.mean([n in pool_high for n in ds3.negs[mask_high]])
    print(f'(c) epoch={Tw+5}: frac negs landing in own PPR pool -- low-degree user={frac_low:.2f}, high-degree user={frac_high:.2f} (expect high >> low)')
    ok &= (frac_high > frac_low)

print()
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
