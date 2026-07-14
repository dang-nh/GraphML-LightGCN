"""One-time GPU-batched Personalized PageRank (PPR) precompute for the degree-aware
curriculum hard-negative sampler. See docs/design/sampler_design.md Sections 1-2.

For each user, ranks all non-interacted items by PPR score on the symmetric-normalized
user-item bipartite graph, builds a percentile band of "hard" (related-but-not-top)
negatives, and dumps a fixed-size uniform sample pool per user plus training degrees.

Usage:
    python scripts/precompute_ppr.py --data third_party/LightGCL/data/yelp --out results/ppr/yelp
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='dir with trnMat.pkl, e.g. third_party/LightGCL/data/yelp')
    p.add_argument('--out', required=True, help='output dir for ppr_pool_<plo>_<phi>.npy and deg.npy')
    p.add_argument('--c', type=float, default=0.15, help='PPR restart/teleport probability')
    p.add_argument('--n_iter', type=int, default=30)
    p.add_argument('--chunk', type=int, default=2048, help='users per chunk')
    p.add_argument('--pool', type=int, default=500, help='fixed pool size per user')
    p.add_argument('--bands', type=str, default='0.05,0.15;0.02,0.10;0.10,0.20',
                    help='semicolon-separated p_lo,p_hi pairs; a band+pool file is written per pair')
    p.add_argument('--cuda', type=str, default='0')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def build_symmetric_adj(train_coo, n_user, n_item, device):
    """Symmetric-normalized bipartite adjacency, N=n_user+n_item, same normalization
    convention as third_party/LightGCL/main.py (D^-1/2 A D^-1/2)."""
    rowD = np.array(train_coo.sum(1)).squeeze()
    colD = np.array(train_coo.sum(0)).squeeze()
    rows, cols = train_coo.row, train_coo.col
    vals = 1.0 / np.sqrt(np.maximum(rowD[rows], 1) * np.maximum(colD[cols], 1))

    N = n_user + n_item
    # block [0:n_user, n_user:N] = R ; block [n_user:N, 0:n_user] = R^T
    r_rows = rows
    r_cols = cols + n_user
    t_rows = cols + n_user
    t_cols = rows

    all_rows = np.concatenate([r_rows, t_rows])
    all_cols = np.concatenate([r_cols, t_cols])
    all_vals = np.concatenate([vals, vals]).astype(np.float32)

    indices = torch.from_numpy(np.vstack([all_rows, all_cols]).astype(np.int64))
    values = torch.from_numpy(all_vals)
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce().to(device)
    return adj, rowD.astype(np.int32)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda:' + args.cuda
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / 'trnMat.pkl', 'rb') as f:
        train = pickle.load(f)
    n_user, n_item = train.shape
    train = train.tocoo()
    print(f'Loaded {data_dir.name}: n_user={n_user} n_item={n_item} nnz={train.nnz}')

    adj, deg = build_symmetric_adj(train, n_user, n_item, device)
    N = n_user + n_item
    print(f'Built symmetric adjacency: N={N} nnz={adj._nnz()}')

    # CSR-style per-user positive item lists, for masking (dense per-chunk boolean mask is
    # cheap at this scale: chunk x n_item bool).
    train_csr = train.tocsr()

    bands = [tuple(float(x) for x in b.split(',')) for b in args.bands.split(';')]
    print(f'Bands to compute: {bands}')

    rng = np.random.default_rng(args.seed)
    pools = {b: np.zeros((n_user, args.pool), dtype=np.int32) for b in bands}

    t0 = time.time()
    n_chunks = int(np.ceil(n_user / args.chunk))
    for ci in range(n_chunks):
        u0 = ci * args.chunk
        u1 = min(u0 + args.chunk, n_user)
        cs = u1 - u0

        E = torch.zeros(N, cs, device=device)
        E[torch.arange(u0, u1, device=device), torch.arange(cs, device=device)] = 1.0
        X = E.clone()
        for _ in range(args.n_iter):
            X = (1 - args.c) * torch.sparse.mm(adj, X) + args.c * E

        # item-node block, transpose to (cs, n_item)
        item_scores = X[n_user:N, :].transpose(0, 1).contiguous()

        # mask positives to -inf so they never enter the band
        mask = torch.from_numpy(train_csr[u0:u1].toarray()).to(device).bool()
        item_scores = item_scores.masked_fill(mask, float('-inf'))

        k_max = int(np.ceil(max(b[1] for b in bands) * n_item))
        top_vals, top_idx = torch.topk(item_scores, k=k_max, dim=1)
        # sort descending within the top-k (topk is not guaranteed sorted for all backends,
        # be explicit)
        sort_ord = torch.argsort(top_vals, dim=1, descending=True)
        top_idx_sorted = torch.gather(top_idx, 1, sort_ord)

        top_idx_np = top_idx_sorted.cpu().numpy()
        for band in bands:
            p_lo, p_hi = band
            lo = int(np.floor(p_lo * n_item))
            hi = int(np.ceil(p_hi * n_item))
            band_items = top_idx_np[:, lo:hi]  # (cs, hi-lo)
            width = band_items.shape[1]
            if width == 0:
                raise ValueError(f'empty band for {band}, widen p_hi-p_lo')
            col_idx = rng.integers(0, width, size=(cs, args.pool))
            sampled = np.take_along_axis(band_items, col_idx, axis=1)
            pools[band][u0:u1] = sampled.astype(np.int32)

        del E, X, item_scores, top_vals, top_idx, sort_ord, top_idx_sorted
        torch.cuda.empty_cache()
        if ci % 4 == 0 or ci == n_chunks - 1:
            print(f'  chunk {ci+1}/{n_chunks} users [{u0}:{u1}) elapsed={time.time()-t0:.1f}s')

    elapsed = time.time() - t0
    print(f'PPR + band construction done in {elapsed:.1f}s')

    np.save(out_dir / 'deg.npy', deg)
    for band, pool in pools.items():
        p_lo, p_hi = band
        fname = f'ppr_pool_{p_lo:.2f}_{p_hi:.2f}.npy'
        np.save(out_dir / fname, pool)
        print(f'Saved {out_dir / fname} shape={pool.shape}')
    print(f'Saved {out_dir / "deg.npy"} shape={deg.shape}')

    # sanity check: no positive leaked into any pool for a sample of users
    sample = rng.choice(n_user, size=min(200, n_user), replace=False)
    for band, pool in pools.items():
        bad = 0
        for u in sample:
            pos = set(train_csr[u].indices.tolist())
            if pos & set(pool[u].tolist()):
                bad += 1
        print(f'Sanity check band {band}: {bad}/{len(sample)} sampled users have a positive leak')


if __name__ == '__main__':
    main()
