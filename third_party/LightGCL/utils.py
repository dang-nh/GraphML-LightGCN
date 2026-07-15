import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

def metrics(uids, predictions, topk, test_labels):
    """Full-ranking Recall@topk / NDCG@topk, averaged over users with >=1 test item.

    Args:
        uids: array of user ids in this batch.
        predictions: (batch, n_item) item ids sorted by descending predicted score
            (train items already masked out), as produced by ``model(..., test=True)``.
        topk: cutoff K (e.g. 20).
        test_labels: list indexed by user id of held-out test item ids.

    Returns:
        (recall, ndcg): aggregate Recall@K and NDCG@K, each averaged over users with
        at least one test item (users with no test items do not contribute).
    """
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num


def metrics_peruser(uids, predictions, topk, test_labels):
    """Like metrics(), but returns per-user recall/ndcg arrays (NaN for users with no
    test items) instead of a single aggregate, for degree-bucket breakdown analysis."""
    recall = np.full(len(uids), np.nan)
    ndcg = np.full(len(uids), np.nan)
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label) > 0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit += 1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc + 2))
            recall[i] = hit / len(label)
            ndcg[i] = dcg / idcg
    return recall, ndcg

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

class TrnData(data.Dataset):
    """Negative sampler for the BPR loss.

    neg_mode='S0' (default) reproduces the original uniform-random rejection sampler,
    just vectorized. Modes S1/S2/S3 implement the degree-aware curriculum hard-negative
    mixture from docs/design/sampler_design.md Section 3:
        P_t(j|u) = (1 - alpha_{u,t}) * Uniform(j) + alpha_{u,t} * PPR(j|u)
    S1: alpha_edge = 1 (pure PPR, no curriculum/gate)
    S2: alpha_edge = s(t) * alpha_bar (global constant)
    S3: alpha_edge = s(t) * sigmoid(a * (log(1+d_u) - log(1+d_mid)))  (full method)
    """

    def __init__(self, coomat, neg_mode='S0', ppr_pool=None, deg=None,
                 alpha_bar=0.5, Tw=10, gate_a=1.0, gate_dmid=None, seed=0):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.n_item = coomat.shape[1]
        self.n_user = coomat.shape[0]

        self.neg_mode = neg_mode
        self.ppr_pool = ppr_pool
        self.alpha_bar = alpha_bar
        self.Tw = Tw
        self.gate_a = gate_a
        self.rng = np.random.default_rng(seed)

        # vectorized positive-membership test: encode (row, col) as a single int64 and
        # binary-search a sorted array, instead of a per-sample dict/dok lookup.
        pos_codes = self.rows.astype(np.int64) * self.n_item + self.cols.astype(np.int64)
        self.pos_codes_sorted = np.sort(np.unique(pos_codes))

        if neg_mode in ('S1', 'S2', 'S3'):
            assert ppr_pool is not None, f'{neg_mode} requires a precomputed ppr_pool'
            assert deg is not None, f'{neg_mode} requires user degrees'
            self.pool_size = ppr_pool.shape[1]
        if neg_mode == 'S3':
            d_mid = gate_dmid if gate_dmid is not None else float(np.median(deg))
            self.base_u = 1.0 / (1.0 + np.exp(-gate_a * (np.log1p(deg) - np.log1p(d_mid))))
        else:
            self.base_u = None

    def _is_positive(self, rows, cols):
        codes = rows.astype(np.int64) * self.n_item + cols.astype(np.int64)
        idx = np.searchsorted(self.pos_codes_sorted, codes)
        idx = np.clip(idx, 0, len(self.pos_codes_sorted) - 1)
        return self.pos_codes_sorted[idx] == codes

    def _sample_uniform(self, rows):
        E = len(rows)
        neg = self.rng.integers(0, self.n_item, size=E).astype(np.int32)
        bad = self._is_positive(rows, neg)
        # rejection resampling for the small fraction of collisions (density is low, so
        # this converges in 1-2 rounds)
        while bad.any():
            n_bad = int(bad.sum())
            neg[bad] = self.rng.integers(0, self.n_item, size=n_bad).astype(np.int32)
            bad_idx = np.nonzero(bad)[0]
            still_bad = self._is_positive(rows[bad_idx], neg[bad_idx])
            bad = np.zeros(E, dtype=bool)
            bad[bad_idx[still_bad]] = True
        return neg

    def neg_sampling(self, epoch=0):
        E = len(self.rows)
        rows = self.rows

        if self.neg_mode == 'S0':
            self.negs = self._sample_uniform(rows)
            return

        s = min(1.0, epoch / self.Tw) if self.Tw > 0 else 1.0
        if self.neg_mode == 'S1':
            alpha_edge = np.ones(E)
        elif self.neg_mode == 'S2':
            alpha_edge = np.full(E, s * self.alpha_bar)
        elif self.neg_mode == 'S3':
            alpha_edge = s * self.base_u[rows]
        else:
            raise ValueError(f'unknown neg_mode {self.neg_mode}')

        use_ppr = self.rng.random(E) < alpha_edge
        neg = np.zeros(E, dtype=np.int32)

        if use_ppr.any():
            ppr_rows = rows[use_ppr]
            col_idx = self.rng.integers(0, self.pool_size, size=len(ppr_rows))
            neg[use_ppr] = self.ppr_pool[ppr_rows, col_idx]
        if (~use_ppr).any():
            neg[~use_ppr] = self._sample_uniform(rows[~use_ppr])

        self.negs = neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]