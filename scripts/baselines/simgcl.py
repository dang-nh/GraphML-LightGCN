"""Standalone SimGCL baseline (Yu et al., SIGIR 2022) for comparison against LightGCL.

Self-contained: does not import from third_party/LightGCL (which another process may be
editing concurrently). Reuses the same data format (trnMat.pkl/tstMat.pkl scipy COO,
shape (n_user, n_item)) and the same symmetric-normalization / full-ranking eval protocol
as the vendored LightGCL repo, for numbers that are directly comparable.
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='dir with trnMat.pkl/tstMat.pkl, e.g. third_party/LightGCL/data/yelp')
    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--temp', type=float, default=0.2)
    p.add_argument('--cl_rate', type=float, default=0.2)
    p.add_argument('--l2', type=float, default=1e-7)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256, help='eval batch of users')
    p.add_argument('--inter_batch', type=int, default=4096, help='train edge batch')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda', type=str, default='0')
    p.add_argument('--out', type=str, default=None, help='override results CSV path')
    return p.parse_args()


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
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
            all_recall += hit / len(label)
            all_ndcg += dcg / idcg
            user_num += 1
    return all_recall / user_num, all_ndcg / user_num


class TrnData(data.Dataset):
    """Plain uniform BPR negative sampling with rejection — SimGCL's own sampler is untouched."""

    def __init__(self, coomat, seed=0):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.rng = np.random.default_rng(seed)
        self.n_item = coomat.shape[1]

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = int(self.rng.integers(self.n_item))
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class SimGCL(nn.Module):
    def __init__(self, n_user, n_item, d, n_layers, eps, temp, cl_rate, adj_norm, train_csr, device):
        super().__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_user, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_item, d)))
        self.n_layers = n_layers
        self.eps = eps
        self.temp = temp
        self.cl_rate = cl_rate
        self.adj_norm = adj_norm
        self.train_csr = train_csr
        self.device = device

    def propagate(self, perturbed=False):
        # bipartite propagation (matches LightGCN/LightGCL): adj_norm is (n_user x n_item);
        # there is no single homogeneous adjacency, so user/item embeddings are updated with
        # two separate matmuls each layer, exactly like the vendored LightGCL repo does.
        eu, ei = self.E_u_0, self.E_i_0
        eu_list, ei_list = [], []
        for _ in range(self.n_layers):
            eu_next = torch.sparse.mm(self.adj_norm, ei)
            ei_next = torch.sparse.mm(self.adj_norm.transpose(0, 1), eu)
            eu, ei = eu_next, ei_next
            if perturbed:
                noise_u = torch.rand_like(eu)
                eu = eu + torch.sign(eu) * F.normalize(noise_u, dim=-1) * self.eps
                noise_i = torch.rand_like(ei)
                ei = ei + torch.sign(ei) * F.normalize(noise_i, dim=-1) * self.eps
            eu_list.append(eu)
            ei_list.append(ei)
        eu_final = torch.stack(eu_list, dim=0).mean(dim=0)
        ei_final = torch.stack(ei_list, dim=0).mean(dim=0)
        return eu_final, ei_final

    def cl_loss(self, u1, i1, u2, i2, uids, iids):
        # in-batch InfoNCE: unique ids in the batch form the negative pool
        def infonce_loss(za, zb, idx):
            idx_u = torch.unique(idx)
            a = F.normalize(za[idx_u], dim=-1)
            b = F.normalize(zb[idx_u], dim=-1)
            logits = a @ b.T / self.temp
            labels = torch.arange(a.shape[0], device=a.device)
            return F.cross_entropy(logits, labels)
        loss_u = infonce_loss(u1, u2, uids)
        loss_i = infonce_loss(i1, i2, iids)
        return loss_u + loss_i

    def forward(self, uids, iids, pos, neg, test=False):
        if test:
            E_u, E_i = self.propagate(perturbed=False)
            preds = E_u[uids] @ E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.tensor(mask, device=self.device)
            preds = preds * (1 - mask) - 1e8 * mask
            return preds.argsort(descending=True)

        E_u, E_i = self.propagate(perturbed=False)
        u_emb, pos_emb, neg_emb = E_u[uids], E_i[pos], E_i[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_bpr = -(pos_scores - neg_scores).sigmoid().log().mean()

        u1, i1 = self.propagate(perturbed=True)
        u2, i2 = self.propagate(perturbed=True)
        loss_cl = self.cl_loss(u1, i1, u2, i2, uids, iids)

        loss_reg = sum(p.norm(2).square() for p in self.parameters())
        loss = loss_bpr + self.cl_rate * loss_cl
        return loss, loss_bpr, self.cl_rate * loss_cl, loss_reg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda:' + args.cuda

    data_dir = Path(args.data)
    with open(data_dir / 'trnMat.pkl', 'rb') as f:
        train = pickle.load(f)
    train_csr = (train != 0).astype(np.float32)
    with open(data_dir / 'tstMat.pkl', 'rb') as f:
        test = pickle.load(f)

    n_user, n_item = train.shape
    print(f'user_num: {n_user} item_num: {n_item} eps: {args.eps} cl_rate: {args.cl_rate} temp: {args.temp}')

    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    train = train.tocoo()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)

    train_data = TrnData(train, seed=args.seed)
    train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

    test = test.tocoo()
    test_labels = [[] for _ in range(n_user)]
    for i in range(len(test.data)):
        test_labels[test.row[i]].append(test.col[i])

    model = SimGCL(n_user, n_item, args.d, args.n_layers, args.eps, args.temp, args.cl_rate,
                    adj_norm, train_csr, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    rows = []
    for epoch in range(args.epoch):
        train_loader.dataset.neg_sampling()
        ep_loss = ep_bpr = ep_cl = ep_reg = 0.0
        for uids, pos, neg in tqdm(train_loader, disable=True):
            uids = uids.long().to(device)
            pos = pos.long().to(device)
            neg = neg.long().to(device)
            iids = torch.cat([pos, neg], dim=0)

            optimizer.zero_grad()
            loss, loss_bpr, loss_cl, loss_reg = model(uids, iids, pos, neg)
            total = loss + args.l2 * loss_reg
            total.backward()
            optimizer.step()

            ep_loss += total.item()
            ep_bpr += loss_bpr.item()
            ep_cl += loss_cl.item()
            ep_reg += loss_reg.item()
        nb = len(train_loader)
        print(f'Epoch: {epoch} Loss: {ep_loss/nb:.4f} Loss_bpr: {ep_bpr/nb:.4f} Loss_cl: {ep_cl/nb:.4f}')

        if epoch % 3 == 0 or epoch == args.epoch - 1:
            test_uids = np.arange(n_user)
            bno = int(np.ceil(len(test_uids) / args.batch))
            r20 = n20 = r40 = n40 = 0.0
            model.eval()
            with torch.no_grad():
                for b in range(bno):
                    s, e = b * args.batch, min((b + 1) * args.batch, len(test_uids))
                    inp = torch.LongTensor(test_uids[s:e]).to(device)
                    preds = model(inp, None, None, None, test=True).cpu().numpy()
                    a, b_ = metrics(test_uids[s:e], preds, 20, test_labels)
                    c, d_ = metrics(test_uids[s:e], preds, 40, test_labels)
                    r20 += a; n20 += b_; r40 += c; n40 += d_
            model.train()
            r20, n20, r40, n40 = r20/bno, n20/bno, r40/bno, n40/bno
            print(f'Test epoch {epoch}: Recall@20 {r20:.4f} Ndcg@20 {n20:.4f} Recall@40 {r40:.4f} Ndcg@40 {n40:.4f}')
            rows.append({'epoch': epoch, 'recall@20': r20, 'ndcg@20': n20, 'recall@40': r40, 'ndcg@40': n40})

    out_dir = Path('results/logs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f'simgcl_{data_dir.name}.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'Saved results to {out_path}')

    last = rows[-1]
    print(f'Final test: Recall@20: {last["recall@20"]} Ndcg@20: {last["ndcg@20"]} '
          f'Recall@40: {last["recall@40"]} Ndcg@40: {last["ndcg@40"]}')


if __name__ == '__main__':
    main()
