import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

class LightGCL(nn.Module):
    """LightGCL: LightGCN message passing contrasted against a truncated-SVD global view.

    Two embedding views are propagated over ``l`` layers: the main LightGCN view
    ``E_u``/``E_i`` (symmetric-normalized adjacency ``adj_norm``, used for ranking) and
    an SVD-reconstructed view ``G_u``/``G_i`` (via the precomputed rank-q factors
    ``u_mul_s``, ``v_mul_s``, ``ut``, ``vt``), aligned with an InfoNCE contrastive loss.
    Only the main view ``E_u``/``E_i`` ever feeds the ranking score, so changing the BPR
    negative sampler (this project's contribution) cannot affect the contrastive branch.

    Args:
        n_u, n_i: number of users / items.
        d: embedding dimension.
        u_mul_s, v_mul_s, ut, vt: precomputed truncated-SVD factors of the normalized
            adjacency (``u_mul_s = U_q Sigma_q``, ``ut = U_q^T``, symmetrically for items).
        train_csr: CSR training interaction matrix, used to mask seen items at test time.
        adj_norm: symmetric-normalized sparse adjacency for LightGCN propagation.
        l: number of propagation layers.
        temp: InfoNCE temperature.
        lambda_1, lambda_2: contrastive loss weight and L2 weight-decay coefficient.
        dropout: edge dropout probability applied to ``adj_norm`` during training.
        batch_user: batch size used when scoring all items at test time.
        device: torch device.
    """

    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device):
        super(LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        """Test mode: rank all items for a batch of users, masking seen training items.
        Train mode: propagate both views for ``l`` layers and return the total loss
        (``loss = loss_r + lambda_1*loss_s + lambda_2*loss_reg``) plus its BPR
        (``loss_r``) and contrastive (``loss_s``) components, given a batch of
        ``(uids, iids)`` positive edges and one sampled ``neg`` item per edge (the
        negative sampler in ``utils.TrnData.neg_sampling`` controls the distribution of
        ``neg``; nothing else in this forward pass changes with the sampler)."""
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            # logsumexp instead of log(exp(.).sum()): the raw exp() overflows to inf once
            # embedding norms grow (no bound over long unregularized training), and inf/nan
            # then poisons the total loss even when lambda_1=0 (0 * nan = nan).
            neg_score = torch.logsumexp(G_u_norm[uids] @ E_u_norm.T / self.temp, dim=1).mean()
            neg_score += torch.logsumexp(G_i_norm[iids] @ E_i_norm.T / self.temp, dim=1).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            score_diff = torch.clamp(pos_scores - neg_scores, -20.0, 20.0)
            loss_r = -F.logsigmoid(score_diff).mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s
