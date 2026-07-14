# Degree-Aware Curriculum Hard-Negative Sampling for LightGCL — Implementation Design

Status: implementation-ready spec for Codex. Read this end-to-end before touching the repo.
Scope: we modify ONLY the BPR negative sampler. The InfoNCE/contrastive path, the SVD
view, the encoder, and the eval protocol are untouched.

> **CORRECTION (measured from the actual clone, third_party/LightGCL/data, 2026-07-14):**
> the proposal's Table 1 stats do NOT match the data that actually ships in the official
> repo. Measured directly from `trnMat.pkl`/`tstMat.pkl`:
> - **Yelp2018**: n_user = 29,601, n_item = 24,734, train_nnz = 1,069,128, test_nnz =
>   305,466, total interactions = 1,374,594, density = 0.188%.
> - **Gowalla**: n_user = 50,821, n_item = 57,440, train_nnz = 1,172,425, test_nnz =
>   130,270, total interactions = 1,302,695, density = 0.045%.
>
> All formulas below are written parametrically (n_user/n_item read from the actual
> matrix shape at runtime) so this does NOT change any code — just do not hardcode the
> proposal's 31,668/38,048 anywhere. The report must cite the measured stats above (with a
> footnote explaining the discrepancy vs. the proposal draft), not the proposal's Table 1.
> The dataset anchors below (31,668/38,048 etc.) are the ORIGINAL proposal assumptions,
> kept for the worked examples in this doc — treat all concrete numbers (POOL sizes, k for
> topk, timing estimates) as illustrative, not exact, and recompute from real shapes.

Dataset anchors (Yelp2018, as assumed in the original proposal draft — see correction
above for actual measured values): n_user = 31,668; n_item = 38,048; |O| = 1,561,406 edges;
mean user degree ≈ 49.3; N = n_user + n_item = 69,716; nnz of symmetric adjacency
Ã = 2·|O| = 3,122,812.

---

## 0. Repo assumptions (VERIFY FIRST — from HKUDS/LightGCL, unseen locally)

These were read from the public repo and drive every hook below. Confirm against the
actual clone in `third_party/LightGCL` before coding.

- `utils.py::TrnData(coomat)` stores `self.rows` (user idx of each edge), `self.cols`
  (pos item idx), `self.dokmat = coomat.todok()` (O(1) membership), and
  `self.negs = np.zeros(len(rows))`. Training is **edge-based**: DataLoader length =
  |O|, one negative per edge. `__getitem__` returns `(row, col, neg)`.
- `TrnData.neg_sampling()` fills `self.negs` and is called **once per epoch** from
  `main.py` (`train_loader.dataset.neg_sampling()`) BEFORE the batch loop. This single
  method is our entire injection point.
- `model.forward(uids, iids, pos, neg, test)`: BPR is
  `loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()` with one neg per edge.
  We do NOT touch the model; we only change which items land in `neg`.
- Train matrix loaded from a pickled scipy sparse COO (`trnMat`), shape (n_user, n_item),
  contiguous 0-based ids. **PPR must be computed on this exact matrix / id space.**
- SVD: `torch.svd_lowrank(adj, q=svd_q)`; `svd_q` is the rank arg for the side ablation.

Checklist to verify: (a) `neg_sampling` really has no args and is called per-epoch;
(b) row=user / col=item orientation; (c) `dokmat.shape[1] == n_item`; (d) data pickle
path + that ids are contiguous and shared with eval; (e) seed is settable for repro.

---

## 1. PPR computation (ONE-TIME per dataset, offline script)

### Graph and operator
Run PPR on the **symmetric-normalized bipartite adjacency** the repo already builds for
propagation: Ã = D^{-1/2} A D^{-1/2}, where A is the (N×N) undirected user↔item adjacency
(blocks R and Rᵀ), D the degree diagonal. Reusing Ã guarantees id-alignment with the
encoder and avoids building a second operator. (Random-walk norm P = D^{-1} A is the
textbook PPR choice and is an acceptable drop-in; symmetric norm is recommended purely to
reuse existing infra. Rankings are near-identical for band selection.)

Convergence note: for symmetric-norm Ã, spectral radius ≤ 1, so (1−c)Ã has radius ≤ 1−c<1
and power iteration converges geometrically at rate (1−c). It does not yield a normalized
probability vector, but we only need per-user **rankings**, which are invariant to that.

### Batched power iteration (GPU, chunked over users)
PPR fixed point (column convention), stacked over all source users:
    X ← (1−c)·Ã·X + c·E ,  X, E ∈ R^{N × n_user}
E is the personalization: column u is a one-hot on user-u's node. Because Ã is symmetric
we use `torch.sparse.mm(A_tilde, X)` directly.

Do NOT materialize full X (N×n_user = 8.83 GB fp32, ×2 for double-buffer = too much).
**Chunk users** into blocks of `CHUNK = 2048`:
  - X_chunk ∈ R^{N × 2048} ≈ 573 MB fp32.
  - E_chunk one-hots only on the user-node block.
  - Iterate `X = (1-c)*torch.sparse.mm(A_tilde, X) + c*E` for `n_iter` steps.
  - Slice the **item-node block** → item scores S_chunk ∈ R^{n_item × 2048}, transpose
    to (2048 × n_item) = 312 MB.
  - Build the band for these users (Section 2), store the candidate pool, free X_chunk.
  - Move to next chunk (≈16 chunks total for Yelp).

### Parameters
- Restart / teleport c = 0.15 (PageRank damping 0.85). Fixed for the project (do NOT tune;
  it is a one-time cost and not a claimed contribution). Higher c (0.2–0.3) localizes more;
  note only.
- Iterations: n_iter = 30 (0.85^30 ≈ 7.6e-3 residual), OR stop when max column L1 delta
  < 1e-4. 30 is safe and cheap; rankings stabilize far earlier than values.
- dtype fp32 (fp16 risks rank ties at the band boundary — avoid).

### Cost / memory
- FLOPs ≈ 2 · n_user · nnz(Ã) · n_iter ≈ 2·31,668·3.12e6·30 ≈ 5.9e15. On an RTX 5090 this
  is single-digit to ~20 minutes wall-clock for sparse-dense mm, dominated by memory BW.
- Peak GPU mem per chunk < 1 GB. Total one-time budget: well under 30 min including band
  extraction and disk write. This comfortably fits the 3-day budget.

### Recommendation
GPU batched power iteration with `torch.sparse.mm`, chunked over users. Rejected
alternatives: (a) scipy/networkx per-user PPR = 31k Python-loop solves → hours–days;
(b) full dense all-pairs = 8.8 GB+ and pointless precision. The repo already puts Ã on GPU,
so this is a ~40-line offline script (`scripts/precompute_ppr.py`) emitting one `.npy`.

---

## 2. Hard-negative band construction (per user, no full dense sort)

Per user-chunk we have item scores S ∈ R^{chunk × n_item}. We need, per user, items whose
PPR rank falls in the percentile band [p_lo·|I|, p_hi·|I|], excluding positives.

Avoid the full O(n_user · n_item log n_item) argsort. Procedure per chunk:
1. Mask positives: set S[user, j] = −inf for j ∈ N_u (gather from CSR of R). This both
   removes known positives from candidacy AND keeps them out of the band automatically.
2. `top_vals, top_idx = torch.topk(S, k = ceil(p_hi·n_item), dim=1)` → top ~15% items
   (k ≈ 5,707 for Yelp). topk over 38k cols is fast and avoids sorting the tail.
3. Sort only those k values per row (cheap), slice off the head `[floor(p_lo·n_item):]`
   (drop the top ~5%, i.e. first ≈1,902) → the band B_u (≈3,805 items/user default).
4. **Downsample to a fixed pool**: since P_PPR(j|u)=Uniform(B_u), we do not need the full
   band online. Draw `POOL = 500` items uniformly from B_u (with replacement if
   |B_u|<500 — only possible for pathological tiny-degree users) and store as int32.

Output artifact: `ppr_pool[n_user, POOL]` int32 (≈63 MB for Yelp), plus `d_u[n_user]`
int32 (user train degrees). One file per (dataset, p_lo, p_hi). Because bands are cheap to
re-slice from the stored top-k values, generate all grid bands in one PPR pass by caching
`top_vals/top_idx` per chunk and slicing multiple (p_lo,p_hi) before freeing.

Positives-in-band caveat: even after excluding N_u, some band items may be *unobserved*
positives — this is the exact false-negative risk the method targets; the top-p_lo cutoff
and the degree gate (Section 3) are the mitigations, not bugs.

---

## 3. Degree-gated curriculum sampler (per-epoch, vectorized numpy)

### Precomputed once (cheap, per (a, d_mid) config — recompute in <1 ms)
- d_u = train degree per user (from artifact).
- Sigmoid gate parameterized by (slope a, midpoint degree d_mid):
      base_u = sigmoid( a · ( log(1+d_u) − log(1+d_mid) ) )
  i.e. b = −a·log(1+d_mid). This pins alpha=0.5 at degree d_mid and makes the grid
  interpretable (default d_mid = median degree). base_u ∈ R^{n_user}, precomputed.
- ppr_pool[n_user, POOL] loaded once.

### Per-epoch `neg_sampling(epoch)` — fully vectorized over all |O| edges
Let E = |O|, rows = edge users (len E), s = min(1, epoch / T_w).
Per-mode alpha for each edge:
- S0: alpha_edge = 0                          (pure uniform, = repo default)
- S1: alpha_edge = 1                          (pure PPR, no curriculum/gate)
- S2: alpha_edge = s · alpha_bar              (global constant alpha_bar)
- S3: alpha_edge = s · base_u[rows]           (degree gate, FULL method)

Sampling (vectorized, ~<1 s/epoch, not a bottleneck):
```
s = min(1.0, epoch / T_w)
alpha_edge = s * gate[rows]                      # gate depends on mode (above)
use_ppr = rng.random(E) < alpha_edge             # bool mask

# PPR branch: index precomputed pool (already excludes positives)
col = rng.integers(0, POOL, size=E)
neg_ppr = ppr_pool[rows, col]

# Uniform branch: vectorized sample + one cleanup pass for the ~0.13% collisions
neg_uni = rng.integers(0, n_item, size=E)
bad = <membership check (rows, neg_uni) in dokmat/CSR>   # ≈ density → ~2k edges
while bad.any():                                          # 1–2 iterations
    neg_uni[bad] = rng.integers(0, n_item, size=bad.sum())
    bad = <recheck only bad>

self.negs = np.where(use_ppr, neg_ppr, neg_uni).astype(np.int32)
```
Notes: use a seeded `np.random.Generator`. Membership check should use a CSR/`indptr`
+ `searchsorted` vectorized test (or keep the repo's dokmat for the tiny collision set).
The PPR branch needs no rejection because the pool already excludes N_u.

### Curriculum / epoch flow
Only two code changes propagate `epoch`:
1. `neg_sampling(self, epoch)` signature.
2. Call site in `main.py`: `train_loader.dataset.neg_sampling(epoch)`.
`s(t)=min(1,t/T_w)` ramps hard-negative probability from 0 to full over T_w epochs, so
early training sees mostly uniform negatives (curriculum warm-up).

---

## 4. Integration plan & checklist

Files to touch (assumption-tagged):
- `scripts/precompute_ppr.py` (NEW): load trnMat → build Ã (reuse repo's normalizer) →
  batched PPR → band → dump `ppr_pool.npy`, `deg.npy`. Standalone, GPU, ~20 min.
- `utils.py::TrnData`:
  - `__init__`: accept `neg_mode` ('S0'|'S1'|'S2'|'S3'), `ppr_pool`, `base_u`
    (or (a,d_mid) to compute it), `alpha_bar`, `T_w`, and a `Generator`.
  - replace `neg_sampling(self)` with `neg_sampling(self, epoch)` per Section 3.
- `main.py`: build/pass the extra args when constructing `TrnData`; pass `epoch` at the
  call site; add CLI flags: `--neg_mode --ppr_pool --p_lo --p_hi --Tw --alpha_bar
  --gate_a --gate_dmid --seed`.

Verify against real repo: (1) neg_sampling call site + arity; (2) row=user/col=item;
(3) that the SAME preprocessed matrix feeds both PPR precompute and training (id-align);
(4) `dokmat.shape[1]==n_item`; (5) global seeding covers the sampler RNG; (6) no other
place samples negatives (e.g. a separate valid/test sampler — leave those uniform).
Non-goal check: confirm the contrastive/InfoNCE in-batch negatives are untouched.

---

## 5. Hyperparameter defaults & (small) screening grid

Fixed (not searched): c=0.15, n_iter=30, POOL=500, dtype fp32.

| Param | Default | Screening grid | Applies to |
|---|---|---|---|
| (p_lo, p_hi) band | (0.05, 0.15) | {(0.05,0.15) default, (0.02,0.10) harder, (0.10,0.20) safer} | S1/S2/S3 |
| T_w (warm-up epochs) | 10 | {5, 10} | S2/S3 |
| alpha_bar | 0.5 | {0.3, 0.5} | S2 |
| gate slope a | 1.0 | {1.0, 1.5} | S3 |
| gate midpoint d_mid | median degree | {median} (fix; tune a only) | S3 |

Sigmoid sanity (a=1.0, d_mid=median≈20–30): tail user d=3 → alpha≈0.17; median → 0.5;
head d=500 → ≈0.96. Exactly the intended "hard for active users, soft for sparse users."

Budget-sized screening (single seed, ≤ ~10 LightGCL runs, fits ≈ half a day):
1. S0 baseline (1 run).
2. S1 over 3 bands → pick best band B* (3 runs).
3. S2 at B*, alpha_bar∈{0.3,0.5} (2 runs).
4. S3 at B*, (a,T_w) ∈ {(1.0,10),(1.5,10),(1.0,5)} (3 runs).
Select best of S1–S3 (expected S3) for the multi-seed final.

---

## 6. Degree-bucket definition

- Bucket users by **training degree** d_u into tertiles by USER COUNT: thresholds =
  33rd and 67th percentiles of {d_u}. tail = bottom third, mid = middle, head = top third.
  (Percentile tertiles chosen over fixed cutoffs for balanced, dataset-agnostic, reproducible
  groups; report the concrete thresholds once computed.)
- Metric: compute per-user Recall@20 / NDCG@20 under the standard full-ranking protocol,
  then average **within each bucket** over test users that have ≥1 test item. Report overall
  + per-bucket. The long-tail claim lives in the **tail** bucket.
- Buckets are fixed by training data → identical across all models, so deltas are comparable.

---

## 7. Compressed 3-day schedule (~1.5–2 GPU-days real)

Core research question = "does an activity-modulated hard-negative curriculum improve
LightGCL accuracy AND long-tail robustness WITHOUT adding false-negative bias?" The
irreducible experiment is **S0 vs S3 on Yelp, multi-seed, with the degree-bucket breakdown.**
Everything else is supporting evidence.

Plan:
- **Day 1** (other agent builds env in parallel): finalize + unit-test
  `precompute_ppr.py` and the `neg_sampling` patch offline. As soon as env is up: run PPR
  precompute for Yelp (~20 min), reproduce LightGCL **S0** to confirm paper numbers +
  runtime/epoch (this calibrates the rest of the schedule).
- **Day 2** (full GPU): AM — single-seed screening (Section 5, ≤10 runs) → pick B* and best
  S3. PM — launch **3-seed finals for S0 and best-S3** (6 runs) + external baselines
  (LightGCN via repo, then SimGCL, MF-BPR) as capacity allows.
- **Day 3** (part GPU + report): collect finals, compute degree-bucket tables, write report.

What to cut if time runs short (drop top-first; NEVER cut the last line):
1. Gowalla (drop first — Yelp is primary).
2. Energy-based SVD rank side-ablation (explicitly minor in the proposal).
3. Trim baselines: keep LightGCL(S0) + LightGCN; drop SimGCL, then MF-BPR. Baselines may
   run 2 seeds (or single) — they are context, not the claim.
4. Shrink screening: fix band=(0.05,0.15), skip S2 alpha_bar sweep, run S0→S1→S3 directly.
5. Reduce final seeds 3→2 — but keep **≥2 seeds for the S0-vs-S3 comparison specifically.**
NEVER cut: **S0 vs S3 on Yelp with ≥2 seeds and the head/mid/tail breakdown.** That single
table is the paper.

### Risks to flag
- **Effect size vs seed noise**: with 2–3 seeds, overall Recall deltas may sit inside ±std.
  Mitigation: foreground the *tail-bucket* delta (where the mechanism should bite hardest)
  and the S0→S1→S2→S3 monotone ablation trend, and report mean±std honestly.
- **PPR band = unobserved positives (false negatives)**: aggressive bands can *lower*
  overall Recall — this is precisely the hypothesis under test; the top-p_lo cutoff + degree
  gate are the designed mitigations. If S1 (pure PPR) underperforms S0 but S3 recovers/beats
  it, that is a *positive* result for the thesis, not a failure.
- **Id / preprocessing mismatch** between the PPR script and the repo's training matrix —
  the single highest-severity correctness bug; verify item 3 of the Section 4 checklist.
- **Unknown per-epoch runtime** on 5090 until the Day-1 S0 repro; the schedule's cut-list is
  ordered to absorb an overrun.
- **SVD/PPR decoupling** (positive note): the sampler is independent of the SVD view, so PPR
  precompute and the curriculum add no per-step cost beyond a <1 s/epoch numpy pass — no
  throughput risk in the training loop.
