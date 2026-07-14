# Degree-Aware Curriculum Hard-Negative Sampling for LightGCL

Class project for Graph Analytics for Big Data (IT5429E). See `docs/proposal_GraphML_LightGCL.pdf`
for the full proposal and `docs/design/sampler_design.md` for the implementation-ready
design of the proposed sampler.

## Layout

- `third_party/LightGCL/` — vendored copy of the official [HKUDS/LightGCL](https://github.com/HKUDS/LightGCL)
  repo (`.git` stripped; we modify it directly). Contains `data/yelp/`, `data/gowalla/`
  (train/test splits, shipped with the repo), `main.py`, `model.py`, `utils.py`.
- `docs/design/` — implementation specs (sampler algorithm, experiment plan).
- `scripts/` — our own scripts (PPR precompute, run orchestration, result aggregation).
- `src/` — our own reusable Python modules, if any grow beyond scripts/.
- `configs/` — hyperparameter configs per run/variant.
- `results/` — logged metrics (CSV) and aggregated tables/figures. Raw per-run logs are
  gitignored; keep only aggregated summaries under version control.
- `report/` — LaTeX source for the final report.

## Environment

RTX 5090 (sm_120) requires a recent PyTorch build with cu128 kernels:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verified working: torch==2.11.0+cu128, numpy==2.5.1, scipy==1.18.0, pandas==3.0.3, tqdm==4.68.4.

## Running the original LightGCL baseline

```
cd third_party/LightGCL
source ../../.venv/bin/activate
python main.py --data yelp
```

~8s/epoch train + ~7s/eval-epoch on the RTX 5090 for Yelp2018 (100 epochs, eval every 3).

## Note on dataset statistics

The proposal's Table 1 (31,668 users / 38,048 items for Yelp2018) does not match what
actually ships in the official LightGCL repo. Measured directly from the vendored data:

| Dataset | #Users | #Items | Train | Test | Total interactions | Density |
|---|---|---|---|---|---|---|
| Yelp2018 | 29,601 | 24,734 | 1,069,128 | 305,466 | 1,374,594 | 0.188% |
| Gowalla | 50,821 | 57,440 | 1,172,425 | 130,270 | 1,302,695 | 0.045% |

The report uses these measured numbers (with a footnote on the discrepancy), since the
proposal itself specifies using the official repo's data as-is.
