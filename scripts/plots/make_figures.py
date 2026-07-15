"""Generate report figures from results/*.csv and results/logs/**/*.log.

Reads only existing experiment outputs (no new training). Writes PDFs into
report/figures/.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
LOGS = RESULTS / 'logs'
FIGDIR = ROOT / 'report' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

COLORS = {
    'mfbpr': '#7f7f7f',
    'lightgcn': '#8c564b',
    'simgcl': '#9467bd',
    'lightgcl_s0': '#1f77b4',
    's0': '#1f77b4',
    's1_band005_015': '#ff7f0e',
    's1_band002_010': '#ff7f0e',
    's1_band010_020': '#ff7f0e',
    's2_a0.3': '#2ca02c',
    's2_a0.5': '#2ca02c',
    's2': '#2ca02c',
    's3_a1.0_tw10': '#d62728',
    's3_a1.5_tw10': '#d62728',
    's3_a1.0_tw5': '#d62728',
    's3': '#d62728',
}

LABELS = {
    'mfbpr': 'MF-BPR',
    'lightgcn': 'LightGCN',
    'simgcl': 'SimGCL',
    'lightgcl_s0': 'LightGCL-S0',
    's2_a0.3': 'S2',
    's3_a1.0_tw10': 'S3',
}


def fig1_main_comparison():
    df = pd.read_csv(RESULTS / 'main_results_summary.csv')
    order = ['mfbpr', 'lightgcn', 'simgcl', 'lightgcl_s0', 's2_a0.3', 's3_a1.0_tw10']
    df = df.set_index('config').loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(df))
    labels = [LABELS[c] for c in df['config']]
    colors = [COLORS[c] for c in df['config']]

    for ax, metric, title in zip(axes, ['recall20', 'ndcg20'], ['Recall@20', 'NDCG@20']):
        ax.bar(x, df[f'{metric}_mean'], yerr=df[f'{metric}_std'], color=colors,
               capsize=3, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title)
    fig.suptitle('Main comparison on Yelp2018 (mean $\\pm$ std across seeds)')
    fig.tight_layout()
    out = FIGDIR / 'main_comparison_bar.pdf'
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_sampler_ablation():
    df = pd.read_csv(RESULTS / 'ablation_summary.csv')
    order = ['lightgcl_s0', 's1_band005_015', 's1_band002_010', 's1_band010_020',
             's2_a0.3', 's2_a0.5', 's3_a1.0_tw10', 's3_a1.5_tw10', 's3_a1.0_tw5']
    df = df.set_index('config').loc[order].reset_index()
    disp_labels = ['S0', 'S1 (0.05,0.15)', 'S1 (0.02,0.10)', 'S1 (0.10,0.20)',
                   r'S2 $\bar\alpha$=0.3', r'S2 $\bar\alpha$=0.5',
                   'S3 a=1.0 Tw=10', 'S3 a=1.5 Tw=10', 'S3 a=1.0 Tw=5']
    colors = [COLORS['s0'], COLORS['s1_band005_015'], COLORS['s1_band002_010'],
              COLORS['s1_band010_020'], COLORS['s2_a0.3'], COLORS['s2_a0.5'],
              COLORS['s3_a1.0_tw10'], COLORS['s3_a1.5_tw10'], COLORS['s3_a1.0_tw5']]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(df))
    std = df['recall20_std'].fillna(0)
    ax.bar(x, df['recall20_mean'], yerr=std, color=colors, capsize=3, width=0.65)
    ax.axhline(df['recall20_mean'].iloc[0], color='black', linestyle='--', linewidth=1,
               alpha=0.6, label='S0 baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(disp_labels, rotation=35, ha='right')
    ax.set_ylabel('Recall@20')
    ax.set_ylim(df['recall20_mean'].min() * 0.95, df['recall20_mean'].max() * 1.03)
    ax.set_title('Sampler ablation: S0 (uniform) vs S1 (pure PPR) vs S2/S3 (curriculum mixtures)')
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    out = FIGDIR / 'sampler_ablation_bar.pdf'
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_degree_bucket():
    df = pd.read_csv(RESULTS / 'degree_bucket_summary.csv')
    df = df[df['bucket'] != 'overall']
    buckets = ['tail', 'mid', 'head']
    configs = ['lightgcl_s0', 's2_a0.3', 's3_a1.0_tw10']
    disp = ['S0', 'S2', 'S3']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.25
    x = np.arange(len(buckets))
    for i, cfg in enumerate(configs):
        sub = df[df['config'] == cfg].set_index('bucket').loc[buckets]
        ax.bar(x + (i - 1) * width, sub['recall20_mean'], width,
               yerr=sub['recall20_std'], capsize=3, color=COLORS[cfg], label=disp[i])
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in buckets])
    ax.set_ylabel('Recall@20')
    ax.set_title('Recall@20 by user-degree bucket (tertiles)')
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGDIR / 'degree_bucket_grouped_bar.pdf'
    fig.savefig(out)
    plt.close(fig)
    return out


EPOCH_RE = re.compile(r'Epoch:\s*(\d+)\s*Loss:\s*([\d.]+)\s*Loss_r:\s*([\d.]+)\s*Loss_s:\s*([\d.]+)')
TEST_RE = re.compile(r'Test of epoch\s*(\d+)\s*:\s*Recall@20:\s*([\d.]+)\s*Ndcg@20:\s*([\d.]+)')


def parse_log(path):
    epochs, loss_r, loss_s = [], [], []
    test_epochs, test_recall = [], []
    text = path.read_text(errors='ignore')
    for m in EPOCH_RE.finditer(text):
        epochs.append(int(m.group(1)))
        loss_r.append(float(m.group(3)))
        loss_s.append(float(m.group(4)))
    for m in TEST_RE.finditer(text):
        test_epochs.append(int(m.group(1)))
        test_recall.append(float(m.group(2)))
    return (np.array(epochs), np.array(loss_r), np.array(loss_s),
            np.array(test_epochs), np.array(test_recall))


def fig4_training_curves():
    configs = {'lightgcl_s0': 's0', 's2_a0.3': 's2', 's3_a1.0_tw10': 's3'}
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    for cfg, key in configs.items():
        logpath = LOGS / 'finals' / f'{cfg}_seed1.log'
        if not logpath.exists():
            logpath = LOGS / 'screening' / f'{cfg}.log'
        epochs, loss_r, loss_s, test_epochs, test_recall = parse_log(logpath)
        color = COLORS[key]
        label = LABELS.get(cfg, cfg)
        axes[0].plot(epochs, loss_r + loss_s, color=color, label=label, linewidth=1.3)
        axes[1].plot(test_epochs, test_recall, color=color, label=label,
                     marker='o', markersize=2, linewidth=1.3)

    axes[0].set_ylabel('Total training loss')
    axes[0].set_title('Training loss vs. epoch (seed 1)')
    axes[0].legend(frameon=False)

    axes[1].set_ylabel('Test Recall@20')
    axes[1].set_xlabel('Epoch')
    axes[1].set_title('Test Recall@20 vs. epoch (seed 1)')
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out = FIGDIR / 'training_curves.pdf'
    fig.savefig(out)
    plt.close(fig)
    return out


def fig5_hyperparam_sweep():
    df = pd.read_csv(RESULTS / 'screening_summary.csv')
    df = df.set_index('config')
    baseline = df.loc['lightgcl_s0', 'recall@20']

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)

    # (a) PPR band width
    band_cfgs = ['s1_band002_010', 's1_band005_015', 's1_band010_020']
    band_labels = ['(0.02,0.10)\nhardest', '(0.05,0.15)\ndefault', '(0.10,0.20)\nsafest']
    axes[0].bar(range(3), [df.loc[c, 'recall@20'] for c in band_cfgs],
                color=COLORS['s1_band005_015'], width=0.6)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(band_labels, fontsize=8)
    axes[0].set_title('S1: PPR band width')
    axes[0].set_ylabel('Recall@20')

    # (b) alpha_bar
    a_cfgs = ['s2_a0.3', 's2_a0.5']
    a_labels = [r'$\bar\alpha$=0.3', r'$\bar\alpha$=0.5']
    axes[1].bar(range(2), [df.loc[c, 'recall@20'] for c in a_cfgs],
                color=COLORS['s2_a0.3'], width=0.5)
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(a_labels)
    axes[1].set_title('S2: mixture weight')

    # (c) gate slope / warmup
    s3_cfgs = ['s3_a1.0_tw10', 's3_a1.5_tw10', 's3_a1.0_tw5']
    s3_labels = ['a=1.0\nTw=10', 'a=1.5\nTw=10', 'a=1.0\nTw=5']
    axes[2].bar(range(3), [df.loc[c, 'recall@20'] for c in s3_cfgs],
                color=COLORS['s3_a1.0_tw10'], width=0.6)
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(s3_labels, fontsize=8)
    axes[2].set_title('S3: gate slope / warm-up')

    all_vals = list(df.loc[band_cfgs + a_cfgs + s3_cfgs, 'recall@20']) + [baseline]
    ylo, yhi = min(all_vals), max(all_vals)
    pad = (yhi - ylo) * 0.3
    for ax in axes:
        ax.axhline(baseline, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_ylim(ylo - pad, yhi + pad)

    fig.suptitle('Single-seed hyperparameter screening sweep (dashed line = S0 baseline)')
    fig.tight_layout()
    out = FIGDIR / 'hyperparam_sweep.pdf'
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    outputs = [
        fig1_main_comparison(),
        fig2_sampler_ablation(),
        fig3_degree_bucket(),
        fig4_training_curves(),
        fig5_hyperparam_sweep(),
    ]
    readme_lines = [
        '# Report figures',
        '',
        '| File | Suggested caption |',
        '|---|---|',
        '| `main_comparison_bar.pdf` | Recall@20 / NDCG@20 across all evaluated models on Yelp2018, mean $\\pm$ std across seeds. |',
        '| `sampler_ablation_bar.pdf` | Recall@20 across the full S0/S1/S2/S3 sampler ablation, dashed line marks the S0 baseline. |',
        '| `degree_bucket_grouped_bar.pdf` | Recall@20 by user-degree tertile (tail/mid/head) for S0, S2, S3. |',
        '| `training_curves.pdf` | Training loss and test Recall@20 vs. epoch for S0, S2, S3 (seed 1). |',
        '| `hyperparam_sweep.pdf` | Single-seed screening sweep over PPR band width (S1), mixture weight (S2), and gate slope/warm-up (S3), vs. the S0 baseline. |',
        '',
    ]
    (FIGDIR / 'README.md').write_text('\n'.join(readme_lines))
    for o in outputs:
        size = o.stat().st_size
        print(f'{o.relative_to(ROOT)}: {size} bytes')


if __name__ == '__main__':
    main()
