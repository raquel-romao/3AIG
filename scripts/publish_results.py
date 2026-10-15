"""Publish aggregate run evidence without copying row-level data or executed notebooks."""
import argparse
import json
from pathlib import Path
import shutil

ALLOWED = ['results.md', 'run_manifest.json', 'oof_metrics.csv', 'fold_metrics.csv',
           'cv_summary.csv', 'feature_importance.csv', 'modality_importance.csv',
           'tfidf_terms.csv', 'embedding_anchors.csv', 'subgroups.csv', 'per_class.csv',
           'calibration.csv', 'complaint_frequency.csv']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/reproduced')
    parser.add_argument('--report-dir', default='reports/reproduced')
    args = parser.parse_args()
    src, dst = Path(args.run_dir), Path(args.report_dir)
    manifest = json.loads((src/'run_manifest.json').read_text())
    if manifest['sample_n'] is not None:
        parser.error('Sampled functional runs must not replace the full-data report')
    dst.mkdir(parents=True, exist_ok=True)
    for name in ALLOWED:
        if not (src/name).is_file():
            raise FileNotFoundError(f'Missing aggregate artifact: {name}')
        shutil.copy2(src/name, dst/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    perf = pd.read_csv(src/'oof_metrics.csv').sort_values('macro_f1')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = perf['model'].str.replace('_', ' ')
    axes[0].barh(labels, perf['macro_f1'], color='#1b6b84')
    axes[0].set(xlabel='Out-of-fold macro-F1', xlim=(0, 1), title='Model comparison')
    axes[1].barh(labels, perf['undertriage_rate'], color='#b86a43')
    axes[1].set(xlabel='Undertriage fraction', title='Ordinal errors')
    axes[1].tick_params(axis='y', labelleft=False)
    fig.suptitle('Synthetic Triagegeist data: random stratified folds', fontsize=12)
    fig.savefig(dst/'model_comparison.png', dpi=180)
    plt.close(fig)
    print(f'Published aggregate tables and figure to {dst}')


if __name__ == '__main__':
    main()
