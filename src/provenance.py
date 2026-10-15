"""Export exact result tables and run metadata without publishing competition rows."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess

import numpy as np
import pandas as pd

from src.config import project_root, as_path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = project_root()
    files = [p for folder in ['src', 'configs', 'scripts'] for p in (root / folder).glob('*')
             if p.is_file() and p.suffix in {'.py', '.yaml', '.sh'}]
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}


def markdown_table(frame):
    def fmt(v):
        return f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def export_run(run_dir, cfg, tables, checks, oof, train_df, *, sample_n, n_splits, split_mode):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(run_dir / f'{name}.csv', index=False)
    # Private diagnostics: never copied to the public aggregate report.
    np.savez_compressed(run_dir / 'oof_predictions.npz', **oof)
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'lightgbm',
                 'sentence-transformers', 'torch', 'matplotlib', 'umap-learn']:
        versions[name] = importlib.metadata.version(name)
    root = project_root()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
    data_dir = as_path(cfg['data']['local_data_dir'])
    data_hashes = {name: digest(data_dir / cfg['data'][key])
                   for name, key in [('train', 'train_file'), ('test', 'test_file'),
                                    ('complaints', 'complaints_file'), ('history', 'patient_history_file')]
                   if key in cfg['data'] and (data_dir / cfg['data'][key]).exists()}
    from src.embeddings import load_embedder
    encoder = load_embedder(cfg['embeddings']['model_name'])
    try:
        encoder_revision = encoder[0].auto_model.config._commit_hash
    except (AttributeError, KeyError, TypeError):
        encoder_revision = None
    manifest = {
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': commit, 'source_sha256': source_hashes(), 'data_sha256': data_hashes,
        'python': platform.python_version(), 'platform': platform.system(), 'packages': versions,
        'config': cfg, 'sample_n': sample_n, 'training_rows': len(train_df),
        'n_splits': n_splits, 'split_mode': split_mode,
        'encoder_revision': encoder_revision,
        'checks': checks,
        'analysis_notebook_source_sha256': hashlib.sha256(json.dumps([
            {'cell_type': c['cell_type'], 'source': c['source']} for c in json.loads(
                (root / 'notebooks/submission_notebook.ipynb').read_text())['cells']
        ], sort_keys=True).encode()).hexdigest(),
        'artifacts_sha256': {p.name: digest(p) for p in sorted(run_dir.iterdir()) if p.suffix in {'.csv', '.npz'}},
        'limitations': ['Synthetic competition data', 'Repeated complaint text may cross random folds',
                        'No demonstrated external clinical validity',
                        'Review thresholds derived descriptively on OOF predictions'],
    }
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    report = '# Reproduced analysis results\n\n'
    report += f"Run: {manifest['run_utc']}. {len(train_df):,} training rows; {n_splits}-fold {split_mode} validation.\n\n"
    report += 'These measurements describe synthetic competition data; they do not establish clinical performance.\n\n'
    report += '## Out-of-fold model comparison\n\n' + markdown_table(tables['oof_metrics']) + '\n\n'
    report += '## Fold variability\n\n' + markdown_table(tables['cv_summary']) + '\n\n'
    report += '## Diagnostics and execution status\n\n' + '\n'.join(f'- {k}: {v}' for k, v in checks.items()) + '\n\n'
    report += ('## Interpretation\n\nCompare text-only and combined models using the table above. '
               'The retained historical submission choice is a fusion fold ensemble, not a claim that fusion wins. '
               'Repeated synthetic complaint patterns can make random-fold validation optimistic. '
               'Site-held-out validation and shuffled-text controls are optional and their execution status is recorded above. '
               'Embedding semantic anchors are exploratory associations.\n\n'
               'Preprocessing was corrected during repository cleanup; these results supersede the historical notebook narrative. '
               'Source, data, model revision and artifact fingerprints are recorded in `run_manifest.json`.\n')
    (run_dir / 'results.md').write_text(report)
    return manifest
