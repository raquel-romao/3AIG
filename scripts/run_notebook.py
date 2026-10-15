"""Execute the canonical notebook in a fresh kernel using this Python environment."""
import argparse
from tempfile import TemporaryDirectory
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/reproduced')
    parser.add_argument('--sample', type=int)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--projections', action='store_true')
    args = parser.parse_args()
    if args.folds < 2 or (args.sample is not None and args.sample < 50):
        parser.error('Use at least two folds and at least 50 sampled rows')
    import nbformat
    from nbclient import NotebookClient
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(TRIAGE_RUN_DIR=str(output), TRIAGE_SPLITS=str(args.folds),
                      TRIAGE_SAMPLE_N=str(args.sample or 0), TRIAGE_SUBMISSION=str(int(args.submission)),
                      TRIAGE_PROJECTIONS=str(int(args.projections)), TOKENIZERS_PARALLELISM='false')
    nb = nbformat.read(root/'notebooks/submission_notebook.ipynb', as_version=4)
    def progress(cell, cell_index):
        if cell.cell_type == 'code':
            print(f'Cell {cell_index + 1}/{len(nb.cells)}: {cell.source.splitlines()[0][:80]}', flush=True)
    with TemporaryDirectory(prefix='triagegeist-kernel-') as d:
        kernel = Path(d)/'kernels'/'triagegeist-local'
        kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Triagegeist local', 'language':'python'}))
        os.environ['JUPYTER_PATH'] = d + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        client = NotebookClient(nb, timeout=3600, kernel_name='triagegeist-local',
                                resources={'metadata':{'path':str(root)}}, on_cell_start=progress)
        try:
            client.execute()
        finally:
            nbformat.write(nb, output/'executed_analysis.ipynb')
    print(f'Completed. Results and manifest: {output}')


if __name__ == '__main__':
    main()
