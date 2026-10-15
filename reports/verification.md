# Verification record

Validation was performed on the repository cleanup in an isolated local working copy before applying it to the review branch. Exact source fingerprints are in `reproduced/run_manifest.json`; a null Git commit there indicates execution before committing the cleanup.

- Seven invented-data unit/contract tests passed using Python 3.11 and the recorded runtime dependencies.
- The documented notebook runner completed a fresh-kernel 1,000-row, two-fold functional run. Those sampled scores are not published as the main results.
- The full notebook was executed in a fresh kernel on all 80,000 training rows using five stratified folds. See the full-data report and manifest for results and executed options.
- Submission export validates the 20,000 test IDs in their input order, the two expected columns, non-missing values and acuity labels 1-5.
- Both notebook variants pass Python syntax checks; their analysis logic is shared, with setup/configuration differences for Kaggle.
- Single-split training/export command entry points respond correctly to `--help`; they are not claimed to reproduce the cross-validated report.

The full executed notebook and row-level outputs are kept locally under ignored `outputs/cleanup_validation_20261015/`. Public reports include aggregates only. CI is configured to run the invented-data tests; no remote CI run or fresh Kaggle-session run was claimed during cleanup.
