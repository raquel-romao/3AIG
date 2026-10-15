# Repository cleanup

This revision preserves the research questions and five model families while making the reported workflow executable and its scope explicit.

## Changes

- One canonical analysis notebook, with the Kaggle variant using the same analysis cells.
- Removed interactive scratch cells and an unverified SHAP block; fixed percentile diagnostics.
- Shared missing-text handling and case-insensitive abbreviation expansion across analysis, training and inference.
- Explicit checks for IDs, conflicting auxiliary-table duplicates, required inputs and prediction schema/order.
- Shared ordinal acuity metrics on actual labels rather than encoded class indices.
- Frozen encoder reused in-process; unique complaints encoded once for CV without label fitting.
- Invalid model/backend names fail explicitly; CatBoost logs are disabled for new runs.
- Working module-based export command and paths relative to the project root.
- Tested dependency versions, a fresh-kernel runner, exact result exports and source/data/model fingerprints.
- Seven invented-data contract tests, a train/predict smoke test within those tests, and a CI test workflow.
- Historical planning/templates/write-ups moved to `reports/archive/`.
- Generated log files removed from version control while keeping local copies.

## Interpretation changes

Abbreviation expansion previously matched uppercase keys against lowercased strings and did not operate as intended. Correcting preprocessing can change scores and predictions. Reproduced outputs supersede historical copied values, which remain available in the archive/Git history.

Full-data results use random stratified folds. Site-held-out and shuffled-text cells remain optional and disabled; they are not claimed as completed. UMAP is optional. Embedding anchors and modality importance shares are exploratory, and no deployment or external clinical benefit is established.

The historical submission choice (structured plus MiniLM fusion) is preserved for the optional fold ensemble. This does not imply it outperforms text-only models. New generated predictions are distinct from any historical competition upload.

## Review before merging

Review the exact reproduced metrics, contribution attribution and licence choice. The cleanup does not claim a verified Kaggle-session run, leaderboard result or clinical validation. The existing raw data, historical prediction files and generated ZIP are retained locally and ignored. The separate Git repository in the parent folder is untouched.
