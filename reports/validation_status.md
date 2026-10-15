# Validation status

The fresh-kernel local run is recorded in `reproduced/run_manifest.json`. The public aggregate tables and figures come from that run. Corrected abbreviation expansion means historical outputs are not assumed numerically identical.

| Analysis | Status in the default workflow |
| --- | --- |
| Five-model comparison with five stratified folds | Executed locally; see manifest and exact tables |
| Training-fold-only preprocessing and classifiers | Implemented and covered by contract tests |
| Ordinal error metrics, OOF confidence, subgroup summaries | Executed locally |
| Feature/term summaries, semantic anchors, K-means clustering | Executed locally; exploratory interpretation |
| Typo/truncation prediction stability | Executed on up to 2,000 reference-fold rows |
| Review flags | Descriptive OOF quantile thresholds; not independently validated |
| UMAP | Optional; not required by default reproduction |
| Site-held-out validation | Optional code retained, disabled in the default run |
| Shuffled-text negative control | Optional code retained, disabled in the default run |
| Local SHAP | Removed from the verified workflow; no verified result |
| Unseen-complaint-group evaluation | Future work |
| External/clinical validation or deployment | Not performed |
| Clean Kaggle-session execution | Not verified in this cleanup |
| Competition upload and leaderboard score | Not verified |

`reproduced/` contains aggregate summaries only. Detailed row-level diagnostics, submissions, the executed notebook and full run outputs stay in ignored `outputs/`. CI runs invented-data tests; it does not access competition data or retrain the full experiment.
