# Reproduced analysis results

Run: 2026-10-15T13:47:42.352016+00:00. 80,000 training rows; 5-fold stratified validation.

These measurements describe synthetic competition data; they do not establish clinical performance.

## Out-of-fold model comparison

| model | accuracy | macro_f1 | off_by_1 | undertriage_rate | severe_undertriage_rate | recall_esi_1_2 |
| --- | --- | --- | --- | --- | --- | --- |
| text_tfidf_only | 0.999363 | 0.998026 | 1.000000 | 0.000638 | 0.000000 | 1.000000 |
| text_emb_only | 0.999337 | 0.997948 | 1.000000 | 0.000663 | 0.000000 | 1.000000 |
| structured_plus_tfidf | 0.997550 | 0.994399 | 0.999900 | 0.001962 | 0.000100 | 0.996399 |
| structured_plus_emb_fusion | 0.995112 | 0.991688 | 0.999900 | 0.004237 | 0.000063 | 0.988836 |
| structured_only | 0.853038 | 0.868646 | 0.998300 | 0.079812 | 0.001275 | 0.979113 |

## Fold variability

| model | folds | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | off_by_1_mean | undertriage_mean | severe_undertriage_mean | recall_esi_1_2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text_tfidf_only | 5 | 0.999362 | 0.000204 | 0.998025 | 0.000636 | 1.000000 | 0.000638 | 0.000000 | 1.000000 |
| text_emb_only | 5 | 0.999337 | 0.000228 | 0.997946 | 0.000710 | 1.000000 | 0.000663 | 0.000000 | 1.000000 |
| structured_plus_tfidf | 5 | 0.997550 | 0.000526 | 0.994397 | 0.001156 | 0.999900 | 0.001962 | 0.000100 | 0.996399 |
| structured_plus_emb_fusion | 5 | 0.995113 | 0.000633 | 0.991685 | 0.001404 | 0.999900 | 0.004237 | 0.000063 | 0.988836 |
| structured_only | 5 | 0.853037 | 0.003205 | 0.868645 | 0.002493 | 0.998300 | 0.079812 | 0.001275 | 0.979113 |

## Diagnostics and execution status

- review_flag_rate: 0.191125
- high_acuity_share_flagged: 0.5787442773054284
- typo_prediction_stability: 0.9625
- shortened_prediction_stability: 0.9325
- site_holdout_executed: False
- shuffled_text_executed: False
- submission_exported: True

## Interpretation

Compare text-only and combined models using the table above. The retained historical submission choice is a fusion fold ensemble, not a claim that fusion wins. Repeated synthetic complaint patterns can make random-fold validation optimistic. Site-held-out validation and shuffled-text controls are optional and their execution status is recorded above. Embedding semantic anchors are exploratory associations.

Preprocessing was corrected during repository cleanup; these results supersede the historical notebook narrative. Source, data, model revision and artifact fingerprints are recorded in `run_manifest.json`.
