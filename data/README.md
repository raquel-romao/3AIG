# Input data

This project uses synthetic competition data from [Kaggle Triagegeist](https://www.kaggle.com/competitions/triagegeist). Acquire it through the competition and follow the applicable data terms. Raw data and patient-level predictions are not committed.

Place these files in `data/raw/`:

| File | Required contents |
| --- | --- |
| `train.csv` | Unique, non-missing `patient_id`, target `triage_acuity` in 1-5, intake variables |
| `test.csv` | Unique, non-missing `patient_id`, compatible intake variables |
| `chief_complaints.csv` | `patient_id`, `chief_complaint_raw` (needed when text is absent from train/test) |
| `patient_history.csv` | `patient_id` and history variables; required when `include_patient_history: true` |
| `sample_submission.csv` | Official submission schema/order reference |

The loader rejects overlapping train/test IDs and conflicting duplicate auxiliary records. Identical duplicate auxiliary records are collapsed. Left joins preserve input order; missing complaint matches become empty strings. Patient history can be disabled explicitly in the configuration.

`disposition` and `ed_los_hours` are excluded in the supplied configurations. Before adapting the project, verify availability at triage time for every added field; a denylist alone does not establish absence of leakage.

Kaggle mounts inputs under `/kaggle/input/`. The Kaggle notebook locates compatible competition files, while local paths are configured in `configs/*.yaml`. See `run_manifest.json` for the input checksums used in a reproduced run.
