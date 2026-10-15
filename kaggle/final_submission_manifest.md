# Kaggle execution and submission

The locally verified workflow and reports are described in the root README. Kaggle-session execution and any competition upload must be verified separately; a local CSV does not prove a submission occurred.

1. Install the runtime and notebook dependencies from the two requirements files, or provide an equivalent Kaggle environment.
2. Attach the competition data and project code (`src/`, `configs/` and requirements). Use the current source, not an old generated ZIP.
3. Open `notebooks/submission_notebook_kaggle.ipynb`. The bootstrap locates code and data under `/kaggle/input/`.
4. Set `TRIAGE_RUN_DIR=/kaggle/working/reproduced` and `TRIAGE_SUBMISSION=1` in the session environment before execution. Allow network access only if weights/packages must be obtained.
5. Run every cell in a fresh session. Check the run manifest, exact aggregate results, submission ID order and labels.
6. The optional fold ensemble writes `submission_notebook_cv.csv` in the run directory. Record its checksum and the Kaggle notebook/version URL if submitting it.

The historical selected model is structured-plus-MiniLM fusion. Text-only models must still be compared; retaining the historical selection does not establish it as best. Source changes and preprocessing corrections can change predictions.

Raw competition data, per-row diagnostics, model artifacts, credentials and generated bundles remain outside the public repository. Aggregate reports and code may be shared subject to the contributors' and competition's applicable terms.
