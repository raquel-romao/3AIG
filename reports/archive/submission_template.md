# 99 - Triagegeist Submission Template

Use this file as a copy-ready blueprint for your final Kaggle notebook.
Keep heavy logic in `src/` and use this notebook for orchestration, evaluation, and clinical interpretation.

---

## 1) Title and clinical framing (Markdown cell)

```md
# Complaint2Concern: Triage acuity prediction with structured inputs + chief complaint embeddings

## Clinical motivation
Emergency triage decisions must prioritize patient safety. Undertriage can delay care for high-risk patients.

This notebook treats free-text chief complaint as a **second clinical signal** that may reveal urgency not fully captured by structured triage variables.
```

---

## 2) Environment and imports (Code cell)

```python
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# If you include src files in notebook folder, adjust path accordingly.
PROJECT_ROOT = Path(".")
if (PROJECT_ROOT / "src").exists():
    sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.data_io import load_train_test
from src.train_baseline import train_baseline
from src.train_fusion import train_fusion
from src.inference import export_submission_from_artifact
```

---

## 3) Config and data paths (Markdown + code)

```md
## Data and leakage guardrails
- This is a triage-time prediction task.
- Features must be available at triage.
- Post-triage fields (e.g., LOS/disposition/final diagnosis) are excluded.
```

```python
# Choose one experiment config for the notebook run.
CFG_PATH = "configs/minilm_catboost.yaml"
cfg = load_config(CFG_PATH)

# On Kaggle, this should point to your competition input dataset path.
# Example:
# cfg["data"]["kaggle_input_dir"] = "/kaggle/input/triagegeist"

train_df, test_df = load_train_test(cfg)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df.head(3)
```

---

## 4) Quick EDA checks (Code cell)

```python
target_col = cfg["data"]["target_col"]
text_col = cfg["data"]["text_col"]

print("Target distribution:")
print(train_df[target_col].value_counts(normalize=True).sort_index())

print("\nSample complaints:")
display(train_df[[text_col]].head(10))
```

---

## 5) Structured or TF-IDF baseline (Markdown + code)

```md
## Baseline model
We first train a lightweight baseline to establish reference performance before using pretrained embeddings.
```

```python
baseline_cfg = load_config("configs/base.yaml")
baseline_result = train_baseline(baseline_cfg)
baseline_result
```

---

## 6) Fusion model (structured + MiniLM embeddings) (Markdown + code)

```md
## Main model: structured + frozen complaint embeddings
Complaint text is encoded with a pretrained sentence-transformer (frozen), then fused with structured features in a tabular classifier.
```

```python
fusion_cfg = load_config("configs/minilm_catboost.yaml")
fusion_result = train_fusion(fusion_cfg)
fusion_result
```

---

## 7) Compare baseline vs fusion (Code cell)

```python
comparison = pd.DataFrame([
    {"model": "baseline", **baseline_result["metrics"]},
    {"model": "fusion", **fusion_result["metrics"]},
])
comparison
```

---

## 8) Safety-oriented analysis placeholders (Markdown cell)

```md
## Safety and interpretability extensions (next steps)

TODOs for final version:
- Undertriage and severe undertriage metrics
- Complaint clustering in embedding space
- Text-vs-structure disagreement analysis
- Review-flag logic for high-risk disagreement cases
```

---

## 9) Export Kaggle submission (Code cell)

```python
submission_path = export_submission_from_artifact(
    cfg=fusion_cfg,
    artifact_path=fusion_cfg["inference"]["artifact_path"],
    output_path=fusion_cfg["inference"]["submission_path"],
)
print("Submission saved to:", submission_path)
pd.read_csv(submission_path).head()
```

---

## 10) Final markdown section for conclusions

```md
## Conclusion
- Chief complaint embeddings improved/maintained performance relative to baseline.
- Text appears to provide complementary concern signal beyond structured features.
- Next focus: explicitly quantify undertriage and disagreement-driven safety flags.
```

---

## Notes for Kaggle packaging

- If Kaggle cannot import local modules directly, copy relevant `src/*.py` files into notebook cells.
- Keep the same function boundaries so local and Kaggle behavior stay aligned.
- Ensure final `submission.csv` columns exactly match competition requirements.
