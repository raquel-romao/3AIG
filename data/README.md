# Data directory

Do not commit Kaggle competition data to git.

Recommended local layout:

```text
data/
├── README.md
├── raw/
│   ├── train.csv
│   └── test.csv
└── processed/
```

For Kaggle notebooks, data is usually mounted under:
- `/kaggle/input/triagegeist/`

Adjust paths in `configs/*.yaml` as needed.
