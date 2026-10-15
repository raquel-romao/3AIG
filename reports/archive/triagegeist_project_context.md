# Triagegeist project context and working plan

## 1. Why this file exists

This file summarizes the context of our discussions about a side project for the Kaggle **Triagegeist** challenge, with a focus on:
- a feasible one-month project
- innovation that is still grounded in a real clinical triage problem
- use of pretrained text models on free-text chief complaints
- a clean GitHub/reproducible project setup
- a practical local-development + Kaggle-submission workflow

The goal is to keep one shared reference for the project vision, scope, repo structure, and next steps.

---

## 2. High-level project background

We initially explored broader ideas around:
- multimodal clinical ML
- computational biology + clinical relevance
- possible LLM use in real healthcare problems
- projects that would be feasible with limited data and resources

After narrowing down the constraints, we moved to the **Kaggle Triagegeist challenge** as a good side project because it is:
- scoped
- clinically relevant
- feasible in about one month
- suitable for learning and portfolio building
- independent of needing clinicians or private datasets

We decided the project should stay:
- technically strong
- clinically grounded
- light enough to run on normal laptops
- innovative in framing and analysis, not just complexity

---

## 3. Competition framing

### Triagegeist challenge focus

The challenge is centered on emergency triage support, especially predicting **triage acuity** from:
- structured patient intake variables
- free-text chief complaint

Important interpretation:
- this is a **triage-time** problem
- features used for the main model should be restricted to information available at the time of triage
- post-triage variables can create leakage

### Leakage logic we clarified

A key example:
- `ED_length` / length of stay cannot be used to predict triage acuity
- because length of stay is only known after the visit unfolds

General rule:

> every feature used for the acuity model should be available at the time the triage decision is made.

So for triage acuity prediction, avoid using:
- ED length / length of stay
- disposition
- final diagnosis
- downstream labs or imaging
- later physician assessments
- anything happening after triage

---

## 4. What we decided not to do

We explicitly moved away from:
- overly broad clinical assistant ideas
- generic chatbots
- large end-to-end LLM systems
- full multimodal clinical pipelines that are too heavy
- solutions that require clinicians to annotate data
- anything too dependent on private or unavailable datasets
- projects that are too complex for a one-month side effort

We also decided that for this Kaggle project:
- the **main target should stay triage acuity**
- disposition and length of stay can be optional downstream extensions, but should not drive the main submission

---

## 5. Main project direction we liked

## Working concept

**Complaint2Concern**

### Core idea

Use the **free-text chief complaint** as a second clinical signal, not just another generic feature.

Hypothesis:

> the complaint text may contain acuity-relevant concern that is not fully captured by the structured intake variables.

This leads to a clinically meaningful question:

> when does the chief complaint suggest more urgency than the structured triage variables alone?

That became the main innovation idea.

---

## 6. Why this feels innovative while staying feasible

We wanted something more interesting than:
- structured baseline only
- plain TF-IDF only
- copying an existing Kaggle notebook

The innovation is not simply:
- “we used a transformer”

The innovation is:
1. use a **pretrained complaint encoder**
2. compare the signal from **text** vs **structured intake**
3. identify **mismatch cases**
4. study whether complaint language can reveal risk that the structured fields missed
5. optionally flag those cases for **review / escalation**

This is strong because it is:
- clinically interpretable
- clearly connected to triage workflow
- feasible on limited compute
- different from a plain leaderboard-oriented model

---

## 7. Practical modeling decision

We said that the most realistic setup is:

### Preferred setup

- frozen pretrained complaint embeddings
- structured triage features
- fusion model with CatBoost or LightGBM

### Why frozen embeddings

We want:
- lightweight training
- laptop-friendly execution
- simple reproducibility
- the ability to use a real pretrained text model without full fine-tuning

So instead of a heavy end-to-end transformer training setup:
- encode each complaint once
- save or cache the embeddings
- use those embeddings in a tabular downstream model

---

## 8. Text modeling options discussed

We discussed several ways to handle chief complaint text:

### a) basic encoding
- label encoding
- one-hot encoding

Not ideal if vocabulary is broad and noisy.

### b) classic text vectorization
- TF-IDF
- CountVectorizer

Good and important as a baseline.

### c) embedding-based NLP
- MiniLM
- BioClinicalBERT
- PubMedBERT-style embeddings

This is the preferred route.

### d) manual feature engineering
- complaint families or groups
- keyword extraction
- category mapping

Still useful as an analysis layer.

### Final preference

Use:
- **pretrained embeddings for the main innovation**
- **TF-IDF as a baseline**
- optional complaint grouping/clustering as an interpretability layer

---

## 9. Preferred pretrained-model strategy

We decided not to make the pretrained model the final classifier.

Instead:
- use the pretrained text model as an **embedding extractor**
- then use a downstream model such as CatBoost or LightGBM

### Best practical route
- MiniLM embeddings + CatBoost/LightGBM
- structured features + text embeddings

### More medical but heavier options
- BioClinicalBERT
- PubMedBERT embeddings

For a one-month project on normal computers, MiniLM or another lightweight sentence-transformer is probably the safest default.

---

## 10. Key analysis questions we liked

A major part of the project should answer these questions:

### Complaint signal questions
- Can we extract useful features from the free-text complaints?
- Can we group complaints into clinically meaningful families?
- Are complaint groups associated with different acuity levels?
- Which complaints are most associated with high acuity?

### Structure vs text questions
- Are the manually extracted structured variables equivalent to what is written in the complaint?
- What does the complaint add beyond the structured inputs?
- Which complaint types contain signal that is not well captured by the structured fields?
- In which cases does text correct errors made by structured-only models?

### Safety question
- Can disagreement between text and structured signals help detect potentially undertriaged patients?

These questions remain central to the notebook and writeup.

---

## 11. Main submission concept

## Final recommended concept

**Complaint2Concern: Using pretrained complaint embeddings to detect hidden acuity signal in emergency triage**

### Proposed project story

We treat free-text chief complaints not just as another predictive feature, but as a second clinical signal that may reveal acuity-relevant concern not fully captured by structured triage intake variables.

### Main components
1. structured-only triage model
2. text-only complaint model
3. fused structured + text model
4. complaint-cluster analysis
5. mismatch analysis between text and structured predictions
6. undertriage-focused evaluation
7. optional review-flag logic

---

## 12. Recommended notebook structure

The Kaggle notebook should feel like a polished clinical-data-science story, not just a pile of experiments.

## Suggested notebook sections

### 1. Clinical motivation
Explain:
- why triage matters
- why undertriage is dangerous
- why chief complaint text may add clinically useful signal

### 2. Task framing and leakage guardrails
Explain:
- this is a triage-time problem
- only triage-available variables are used
- examples of leakage to exclude

### 3. Data overview and EDA
Show:
- target class distribution
- structured-feature summaries
- complaint examples
- complaint frequencies
- missingness patterns

### 4. Text preprocessing
Keep this light and realistic:
- lowercase
- whitespace cleanup
- punctuation cleanup
- optional abbreviation normalization

### 5. Baselines
Build:
- structured-only baseline
- TF-IDF text-only baseline
- structured + TF-IDF baseline

### 6. Pretrained embedding model
Build:
- complaint embeddings with a frozen pretrained model
- fused structured + embedding model

### 7. Main comparison
Compare:
- structured only
- TF-IDF only
- structured + TF-IDF
- structured + pretrained embeddings

### 8. Complaint clustering
Use the embeddings to:
- cluster complaints
- inspect representative examples
- identify complaint families
- relate clusters to triage acuity

### 9. Text vs structure mismatch analysis
Study:
- where text predicts more concern than structured features
- whether these are higher-acuity cases
- whether these cases are enriched for model errors or undertriage

### 10. Safety-oriented evaluation
Include:
- macro-F1
- accuracy
- confusion matrix
- off-by-1 accuracy
- undertriage rate
- severe undertriage analysis
- recall for higher acuity levels

### 11. Optional review flag
Create a simple rule for:
- low-confidence cases
- strong disagreement cases
- cases where text suggests higher concern than structured inputs

### 12. Robustness and subgroup checks
If feasible:
- typo robustness
- shortened complaint robustness
- subgroup performance by age / sex / site

### 13. Limitations and deployment concept
Explain:
- synthetic nature of the data
- triage support vs clinician replacement
- how this could look in practice as a triage copilot

---

## 13. What makes the project impactful

The project should feel impactful because it focuses on:
- clinically meaningful triage support
- hidden signal in complaint language
- undertriage reduction
- review-worthy mismatch cases
- interpretability and workflow relevance

The strongest message is:

> the free-text complaint can function as a second clinical lens, helping identify concern that structured intake fields may under-represent.

---

## 14. Evaluation philosophy

We discussed that plain accuracy is not enough.

### Metrics to include
- multiclass accuracy
- macro-F1
- confusion matrix
- off-by-1 accuracy

### Safety-oriented metrics
- undertriage rate
- severe undertriage rate
- recall for more urgent cases
- calibration if possible

### Important emphasis
Not all triage mistakes are clinically equivalent.

A model that slightly improves general accuracy but reduces unsafe undertriage is more interesting than one that only optimizes leaderboard performance.

---

## 15. Optional extension tasks we discussed

We discussed the possibility of later predicting:
- triage acuity
- disposition
- length of stay

### Final decision
For the Kaggle project:
- keep **triage acuity** as the main target
- consider disposition and LOS as future extensions only

Reason:
- once disposition and especially LOS are included, the project becomes more of an early ED trajectory model rather than a focused triage assistant

Also:
- those downstream outcomes depend more on later workflow, tests, treatment, and operations
- they complicate the story and introduce more leakage risk

---

## 16. Recommended development workflow

### Main decision
**Develop locally, submit on Kaggle**

This was our recommended workflow.

### Why
Local development is better for:
- git
- branching
- debugging
- testing multiple models
- running preprocessing and embedding scripts cleanly

Kaggle is necessary for:
- the final public notebook
- end-to-end reproducibility in the competition environment
- submission artifacts

### Practical split

#### Work locally for:
- EDA
- model development
- baselines
- embedding experiments
- clustering
- figure drafting
- writeup drafting

#### Finalize on Kaggle for:
- clean public notebook
- final reproducible run
- submission.csv creation
- polished writeup presentation

---

## 17. Recommended repository structure

```text
triagegeist-complaint2concern/
├── README.md
├── .gitignore
├── requirements.txt
├── environment.yml
├── LICENSE
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda_local.ipynb
│   ├── 02_baselines_local.ipynb
│   ├── 03_embeddings_local.ipynb
│   └── 99_submission_template.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_io.py
│   ├── preprocess.py
│   ├── features_structured.py
│   ├── features_text.py
│   ├── embeddings.py
│   ├── train_baseline.py
│   ├── train_fusion.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── configs/
│   ├── base.yaml
│   ├── tfidf_lgbm.yaml
│   └── minilm_catboost.yaml
├── reports/
│   ├── figures/
│   └── writeup_draft.md
├── models/
│   └── README.md
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── run_local.sh
│   ├── export_submission.py
│   └── make_kaggle_dataset.sh
└── kaggle/
    ├── notebook_outline.md
    └── submission_checklist.md
```

---

## 18. What each part of the repo is for

### `src/`
Main project logic. Keep notebooks thin and put real code here.

### `notebooks/`
Exploration and final Kaggle notebook template.

### `configs/`
Experiment configuration files.

### `reports/`
Draft writeups and figures.

### `data/`
No competition data committed here. Only a README explaining how to obtain it.

### `outputs/`
Temporary outputs and local artifacts.

### `models/`
Model notes; do not commit large weights by default.

### `scripts/`
Convenience scripts for running locally and exporting predictions.

### `kaggle/`
Notebook outline and final submission notes.

---

## 19. Minimal starter files we wanted

These files should exist early:

- `README.md`
- `.gitignore`
- `requirements.txt`
- `environment.yml`
- `data/README.md`
- `reports/writeup_draft.md`
- `src/__init__.py`
- `src/config.py`
- `src/data_io.py`
- `src/preprocess.py`
- `src/features_structured.py`
- `src/features_text.py`
- `src/embeddings.py`
- `src/train_baseline.py`
- `src/train_fusion.py`
- `src/evaluate.py`
- `src/inference.py`
- `src/utils.py`
- `configs/base.yaml`
- `configs/tfidf_lgbm.yaml`
- `configs/minilm_catboost.yaml`
- `scripts/run_local.sh`
- `scripts/export_submission.py`
- `kaggle/notebook_outline.md`
- `kaggle/submission_checklist.md`

---

## 20. Coding principles for the project

### Keep notebooks thin
The final Kaggle notebook should mostly:
- load data
- call functions from `src/`
- show plots/tables
- train/evaluate
- export `submission.csv`

### Put logic in modules
Important logic should live in:
- preprocessing modules
- feature modules
- training modules
- evaluation modules
- inference/export modules

### Design for both local and Kaggle paths
Path handling should support:
- local project paths
- Kaggle input paths

Avoid hard-coding only one environment.

### Use simple defaults
Preferred tools:
- pandas
- numpy
- scikit-learn
- CatBoost or LightGBM
- sentence-transformers
- YAML configs

---

## 21. Practical modeling plan

## Phase 1: establish baselines
- structured-only baseline
- text-only TF-IDF baseline
- fused TF-IDF baseline

## Phase 2: add pretrained embeddings
- generate complaint embeddings
- train fused structured + embedding model
- compare against TF-IDF

## Phase 3: analysis layer
- cluster complaints in embedding space
- identify important complaint families
- analyze text-vs-structure mismatch
- analyze undertriage-sensitive error patterns

## Phase 4: polish
- add figures
- build review-flag logic
- write clear conclusions
- prepare final Kaggle notebook and writeup

---

## 22. One-month suggested timeline

### Week 1
- set up repo
- inspect data
- define leakage rules
- build structured baseline
- build TF-IDF baseline

### Week 2
- implement pretrained embeddings
- build fused model
- compare models
- draft core plots

### Week 3
- complaint clustering
- mismatch analysis
- undertriage analysis
- optional review flag

### Week 4
- robustness checks
- subgroup checks
- final notebook cleanup
- final writeup
- Kaggle submission

---

## 23. Project title ideas connected to the final concept

Main favorite:
- **Complaint2Concern**

Alternatives:
- **Text Gaps in Triage**
- **What the Complaint Adds**
- **Second Signal Triage**
- **Chief Concern**

For a team name, earlier ideas included:
- **TriAIge**
- **Triagent**
- **AcuityAI**

---

## 24. Final anchor sentence for the project

This is the main sentence we wanted the project to revolve around:

> We treat the free-text chief complaint not just as another feature, but as a second clinical signal that can reveal acuity-relevant concern not fully captured by structured triage intake variables.

---

## 25. Prompt to request full project scaffolding

Use the following prompt when asking for help setting up the project code and local folder organization:

```text
I am starting a Kaggle project for the Triagegeist emergency-triage challenge and I want you to help me set up the project structure on my computer in a clean, reproducible way.

Project concept:
- Main target: triage acuity prediction
- Main idea: use structured intake variables plus free-text chief complaint
- Innovation: use a pretrained complaint embedding model as a second clinical signal, then compare what the text suggests versus what the structured variables suggest
- Goal: detect when the chief complaint implies more concern than the structured inputs alone
- Practical constraint: everything should be light enough to run on a normal laptop
- Preferred modeling approach: frozen pretrained text embeddings + CatBoost/LightGBM fusion model
- We will develop locally but make the final notebook run end-to-end on Kaggle

Please help me scaffold the project with the following requirements:

1. Create a clean folder structure for a Python ML project with:
- src/
- notebooks/
- configs/
- reports/
- data/
- outputs/
- scripts/
- kaggle/

2. Create starter files including:
- README.md
- .gitignore
- requirements.txt
- environment.yml
- data/README.md
- reports/writeup_draft.md
- src/__init__.py
- src/config.py
- src/data_io.py
- src/preprocess.py
- src/features_structured.py
- src/features_text.py
- src/embeddings.py
- src/train_baseline.py
- src/train_fusion.py
- src/evaluate.py
- src/inference.py
- src/utils.py
- configs/base.yaml
- configs/tfidf_lgbm.yaml
- configs/minilm_catboost.yaml
- scripts/run_local.sh
- scripts/export_submission.py
- kaggle/notebook_outline.md
- kaggle/submission_checklist.md

3. In the Python files, add minimal but working starter code and docstrings for:
- reading train/test data from either local paths or Kaggle paths
- preprocessing structured features
- preprocessing complaint text
- generating TF-IDF features
- generating frozen pretrained embeddings for complaints
- training a structured baseline model
- training a fusion model with structured features + text embeddings
- evaluating multiclass triage acuity performance
- exporting Kaggle submission files

4. Keep the notebooks thin. Put most logic in src/.

5. Use practical defaults that run locally, with a preference for:
- pandas
- scikit-learn
- CatBoost or LightGBM
- sentence-transformers
- YAML configs

6. Add comments showing where I should later plug in:
- undertriage metrics
- complaint clustering
- text-vs-structure disagreement analysis
- review-flag logic

7. Make the setup friendly for GitHub and local development, but ensure the final pipeline can also be used from a Kaggle notebook.

Please output:
- the full folder tree
- the content of each starter file
- and short instructions on how to initialize the project locally.
```

---

## 26. Final practical recommendation

### Development
Do most work locally.

### Submission
Prepare one clean Kaggle notebook that:
- runs end-to-end
- reproduces the core pipeline
- generates the final evaluation outputs
- writes `submission.csv`
- supports the public writeup

### Core message
Keep the project focused and coherent:
- one main target
- one strong clinical idea
- one lightweight pretrained text component
- one meaningful innovation layer based on text-vs-structure mismatch

