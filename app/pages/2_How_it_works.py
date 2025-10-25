import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st

st.title("ℹ️ How it works")

st.markdown(
    """
### Supervised mode

- Deterministic **70/10/20 split** (Train / Validation / Test) with a fixed random seed, stratified by `label`.
- Train a **class-weighted Logistic Regression** model on either:
  - TF-IDF features (fast, CPU-only) or
  - MiniLM sentence embeddings (semantic, requires `torch`).
- On Validation, choose a single decision threshold **τ\*** such that:
  - Recall ≥ R (R = your slider value, e.g. 0.95),
  - and Precision is maximal under that recall constraint.
- Freeze τ\*. Apply τ\* to the untouched Test split.
- Report:
  - Test precision / recall / F1 / accuracy,
  - Workload metrics:
    - **k_R** (how deep you need to screen to recover R fraction of all relevant records),
    - **WSS@R** (work saved),
    - a conservative "stop depth" (bootstrap high quantile of k_R).
- Export a table with `score` and `pred_binary = 1(score ≥ τ*)`.

This mimics a recall-targeted screening assistant.

---

### Unsupervised mode

- No labels required.
- You provide:
  - **Positive vocabulary** (push relevance score UP),
  - **Negative vocabulary** (push relevance score DOWN).
- We compute a TF–IDF cosine-like relevance score using those vocabularies.
- (Optional) We add a semantic similarity channel using MiniLM.
- We fuse these into a single `score`, sort all abstracts,
  and mark the first K of them with `screen_flag = 1`.

`screen_flag = 1` means: "read this in the first manual batch."

This is how you bootstrap labeling without doing full manual review of everything.

---

### Practical workflow

1. Run **Unsupervised** first → download ranked file → screen the items with `screen_flag = 1`.
2. Add your decisions as `label` (1/0).
3. Run **Supervised** → now you get `pred_binary` and workload metrics under Recall≥R.
"""
)

st.info(
    "The separate ScreeningPipeline repository contains the full research pipeline "
    "(MiniLM baseline, LightHybrid fusion, transformer-heavy variants, PR curves with "
    "uncertainty bands, DeLong tests, etc.). ScreenFlow is the operational UI layer."
)
