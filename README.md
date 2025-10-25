# ScreenFlow App 🧠  
Interactive Abstract Screening UI

ScreenFlow is a lightweight Streamlit app for fast, reproducible abstract triage in systematic reviews. It supports two modes:

---

## 1. Supervised (with labels)

Use this mode if you already manually labeled at least some abstracts.

### What it does
1. Builds a deterministic, stratified **70/10/20 split** into Train / Validation / Test using a fixed random seed (stratified by `label`).
2. Trains a **class-weighted Logistic Regression** classifier on either:
   - **TF‑IDF features** (fast, CPU friendly), or
   - **MiniLM sentence embeddings** (requires `sentence-transformers` and `torch`).
3. On the Validation split, it finds a single **frozen decision threshold τ\*** such that:
   - **Recall ≥ R** (R = your “Target recall (R)” slider, e.g. 0.95), and  
   - Precision is as high as possible under that recall constraint.
4. That τ\* is then applied unchanged to the fully held-out Test split.
5. It reports:
   - Per-record **score** (predicted relevance probability),
   - Per-record **binary decision** `pred_binary` = 1(score ≥ τ\*),
   - Test performance metrics (precision, recall, F1, accuracy),
   - Workload metrics:
     - **k_R** = smallest number of top-ranked abstracts you would need to read to recover at least R fraction of all relevant studies,
     - **WSS@R** = work saved over screening everything,
     - and a conservative upper bound on how deep you should read.

### Why this matters
This mode simulates a “high-recall screening assistant”:  
set a required recall (e.g. 0.95), and the app will tell you which items it would auto-include (`pred_binary = 1`) and how much human effort you save.

---

## 2. Unsupervised (no labels)

Use this mode if you DO NOT have labels yet.

### What it does
1. You provide two vocabularies:
   - **Positive vocabulary**: terms you *do* care about (should push an abstract up the list),
   - **Negative vocabulary**: terms you want to *down-rank* (off-topic terms).
2. The app builds a TF‑IDF representation and computes a cosine-like relevance score where:
   - matches to positive terms increase the score,
   - matches to negative terms decrease the score.
3. (Optional) You can enable **semantic MiniLM similarity** for an additional semantic score based on sentence embeddings.  
   - This requires `sentence-transformers` and `torch`.  
   - If not installed, leave it off — TF‑IDF alone still works.
4. The TF‑IDF score and (optionally) the MiniLM semantic score are fused into a single **score**.
5. The app sorts all abstracts by this score and then sets:
   - **`screen_flag = 1`** for the top **K** abstracts,  
   - **`screen_flag = 0`** for the rest.

K is chosen by you via the slider **“Top‑K to flag for first screening”**.

### How to interpret `screen_flag`
`screen_flag = 1` means  
> “Read this in the first manual batch.”

This is NOT saying “this is definitely relevant,” it’s saying “start here.”  
This is exactly what you need at the very beginning of a review, before you have any labels.

---

## Typical workflow

1. **Start with Unsupervised mode**  
   - Upload CSV with `id,title,abstract`.  
   - Enter positive and negative vocabularies.  
   - Set Top‑K (e.g. 200).  
   - Download the ranked CSV. The first K rows (`screen_flag = 1`) are your prioritized first-pass review set.

2. **Manually screen those K abstracts**  
   - Assign `label = 1` (relevant) / `0` (not relevant).

3. **Then use Supervised mode**  
   - Upload CSV with `id,title,abstract,label`.  
   - The app will train a model, pick τ\* based on Recall ≥ R,  
     and produce `pred_binary` for each abstract.  
   - You also get performance metrics on a held-out test split and workload estimates (k_R, WSS@R).

This matches how high-recall screening tools are actually used:  
first prioritization, then supervised high-recall classification.

---

## Input data format

### Unsupervised mode
Your CSV must include:
- `id`
- `title`
- `abstract`

Example:
```csv
id,title,abstract
PMID123,"Glymphatic transport in humans","We assess CSF clearance using ASL and diffusion MRI..."
PMID124,"Intraoperative cochlear nerve monitoring","We evaluate ABR and CNAP monitoring during VS surgery..."
```

### Supervised mode
Same as above, plus a binary column:
- `label` (1 = relevant, 0 = not relevant)

Example:
```csv
id,title,abstract,label
PMID123,"Glymphatic transport in humans","We assess CSF clearance using ASL and diffusion MRI...",1
PMID124,"Intraoperative cochlear nerve monitoring","We evaluate ABR and CNAP monitoring during VS surgery...",0
```

**Note:** Supervised mode uses a stratified 70/10/20 split.  
If you only have a tiny dataset (like <30 items total, or only one class = all zeros or all ones), supervised mode may fail because it cannot create stratified train/val/test splits. Unsupervised mode has no such restriction.

---

## Running the app locally

You’ll need Python 3.10+.

```bash
# 1) Create and activate a fresh virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# 2) Install core dependencies (CPU-friendly)
pip install -r requirements.txt

# 3) Launch the app
streamlit run app/0_🧠_ScreenFlow.py
```

This will open ScreenFlow in your browser (usually http://localhost:8501).

### Optional: semantic embeddings / MiniLM
If you want to:
- enable “Use semantic MiniLM similarity” in Unsupervised mode, or
- choose “MiniLM (semantic)” representation in Supervised mode,

then install the embedding stack:

```bash
pip install sentence-transformers transformers torch
```

If `torch` installation is painful (e.g. on CPU-only Windows), just skip this step and keep the semantic option off. The TF‑IDF path still works end-to-end.

---

## Outputs you get

### Unsupervised mode
Downloadable CSV (`ranked_unsupervised.csv`) with:
- `id`, `title`, `abstract`
- `score` (relevance score after fusion)
- `tfidf_prior` (normalized TF‑IDF prior from vocab terms)
- `semantic_prior` (normalized semantic similarity, or 0 if disabled)
- `screen_flag`
  - 1 = in the “Top‑K to read first”
  - 0 = lower priority

This is your first-pass manual screening batch.

### Supervised mode
Downloadable CSV (`ranked_supervised.csv`) with:
- `id`, `title`, `abstract`
- `score` (predicted probability of relevance on the held-out Test split)
- `pred_binary`
  - 1 = model says “include / relevant” using τ\*
  - 0 = model says “exclude / not relevant”

You also get a downloadable JSON (`metrics_test.json`) containing:
- Frozen threshold `τ*`
- Precision, recall, F1, accuracy on the Test split
- Workload metrics:
  - **k_R**
  - **WSS@R**
  - conservative “stop depth” (90th percentile bootstrap upper bound on k_R)

---

## Interpreting workload metrics

- **Recall ≥ R** means: we try to recover at least R fraction (e.g. 95%) of all true relevant studies.
- **k_R** is the number of highest-ranked abstracts you must check to hit that recall.
- **WSS@R** (“work saved over sampling at Recall R”) is 1 - k_R / N, i.e. roughly how much manual work you avoid compared to brute-force reading everything.
- The conservative stop depth (“top X records”) is a bootstrap-based high quantile of k_R:  
  *“Read at least this deep to be safe.”*

This is directly aligned with high-recall systematic review practice.

---

## License

MIT License (see `LICENSE`).

---

## Relationship to the main pipeline repo

This UI is meant for interactive triage and high-recall screening.  
For full reproducible benchmarking and modeling (MiniLM baseline, LightHybrid fusion, transformer-heavy variants, DeLong test, PR curves with CIs, etc.), use the separate `ScreeningPipeline` repository described in the manuscript.

That repository is designed for full scientific reproducibility;  
this one is designed for actual day-to-day screening.

---
