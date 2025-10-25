import sys, pathlib
# make sure the `app` dir is on sys.path so we can import utils/*
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))

import streamlit as st

st.set_page_config(
    page_title="ScreenFlow — Abstract Screening UI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 ScreenFlow — Abstract Screening UI")

st.markdown(
    """
ScreenFlow helps you prioritize and screen abstracts with high recall.

- **Supervised (with labels)**  
  You already have relevance labels (0/1).  
  We train on Train, pick a frozen decision threshold τ* on Validation to
  guarantee high recall (e.g. ≥0.95), then evaluate on an untouched Test split.
  You get per-record `pred_binary` and workload metrics.

- **Unsupervised (no labels)**  
  You don't have any labels yet.  
  You define positive and negative vocabulary, we score + rank all abstracts,
  and we flag the top-K as your "read these first" batch via `screen_flag = 1`.

Use the **Screening** page in the sidebar to upload your CSV and run.
"""
)

st.info(
    "Supervised mode requires a `label` column (1 = relevant, 0 = not relevant). "
    "Unsupervised mode only needs `id,title,abstract` plus your keyword priors."
)

st.caption(
    "For full reproducible benchmarking and modeling (MiniLM baseline, LightHybrid, "
    "transformer-heavy models, WSS@95, Recall@k, etc.), please refer to the "
    "separate ScreeningPipeline repository described in the manuscript."
)
