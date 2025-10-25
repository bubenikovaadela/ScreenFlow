from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .metrics import (
    select_threshold,
    compute_binary_metrics,
    compute_workload,
    bootstrap_kR_hi,
)

def normalize_text(title: str, abstract: str) -> str:
    t = (str(title) if pd.notna(title) else "").strip()
    a = (str(abstract) if pd.notna(abstract) else "").strip()
    x = (t + " " + a).lower()
    return " ".join(x.split())


def make_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = [
        normalize_text(t, a)
        for t, a in zip(df.get("title", ""), df.get("abstract", ""))
    ]
    return df


def split_stratified_70_10_20(df: pd.DataFrame,
                              seed: int = 42) -> Tuple[pd.DataFrame,
                                                       pd.DataFrame,
                                                       pd.DataFrame]:
    """Deterministic 70/10/20 split: Train / Val / Test.
    1) Split 80/20 into trainval/test,
    2) then split trainval 87.5/12.5 into train/val.
    Both splits stratify on `label`.
    """
    y = df["label"].astype(int).to_numpy()
    idx = np.arange(len(df))

    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, y,
        test_size=0.2,
        stratify=y,
        random_state=seed
    )

    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval,
        y_trainval,
        test_size=0.125,  # 0.125 of 0.8 = 0.1 absolute
        stratify=y_trainval,
        random_state=seed
    )

    dtr = df.iloc[idx_train].reset_index(drop=True)
    dval = df.iloc[idx_val].reset_index(drop=True)
    dte = df.iloc[idx_test].reset_index(drop=True)
    return dtr, dval, dte


def fit_tfidf(train_texts: pd.Series) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    vec.fit(train_texts.tolist())
    return vec


def embed_minilm(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "MiniLM embeddings requested but 'sentence-transformers' is not installed. "
            "Either install it (plus torch), or choose 'TF-IDF (fast, CPU)'."
        ) from e
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return X


def supervised_run(df_in: pd.DataFrame,
                   target_recall: float = 0.95,
                   seed: int = 42,
                   representation: str = "tfidf",
                   bootstrap_B: int = 300
                   ) -> Tuple[Dict, pd.DataFrame]:
    """Train on TRAIN, pick τ* on VAL to satisfy Recall≥target_recall
    while maximizing Precision, then evaluate on TEST and return:
    - metrics dict
    - ranked dataframe with score + pred_binary.
    representation ∈ {"tfidf", "minilm"}.
    """

    assert "label" in df_in.columns, \
        "Supervised mode requires a 'label' column (0/1)."

    df = make_text_column(df_in)

    # Split into train/val/test
    dtr, dval, dte = split_stratified_70_10_20(df, seed=seed)
    y_train = dtr["label"].astype(int).to_numpy()
    y_val = dval["label"].astype(int).to_numpy()
    y_test = dte["label"].astype(int).to_numpy()

    # Vectorize text
    if representation == "tfidf":
        vec = fit_tfidf(dtr["text"])
        Xtr = vec.transform(dtr["text"])
        Xv  = vec.transform(dval["text"])
        Xt  = vec.transform(dte["text"])
    elif representation == "minilm":
        Xtr = embed_minilm(dtr["text"].tolist())
        Xv  = embed_minilm(dval["text"].tolist())
        Xt  = embed_minilm(dte["text"].tolist())
    else:
        raise ValueError("Unknown representation. Use 'tfidf' or 'minilm'.")

    # Train classifier
    clf = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=seed
    )
    clf.fit(Xtr, y_train)

    # Choose τ* on validation
    val_probs = clf.predict_proba(Xv)[:, 1]
    tau = select_threshold(y_val, val_probs, recall_target=target_recall)

    # Apply τ* to held-out test
    test_probs = clf.predict_proba(Xt)[:, 1]

    bin_metrics = compute_binary_metrics(y_test, test_probs, tau)
    workload = compute_workload(
        y_test, test_probs,
        recall_target=target_recall
    )
    kR_hi = bootstrap_kR_hi(
        y_test, test_probs,
        recall_target=target_recall,
        B=bootstrap_B,
        quantile=0.9,
        rng=seed
    )

    # Ranked output table
    out = dte[["id", "title", "abstract"]].copy()
    out["score"] = test_probs
    out["pred_binary"] = (test_probs >= tau).astype(int)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    metrics = {
        "tau_star": float(tau),
        "target_recall": float(target_recall),
        "representation": representation,
        **bin_metrics,
        **workload,
        "k_R_hi_90pct": int(kR_hi),
    }

    return metrics, out
