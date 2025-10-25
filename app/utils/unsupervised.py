from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

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

def tfidf_query_similarity(texts: List[str],
                           pos_terms: List[str],
                           neg_terms: List[str]) -> np.ndarray:
    """TF-IDF cosine-like similarity with positive and negative priors.
    Positive terms increase the score; negative terms decrease it.
    """
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(texts)
    vocab = vec.vocabulary_

    q = np.zeros((len(vocab),), dtype=float)

    # positive priors -> push score up
    for w in pos_terms:
        w = w.strip().lower()
        if w in vocab:
            q[vocab[w]] += 1.0

    # negative priors -> push score down
    for w in neg_terms:
        w = w.strip().lower()
        if w in vocab:
            q[vocab[w]] -= 1.0

    # normalize query vector
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    # cosine similarity of each abstract vs q
    row_norm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    dots = X.dot(q)
    sims = np.zeros_like(dots, dtype=float)
    nz = row_norm > 0
    sims[nz] = dots[nz] / row_norm[nz]
    return sims

def semantic_similarity(texts: List[str],
                        query_text: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # semantic disabled -> all zeros
        return np.zeros((len(texts),), dtype=float)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    q = model.encode(
        [query_text],
        show_progress_bar=False,
        normalize_embeddings=True
    )[0]
    return X @ q

def minmax01(x: np.ndarray) -> np.ndarray:
    lo = np.min(x)
    hi = np.max(x)
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.zeros_like(x)

def unsupervised_run(df_in: pd.DataFrame,
                     pos_terms: List[str],
                     neg_terms: List[str],
                     top_k: int = 200,
                     weight_semantic: float = 0.5,
                     use_semantic: bool = True
                     ) -> Tuple[pd.DataFrame, int]:
    """Rank records using positive/negative vocabulary (TF-IDF similarity),
    optionally fuse semantic MiniLM similarity,
    then flag the first K items with screen_flag=1.
    """
    df = make_text_column(df_in)
    texts = df["text"].tolist()

    tfidf_sim = tfidf_query_similarity(texts, pos_terms, neg_terms)
    tfidf_sim_n = minmax01(tfidf_sim)

    if use_semantic:
        sem_sim = semantic_similarity(texts, " ".join(pos_terms))
        sem_sim_n = minmax01(sem_sim)
    else:
        sem_sim_n = np.zeros_like(tfidf_sim_n)

    w_sem = max(0.0, min(1.0, float(weight_semantic)))
    w_tfidf = 1.0 - w_sem
    fused = w_sem * sem_sim_n + w_tfidf * tfidf_sim_n

    out = df[["id", "title", "abstract"]].copy()
    out["tfidf_prior"] = tfidf_sim_n
    out["semantic_prior"] = sem_sim_n
    out["score"] = fused

    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    cutoff = min(int(top_k), len(out))
    out["screen_flag"] = 0
    if cutoff > 0:
        out.loc[:cutoff-1, "screen_flag"] = 1

    return out, cutoff
