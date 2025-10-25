from __future__ import annotations
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def select_threshold(y_val: np.ndarray,
                     scores_val: np.ndarray,
                     recall_target: float = 0.95) -> float:
    """Pick the highest-precision threshold with Recall≥target on validation."""
    y_val = np.asarray(y_val).astype(int)
    scores_val = np.asarray(scores_val, dtype=float)

    thr_candidates = np.unique(scores_val)[::-1]
    best_thr = None
    best_prec = -1.0

    tp_total = (y_val == 1).sum()
    for thr in thr_candidates:
        y_pred = scores_val >= thr
        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = tp_total - tp
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if recall < recall_target:
            continue
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        if (precision > best_prec) or (
            np.isclose(precision, best_prec) and best_thr is not None and thr > best_thr
        ):
            best_prec = precision
            best_thr = thr

    if best_thr is None:
        best_thr = thr_candidates[-1] if len(thr_candidates) else 0.5
    return float(best_thr)


def compute_binary_metrics(y_true: np.ndarray,
                           y_score: np.ndarray,
                           thr: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except Exception:
        auc_roc = float("nan")
    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = float("nan")

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(auc_roc),
        "average_precision": float(ap),
    }


def compute_workload(y_true: np.ndarray,
                     y_score: np.ndarray,
                     recall_target: float = 0.95,
                     ks: List[int] | None = None):
    """Return k_R, WSS@R, and Recall@k for several k values."""
    if ks is None:
        ks = [50, 100, 200, 500, 1000]

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    N = len(y_true)
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    cum_rel = np.cumsum(y_sorted)
    total_rel = int(cum_rel[-1]) if len(cum_rel) else 0

    if total_rel > 0:
        recall_curve = cum_rel / total_rel
        hits = np.where(recall_curve >= recall_target)[0]
        k_R = int(hits[0] + 1) if len(hits) else N
    else:
        k_R = N

    wss = 1.0 - (k_R / N) if N > 0 else 0.0

    rec_at = {}
    for k in ks:
        kk = min(k, N)
        if total_rel > 0 and kk > 0:
            rec_at[f"recall@{k}"] = float(cum_rel[kk-1] / total_rel)
        else:
            rec_at[f"recall@{k}"] = 0.0

    return {
        "k_R": int(k_R),
        "WSS_at_recall": float(wss),
        "recall_target": float(recall_target),
        **rec_at,
    }


def bootstrap_kR_hi(y_true: np.ndarray,
                    y_score: np.ndarray,
                    recall_target: float = 0.95,
                    B: int = 300,
                    quantile: float = 0.9,
                    rng: int | None = 42) -> int:
    """Bootstrap a conservative upper bound on k_R (e.g. 90th percentile)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rs = np.random.RandomState(rng)
    N = len(y_true)
    ks = []
    for _ in range(B):
        idx = rs.randint(0, N, size=N)
        kR = compute_workload(
            y_true[idx], y_score[idx], recall_target
        )["k_R"]
        ks.append(kR)
    return int(np.quantile(ks, quantile))
