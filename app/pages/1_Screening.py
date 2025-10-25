import sys, pathlib
# ensure `app/` is on path when Streamlit runs this page
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import io
import pandas as pd
import streamlit as st

from utils.supervised import supervised_run
from utils.unsupervised import unsupervised_run

st.title("📄 Screening")

with st.sidebar:
    st.header("Run settings")

    mode = st.radio("Mode", ["Supervised (with labels)", "Unsupervised (no labels)"])

    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    if mode.startswith("Supervised"):
        target_recall = st.slider(
            "Target recall (R)",
            0.50, 0.99, 0.95, 0.01,
            help="We will choose τ* so that Recall ≥ R on validation while maximizing Precision."
        )
        representation = st.selectbox(
            "Text representation",
            ["TF-IDF (fast, CPU)", "MiniLM (semantic, downloads model)"],
            help="TF-IDF uses bag-of-words features; MiniLM uses sentence embeddings (requires torch)."
        )
        bootstrap_B = st.slider(
            "Bootstrap B (for conservative stop depth)",
            50, 1000, 300, 50,
            help="We resample the test set B times to estimate a high-quantile upper bound on how deep you need to read."
        )
    else:
        pos_terms_raw = st.text_input(
            "Positive vocabulary (comma-separated)",
            "glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy",
            help="Terms you DO care about. Matching these pushes the score UP."
        )
        neg_terms_raw = st.text_input(
            "Negative vocabulary (comma-separated)",
            "cardiac,heart,coronary,ophthalmology,renal,breast",
            help="Terms you do NOT care about. Matching these pushes the score DOWN."
        )

        use_semantic = st.checkbox(
            "Use semantic MiniLM similarity (downloads model)",
            value=False,
            help="Adds a semantic similarity channel using MiniLM embeddings. Requires sentence-transformers + torch."
        )
        w_sem = st.slider(
            "Weight of semantic similarity",
            0.0, 1.0, 0.5, 0.05,
            disabled=not use_semantic,
            help="0 = TF-IDF only. 1 = semantic only. Values in between = fusion."
        )

        top_k = st.slider(
            "Top-K to flag for first screening",
            50, 1000, 200, 50,
            help="We'll set screen_flag = 1 for the top-K ranked abstracts so you know what to read first."
        )

uploaded = st.file_uploader(
    "Upload your CSV (id, title, abstract[, label])",
    type=["csv"]
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    required_cols = {"id", "title", "abstract"}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"CSV must contain columns: {sorted(list(required_cols))}")
        st.stop()

    if st.button("Run screening"):
        if mode.startswith("Supervised"):
            if "label" not in df.columns:
                st.error("Supervised mode requires a 'label' column (0/1).")
                st.stop()

            rep = "tfidf" if representation.startswith("TF-IDF") else "minilm"

            with st.spinner("Training model, selecting τ*, evaluating on held-out test..."):
                metrics, ranked = supervised_run(
                    df,
                    target_recall=float(target_recall),
                    seed=int(random_state),
                    representation=rep,
                    bootstrap_B=int(bootstrap_B)
                )

            st.success(
                "Frozen τ* = {:.4f} (chosen on validation with Recall≥{:.2f}).\n"
                "TEST metrics — precision: {:.3f}, recall: {:.3f}, F1: {:.3f}, accuracy: {:.3f}.\n"
                "Workload — k_R: {}, WSS@R: {:.3f}; Conservative stop depth (90th percentile bootstrap): top {} records."
                .format(
                    metrics["tau_star"],
                    metrics["target_recall"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["accuracy"],
                    metrics["k_R"],
                    metrics["WSS_at_recall"],
                    metrics["k_R_hi_90pct"]
                )
            )

            st.caption("Preview of top-ranked records from the held-out Test split:")
            st.dataframe(ranked.head(50))

            st.download_button(
                "Download ranked (with pred_binary)",
                ranked.to_csv(index=False).encode("utf-8"),
                "ranked_supervised.csv",
                "text/csv"
            )

            st.download_button(
                "Download metrics (JSON)",
                io.BytesIO(bytes(str(metrics), "utf-8")),
                "metrics_test.json",
                "application/json"
            )

        else:
            pos_terms = [w.strip() for w in pos_terms_raw.split(",") if w.strip()]
            neg_terms = [w.strip() for w in neg_terms_raw.split(",") if w.strip()]

            with st.spinner("Scoring and ranking with your positive/negative vocabulary..."):
                ranked, cutoff = unsupervised_run(
                    df,
                    pos_terms=pos_terms,
                    neg_terms=neg_terms,
                    top_k=int(top_k),
                    weight_semantic=float(w_sem),
                    use_semantic=bool(use_semantic)
                )

            st.success(
                f"Flagged top {cutoff} records as initial manual screening batch (screen_flag = 1)."
            )

            st.caption("Preview of top-ranked records:")
            st.dataframe(ranked.head(50))

            st.download_button(
                "Download ranked (with screen_flag)",
                ranked.to_csv(index=False).encode("utf-8"),
                "ranked_unsupervised.csv",
                "text/csv"
            )
else:
    st.info("Upload a CSV to begin. See README for the required columns and examples.")
