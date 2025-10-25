import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st

st.title("🧪 Model guide")

st.markdown(
    """
### TF-IDF (fast, CPU)
- No heavy downloads.
- Very fast to run locally.
- Works in both Unsupervised mode (keyword-driven relevance) and Supervised mode (Logistic Regression on TF-IDF features).

### MiniLM (semantic)
- Captures semantic similarity, not just exact keyword matches.
- Helps when authors describe the same concept with different wording.
- Requires `sentence-transformers` and `torch`, so it's heavier.
- Available in both modes:
  - In **Unsupervised**, if you enable *Use semantic MiniLM similarity*.
  - In **Supervised**, if you choose *MiniLM (semantic)* as the text representation.

---

### Binary outputs

- **pred_binary (Supervised)**  
  1 = model says "include / relevant" using the frozen τ* threshold  
  chosen on Validation to satisfy Recall ≥ R.  
  0 = model says "exclude / not relevant".

- **screen_flag (Unsupervised)**  
  1 = this abstract is in the top-K ranked items and should be read in the first manual batch.  
  0 = lower priority.

Important: `screen_flag = 1` does **not** mean "definitely relevant".  
It just tells you where to start reading.
"""
)
