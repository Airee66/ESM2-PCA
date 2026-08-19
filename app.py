#!/usr/bin/env python3

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parent
EXTRACTOR = REPO_ROOT / "scripts" / "extract_esm2_mean.py"


@st.cache_resource
def load_model_description():
    return "ESM-2 8M (esm2_t6_8M_UR50D)"


def extract_fasta_embeddings(fasta_path: Path, out_dir: Path):
    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(EXTRACTOR),
        str(fasta_path),
        str(out_dir),
        "--nogpu",
    ]
    subprocess.run(command, check=True)

    emb_path = out_dir / "embeddings.npy"
    labels_path = out_dir / "labels.csv"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings were not produced at {emb_path}")

    X = np.load(emb_path)
    labels = np.genfromtxt(labels_path, dtype=str, delimiter=",", comments=None)
    if labels.size == 1 and labels.shape == ():
        labels = np.array([str(labels)])
    return X, labels


def compute_pca(X: np.ndarray, n_components: int = 2):
    if X.shape[0] < 2:
        raise ValueError("At least 2 sequences are required to compute PCA.")
    pca = PCA(n_components=n_components, random_state=0)
    projection = pca.fit_transform(X)
    return projection, pca.explained_variance_ratio_


def draw_pca_plot(projection: np.ndarray, labels: Optional[np.ndarray], title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = projection[:, 0]
    ys = projection[:, 1]

    if labels is not None:
        for x, y, label in zip(xs, ys, labels):
            ax.scatter(x, y, s=60, alpha=0.8)
            ax.annotate(str(label), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    else:
        ax.scatter(xs, ys, s=60, alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="ESM-2 FASTA PCA", page_icon="🧬", layout="wide")
st.title("ESM-2 FASTA PCA Explorer")
st.caption("Upload a FASTA file to get ESM-2 embeddings and a PCA projection for downstream ML or visualization.")

with st.sidebar:
    st.subheader("Settings")
    model_name = st.selectbox("Model", ["esm2_t6_8M_UR50D"], index=0)
    n_components = st.slider("PCA components", min_value=2, max_value=5, value=2)
    st.caption(f"Using {model_name} with mean-pooled sequence embeddings.")

uploaded_file = st.file_uploader("Drop a FASTA file here", type=["fasta", "fa", "faa", "txt"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        fasta_path = tmpdir_path / uploaded_file.name
        fasta_path.write_text(uploaded_file.getvalue().decode("utf-8", errors="ignore"), encoding="utf-8")
        output_dir = tmpdir_path / "esm2_out"

        try:
            with st.spinner("Extracting ESM-2 embeddings from the uploaded FASTA..."):
                X, labels = extract_fasta_embeddings(fasta_path, output_dir)
            st.success(f"Extracted {X.shape[0]} sequences with embedding dimension {X.shape[1]}.")

            if X.shape[0] < 2:
                st.warning("At least two sequences are needed for PCA.")
                st.stop()

            with st.spinner("Running PCA..."):
                projection, variance = compute_pca(X, n_components=n_components)

            fig = draw_pca_plot(projection, labels, f"ESM-2 PCA of {uploaded_file.name}")
            st.pyplot(fig)

            st.subheader("PCA results")
            st.write("Explained variance ratios:")
            variance_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(len(variance))],
                "ExplainedVarianceRatio": variance,
            })
            st.dataframe(variance_df, use_container_width=True)

            pca_df = pd.DataFrame(projection, columns=[f"PC{i+1}" for i in range(projection.shape[1])])
            if labels is not None:
                pca_df.insert(0, "label", labels.tolist())
            st.download_button(
                label="Download PCA coordinates (.csv)",
                data=pca_df.to_csv(index=False),
                file_name="esm2_pca_projection.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download embeddings (.csv)",
                data=pd.DataFrame(X).to_csv(index=False),
                file_name="esm2_embeddings.csv",
                mime="text/csv",
            )

        except subprocess.CalledProcessError as exc:
            st.error(f"ESM extraction failed. Please check the FASTA input and model availability. Exit code: {exc.returncode}")
        except Exception as exc:
            st.exception(exc)
else:
    st.info("Upload a FASTA file to begin. The app will compute mean-pooled ESM-2 embeddings, run PCA, and display the results.")
