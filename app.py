#!/usr/bin/env python3

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


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
    """Compute PCA projection. Try scikit-learn if available, otherwise fall back to a NumPy SVD implementation.

    Returns:
        projection: (n_samples, n_components) array
        explained_variance_ratio: length-n_components array
    """
    if X.shape[0] < 2:
        raise ValueError("At least 2 sequences are required to compute PCA.")

    # Prefer scikit-learn when available for stability and feature parity
    try:
        from sklearn.decomposition import PCA as SKPCA

        pca = SKPCA(n_components=n_components, random_state=0)
        projection = pca.fit_transform(X)
        return projection, pca.explained_variance_ratio_
    except Exception:
        # Fallback: compute PCA via SVD on mean-centered data
        Xc = X - np.mean(X, axis=0, keepdims=True)
        # compute compact SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        components = Vt[:n_components]
        projection = Xc.dot(components.T)
        # compute explained variance ratio from singular values
        # variance per component = (S**2) / (n_samples - 1)
        var_comp = (S ** 2) / max(1, (X.shape[0] - 1))
        total_var = var_comp.sum()
        explained = (var_comp[:n_components] / total_var) if total_var > 0 else np.zeros(n_components)
        return projection, explained


def draw_pca_plot(projection: np.ndarray, labels: Optional[np.ndarray], title: str):
    # If Plotly isn't available in the runtime, return None so caller can fall back
    if not PLOTLY_AVAILABLE:
        return None

    # Use Plotly for interactive and lightweight plotting in Streamlit
    df = pd.DataFrame(projection, columns=["PC1", "PC2"] if projection.shape[1] >= 2 else ["PC1"])  # type: ignore
    if labels is not None:
        df.insert(0, "label", labels)
        hover_data = {"label": True}
    else:
        hover_data = None

    if projection.shape[1] >= 2:
        fig = px.scatter(df, x="PC1", y="PC2", text=(df["label"] if "label" in df.columns else None), title=title)
        fig.update_traces(textposition="top center")
    else:
        # Single-component fallback: show as bar chart
        fig = px.bar(df, x=df.index, y="PC1", text=(df["label"] if "label" in df.columns else None), title=title)

    fig.update_layout(width=800, height=600)
    return fig


st.set_page_config(page_title="ESM-2 FASTA PCA", page_icon="🧬", layout="wide")
st.title("ESM-2 FASTA PCA Explorer")
st.caption("Upload a FASTA file to get ESM-2 embeddings and a PCA projection for downstream ML or visualization.")

with st.sidebar:
    st.subheader("Settings")
    # Use fixed 3-mer features for the demo (k=3). Remove interactive k selection to simplify UI.
    k = 3
    n_components = st.slider("PCA components", min_value=2, max_value=5, value=2)
    n_clusters = st.slider("Number of clusters (for coloring)", min_value=1, max_value=10, value=3)
    st.caption("Using 3-mer (k=3) featureization. PCA is computed from k-mer frequencies.")

uploaded_file = st.file_uploader("Drop a FASTA file here", type=["fasta", "fa", "faa", "txt"]) 
# Provide a quick example option that uses the included ha_random_20_cow_chicken_human.fasta in the repo
use_example = st.button("Use example FASTA (ha_random_20, 60 sequences)")

# Helper functions for FASTA parsing, k-mer featurization, and simple clustering

def read_fasta(path: Path):
    records = []
    header = None
    seq_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line.upper())
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def compute_kmer_matrix(seqs, k=3):
    # seqs: list of sequences (strings)
    n = len(seqs)
    kmer_counts_list = []
    kmer_set = set()
    for s in seqs:
        s = s.upper()
        counts = {}
        L = len(s)
        for i in range(0, max(0, L - k + 1)):
            kmer = s[i:i+k]
            if not kmer.isalpha():
                continue
            counts[kmer] = counts.get(kmer, 0) + 1
            kmer_set.add(kmer)
        kmer_counts_list.append((counts, max(1, L - k + 1)))
    kmers = sorted(kmer_set)
    X = np.zeros((n, len(kmers)), dtype=float)
    for i, (counts, denom) in enumerate(kmer_counts_list):
        for j, kmer in enumerate(kmers):
            X[i, j] = counts.get(kmer, 0) / denom
    return X, kmers


def compute_clusters(X, n_clusters=3):
    # Try scikit-learn's KMeans first, fallback to a simple numpy implementation
    if n_clusters <= 1:
        return np.zeros(X.shape[0], dtype=int)
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=0)
        labels = km.fit_predict(X)
        return labels
    except Exception:
        # Simple k-means (Lloyd) with random init
        rng = np.random.default_rng(0)
        n_samples = X.shape[0]
        # initialize centroids by sampling points
        centroids = X[rng.choice(n_samples, size=n_clusters, replace=False)]
        for _ in range(100):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            assigns = dists.argmin(axis=1)
            new_centroids = np.array([X[assigns == i].mean(axis=0) if np.any(assigns == i) else centroids[i] for i in range(n_clusters)])
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        return assigns


# Processing flows: uploaded FASTA or example FASTA
if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        fasta_path = tmpdir_path / uploaded_file.name
        fasta_path.write_text(uploaded_file.getvalue().decode("utf-8", errors="ignore"), encoding="utf-8")
        records = read_fasta(fasta_path)
        headers, seqs = zip(*records) if records else ([], [])
        if len(seqs) == 0:
            st.error("No sequences found in uploaded FASTA.")
        else:
            with st.spinner("Computing k-mer features..."):
                X, kmers = compute_kmer_matrix(list(seqs), k=k)
            st.success(f"Computed k={k} k-mer features: matrix shape {X.shape}.")

            if X.shape[0] < 2:
                st.warning("At least two sequences are needed for PCA.")
            else:
                with st.spinner("Running PCA..."):
                    projection, variance = compute_pca(X, n_components=n_components)
                # clustering for coloring
                clusters = compute_clusters(X, n_clusters=n_clusters)
                fig = draw_pca_plot(projection, labels=np.array([str(h) for h in headers]), title=f"k={k} PCA of {uploaded_file.name}")
                if fig is not None:
                    # add cluster coloring
                    try:
                        fig.data[0].marker.color = clusters
                    except Exception:
                        pass
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    pca_tbl = pd.DataFrame(projection, columns=[f"PC{i+1}" for i in range(projection.shape[1])])
                    pca_tbl.insert(0, "label", list(headers))
                    pca_tbl["cluster"] = clusters
                    st.dataframe(pca_tbl)

                st.subheader("PCA results")
                st.write("Explained variance ratios:")
                variance_df = pd.DataFrame({
                    "Component": [f"PC{i+1}" for i in range(len(variance))],
                    "ExplainedVarianceRatio": variance,
                })
                st.dataframe(variance_df, use_container_width=True)

                # download outputs
                pca_df = pd.DataFrame(projection, columns=[f"PC{i+1}" for i in range(projection.shape[1])])
                pca_df.insert(0, "label", list(headers))
                pca_df["cluster"] = clusters
                st.download_button(label="Download PCA coordinates (.csv)", data=pca_df.to_csv(index=False), file_name="kmer_pca_projection.csv", mime="text/csv")
                emb_df = pd.DataFrame(X, columns=kmers)
                emb_df.insert(0, "label", list(headers))
                st.download_button(label="Download k-mer features (.csv)", data=emb_df.to_csv(index=False), file_name="kmer_features.csv", mime="text/csv")

elif use_example:
    example_src = REPO_ROOT / "ha_random_20_cow_chicken_human.fasta"
    if not example_src.exists():
        st.error("Example FASTA not found in the repository. Please ensure ha_random_20_cow_chicken_human.fasta exists.")
    else:
        records = read_fasta(example_src)
        headers, seqs = zip(*records) if records else ([], [])
        if len(seqs) == 0:
            st.error("No sequences found in example FASTA.")
        else:
            with st.spinner("Computing k-mer features for example..."):
                X, kmers = compute_kmer_matrix(list(seqs), k=k)
            st.success(f"Computed k={k} k-mer features for example: matrix shape {X.shape}.")

            if X.shape[0] < 2:
                st.warning("At least two sequences are needed for PCA.")
            else:
                with st.spinner("Running PCA..."):
                    projection, variance = compute_pca(X, n_components=n_components)
                clusters = compute_clusters(X, n_clusters=n_clusters)
                fig = draw_pca_plot(projection, labels=np.array([str(h) for h in headers]), title=f"k={k} PCA of example")
                if fig is not None:
                    try:
                        fig.data[0].marker.color = clusters
                    except Exception:
                        pass
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    pca_tbl = pd.DataFrame(projection, columns=[f"PC{i+1}" for i in range(projection.shape[1])])
                    pca_tbl.insert(0, "label", list(headers))
                    pca_tbl["cluster"] = clusters
                    st.dataframe(pca_tbl)

                st.subheader("PCA results")
                st.write("Explained variance ratios:")
                variance_df = pd.DataFrame({
                    "Component": [f"PC{i+1}" for i in range(len(variance))],
                    "ExplainedVarianceRatio": variance,
                })
                st.dataframe(variance_df, use_container_width=True)

                pca_df = pd.DataFrame(projection, columns=[f"PC{i+1}" for i in range(projection.shape[1])])
                pca_df.insert(0, "label", list(headers))
                pca_df["cluster"] = clusters
                st.download_button(label="Download PCA coordinates (.csv)", data=pca_df.to_csv(index=False), file_name="kmer_pca_projection_example.csv", mime="text/csv")
                emb_df = pd.DataFrame(X, columns=kmers)
                emb_df.insert(0, "label", list(headers))
                st.download_button(label="Download k-mer features (.csv)", data=emb_df.to_csv(index=False), file_name="kmer_features_example.csv", mime="text/csv")

else:
    st.info("Upload a FASTA file to begin, or click 'Use example FASTA'. The app will compute k-mer features, run PCA, and display the results.")
