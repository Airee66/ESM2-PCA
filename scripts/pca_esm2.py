#!/usr/bin/env python3
# Run PCA on ESM-2 embeddings and save a machine-learning-ready projection.

import argparse
import csv
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTRACTOR_PATH = REPO_ROOT / "scripts" / "extract_esm2_mean.py"


def load_embeddings(input_path):
    p = pathlib.Path(input_path)
    if p.suffix == ".npy":
        X = np.load(p)
        return X, None
    if p.suffix == ".npz":
        data = np.load(p)
        if "X" in data:
            X = data["X"]
        elif "embeddings" in data:
            X = data["embeddings"]
        else:
            raise ValueError(f"NPZ file {p} does not contain an 'X' or 'embeddings' array.")
        labels = data["labels"] if "labels" in data else None
        return X, labels
    if p.suffix in {".csv", ".tsv"}:
        sep = "," if p.suffix == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep, header=None)
        X = df.to_numpy(dtype=float)
        return X, None
    raise ValueError(f"Unsupported embedding format: {p}")


def load_labels(labels_path, fallback_labels):
    if labels_path is None:
        return fallback_labels
    if labels_path.suffix == ".csv":
        arr = np.genfromtxt(labels_path, dtype=str, delimiter=",", comments=None)
        if arr.ndim == 0:
            return [str(arr)]
        return arr.tolist()
    if labels_path.suffix == ".npy":
        return np.load(labels_path).tolist()
    return None


def ensure_embeddings(input_path, output_dir, model_name):
    p = pathlib.Path(input_path)
    if p.suffix.lower() in {".fasta", ".fa", ".faa", ".txt"}:
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(EXTRACTOR_PATH),
            str(p),
            str(out_dir),
            "--model",
            model_name,
            "--nogpu",
        ]
        subprocess.run(cmd, check=True)
        emb_path = out_dir / "embeddings.npy"
        labels_path = out_dir / "labels.csv"
        if not emb_path.exists():
            raise FileNotFoundError(f"Extractor did not create {emb_path}")
        return str(emb_path), str(labels_path)
    return str(p), None


def save_projection_csv(out_dir, projection, labels, prefix="pca"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(projection, columns=[f"PC{i+1}" for i in range(projection.shape[1])])
    if labels is not None:
        df.insert(0, "label", labels)
    df.to_csv(out_dir / f"{prefix}.csv", index=False)


def save_variance_csv(out_dir, explained_variance):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(explained_variance))], "explained_variance_ratio": explained_variance})
    df.to_csv(out_dir / "pca_variance.csv", index=False)


def plot_pca(out_dir, projection, labels, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if projection.shape[1] >= 2:
        xs = projection[:, 0]
        ys = projection[:, 1]
        if labels is not None:
            for x, y, label in zip(xs, ys, labels):
                ax.scatter(x, y, s=50, alpha=0.8)
                ax.annotate(str(label), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        else:
            ax.scatter(xs, ys, s=50, alpha=0.8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / "pca_plot.png", dpi=200)
        plt.close(fig)
    else:
        raise ValueError("PCA projection must have at least 2 components to plot.")


def create_parser():
    parser = argparse.ArgumentParser(description="Run PCA on ESM-2 embeddings and save plot/CSV outputs")
    parser.add_argument("input", help="FASTA file or embedding matrix (.npy/.npz/.csv/.tsv)")
    parser.add_argument("--output-dir", default=None, help="directory to save outputs (defaults to input parent or fasta output dir)")
    parser.add_argument("--components", type=int, default=2, help="number of PCA components to retain (default: 2)")
    parser.add_argument("--model", type=str, default="esm2_t6_8M_UR50D", help="ESM model name when input is FASTA")
    parser.add_argument("--labels", default=None, help="optional labels CSV file matching embedding rows")
    parser.add_argument("--no-plot", action="store_true", help="skip generating the PCA scatter plot")
    return parser


def main():
    args = create_parser().parse_args()
    input_path = pathlib.Path(args.input)

    if args.output_dir is None:
        if input_path.suffix.lower() in {".fasta", ".fa", ".faa", ".txt"}:
            out_dir = input_path.parent / "pca_output"
        else:
            out_dir = input_path.parent / f"{input_path.stem}_pca"
    else:
        out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() in {".fasta", ".fa", ".faa", ".txt"}:
        emb_path, labels_path = ensure_embeddings(input_path, out_dir / "esm2_embeddings", args.model)
        X, fallback_labels = load_embeddings(emb_path)
        labels = load_labels(pathlib.Path(labels_path) if labels_path else None, fallback_labels)
    else:
        X, fallback_labels = load_embeddings(input_path)
        labels = load_labels(pathlib.Path(args.labels) if args.labels else None, fallback_labels)

    if X.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2D, got shape {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("At least 2 sequences are required for PCA.")
    if args.components < 1 or args.components >= min(X.shape):
        raise ValueError(f"components must be between 1 and {min(X.shape)-1}.")

    pca = PCA(n_components=args.components, random_state=0)
    projection = pca.fit_transform(X)

    save_projection_csv(out_dir, projection, labels, prefix="pca_projection")
    save_variance_csv(out_dir, pca.explained_variance_ratio_)
    np.save(out_dir / "pca_components.npy", projection)

    if not args.no_plot:
        plot_pca(out_dir, projection, labels, title="ESM-2 PCA")

    print(f"PCA complete: {X.shape} -> {projection.shape}")
    print(f"Saved outputs to {out_dir}")
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")


if __name__ == "__main__":
    main()
