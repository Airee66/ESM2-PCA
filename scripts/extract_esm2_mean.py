#!/usr/bin/env python3
# Extract per-sequence mean-pooled ESM-2 embeddings (default: esm2_t6_8M_UR50D)
# Saves one .npy file per FASTA entry containing the mean-pooled embedding vector.

import argparse
import pathlib
import re

import numpy as np
import torch

from esm import FastaBatchedDataset, pretrained


def sanitize_label(label):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    return safe or "sequence"


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract mean-pooled ESM-2 embeddings and save a machine-learning-ready matrix"
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="input FASTA file",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for .npy per-sequence embeddings and aggregated matrix outputs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_t6_8M_UR50D",
        help="pretrained model name (default: esm2_t6_8M_UR50D)",
    )
    parser.add_argument(
        "--toks_per_batch",
        type=int,
        default=4096,
        help="max tokens per batch (controls batching)",
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than this length",
    )
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="do not use GPU even if available",
    )
    return parser


def run(args):
    print(f"Loading model {args.model} ...")
    model, alphabet = pretrained.load_model_and_alphabet(args.model)
    model.eval()

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(args.truncation_seq_length),
        batch_sampler=batches,
    )

    print(f"Read {args.fasta_file} with {len(dataset)} sequences")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use final layer by default (official scripts use the final layer index)
    final_layer_idx = model.num_layers
    repr_layers = [final_layer_idx]

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1} / {len(batches)} ({toks.size(0)} sequences)")
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                rep_tensor = representations[repr_layers[0]]
                seq_rep = rep_tensor[i, 1 : truncate_len + 1].mean(0).clone()
                arr = seq_rep.cpu().numpy()

                safe_label = sanitize_label(label)
                out_file = args.output_dir / f"{safe_label}.npy"
                np.save(out_file, arr)

                all_labels.append(label)
                all_embeddings.append(arr)

    if len(all_embeddings) == 0:
        raise ValueError(f"No sequences were processed from FASTA file: {args.fasta_file}")

    matrix = np.vstack(all_embeddings)
    np.save(args.output_dir / "embeddings.npy", matrix)
    np.savez_compressed(args.output_dir / "embeddings.npz", X=matrix, labels=np.asarray(all_labels, dtype=object))
    np.savetxt(args.output_dir / "embeddings.csv", matrix, delimiter=",", fmt="%.10f")
    np.savetxt(args.output_dir / "labels.csv", np.asarray(all_labels, dtype=object), delimiter=",", fmt="%s")

    print(f"Done. Saved per-sequence embeddings and ML-ready matrix to {args.output_dir}")
    print(f"Matrix shape: {matrix.shape} (n_sequences x embedding_dim)")


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
