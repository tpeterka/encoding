"""
FAISS vector database for MFA patch embeddings.
Each vector is the latent space embedding inferred from a trained autoencoder embedding model.

Loads one MFA file, extracts scalar control-point patches, embeds them with a
trained patch autoencoder encoder, and stores the latent vectors in a FAISS
index. Control-point lattice bounds are saved alongside the index so search
results can be mapped back to source patches.
"""

import argparse
import os

import diy
import faiss
import mfa
import numpy as np
import torch

from patch_data import extract_block_patches
from patch_model import PatchAutoencoder


def load_patch_encoder(checkpoint_path, device, patch_x, patch_y, patch_z):
    # Training checkpoints include numpy metadata alongside weights, so opt out of
    # PyTorch 2.6's default weights_only=True behavior when loading local files.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    patch_shape = checkpoint["patch_shape"]
    expected_shape = [patch_x, patch_y, patch_z]
    if list(patch_shape) != expected_shape:
        raise ValueError(
            f"Checkpoint patch shape {patch_shape} does not match requested patch shape {expected_shape}"
        )

    input_dim = checkpoint["input_dim"]
    expected_input_dim = patch_x * patch_y * patch_z
    if input_dim != expected_input_dim:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} does not match requested patch volume {expected_input_dim}"
        )

    model = PatchAutoencoder(input_dim=input_dim, latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    norm_mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    norm_std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    if norm_mean.shape != (input_dim,) or norm_std.shape != (input_dim,):
        raise ValueError(
            f"Normalization stats shape mismatch: mean={norm_mean.shape}, std={norm_std.shape}, expected {(input_dim,)}"
        )

    return model, norm_mean, norm_std, checkpoint["latent_dim"]


def encode_patch_vectors(patch_vectors, model, norm_mean, norm_std, device, batch_size):
    normalized = (patch_vectors - norm_mean) / norm_std
    normalized = normalized.astype(np.float32, copy=False)

    latents = []
    with torch.no_grad():
        for start in range(0, normalized.shape[0], batch_size):
            stop = min(start + batch_size, normalized.shape[0])
            batch = torch.from_numpy(normalized[start:stop]).to(device)
            latents.append(model.encode(batch).cpu().numpy())

    latent_vectors = np.vstack(latents).astype(np.float32, copy=False)
    return latent_vectors


def build_embedding_database(
    infile,
    encoder_ckpt,
    output_dir="output",
    patch_x=4,
    patch_y=4,
    patch_z=4,
    batch_size=4096,
):
    w = diy.mpi.MPIComm()
    master = diy.Master(w)
    assigner = diy.ContiguousAssigner(w.size, -1)

    print(f"Loading MFA blocks from {infile}")
    diy.read_blocks(infile, assigner, master, load=mfa.load_block)

    all_patch_vectors = []
    all_patch_bounds = []

    def collect_block_patches(block, cp):
        patch_vectors, patch_bounds = extract_block_patches(block, cp, patch_x, patch_y, patch_z)
        all_patch_vectors.append(patch_vectors)
        all_patch_bounds.append(patch_bounds)

    master.foreach(collect_block_patches)

    if not all_patch_vectors:
        raise ValueError(f"No blocks were loaded from {infile}")

    patch_vectors = np.vstack(all_patch_vectors).astype(np.float32, copy=False)
    patch_bounds = np.vstack(all_patch_bounds)
    patch_dim = patch_x * patch_y * patch_z

    print(
        f"\nRaw patch vectors: shape={patch_vectors.shape}, min={patch_vectors.min():.6f}, "
        f"max={patch_vectors.max():.6f}"
    )
    print(
        f"Patch bounds: shape={patch_bounds.shape} "
        f"(each row: [gid, ix_min, iy_min, iz_min, ix_max, iy_max, iz_max])"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_mean, norm_std, latent_dim = load_patch_encoder(
        encoder_ckpt, device, patch_x, patch_y, patch_z
    )
    latent_vectors = encode_patch_vectors(
        patch_vectors, model, norm_mean, norm_std, device, batch_size
    )

    if latent_vectors.shape != (patch_vectors.shape[0], latent_dim):
        raise ValueError(
            f"Unexpected latent vector shape {latent_vectors.shape}, expected {(patch_vectors.shape[0], latent_dim)}"
        )

    print(
        f"Latent vectors: shape={latent_vectors.shape}, min={latent_vectors.min():.6f}, "
        f"max={latent_vectors.max():.6f}"
    )

    index = faiss.IndexFlatL2(latent_dim)
    index.add(latent_vectors)
    print(f"FAISS index: {index.ntotal} vectors, dimension={index.d}")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(infile))[0]

    index_path = os.path.join(output_dir, f"mfa_patch_latents_{stem}.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    bounds_path = os.path.join(output_dir, f"mfa_patch_ctrl_bounds_{stem}.npy")
    np.save(bounds_path, patch_bounds)
    print(f"Saved patch control-point bounds to {bounds_path}")

    return index, patch_bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index over MFA patch embeddings")
    parser.add_argument("--infile", required=True, help="input MFA file")
    parser.add_argument("--encoder-ckpt", required=True, help="trained patch autoencoder checkpoint")
    parser.add_argument("--output-dir", default="output", help="directory for FAISS index and metadata")
    parser.add_argument("--patch-x", type=int, default=4, help="patch size in x")
    parser.add_argument("--patch-y", type=int, default=4, help="patch size in y")
    parser.add_argument("--patch-z", type=int, default=4, help="patch size in z")
    parser.add_argument("--batch-size", type=int, default=4096, help="encoder inference batch size")
    args = parser.parse_args()

    build_embedding_database(
        infile=args.infile,
        encoder_ckpt=args.encoder_ckpt,
        output_dir=args.output_dir,
        patch_x=args.patch_x,
        patch_y=args.patch_y,
        patch_z=args.patch_z,
        batch_size=args.batch_size,
    )
