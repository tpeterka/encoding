"""
FAISS vector database for patch embeddings of adaptively encoded MFA models.

Loads one adaptive MFA file, extracts scalar control-point patches from every
tensor product in the model, embeds them with a trained patch autoencoder
encoder (reused from mfa_faiss), and stores the latent vectors in a single
FAISS index. Rows are grouped by tensor in tensor order.

Saved alongside the index:
    patch bounds : one row per vector,
                   [gid, tensor_idx, ix_min, iy_min, iz_min, ix_max, iy_max, iz_max]
                   (control-point indices local to the parent tensor)
    tensor info  : one row per tensor,
                   [gid, tensor_idx, nx, ny, nz, npx, npy, npz, row_start, row_count]
                   (row_start/row_count locate the tensor's vectors in the index)
"""

import argparse
import importlib.util
import os
import sys

import diy
import faiss
import mfa
import numpy as np
import torch

from tensor_patch_data import PATCH_BOUNDS_COLUMNS, TENSOR_INFO_COLUMNS, extract_block_patches

# reuse the encoder loading and inference helpers from mfa_faiss without duplicating code
# mfa_faiss/embed_vector_db.py has the same module name as this file, so load it under a distinct name
MFA_FAISS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mfa_faiss")
sys.path.append(MFA_FAISS_DIR)
_spec = importlib.util.spec_from_file_location(
    "mfa_faiss_embed_vector_db", os.path.join(MFA_FAISS_DIR, "embed_vector_db.py")
)
_mfa_faiss_embed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mfa_faiss_embed)
load_patch_encoder = _mfa_faiss_embed.load_patch_encoder
encode_patch_vectors = _mfa_faiss_embed.encode_patch_vectors


def timestep_from_infile(infile):
    """
    Extract the timestep from an MFA file name of the form <timestep>-<anything>.mfa,
    eg. 0-adaptive-approx.mfa -> "0".
    """
    return os.path.splitext(os.path.basename(infile))[0].split("-")[0]


def build_embedding_database(
    infile,
    encoder_ckpt,
    output_dir="output",
    patch_x=4,
    patch_y=4,
    patch_z=4,
    batch_size=4096,
    timestep=None,
):
    w = diy.mpi.MPIComm()
    master = diy.Master(w)
    assigner = diy.ContiguousAssigner(w.size, -1)

    print(f"Loading MFA blocks from {infile}")
    diy.read_blocks(infile, assigner, master, load=mfa.load_block)

    all_patch_vectors = []
    all_patch_bounds = []
    all_tensor_info = []
    n_rows = [0]

    def collect_block_patches(block, cp):
        patch_vectors, patch_bounds, tensor_info = extract_block_patches(
            block, cp, patch_x, patch_y, patch_z, row_offset=n_rows[0]
        )
        n_rows[0] += patch_vectors.shape[0]
        all_patch_vectors.append(patch_vectors)
        all_patch_bounds.append(patch_bounds)
        all_tensor_info.append(tensor_info)

    master.foreach(collect_block_patches)

    if not all_patch_vectors:
        raise ValueError(f"No blocks were loaded from {infile}")

    patch_vectors = np.vstack(all_patch_vectors).astype(np.float32, copy=False)
    patch_bounds = np.vstack(all_patch_bounds)
    tensor_info = np.vstack(all_tensor_info)

    if patch_vectors.shape[0] == 0:
        raise ValueError(f"No patches could be extracted from {infile}")
    if int(tensor_info[:, -1].sum()) != patch_vectors.shape[0]:
        raise ValueError("Sum of per-tensor patch counts does not match total patch count")

    print(
        f"\nRaw patch vectors: shape={patch_vectors.shape}, min={patch_vectors.min():.6f}, "
        f"max={patch_vectors.max():.6f}"
    )
    print(f"Patch bounds: shape={patch_bounds.shape} (each row: {list(PATCH_BOUNDS_COLUMNS)})")
    print(f"Tensor info: shape={tensor_info.shape} (each row: {list(TENSOR_INFO_COLUMNS)})")

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
    for gid, tensor_idx, *_, row_start, row_count in tensor_info:
        if row_count == 0:
            continue
        lat = latent_vectors[row_start:row_start + row_count]
        print(
            f"  gid={gid} tensor {tensor_idx}: {row_count} vectors, "
            f"latent min={lat.min():.6f}, max={lat.max():.6f}"
        )

    index = faiss.IndexFlatL2(latent_dim)
    index.add(latent_vectors)
    print(f"FAISS index: {index.ntotal} vectors, dimension={index.d}")

    os.makedirs(output_dir, exist_ok=True)
    stem = timestep if timestep is not None else timestep_from_infile(infile)

    index_path = os.path.join(output_dir, f"mfa_adaptive_patch_latents_{stem}.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    bounds_path = os.path.join(output_dir, f"mfa_adaptive_patch_ctrl_bounds_{stem}.npy")
    np.save(bounds_path, patch_bounds)
    print(f"Saved patch control-point bounds to {bounds_path}")

    tensor_info_path = os.path.join(output_dir, f"mfa_adaptive_tensor_info_{stem}.npy")
    np.save(tensor_info_path, tensor_info)
    print(f"Saved per-tensor info to {tensor_info_path}")

    return index, patch_bounds, tensor_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a FAISS index over patch embeddings of all tensors in an adaptive MFA model"
    )
    parser.add_argument("--infile", required=True, help="input adaptive MFA file")
    parser.add_argument("--encoder-ckpt", required=True, help="trained patch autoencoder checkpoint")
    parser.add_argument("--output-dir", default="output", help="directory for FAISS index and metadata")
    parser.add_argument("--patch-x", type=int, default=4, help="patch size in x")
    parser.add_argument("--patch-y", type=int, default=4, help="patch size in y")
    parser.add_argument("--patch-z", type=int, default=4, help="patch size in z")
    parser.add_argument("--batch-size", type=int, default=4096, help="encoder inference batch size")
    parser.add_argument(
        "--timestep",
        default=None,
        help="timestep label for output file names (default: leading part of infile name before the first '-')",
    )
    args = parser.parse_args()

    build_embedding_database(
        infile=args.infile,
        encoder_ckpt=args.encoder_ckpt,
        output_dir=args.output_dir,
        patch_x=args.patch_x,
        patch_y=args.patch_y,
        patch_z=args.patch_z,
        batch_size=args.batch_size,
        timestep=args.timestep,
    )
