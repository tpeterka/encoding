"""
Per-tensor norms of patch embeddings for one timestep of an adaptive MFA model.

Loads the FAISS index and metadata written by embed_vector_db.py. For each
tensor t, the latent vectors of its patches form a matrix V_t of shape
(n_patches_t, latent_dim), and the script computes:

    frobenius_norm      ||V_t||_F = sqrt(sum_i ||v_i||^2)
    rms_patch_norm      ||V_t||_F / sqrt(n_patches_t), comparable across tensors
                        with different patch counts
    per-patch norms     ||v_i|| for every patch (saved for later queries)
"""

import argparse
import os

import faiss
import numpy as np


def extract_vectors(index):
    """
    Extract all vectors from a FAISS IndexFlatL2 as a numpy array of shape (n, d).
    """
    return index.reconstruct_n(0, index.ntotal).astype(np.float32, copy=False)


def compute_tensor_norms(timestep, output_dir="output"):
    stem = f"{timestep}"
    index_path = os.path.join(output_dir, f"mfa_adaptive_patch_latents_{stem}.index")
    bounds_path = os.path.join(output_dir, f"mfa_adaptive_patch_ctrl_bounds_{stem}.npy")
    tensor_info_path = os.path.join(output_dir, f"mfa_adaptive_tensor_info_{stem}.npy")

    print(f"Loading {index_path} ...")
    index = faiss.read_index(index_path)
    print(f"Loading {bounds_path} ...")
    bounds = np.load(bounds_path)
    print(f"Loading {tensor_info_path} ...")
    tensor_info = np.load(tensor_info_path)

    if bounds.ndim != 2 or bounds.shape[1] != 8:
        raise ValueError(f"Expected patch bounds with shape (n, 8), got {bounds.shape}")
    if tensor_info.ndim != 2 or tensor_info.shape[1] != 10:
        raise ValueError(f"Expected tensor info with shape (n_tensors, 10), got {tensor_info.shape}")
    if bounds.shape[0] != index.ntotal:
        raise ValueError(
            f"Patch-bounds row count {bounds.shape[0]} does not match index size {index.ntotal}"
        )
    if int(tensor_info[:, 9].sum()) != index.ntotal:
        raise ValueError("Sum of per-tensor patch counts does not match index size")

    vectors = extract_vectors(index)
    patch_norms = np.linalg.norm(vectors, axis=1)

    n_tensors = tensor_info.shape[0]
    frobenius_norms = np.zeros(n_tensors, dtype=np.float64)
    rms_patch_norms = np.zeros(n_tensors, dtype=np.float64)
    patch_counts = tensor_info[:, 9].copy()

    print(f"\nPer-tensor latent norms: timestep {timestep}, {index.ntotal} vectors, dimension {index.d}")
    print(f"  {'gid':>4} {'tensor':>6} {'nctrl_pts':>16} {'patches':>8} {'frobenius':>14} {'rms_patch':>12}")
    for t, (gid, tensor_idx, nx, ny, nz, _, _, _, row_start, row_count) in enumerate(tensor_info):
        if row_count > 0:
            rows = slice(row_start, row_start + row_count)
            if not np.all(bounds[rows, 0] == gid) or not np.all(bounds[rows, 1] == tensor_idx):
                raise ValueError(f"Patch bounds rows {row_start}..{row_start + row_count} do not belong to tensor {tensor_idx}")
            frobenius_norms[t] = np.linalg.norm(vectors[rows].astype(np.float64))
            rms_patch_norms[t] = frobenius_norms[t] / np.sqrt(row_count)
        print(
            f"  {gid:>4} {tensor_idx:>6} {f'{nx}x{ny}x{nz}':>16} {row_count:>8} "
            f"{frobenius_norms[t]:>14.6f} {rms_patch_norms[t]:>12.6f}"
        )

    total_norm = np.linalg.norm(vectors.astype(np.float64))
    print(f"  Frobenius norm of all vectors: {total_norm:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    output_stem = f"mfa_adaptive_tensor_norms_{stem}"
    norms_path = os.path.join(output_dir, f"{output_stem}.npy")
    metadata_path = os.path.join(output_dir, f"{output_stem}.npz")

    np.save(norms_path, frobenius_norms)
    np.savez(
        metadata_path,
        frobenius_norms=frobenius_norms,
        rms_patch_norms=rms_patch_norms,
        patch_counts=patch_counts,
        patch_norms=patch_norms,
        patch_bounds=bounds,
        tensor_info=tensor_info,
        timestep=np.asarray(timestep),
    )
    print(f"Saved per-tensor Frobenius norms to {norms_path}")
    print(f"Saved metadata to {metadata_path}")

    return frobenius_norms, rms_patch_norms, patch_norms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-tensor Frobenius norms of patch embeddings for one timestep"
    )
    parser.add_argument("timestep", help="Timestep")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing FAISS index files (default: output)",
    )
    args = parser.parse_args()

    compute_tensor_norms(args.timestep, output_dir=args.output_dir)
