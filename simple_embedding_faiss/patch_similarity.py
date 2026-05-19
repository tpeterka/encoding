"""
Patch Similarity Between Two Timesteps

Loads two FAISS patch databases (one per timestep), extracts the patch
vectors, and computes an aggregate L2 distance metric between corresponding
patches. Assumes both timesteps have the same number of patches at the same
spatial locations.

The aggregate metric is the L2 norm of the per-patch L2 distances, which is
equivalent to the Frobenius norm of the difference between the two patch
vector matrices: D = ||V0 - V1||_F = sqrt(sum_i ||v0_i - v1_i||^2).

Usage:
    python patch_similarity.py <timestep_a> <timestep_b>
"""

import os
import argparse
import numpy as np
import faiss


def extract_vectors(index):
    """
    Extract all vectors from a FAISS IndexFlatL2 as a numpy array.

    Returns: numpy array of shape (n, d), dtype float32
    """
    n = index.ntotal
    d = index.d
    vectors = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        vectors[i] = index.reconstruct(i)
    return vectors


def compute_patch_similarity(timestep_a, timestep_b, output_dir="output"):
    """
    Compute the aggregate L2 distance between corresponding patches
    of two timesteps.

    Parameters
    ----------
    timestep_a, timestep_b : int or str
        The two timestep identifiers.
    output_dir : str
        Directory containing the FAISS index files.

    Returns
    -------
    aggregate_distance : float
        The Frobenius norm of the difference between the two patch matrices.
    per_patch_distances : numpy array of shape (n_patches,)
        The L2 distance for each corresponding pair of patches.
    """
    # --- Load FAISS indices ---
    path_a = os.path.join(output_dir, f"patches_{timestep_a}.index")
    path_b = os.path.join(output_dir, f"patches_{timestep_b}.index")

    print(f"Loading {path_a} ...")
    index_a = faiss.read_index(path_a)
    print(f"Loading {path_b} ...")
    index_b = faiss.read_index(path_b)

    # --- Validate ---
    assert index_a.ntotal == index_b.ntotal, (
        f"Patch count mismatch: timestep {timestep_a} has {index_a.ntotal}, "
        f"timestep {timestep_b} has {index_b.ntotal}"
    )
    assert index_a.d == index_b.d, (
        f"Dimension mismatch: timestep {timestep_a} has d={index_a.d}, "
        f"timestep {timestep_b} has d={index_b.d}"
    )

    n_patches = index_a.ntotal
    dim = index_a.d
    print(f"Patches: {n_patches}, dimension: {dim}")

    # --- Extract vectors ---
    vectors_a = extract_vectors(index_a)
    vectors_b = extract_vectors(index_b)

    # --- Compute distances ---
    diff = vectors_a - vectors_b
    per_patch_distances = np.linalg.norm(diff, axis=1)
    aggregate_distance = np.linalg.norm(diff)

    # --- Report ---
    print(f"\nPatch similarity: timestep {timestep_a} vs {timestep_b}")
    print(f"  Per-patch L2 distance:  min={per_patch_distances.min():.6f}, "
          f"max={per_patch_distances.max():.6f}, "
          f"mean={per_patch_distances.mean():.6f}")
    print(f"  Aggregate L2 distance:  {aggregate_distance:.6f}")

    return aggregate_distance, per_patch_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute patch similarity between two timesteps"
    )
    parser.add_argument("timestep_a", help="First timestep")
    parser.add_argument("timestep_b", help="Second timestep")
    parser.add_argument("--output-dir", default="output",
                        help="Directory containing FAISS index files (default: output)")
    args = parser.parse_args()

    compute_patch_similarity(args.timestep_a, args.timestep_b, args.output_dir)
