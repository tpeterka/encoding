"""
Patch Similarity Between Two Timesteps

Loads two FAISS patch databases (one per timestep), extracts the stored
vectors, and computes an aggregate L2 distance metric between corresponding
patches. Assumes both timesteps have the same number of patches at the same
spatial locations.

The aggregate metric is the L2 norm of the per-patch L2 distances, which is
equivalent to the Frobenius norm of the difference between the two stored
vector matrices: D = ||V0 - V1||_F = sqrt(sum_i ||v0_i - v1_i||^2).

The script supports two representations:
    concat : raw concatenated control-point patch vectors
    embed  : latent patch embeddings produced by embed_vector_db.py
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


def load_and_validate_bounds(timestep_a, timestep_b, resolution, output_dir):
    bounds_path_a = os.path.join(
        output_dir, f"mfa_patch_ctrl_bounds_{timestep_a}-{resolution}-approx.npy"
    )
    bounds_path_b = os.path.join(
        output_dir, f"mfa_patch_ctrl_bounds_{timestep_b}-{resolution}-approx.npy"
    )

    print(f"Loading {bounds_path_a} ...")
    bounds_a = np.load(bounds_path_a)
    print(f"Loading {bounds_path_b} ...")
    bounds_b = np.load(bounds_path_b)

    if bounds_a.shape != bounds_b.shape:
        raise ValueError(
            f"Patch-bounds shape mismatch: timestep {timestep_a} has {bounds_a.shape}, "
            f"timestep {timestep_b} has {bounds_b.shape}"
        )
    if not np.array_equal(bounds_a, bounds_b):
        mismatch = np.argwhere(bounds_a != bounds_b)[0]
        row, col = int(mismatch[0]), int(mismatch[1])
        raise ValueError(
            f"Patch bounds differ at row {row}, column {col}: "
            f"timestep {timestep_a} has {bounds_a[row, col]}, "
            f"timestep {timestep_b} has {bounds_b[row, col]}"
        )

    print(f"Patch bounds verified: shape={bounds_a.shape}")
    return bounds_a


def compute_patch_similarity(
    timestep_a,
    timestep_b,
    resolution,
    mode="concat",
    output_dir="output",
    skip_bounds_check=False,
):
    """
    Compute the aggregate L2 distance between corresponding stored vectors
    of two timesteps.

    Parameters
    ----------
    timestep_a, timestep_b : int or str
        The two timestep identifiers.
    resolution : str or None
        Resolution suffix used by concat mode's existing MFA filenames.
    mode : str
        Either "concat" for raw concatenated patch vectors or "embed" for
        latent patch embeddings.
    output_dir : str
        Directory containing the FAISS index files.
    skip_bounds_check : bool
        If True, skip validating that saved patch bounds match exactly between
        the two timesteps before comparing vectors.

    Returns
    -------
    aggregate_distance : float
        The Frobenius norm of the difference between the two stored vector
        matrices.
    per_patch_distances : numpy array of shape (n_patches,)
        The L2 distance for each corresponding pair of patches.
    """
    if mode == "concat":
        if not resolution:
            raise ValueError("concat mode requires --resolution")
        path_a = os.path.join(output_dir, f"mfa_patches_{timestep_a}-{resolution}-approx.index")
        path_b = os.path.join(output_dir, f"mfa_patches_{timestep_b}-{resolution}-approx.index")
        representation = "concatenated patch vectors"
    elif mode == "embed":
        path_a = os.path.join(output_dir, f"mfa_patch_latents_{timestep_a}-{resolution}-approx.index")
        path_b = os.path.join(output_dir, f"mfa_patch_latents_{timestep_b}-{resolution}-approx.index")
        representation = "latent patch embeddings"
    else:
        raise ValueError(f"Unsupported mode {mode!r}; expected 'concat' or 'embed'")

    if skip_bounds_check:
        print("Skipping patch-bounds validation")
        bounds = None
    else:
        bounds = load_and_validate_bounds(timestep_a, timestep_b, resolution, output_dir)

    print(f"Loading {path_a} ...")
    index_a = faiss.read_index(path_a)
    print(f"Loading {path_b} ...")
    index_b = faiss.read_index(path_b)

    if index_a.ntotal != index_b.ntotal:
        raise ValueError(
            f"Patch count mismatch: timestep {timestep_a} has {index_a.ntotal}, "
            f"timestep {timestep_b} has {index_b.ntotal}"
        )
    if index_a.d != index_b.d:
        raise ValueError(
            f"Dimension mismatch: timestep {timestep_a} has d={index_a.d}, "
            f"timestep {timestep_b} has d={index_b.d}"
        )

    n_patches = index_a.ntotal
    dim = index_a.d
    if bounds is not None and bounds.shape[0] != n_patches:
        raise ValueError(
            f"Patch-bounds row count {bounds.shape[0]} does not match index size {n_patches}"
        )
    print(f"Patches: {n_patches}, dimension: {dim}, representation: {representation}")

    vectors_a = extract_vectors(index_a)
    vectors_b = extract_vectors(index_b)

    diff = vectors_a - vectors_b
    per_patch_distances = np.linalg.norm(diff, axis=1)
    aggregate_distance = np.linalg.norm(diff)

    print(f"\nPatch similarity ({representation}): timestep {timestep_a} vs {timestep_b}")
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
    parser.add_argument("--resolution", required=True, help='resolution (string), eg "64x64x128"')
    parser.add_argument(
        "--mode",
        choices=("concat", "embed"),
        default="concat",
        help="which index representation to compare (default: concat)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing FAISS index files (default: output)",
    )
    parser.add_argument(
        "--skip-bounds-check",
        action="store_true",
        help="compare FAISS vectors without validating saved patch bounds first",
    )
    args = parser.parse_args()

    compute_patch_similarity(
        args.timestep_a,
        args.timestep_b,
        args.resolution,
        mode=args.mode,
        output_dir=args.output_dir,
        skip_bounds_check=args.skip_bounds_check,
    )
