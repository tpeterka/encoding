"""
Compute patch distances within a single timestep relative to one reference patch.

Loads one FAISS patch database for a timestep, selects the unique patch that
contains a target control-point coordinate, and computes the L2 distance from
every stored vector to that reference vector.

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


def parse_reference_point(text):
    """
    Parse a reference point string formatted as gid,ix,iy,iz.
    """
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--reference-point must have exactly four comma-separated integers: gid,ix,iy,iz"
        )

    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "--reference-point must contain only integers: gid,ix,iy,iz"
        ) from exc


def find_reference_index(bounds, reference_point):
    """
    Find a patch whose bounds contain [gid, ix, iy, iz]. If multiple overlapping
    patches contain the point, choose the one with the smallest lower bound.
    """
    gid, ix, iy, iz = (int(value) for value in reference_point)
    matches = (
        (bounds[:, 0] == gid)
        & (bounds[:, 1] <= ix)
        & (ix <= bounds[:, 4])
        & (bounds[:, 2] <= iy)
        & (iy <= bounds[:, 5])
        & (bounds[:, 3] <= iz)
        & (iz <= bounds[:, 6])
    )
    match_indices = np.flatnonzero(matches)

    if match_indices.size == 0:
        raise ValueError(
            "Reference patch not found containing point "
            f"{(gid, ix, iy, iz)}"
        )
    if match_indices.size == 1:
        return int(match_indices[0])

    candidate_bounds = bounds[match_indices, 1:4]
    order = np.lexsort((candidate_bounds[:, 2], candidate_bounds[:, 1], candidate_bounds[:, 0]))
    selected_index = int(match_indices[order[0]])
    print(
        "Reference point is contained in multiple patches; selecting the one with "
        f"the smallest lower bound at index {selected_index}"
    )
    return selected_index


def build_input_paths(timestep, resolution, mode, output_dir):
    if mode == "concat":
        if not resolution:
            raise ValueError("concat mode requires --resolution")
        index_path = os.path.join(
            output_dir, f"mfa_patches_{timestep}-{resolution}-approx.index"
        )
        bounds_path = os.path.join(
            output_dir, f"mfa_patch_ctrl_bounds_{timestep}-{resolution}-approx.npy"
        )
        representation = "concatenated patch vectors"
        output_stem = f"mfa_space_patch_distances_{timestep}-{resolution}-approx"
    elif mode == "embed":
        index_path = os.path.join(
            output_dir, f"mfa_patch_latents_{timestep}-{resolution}-approx.index"
        )
        bounds_path = os.path.join(
            output_dir, f"mfa_patch_ctrl_bounds_{timestep}-{resolution}-approx.npy"
        )
        representation = "latent patch embeddings"
        output_stem = f"mfa_space_patch_latent_distances_{timestep}-{resolution}-approx"
    else:
        raise ValueError(f"Unsupported mode {mode!r}; expected 'concat' or 'embed'")

    return index_path, bounds_path, representation, output_stem


def compute_space_patch_similarity(
    timestep,
    resolution,
    reference_point,
    mode="concat",
    output_dir="output",
):
    """
    Compute L2 distance from every stored patch vector to a reference patch.

    Parameters
    ----------
    timestep : int or str
        The timestep identifier.
    resolution : str
        Resolution suffix used by the existing MFA filenames.
    reference_point : tuple[int, int, int, int]
        Target control-point coordinate as (gid, ix, iy, iz). The selected
        reference patch contains this point; if multiple overlapping patches
        contain it, the one with the smallest lower bound is chosen.
    mode : str
        Either "concat" for raw concatenated patch vectors or "embed" for
        latent patch embeddings.
    output_dir : str
        Directory containing the FAISS index files and patch metadata.

    Returns
    -------
    distances : numpy array of shape (n_patches,)
        L2 distance from each stored patch vector to the reference patch.
    reference_index : int
        Index of the reference patch in the FAISS index / bounds array.
    distances_path : str
        Path to the saved raw distance .npy file.
    metadata_path : str
        Path to the saved metadata .npz file.
    """
    index_path, bounds_path, representation, output_stem = build_input_paths(
        timestep, resolution, mode, output_dir
    )

    print(f"Loading {bounds_path} ...")
    bounds = np.load(bounds_path)
    if bounds.ndim != 2 or bounds.shape[1] != 7:
        raise ValueError(
            f"Expected patch bounds with shape (n, 7), got {bounds.shape}"
        )

    print(f"Loading {index_path} ...")
    index = faiss.read_index(index_path)

    n_patches = index.ntotal
    dim = index.d
    if bounds.shape[0] != n_patches:
        raise ValueError(
            f"Patch-bounds row count {bounds.shape[0]} does not match index size {n_patches}"
        )

    reference_index = find_reference_index(bounds, reference_point)
    reference_bounds_row = bounds[reference_index].copy()

    print(f"Patches: {n_patches}, dimension: {dim}, representation: {representation}")
    print(f"Reference patch index: {reference_index}")
    print(f"Reference target point: {tuple(int(value) for value in reference_point)}")
    print(
        "Reference patch bounds: "
        f"[gid, ix_min, iy_min, iz_min, ix_max, iy_max, iz_max] = {reference_bounds_row.tolist()}"
    )

    vectors = extract_vectors(index)
    diff = vectors - vectors[reference_index]
    distances = np.linalg.norm(diff, axis=1)

    print(f"\nSpatial patch similarity ({representation}): timestep {timestep}")
    print(f"  Per-patch L2 distance to reference: min={distances.min():.6f}, "
          f"max={distances.max():.6f}, mean={distances.mean():.6f}")
    print(f"  Reference patch self-distance:      {distances[reference_index]:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    distances_path = os.path.join(output_dir, f"{output_stem}.npy")
    metadata_path = os.path.join(output_dir, f"{output_stem}.npz")

    np.save(distances_path, distances)
    np.savez(
        metadata_path,
        distances=distances,
        patch_bounds=bounds,
        reference_index=np.int64(reference_index),
        reference_point=np.asarray(reference_point, dtype=np.int64),
        reference_bounds_row=reference_bounds_row,
        timestep=np.asarray(timestep),
        resolution=np.asarray(resolution),
        mode=np.asarray(mode),
    )

    print(f"Saved raw distances to {distances_path}")
    print(f"Saved metadata to {metadata_path}")

    return distances, reference_index, distances_path, metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute patch distances within one timestep relative to the patch containing a reference point"
    )
    parser.add_argument("timestep", help="Timestep")
    parser.add_argument("--resolution", required=True, help='resolution (string), eg "64x64x128"')
    parser.add_argument(
        "--reference-point",
        required=True,
        help="target control-point coordinate as gid,ix,iy,iz",
    )
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
    args = parser.parse_args()

    compute_space_patch_similarity(
        args.timestep,
        args.resolution,
        parse_reference_point(args.reference_point),
        mode=args.mode,
        output_dir=args.output_dir,
    )
