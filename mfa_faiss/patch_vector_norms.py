"""
Compute per-patch vector norms for one timestep and save patch-space metadata.

Loads one FAISS patch database and matching bounds metadata, reconstructs all
stored vectors, computes their L2 norms, and saves both a raw .npy array and a
metadata .npz that can be converted with space_patch_similarity_to_vtk.py.
"""

import argparse
import os
import numpy as np
import faiss

from space_patch_similarity import build_input_paths, extract_vectors


def build_output_stem(timestep, resolution, mode):
    if mode == "concat":
        return f"mfa_patch_norms_{timestep}-{resolution}-approx"
    if mode == "embed":
        return f"mfa_patch_latent_norms_{timestep}-{resolution}-approx"
    raise ValueError(f"Unsupported mode {mode!r}; expected 'concat' or 'embed'")


def compute_patch_vector_norms(timestep, resolution, mode="concat", output_dir="output"):
    index_path, bounds_path, representation, _ = build_input_paths(
        timestep, resolution, mode, output_dir
    )
    output_stem = build_output_stem(timestep, resolution, mode)

    print(f"Loading {bounds_path} ...")
    bounds = np.load(bounds_path)
    if bounds.ndim != 2 or bounds.shape[1] != 7:
        raise ValueError(f"Expected patch bounds with shape (n, 7), got {bounds.shape}")

    print(f"Loading {index_path} ...")
    index = faiss.read_index(index_path)
    if bounds.shape[0] != index.ntotal:
        raise ValueError(
            f"Patch-bounds row count {bounds.shape[0]} does not match index size {index.ntotal}"
        )

    vectors = extract_vectors(index)
    norms = np.linalg.norm(vectors, axis=1)

    print("Patch vector norm summary")
    print(f"  timestep: {timestep}")
    print(f"  resolution: {resolution}")
    print(f"  mode: {mode}")
    print(f"  representation: {representation}")
    print(f"  vector array shape: {vectors.shape}")
    print(
        f"  norm stats: min={float(norms.min()):.9g}, max={float(norms.max()):.9g}, "
        f"mean={float(norms.mean()):.9g}, std={float(norms.std()):.9g}"
    )

    os.makedirs(output_dir, exist_ok=True)
    norms_path = os.path.join(output_dir, f"{output_stem}.npy")
    metadata_path = os.path.join(output_dir, f"{output_stem}.npz")

    np.save(norms_path, norms)
    np.savez(
        metadata_path,
        distances=norms,
        patch_bounds=bounds,
        timestep=np.asarray(timestep),
        resolution=np.asarray(resolution),
        mode=np.asarray(mode),
    )

    print(f"Saved raw norms to {norms_path}")
    print(f"Saved metadata to {metadata_path}")

    return norms, norms_path, metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-patch vector norms for one timestep"
    )
    parser.add_argument("timestep", help="Timestep")
    parser.add_argument("--resolution", required=True, help='resolution (string), eg "64x64x128"')
    parser.add_argument(
        "--mode",
        choices=("concat", "embed"),
        default="concat",
        help="which index representation to inspect (default: concat)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing FAISS index files (default: output)",
    )
    args = parser.parse_args()

    compute_patch_vector_norms(
        args.timestep,
        args.resolution,
        mode=args.mode,
        output_dir=args.output_dir,
    )
