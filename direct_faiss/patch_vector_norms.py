"""
Compute per-patch vector norms for one timestep and save patch-space metadata.

Loads one FAISS patch database and matching coordinate metadata, reconstructs all
stored vectors, computes their L2 norms, and saves both a raw .npy array and a
metadata .npz that can be converted with space_patch_similarity_to_vtk.py.
"""

import os
import argparse
import numpy as np
import faiss

from patch_similarity import extract_vectors


def compute_patch_vector_norms(timestep, output_dir="output"):
    index_path = os.path.join(output_dir, f"patches_{timestep}.index")
    coords_path = os.path.join(output_dir, f"patch_coords_{timestep}.npy")

    print(f"Loading {coords_path} ...")
    patch_coords = np.load(coords_path)
    if patch_coords.ndim != 2 or patch_coords.shape[1] != 6:
        raise ValueError(
            f"Expected patch_coords with shape (n, 6), got {patch_coords.shape}"
        )

    print(f"Loading {index_path} ...")
    index = faiss.read_index(index_path)
    if patch_coords.shape[0] != index.ntotal:
        raise ValueError(
            f"Patch-coordinate row count {patch_coords.shape[0]} does not match index size {index.ntotal}"
        )

    vectors = extract_vectors(index)
    norms = np.linalg.norm(vectors, axis=1)

    print("Patch vector norm summary")
    print(f"  timestep: {timestep}")
    print(f"  vector array shape: {vectors.shape}")
    print(
        f"  norm stats: min={float(norms.min()):.9g}, max={float(norms.max()):.9g}, "
        f"mean={float(norms.mean()):.9g}, std={float(norms.std()):.9g}"
    )

    os.makedirs(output_dir, exist_ok=True)
    norms_path = os.path.join(output_dir, f"patch_norms_{timestep}.npy")
    metadata_path = os.path.join(output_dir, f"patch_norms_{timestep}.npz")

    np.save(norms_path, norms)
    np.savez(
        metadata_path,
        distances=norms,
        patch_coords=patch_coords,
        timestep=np.asarray(timestep),
    )

    print(f"Saved raw norms to {norms_path}")
    print(f"Saved metadata to {metadata_path}")

    return norms, norms_path, metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-patch vector norms for one timestep"
    )
    parser.add_argument("timestep", help="Timestep")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing FAISS index files (default: output)",
    )
    args = parser.parse_args()

    compute_patch_vector_norms(
        args.timestep,
        output_dir=args.output_dir,
    )
