"""
FAISS Vector Database for Hash Embedding Patches

Loads the learned embedding table from output/embedding_table.npy,
extracts overlapping 3D patches from the feature grid, and stores them
in a FAISS vector database. Each patch vector is the concatenation of
n_features values at every grid point in the patch.

Patches are placed with stride = (patch_size - 1) along each axis so that
boundary grid points are shared between adjacent patches. If the remaining
grid points at the end of an axis don't form a full patch, they are skipped.

Patch coordinate metadata (min/max corners in normalized [-1, 1] space) is
saved to a separate numpy file alongside the FAISS index.
"""

import os
import numpy as np
import faiss
import argparse

def grid_index_to_coord(ix, iy, iz, res_x, res_y, res_z):
    """
    Convert integer grid indices to normalized [-1, 1] coordinates.

    The mapping matches what model.py / utility.py use:
        coord = index / (res - 1) * 2 - 1
    """
    x = ix / (res_x - 1) * 2.0 - 1.0
    y = iy / (res_y - 1) * 2.0 - 1.0
    z = iz / (res_z - 1) * 2.0 - 1.0
    return x, y, z


def linear_index(ix, iy, iz, res_y, res_z):
    """
    Row-major linear index matching hash_linear_3d() in model.py:
        index = ix * (res_y * res_z) + iy * res_z + iz
    """
    return ix * (res_y * res_z) + iy * res_z + iz


def build_patch_database(timestep=0,
    embedding_path="output/embedding_table_0.npy",
    output_dir="output",
    res_x=64,
    res_y=64,
    res_z=128,
    n_features=2,
    patch_x=4,
    patch_y=4,
    patch_z=4,
):
    """
    Build a FAISS index from overlapping 3D patches of the embedding grid.

    Parameters
    ----------
    embedding_path : str
        Path to the saved embedding table (.npy file).
    output_dir : str
        Directory for output files.
    res_x, res_y, res_z : int
        Grid resolution along each axis (must match training config).
    n_features : int
        Number of features per grid vertex (must match training config).
    patch_x, patch_y, patch_z : int
        Patch size (in grid points) along each axis. Default 4x4x4.
    """
    # --- Load embedding table ---
    embedding_table = np.load(embedding_path)
    expected_shape = (res_x * res_y * res_z, n_features)
    assert embedding_table.shape == expected_shape, (
        f"Shape mismatch: got {embedding_table.shape}, expected {expected_shape}"
    )
    print(f"Loaded embedding table: path={embedding_path}, "
          f"shape={embedding_table.shape}, "
          f"dtype={embedding_table.dtype}")

    # --- Compute patch layout ---
    # Stride = patch_size - 1 so boundaries overlap
    stride_x = patch_x - 1
    stride_y = patch_y - 1
    stride_z = patch_z - 1

    # Number of patches per axis (only full patches)
    n_patches_x = (res_x - patch_x) // stride_x + 1
    n_patches_y = (res_y - patch_y) // stride_y + 1
    n_patches_z = (res_z - patch_z) // stride_z + 1
    n_patches_total = n_patches_x * n_patches_y * n_patches_z

    # Dimension of each patch vector
    patch_dim = patch_x * patch_y * patch_z * n_features

    print(f"Grid resolution: ({res_x}, {res_y}, {res_z})")
    print(f"Patch size: ({patch_x}, {patch_y}, {patch_z})")
    print(f"Stride: ({stride_x}, {stride_y}, {stride_z})")
    print(f"Patches per axis: ({n_patches_x}, {n_patches_y}, {n_patches_z})")
    print(f"Total patches: {n_patches_total}")
    print(f"Patch vector dimension: {patch_dim}")

    # Check for skipped grid points
    covered_x = (n_patches_x - 1) * stride_x + patch_x
    covered_y = (n_patches_y - 1) * stride_y + patch_y
    covered_z = (n_patches_z - 1) * stride_z + patch_z
    if covered_x < res_x:
        print(f"  Note: {res_x - covered_x} grid point(s) skipped at end of x-axis")
    if covered_y < res_y:
        print(f"  Note: {res_y - covered_y} grid point(s) skipped at end of y-axis")
    if covered_z < res_z:
        print(f"  Note: {res_z - covered_z} grid point(s) skipped at end of z-axis")

    # --- Extract patches ---
    # Patch vectors: (n_patches_total, patch_dim)
    patch_vectors = np.empty((n_patches_total, patch_dim), dtype=np.float32)
    # Patch coordinates: (n_patches_total, 6) = [x_min, y_min, z_min, x_max, y_max, z_max]
    patch_coords = np.empty((n_patches_total, 6), dtype=np.float64)

    patch_idx = 0
    for px in range(n_patches_x):
        ix_start = px * stride_x
        for py in range(n_patches_y):
            iy_start = py * stride_y
            for pz in range(n_patches_z):
                iz_start = pz * stride_z

                # Gather n_features for all grid points in this patch
                features = []
                for dx in range(patch_x):
                    for dy in range(patch_y):
                        for dz in range(patch_z):
                            ix = ix_start + dx
                            iy = iy_start + dy
                            iz = iz_start + dz
                            idx = linear_index(ix, iy, iz, res_y, res_z)
                            features.append(embedding_table[idx])

                patch_vectors[patch_idx] = np.concatenate(features)

                # Compute min/max corner coordinates in normalized space
                x_min, y_min, z_min = grid_index_to_coord(
                    ix_start, iy_start, iz_start, res_x, res_y, res_z
                )
                x_max, y_max, z_max = grid_index_to_coord(
                    ix_start + patch_x - 1,
                    iy_start + patch_y - 1,
                    iz_start + patch_z - 1,
                    res_x, res_y, res_z,
                )
                patch_coords[patch_idx] = [x_min, y_min, z_min, x_max, y_max, z_max]

                patch_idx += 1

    assert patch_idx == n_patches_total

    print(f"\nPatch vectors: shape={patch_vectors.shape}, "
          f"min={patch_vectors.min():.6f}, max={patch_vectors.max():.6f}")

    # --- Build FAISS index ---
    index = faiss.IndexFlatL2(patch_dim)
    index.add(patch_vectors)
    print(f"FAISS index: {index.ntotal} vectors, dimension={index.d}")

    # --- Save outputs ---
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(output_dir, f"patches_{timestep}.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    coords_path = os.path.join(output_dir, f"patch_coords_{timestep}.npy")
    np.save(coords_path, patch_coords)
    print(f"Saved patch coordinates to {coords_path}")
    print(f"  patch_coords shape: {patch_coords.shape}  "
          f"(each row: [x_min, y_min, z_min, x_max, y_max, z_max])")

    return index, patch_coords


if __name__ == "__main__":
    # --- Configuration (must match train.py) ---
    res_x = 64  # grid points, not spaces)
    res_y = 64
    res_z = 128
    n_features = 2
    timestep = 0

    # Patch size (variable, default 4x4x4) (grid points, not spaces)
    patch_x = 4
    patch_y = 4
    patch_z = 4

    # parse timestep (positional) argument
    parser=argparse.ArgumentParser(description="vector database creation script")
    parser.add_argument("timestep", help="timestep")
    args = parser.parse_args()
    timestep = args.timestep

    build_patch_database(timestep=timestep,
        embedding_path=f"output/embedding_table_{timestep}.npy",
        output_dir="output",
        res_x=res_x,
        res_y=res_y,
        res_z=res_z,
        n_features=n_features,
        patch_x=patch_x,
        patch_y=patch_y,
        patch_z=patch_z,
    )
