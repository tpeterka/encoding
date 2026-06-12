"""
FAISS Vector Database for Raw Data Patches

Loads raw scalar data from a binary file (data/train_<timestep>.bin),
extracts overlapping 4x4x4 patches of data values, and stores them
in a FAISS vector database. Each patch vector contains one scalar value
per grid point (64 elements for a 4x4x4 patch).

The binary file format is rows of [x, y, z, value] (float32), written in
row-major order (x varies slowest, z varies fastest). Only the value
column is used; the spatial coordinates are recomputed from grid indices.

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


def build_patch_database(timestep=0,
    data_path="data/train_0.bin",
    output_dir="output",
    res_x=128,
    res_y=128,
    res_z=256,
    patch_x=4,
    patch_y=4,
    patch_z=4,
):
    """
    Build a FAISS index from overlapping 3D patches of raw scalar data.

    Parameters
    ----------
    data_path : str
        Path to the binary data file (.bin). Format: rows of
        [x, y, z, value] as float32, written in row-major order.
    output_dir : str
        Directory for output files.
    res_x, res_y, res_z : int
        Grid resolution along each axis.
    patch_x, patch_y, patch_z : int
        Patch size (in grid points) along each axis. Default 4x4x4.
    """
    # --- Load raw data from binary file ---
    raw = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)
    expected_rows = res_x * res_y * res_z
    assert raw.shape[0] == expected_rows, (
        f"Row count mismatch: got {raw.shape[0]}, expected {expected_rows}"
    )
    # Extract only the data values (column 3) and reshape to 3D grid
    data = raw[:, 3].reshape((res_x, res_y, res_z))
    print(f"Loaded data: path={data_path}, "
          f"shape={data.shape}, "
          f"min={data.min():.6f}, max={data.max():.6f}")

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

    # Dimension of each patch vector (one scalar per grid point)
    patch_dim = patch_x * patch_y * patch_z

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

                # Extract the 3D sub-block and flatten to a vector
                patch_vectors[patch_idx] = data[
                    ix_start:ix_start + patch_x,
                    iy_start:iy_start + patch_y,
                    iz_start:iz_start + patch_z,
                ].ravel()

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
    # --- Configuration ---
    res_x = 128  # grid points, not spaces
    res_y = 128
    res_z = 256

    # Patch size (variable, default 4x4x4) (grid points, not spaces)
    patch_x = 4
    patch_y = 4
    patch_z = 4

    # parse timestep (positional) argument
    parser = argparse.ArgumentParser(description="vector database creation script")
    parser.add_argument("timestep", help="timestep")
    args = parser.parse_args()
    timestep = args.timestep

    build_patch_database(timestep=timestep,
        data_path=f"data/train_{timestep}.bin",
        output_dir="output",
        res_x=res_x,
        res_y=res_y,
        res_z=res_z,
        patch_x=patch_x,
        patch_y=patch_y,
        patch_z=patch_z,
    )
