"""
Generate synthetic patch-space distance data for debugging.

Writes a flat .npy distance array and a matching .npz metadata file using the
same row ordering as patch_data.py: px outermost, then py, then pz fastest.
The field values are simple and controlled so downstream reconstruction and VTK
conversion can be validated independently of MFA/FAISS logic.
"""

import os
import argparse
import numpy as np


DEFAULT_SHAPE = (3, 4, 5)
DEFAULT_ZERO_INDEX = (1, 2, 3)
DEFAULT_GID = 0


def parse_triplet(text, flag_name):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{flag_name} must have exactly three comma-separated integers")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{flag_name} must contain only integers") from exc


def build_synthetic_arrays(shape, zero_index, gid):
    nx, ny, nz = shape
    zx, zy, zz = zero_index
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Shape must be positive in all dimensions, got {shape}")
    if not (0 <= zx < nx and 0 <= zy < ny and 0 <= zz < nz):
        raise ValueError(f"Zero index {zero_index} is outside shape {shape}")

    n_total = nx * ny * nz
    distances = np.ones(n_total, dtype=np.float32)
    patch_bounds = np.empty((n_total, 7), dtype=np.int64)

    flat_index = 0
    reference_index = None
    reference_bounds_row = None
    for px in range(nx):
        for py in range(ny):
            for pz in range(nz):
                patch_bounds[flat_index] = [gid, px, py, pz, px, py, pz]
                if (px, py, pz) == zero_index:
                    distances[flat_index] = 0.0
                    reference_index = flat_index
                    reference_bounds_row = patch_bounds[flat_index].copy()
                flat_index += 1

    assert flat_index == n_total
    if reference_index is None or reference_bounds_row is None:
        raise RuntimeError("Failed to assign synthetic reference patch")

    return distances, patch_bounds, reference_index, reference_bounds_row


def default_stem(output_dir, shape, zero_index):
    return os.path.join(
        output_dir,
        f"synthetic_space_patch_{shape[0]}x{shape[1]}x{shape[2]}_zero_{zero_index[0]}_{zero_index[1]}_{zero_index[2]}",
    )


def generate_synthetic_space_patch_data(output_dir, shape, zero_index, gid=DEFAULT_GID):
    os.makedirs(output_dir, exist_ok=True)

    distances, patch_bounds, reference_index, reference_bounds_row = build_synthetic_arrays(
        shape, zero_index, gid
    )

    stem = default_stem(output_dir, shape, zero_index)
    npy_path = f"{stem}.npy"
    npz_path = f"{stem}.npz"

    np.save(npy_path, distances)
    np.savez(
        npz_path,
        distances=distances,
        patch_bounds=patch_bounds,
        reference_index=np.int64(reference_index),
        reference_point=np.asarray((gid, zero_index[0], zero_index[1], zero_index[2]), dtype=np.int64),
        reference_bounds_row=reference_bounds_row,
        timestep=np.asarray("synthetic"),
        resolution=np.asarray(f"{shape[0]}x{shape[1]}x{shape[2]}"),
        mode=np.asarray("synthetic"),
    )

    print("Generated synthetic patch-space data")
    print(f"  shape: {shape}")
    print(f"  gid: {int(gid)}")
    print(f"  zero patch-space index: {zero_index}")
    print(f"  zero flat index: {int(reference_index)}")
    print(f"  output .npy: {npy_path}")
    print(f"  output .npz: {npz_path}")

    return npy_path, npz_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic patch-space .npy and .npz files for debugging"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="directory for generated files (default: output)",
    )
    parser.add_argument(
        "--shape",
        default="3,4,5",
        help="synthetic patch-space shape as x,y,z (default: 3,4,5)",
    )
    parser.add_argument(
        "--zero-index",
        default="1,2,3",
        help="patch-space index to set to zero as x,y,z (default: 1,2,3)",
    )
    parser.add_argument(
        "--gid",
        type=int,
        default=0,
        help="gid to store in synthetic patch bounds (default: 0)",
    )
    args = parser.parse_args()

    generate_synthetic_space_patch_data(
        output_dir=args.output_dir,
        shape=parse_triplet(args.shape, "--shape"),
        zero_index=parse_triplet(args.zero_index, "--zero-index"),
        gid=args.gid,
    )
