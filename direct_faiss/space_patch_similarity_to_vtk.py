"""
Convert patch-space scalar metadata to a VTK structured-points volume.

Reads the metadata .npz produced by patch_vector_norms.py, reconstructs a regular
3D patch-space array from patch lower-corner coordinates, and writes it as a VTK
image for visualization in ParaView.
"""

import os
import argparse
import numpy as np

from utility import np_array_to_vtk


def load_distance_metadata(metadata_file):
    print(f"Loading {metadata_file} ...")
    with np.load(metadata_file) as metadata:
        if "distances" not in metadata or "patch_coords" not in metadata:
            raise ValueError(
                f"Expected 'distances' and 'patch_coords' arrays in {metadata_file}"
            )
        distances = np.asarray(metadata["distances"], dtype=np.float32)
        patch_coords = np.asarray(metadata["patch_coords"], dtype=np.float64)

    if distances.ndim != 1:
        raise ValueError(f"Expected distances to be 1D, got shape {distances.shape}")
    if patch_coords.ndim != 2 or patch_coords.shape[1] != 6:
        raise ValueError(
            f"Expected patch_coords with shape (n, 6), got {patch_coords.shape}"
        )
    if distances.shape[0] != patch_coords.shape[0]:
        raise ValueError(
            f"Distance count {distances.shape[0]} does not match patch_coords row count {patch_coords.shape[0]}"
        )

    return distances, patch_coords


def build_patch_volume(distances, patch_coords):
    x_vals = np.unique(patch_coords[:, 0])
    y_vals = np.unique(patch_coords[:, 1])
    z_vals = np.unique(patch_coords[:, 2])

    shape = (len(x_vals), len(y_vals), len(z_vals))
    expected_count = int(np.prod(shape))
    if expected_count != patch_coords.shape[0]:
        raise ValueError(
            f"Patch coordinates do not form a complete regular patch grid: expected {expected_count} "
            f"patches from shape {shape}, found {patch_coords.shape[0]}"
        )

    x_index = {float(value): idx for idx, value in enumerate(x_vals.tolist())}
    y_index = {float(value): idx for idx, value in enumerate(y_vals.tolist())}
    z_index = {float(value): idx for idx, value in enumerate(z_vals.tolist())}

    volume = np.empty(shape, dtype=np.float32)
    filled = np.zeros(shape, dtype=bool)

    for distance, row in zip(distances, patch_coords):
        px = x_index[float(row[0])]
        py = y_index[float(row[1])]
        pz = z_index[float(row[2])]
        if filled[px, py, pz]:
            raise ValueError(
                f"Duplicate patch lower corner at {(float(row[0]), float(row[1]), float(row[2]))}"
            )
        volume[px, py, pz] = distance
        filled[px, py, pz] = True

    if not np.all(filled):
        missing = np.argwhere(~filled)[0]
        raise ValueError(
            f"Missing patch entry at patch-space index {tuple(int(v) for v in missing.tolist())}"
        )

    return volume, shape, x_vals, y_vals, z_vals


def default_output_path(metadata_file):
    stem, _ = os.path.splitext(metadata_file)
    return f"{stem}.vtk"


def convert_space_patch_similarity_to_vtk(metadata_file, output_path=None):
    distances, patch_coords = load_distance_metadata(metadata_file)
    volume, shape, x_vals, y_vals, z_vals = build_patch_volume(distances, patch_coords)

    if output_path is None:
        output_path = default_output_path(metadata_file)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np_array_to_vtk(volume, output_path, spacing_x=1.0, spacing_y=1.0, spacing_z=1.0)

    print(f"Patch-space dimensions: {shape}")
    print(
        "Lower-corner ranges: "
        f"x=[{float(x_vals[0]):.9g}, {float(x_vals[-1]):.9g}], "
        f"y=[{float(y_vals[0]):.9g}, {float(y_vals[-1]):.9g}], "
        f"z=[{float(z_vals[0]):.9g}, {float(z_vals[-1]):.9g}]"
    )
    print("Patch-space spacing: (1.0, 1.0, 1.0)")
    print(f"Saved VTK volume to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert patch-space scalar metadata to a VTK structured-points volume"
    )
    parser.add_argument("metadata_file", help="input .npz metadata file from patch_vector_norms.py")
    parser.add_argument("--output", help="output .vtk file path")
    args = parser.parse_args()

    convert_space_patch_similarity_to_vtk(
        args.metadata_file,
        output_path=args.output,
    )
