"""
Convert patch-space similarity distances to a VTK structured-points volume.

Reads the metadata .npz produced by space_patch_similarity.py, selects one gid,
constructs a regular 3D patch-space array of distances, and writes it as a
VTK image for visualization in ParaView.
"""

import os
import argparse
import numpy as np
import vtk
from vtk.util import numpy_support


def np_array_to_vtk(data, file_name, spacing_x=1.0, spacing_y=1.0, spacing_z=1.0):
    """
    Convert a 3D numpy array to a VTK structured points file (.vtk).
    """
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=data.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(data.shape)
    image_data.SetSpacing(spacing_x, spacing_y, spacing_z)

    image_data.SetOrigin(0.0, 0.0, 0.0)

    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
    vtk_array = image_data.GetPointData().GetScalars()
    vtk_array.DeepCopy(vtk_data_array)

    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(image_data)
    writer.Write()


def load_distance_metadata(metadata_file):
    print(f"Loading {metadata_file} ...")
    with np.load(metadata_file) as metadata:
        if "distances" not in metadata or "patch_bounds" not in metadata:
            raise ValueError(
                f"Expected 'distances' and 'patch_bounds' arrays in {metadata_file}"
            )
        distances = np.asarray(metadata["distances"], dtype=np.float32)
        patch_bounds = np.asarray(metadata["patch_bounds"], dtype=np.int64)

    if distances.ndim != 1:
        raise ValueError(f"Expected distances to be 1D, got shape {distances.shape}")
    if patch_bounds.ndim != 2 or patch_bounds.shape[1] != 7:
        raise ValueError(
            f"Expected patch_bounds with shape (n, 7), got {patch_bounds.shape}"
        )
    if distances.shape[0] != patch_bounds.shape[0]:
        raise ValueError(
            f"Distance count {distances.shape[0]} does not match patch_bounds row count {patch_bounds.shape[0]}"
        )

    return distances, patch_bounds


def build_patch_volume(distances, patch_bounds, gid):
    gid = int(gid)
    gid_mask = patch_bounds[:, 0] == gid
    if not np.any(gid_mask):
        available_gids = np.unique(patch_bounds[:, 0]).tolist()
        raise ValueError(
            f"No patches found for gid={gid}; available gids: {available_gids}"
        )

    distances_gid = distances[gid_mask]
    bounds_gid = patch_bounds[gid_mask]

    ix_vals = np.unique(bounds_gid[:, 1])
    iy_vals = np.unique(bounds_gid[:, 2])
    iz_vals = np.unique(bounds_gid[:, 3])

    shape = (len(ix_vals), len(iy_vals), len(iz_vals))
    expected_count = int(np.prod(shape))
    if expected_count != bounds_gid.shape[0]:
        raise ValueError(
            f"gid={gid} does not form a complete regular patch grid: expected {expected_count} "
            f"patches from shape {shape}, found {bounds_gid.shape[0]}"
        )

    x_index = {int(value): idx for idx, value in enumerate(ix_vals.tolist())}
    y_index = {int(value): idx for idx, value in enumerate(iy_vals.tolist())}
    z_index = {int(value): idx for idx, value in enumerate(iz_vals.tolist())}

    volume = np.empty(shape, dtype=np.float32)
    filled = np.zeros(shape, dtype=bool)

    for distance, row in zip(distances_gid, bounds_gid):
        px = x_index[int(row[1])]
        py = y_index[int(row[2])]
        pz = z_index[int(row[3])]
        if filled[px, py, pz]:
            raise ValueError(
                f"Duplicate patch lower bound for gid={gid} at {(int(row[1]), int(row[2]), int(row[3]))}"
            )
        volume[px, py, pz] = distance
        filled[px, py, pz] = True

    if not np.all(filled):
        missing = np.argwhere(~filled)[0]
        raise ValueError(
            f"Missing patch entry for gid={gid} at patch-space index {tuple(int(v) for v in missing.tolist())}"
        )

    return volume, shape, ix_vals, iy_vals, iz_vals


def default_output_path(metadata_file, gid):
    stem, _ = os.path.splitext(metadata_file)
    return f"{stem}_gid{int(gid)}.vtk"


def convert_space_patch_similarity_to_vtk(metadata_file, gid=0, output_path=None):
    distances, patch_bounds = load_distance_metadata(metadata_file)
    volume, shape, ix_vals, iy_vals, iz_vals = build_patch_volume(distances, patch_bounds, gid)

    if output_path is None:
        output_path = default_output_path(metadata_file, gid)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np_array_to_vtk(volume, output_path, spacing_x=1.0, spacing_y=1.0, spacing_z=1.0)

    print(f"gid={int(gid)} patch-space dimensions: {shape}")
    print(
        "Lower-bound ranges: "
        f"x=[{int(ix_vals[0])}, {int(ix_vals[-1])}], "
        f"y=[{int(iy_vals[0])}, {int(iy_vals[-1])}], "
        f"z=[{int(iz_vals[0])}, {int(iz_vals[-1])}]"
    )
    print("Patch-space spacing: (1.0, 1.0, 1.0)")
    print(f"Saved VTK volume to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert patch-space similarity metadata to a VTK structured-points volume"
    )
    parser.add_argument("metadata_file", help="input .npz metadata file from space_patch_similarity.py")
    parser.add_argument("--gid", type=int, default=0, help="gid to extract (default: 0)")
    parser.add_argument("--output", help="output .vtk file path")
    args = parser.parse_args()

    convert_space_patch_similarity_to_vtk(
        args.metadata_file,
        gid=args.gid,
        output_path=args.output,
    )
