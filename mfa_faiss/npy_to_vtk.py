"""
Diagnostic converter from a NumPy .npy array to a VTK structured-points file.

This script is intentionally minimal. It reads the .npy array exactly as saved,
reshapes it to a hard-coded 3D volume, and writes a .vtk file without using any
patch-bound metadata. It is intended to compare raw ParaView import behavior
against direct VTK export.
"""

import os
import argparse
import numpy as np
import vtk
from vtk.util import numpy_support


DEFAULT_SHAPE = (21, 21, 42)


def parse_shape(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--shape must have exactly three comma-separated integers: x,y,z")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--shape must contain only integers: x,y,z") from exc
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"--shape must be positive in all dimensions, got {shape}")
    return shape


def default_output_path(npy_file):
    stem, _ = os.path.splitext(npy_file)
    return f"{stem}.vtk"


def np_array_to_vtk(data, file_name, spacing_x=1.0, spacing_y=1.0, spacing_z=1.0):
    """
    Convert a 3D numpy array to a VTK structured points file (.vtk).

    This diagnostic path uses C-order flattening to match the raw-image layout
    that was verified in ParaView for the saved .npy payload.
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


def convert_npy_to_vtk(npy_file, shape=DEFAULT_SHAPE, output_path=None):
    print(f"Loading {npy_file} ...")
    array = np.load(npy_file)
    expected_size = int(np.prod(shape))
    if array.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} values for shape {shape}, got {array.size}"
        )

    volume = np.asarray(array, dtype=np.float32).reshape(shape, order="C")

    if output_path is None:
        output_path = default_output_path(npy_file)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np_array_to_vtk(volume, output_path, spacing_x=1.0, spacing_y=1.0, spacing_z=1.0)

    print(f"Input array shape: {array.shape}")
    print(f"VTK volume shape: {volume.shape}")
    print(f"VTK scalar dtype: {volume.dtype}")
    print("Data reshape order: C-order")
    print("VTK flatten order: C-order")
    print("Origin: (0.0, 0.0, 0.0)")
    print("Spacing: (1.0, 1.0, 1.0)")
    print(f"Saved VTK file to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic converter from a NumPy .npy array to a VTK structured-points file"
    )
    parser.add_argument("npy_file", help="input .npy file")
    parser.add_argument(
        "--shape",
        default="21,21,42",
        help="volume shape as x,y,z (default: 21,21,42)",
    )
    parser.add_argument("--output", help="output .vtk file path")
    args = parser.parse_args()

    convert_npy_to_vtk(
        args.npy_file,
        shape=parse_shape(args.shape),
        output_path=args.output,
    )
