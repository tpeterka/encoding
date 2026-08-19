"""
Synthetic NumPy-to-VTK axis-mapping test.

Writes several small VTK image datasets containing a single nonzero marker so
it is easy to inspect in ParaView which NumPy axis convention matches the
expected rendered coordinates.
"""

import os
import argparse
import numpy as np
import vtk
from vtk.util import numpy_support


DEFAULT_SHAPE = (3, 4, 5)
DEFAULT_MARKER_INDEX = (1, 2, 3)
MARKER_VALUE = np.float32(1.0)


VARIANTS = (
    {
        "name": "shape_xyz_flat_C",
        "description": "SetDimensions(shape), no transpose, flatten C-order",
        "transform": lambda data: data,
        "dimensions": lambda arr: tuple(int(v) for v in arr.shape),
        "flatten_order": "C",
    },
    {
        "name": "shape_xyz_flat_F",
        "description": "SetDimensions(shape), no transpose, flatten F-order",
        "transform": lambda data: data,
        "dimensions": lambda arr: tuple(int(v) for v in arr.shape),
        "flatten_order": "F",
    },
    {
        "name": "transpose_zyx_flat_C",
        "description": "Transpose (z,y,x), SetDimensions(transposed shape), flatten C-order",
        "transform": lambda data: data.transpose(2, 1, 0),
        "dimensions": lambda arr: tuple(int(v) for v in arr.shape),
        "flatten_order": "C",
    },
    {
        "name": "transpose_zyx_flat_F",
        "description": "Transpose (z,y,x), SetDimensions(transposed shape), flatten F-order",
        "transform": lambda data: data.transpose(2, 1, 0),
        "dimensions": lambda arr: tuple(int(v) for v in arr.shape),
        "flatten_order": "F",
    },
    {
        "name": "shape_reversed_flat_C",
        "description": "SetDimensions(shape reversed), no transpose, flatten C-order",
        "transform": lambda data: data,
        "dimensions": lambda arr: tuple(int(v) for v in arr.shape[::-1]),
        "flatten_order": "C",
    },
    {
        "name": "shape_reversed_flat_F",
        "description": "SetDimensions(shape reversed), no transpose, flatten F-order",
        "transform": lambda data: data,
        "dimensions": lambda arr: tuple(int(v) for v in arr.shape[::-1]),
        "flatten_order": "F",
    },
)


def parse_triplet(text, flag_name):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{flag_name} must have exactly three comma-separated integers")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{flag_name} must contain only integers") from exc


def point_id_for_index(index, shape):
    ix, iy, iz = index
    nx, ny, nz = shape
    return ix + nx * (iy + ny * iz)


def write_vtk(data, dimensions, output_path, flatten_order):
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=data.ravel(order=flatten_order),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dimensions)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)
    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
    image_data.GetPointData().GetScalars().DeepCopy(vtk_data_array)

    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(image_data)
    writer.Write()


def build_base_volume(shape, marker_index):
    if any(idx < 0 for idx in marker_index):
        raise ValueError(f"Marker index must be non-negative, got {marker_index}")
    if any(idx >= dim for idx, dim in zip(marker_index, shape)):
        raise ValueError(
            f"Marker index {marker_index} is outside shape {shape}"
        )

    data = np.zeros(shape, dtype=np.float32)
    data[marker_index] = MARKER_VALUE
    return data


def run_axis_mapping_test(output_dir, shape=DEFAULT_SHAPE, marker_index=DEFAULT_MARKER_INDEX):
    os.makedirs(output_dir, exist_ok=True)
    base = build_base_volume(shape, marker_index)

    print("Synthetic VTK axis-mapping test")
    print(f"  base shape: {shape}")
    print(f"  marker NumPy index: {marker_index}")
    print(f"  marker flat point id if interpreted as x-fastest image data: {point_id_for_index(marker_index, shape)}")
    print(f"  output dir: {output_dir}")

    summary_lines = []
    for variant in VARIANTS:
        transformed = variant["transform"](base)
        dimensions = variant["dimensions"](transformed)
        output_path = os.path.join(output_dir, f"axis_test_{variant['name']}.vtk")
        write_vtk(
            transformed,
            dimensions=dimensions,
            output_path=output_path,
            flatten_order=variant["flatten_order"],
        )
        summary_lines.append(
            f"  {variant['name']}: {variant['description']} -> dimensions={dimensions}, file={output_path}"
        )

    print("Generated variants:")
    for line in summary_lines:
        print(line)
    print("Open these in ParaView and use Find Data or Spreadsheet View to locate the single point with value 1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write synthetic VTK files to diagnose NumPy-to-VTK axis mapping"
    )
    parser.add_argument(
        "--output-dir",
        default="output/axis_test",
        help="directory for generated .vtk files (default: output/axis_test)",
    )
    parser.add_argument(
        "--shape",
        default="3,4,5",
        help="base NumPy shape as x,y,z (default: 3,4,5)",
    )
    parser.add_argument(
        "--marker-index",
        default="1,2,3",
        help="marker NumPy index as x,y,z (default: 1,2,3)",
    )
    args = parser.parse_args()

    run_axis_mapping_test(
        output_dir=args.output_dir,
        shape=parse_triplet(args.shape, "--shape"),
        marker_index=parse_triplet(args.marker_index, "--marker-index"),
    )
