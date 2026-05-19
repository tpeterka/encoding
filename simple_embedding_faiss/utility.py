"""
Simplified F-Hash: Utility functions
Data format conversion and VTK output
"""

import numpy as np
import vtk
from vtk.util import numpy_support


def save_training_data(data, path, x_size, y_size, z_size):
    """
    Convert a 3D numpy volume into a binary training file.
    Each voxel produces a row: [x_norm, y_norm, z_norm, value] (float32).
    Spatial coordinates are scaled to [-1, 1].

    data: 3D numpy array of shape (x_size, y_size, z_size)
    path: output binary file path
    x_size, y_size, z_size: volume dimensions
    """
    n_samples = x_size * y_size * z_size
    points = np.zeros((n_samples, 4), dtype=np.float32)

    count = 0
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                points[count, 0] = x / (x_size - 1) * 2 - 1  # scale to [-1, 1]
                points[count, 1] = y / (y_size - 1) * 2 - 1
                points[count, 2] = z / (z_size - 1) * 2 - 1
                points[count, 3] = data[x, y, z]
                count += 1

    points.tofile(path)
    return n_samples


def np_array_to_vtk(data, file_name, spacing_x=1, spacing_y=1, spacing_z=1):
    """
    Convert a 3D numpy array to a VTK structured points file (.vtk).

    data: 3D numpy array
    file_name: output .vtk file path
    spacing_x/y/z: voxel spacing in each dimension
    """
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT
    )

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(data.shape)
    image_data.SetSpacing(spacing_x, spacing_y, spacing_z)

    # Set origin at center of volume
    x_origin = (data.shape[0] - 1) * spacing_x / 2
    y_origin = (data.shape[1] - 1) * spacing_y / 2
    z_origin = (data.shape[2] - 1) * spacing_z / 2
    image_data.SetOrigin(-x_origin, -y_origin, -z_origin)

    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

    vtk_array = image_data.GetPointData().GetScalars()
    vtk_array.DeepCopy(vtk_data_array)

    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(image_data)
    writer.Write()
