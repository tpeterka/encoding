"""
Author: Jianxin Sun
Email: sunjianxin66@gmail.com
Description:
    Data format conversion
"""

import numpy as np
from tqdm import tqdm
import vtk
from vtk.util import numpy_support

def saveToTrainingData(data, path, x_size, y_size, z_size):
    points = np.zeros((x_size*y_size*z_size, 4), dtype=np.float32)
    count = 0
    # for x in tqdm(range(x_size)):
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                points[count, 0] = x/(x_size - 1)*2 - 1 # scale to [-1, 1]
                points[count, 1] = y/(y_size - 1)*2 - 1 # scale to [-1, 1]
                points[count, 2] = z/(z_size - 1)*2 - 1 # scale to [-1, 1]
                points[count, 3] = data[x, y, z]
                count = count+1
    points.tofile(path)

def saveToTrainingData4D(data, path, time_size, x_size, y_size, z_size):
    points = np.zeros((time_size*x_size*y_size*z_size, 5), dtype=np.float32)
    count = 0
    # for x in tqdm(range(x_size)):
    for t in range(time_size):
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    points[count, 0] = t
                    points[count, 1] = x/(x_size - 1)*2 - 1 # scale to [-1, 1]
                    points[count, 2] = y/(y_size - 1)*2 - 1 # scale to [-1, 1]
                    points[count, 3] = z/(z_size - 1)*2 - 1 # scale to [-1, 1]
                    points[count, 4] = data[t, x, y, z]
                    count = count+1
    points.tofile(path)

def saveToTrainingDataThreshold(data, path, x_size, y_size, z_size, threshold):
    size = np.sum(data >= threshold)
    points = np.zeros((size, 4), dtype=np.float32)
    count = 0
    # for x in tqdm(range(x_size)):
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if (data[x, y, z] >= threshold):
                    points[count, 0] = x/(x_size - 1)*2 - 1 # scale to [-1, 1]
                    points[count, 1] = y/(y_size - 1)*2 - 1 # scale to [-1, 1]
                    points[count, 2] = z/(z_size - 1)*2 - 1 # scale to [-1, 1]
                    points[count, 3] = data[x, y, z]
                    count = count+1
    # print("feature sample size:", points.shape)
    points.tofile(path)

def saveToTrainingDataFeature(feature_xyzs, feature_values, path,  x_size, y_size, z_size):
    size = len(feature_xyzs)
    points = np.zeros((size, 4), dtype=np.float32)
    for i in range(size):
        points[i, 0] = feature_xyzs[i][0]/(x_size - 1)*2 - 1 # scale to [-1, 1]
        points[i, 1] = feature_xyzs[i][1]/(y_size - 1)*2 - 1 # scale to [-1, 1]
        points[i, 2] = feature_xyzs[i][2]/(z_size - 1)*2 - 1 # scale to [-1, 1]
        points[i, 3] = feature_values[i]
    points.tofile(path)


def npArray2Vtk(data, fileName, spacing_x, spacing_y, spacing_z):
    # Convert the numpy array to a VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    # Create a VTK image data object
    image_data = vtk.vtkImageData()

    # Set the dimensions of the image data (same as the shape of the numpy array)
    image_data.SetDimensions(data.shape)
    image_data.SetSpacing(spacing_x, spacing_y, spacing_z)
    
    #Set origin
    x_origin = (data.shape[0] - 1)*spacing_x/2
    y_origin = (data.shape[1] - 1)*spacing_y/2
    z_origin = (data.shape[2] - 1)*spacing_z/2
    image_data.SetOrigin(-x_origin, -y_origin, -z_origin)

    # Allocate scalars for the image data
    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Get the VTK array from the image data object and set its values to the converted numpy array
    vtk_array = image_data.GetPointData().GetScalars()
    vtk_array.DeepCopy(vtk_data_array)

    # Create a VTK writer to save the image data to a .vtk file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(fileName)
    writer.SetInputData(image_data)

    # Write the .vtk file
    writer.Write()

    # print("3D volume data has been saved to", fileName)