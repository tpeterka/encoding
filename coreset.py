"""
Author: Jianxin Sun
Email: sunjianxin66@gmail.com
Description:
    Coreset Selection
"""

import json
import numpy as np
from tqdm import tqdm
from utility import npArray2Vtk, saveToTrainingData

def getArgonBubbleData():
    '''
    Load the Argon Bubble dataset
    '''
    timestep_key = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    x_size = 128
    y_size = 128
    z_size = 256
    t_size = len(timestep_key)
    data = np.zeros((t_size, x_size, y_size, z_size))

    # Load key frames
    for i in range(t_size):
        filePath = "data/argon_bubble/rescaled2_" + str(timestep_key[i]) + ".dat"
        f = open(filePath, "r")
        data_cur = np.fromfile(f, dtype=np.float32)
        data_cur = np.reshape(data_cur, (128, 128, 256), order='F')
        data[i, :, :, :] = data_cur
    
    # Normalization
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    data = data.astype('float32') 
    print("min, max", np.min(data), np.max(data))

    return data

def save2Vtk(path, datas):
    '''
    Save raw volume into .vtk file for visualization
    path: folder path to save to
    datas: list of volumes
    '''
    spacing = 1
    t_size, _, _, _ = datas.shape
    for i in tqdm(range(t_size)):
        filePath = path + "/" + str(i) + ".vtk"
        npArray2Vtk(datas[i], filePath, spacing, spacing, spacing)

def save2Bin(path, datas):
    '''
    Save raw volume into .bin file for training
    path: folder path to save to
    datas: list of volumes
    '''
    t_size, x_size, y_size, z_size = datas.shape
    for i in tqdm(range(t_size)):
        filePath = path + "/" + str(i) + ".bin"
        saveToTrainingData(datas[i], filePath, x_size, y_size, z_size)


def getSegmentFeatures(data, threshold, t_size, x_size, y_size, z_size):
    '''
    Find the x, y, z indices of all points in the feaure boundary [ >= threshold ]
    Find the values of all points in the feature boundary
    Find the lower and upper bounds of x, y, z dimension
    data: list of 3d volume sequence across time step
    t_size: number of time step
    x/y/z_size: size of x, y, z dimension of volume
    '''
    
    x_lowers = []
    x_uppers = []
    y_lowers = []
    y_uppers = []
    z_lowers = []
    z_uppers = []

    masks = []
    masks_dilation = []
    
    for t in tqdm(range(t_size)): # 9
        data_cur = data[t]
        x_lower = 1000
        x_upper = 0
        y_lower = 1000
        y_upper = 0
        z_lower = 1000
        z_upper = 0
    
        mask = np.zeros((x_size, y_size, z_size), dtype=np.int32)
        mask_dilation = np.zeros((x_size, y_size, z_size), dtype=np.int32)
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    if (data_cur[x, y, z] >= threshold):
                        mask[x, y, z] = 1
                        # adding dilation
                        xs = [x - 1, x, x + 1]
                        ys = [y - 1, y, y + 1]
                        zs = [z - 1, z, z + 1]
                        for x_n in xs:
                            for y_n in ys:
                                for z_n in zs:
                                    if ((x_n >= 0 and x_n <= x_size - 1) and (y_n >= 0 and y_n <= y_size - 1) and (z_n >= 0 and z_n <= z_size - 1)):
                                        if (x_n < x_lower):
                                            x_lower = x_n
                                        if (x_n > x_upper):
                                            x_upper = x_n
                                        if (y_n < y_lower):
                                            y_lower = y_n
                                        if (y_n > y_upper):
                                            y_upper = y_n
                                        if (z_n < z_lower):
                                            z_lower = z_n
                                        if (z_n > z_upper):
                                            z_upper = z_n
                                        mask_dilation[x_n, y_n, z_n] = 1

        x_lowers.append(x_lower)
        x_uppers.append(x_upper)
        y_lowers.append(y_lower)
        y_uppers.append(y_upper)
        z_lowers.append(z_lower)
        z_uppers.append(z_upper)
        masks.append(mask)
        masks_dilation.append(mask_dilation)

    return x_lowers, x_uppers, y_lowers, y_uppers, z_lowers, z_uppers, masks, masks_dilation

def getTrainingDataFormate2(data,
                            x_lowers,
                            x_uppers,
                            y_lowers,
                            y_uppers,
                            z_lowers,
                            z_uppers,
                            global_x_size,
                            global_y_size,
                            global_z_size):
    '''
    Create training data pair (x, y, z) -> value
    data: list of 3d volume sequence across time step
    x/y/z_lowers: lower bound on x/y/z dimension
    x/y/z_uppers: upper bound on x/y/z dimension
    global_x/y/z_size: original volume dimension
    '''

    # Temporal Fusion: find the dimension of Feature Bounding Box (FBB)
    x_lower_all = min(x_lowers)
    x_upper_all = max(x_uppers)
    y_lower_all = min(y_lowers)
    y_upper_all = max(y_uppers)
    z_lower_all = min(z_lowers)
    z_upper_all = max(z_uppers)

    all_x_size = x_upper_all - x_lower_all + 1
    all_y_size = y_upper_all - y_lower_all + 1
    all_z_size = z_upper_all - z_lower_all + 1   
    x_all_interval = 2/(all_x_size - 1)
    y_all_interval = 2/(all_y_size - 1)
    z_all_interval = 2/(all_z_size - 1)
    
    xyzs_global_list = []
    xyzs_local_list = []
    vs_list = []
    t_size = len(x_lowers)
    x_global_interval = 2/(global_x_size - 1)
    y_global_interval = 2/(global_y_size - 1)
    z_global_interval = 2/(global_z_size - 1)

    for t in range(t_size):
        data_cur = data[t]
        x_lower = x_lowers[t]
        x_upper = x_uppers[t]
        y_lower = y_lowers[t]
        y_upper = y_uppers[t]
        z_lower = z_lowers[t]
        z_upper = z_uppers[t]
        
        f_x_size = x_upper - x_lower + 1
        f_y_size = y_upper - y_lower + 1
        f_z_size = z_upper - z_lower + 1

        # Get the global positions
        size = f_x_size*f_y_size*f_z_size
        xyzs_global = np.zeros((size, 4), dtype=np.float32)
        vs = np.zeros((size, 1), dtype=np.float32)
        count = 0
        for x in range(f_x_size):
            for y in range(f_y_size):
                for z in range(f_z_size):
                    x_global = x_lower + x
                    y_global = y_lower + y
                    z_global = z_lower + z
                    x_pos = x_global*x_global_interval - 1
                    y_pos = y_global*y_global_interval - 1
                    z_pos = z_global*z_global_interval - 1
                    xyzs_global[count, 0] = t
                    xyzs_global[count, 1] = x_pos
                    xyzs_global[count, 2] = y_pos
                    xyzs_global[count, 3] = z_pos
                    vs[count, 0] = data_cur[x_global, y_global, z_global]
                    count += 1
        xyzs_global_list.append(xyzs_global)
        vs_list.append(vs)

        # Position Translation: get the local positions
        xyzs_local = np.zeros((size, 4), dtype=np.float32)
        count = 0
        for x in range(f_x_size):
            for y in range(f_y_size):
                for z in range(f_z_size):
                    x_all = x_lower - x_lower_all + x
                    y_all = y_lower - y_lower_all + y
                    z_all = z_lower - z_lower_all + z
                    x_pos = x_all*x_all_interval - 1
                    y_pos = y_all*y_all_interval - 1
                    z_pos = z_all*z_all_interval - 1
                    xyzs_local[count, 0] = t
                    xyzs_local[count, 1] = x_pos
                    xyzs_local[count, 2] = y_pos
                    xyzs_local[count, 3] = z_pos
                    count += 1
        xyzs_local_list.append(xyzs_local)

    return xyzs_global_list, xyzs_local_list, vs_list, x_lower_all, x_upper_all, y_lower_all, y_upper_all, z_lower_all, z_upper_all

def divide(r, res):
    '''
    Input encoding grid resolution divider. fold = 2
    r: current resolution
    res: resolution list
    '''
    res.append(r)
    if (r == 2):
        return
    if (r%2 == 0): # even number
        divide(r/2, res)
    else:
        r += 1
        divide(r/2, res)

def getMultiRes(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, t_lower, t_upper):
    '''
    Calculate the spatial multi-resolution setting
    x/y/z_lower: lower bound on x/y/z dimension
    x/y/z_upper: upper bound on x/y/z dimension
    '''
    x_diff = x_upper - x_lower + 1
    y_diff = y_upper - y_lower + 1
    z_diff = z_upper - z_lower + 1
    t_diff = t_upper - t_lower + 1

    x_ress = []
    divide(x_diff, x_ress)
    x_len = len(x_ress)

    y_ress = []
    divide(y_diff, y_ress)
    y_len = len(y_ress)

    z_ress = []
    divide(z_diff, z_ress)
    z_len = len(z_ress)

    t_ress = []
    divide(t_diff, t_ress)
    t_len = len(t_ress)

    # find the size of the shortest list
    length = min(x_len, y_len, z_len)

    if t_len < length:
        padding_len = length - t_len
        for i in range(padding_len):
            t_ress.append(2)
    else:
        t_ress = t_ress[0:length]
    # print(length)
    
    return x_ress[0:length], y_ress[0:length], z_ress[0:length], t_ress

if __name__ == "__main__":
    # Load key frames of Argon Bubble dataset
    datas = getArgonBubbleData()
    # Save data into .vtk for visualization
    print("Save frames as .vtk files")
    save2Vtk("data/argon_bubble/argon_128x128x256", datas)
    # Save data into .bin for training
    print("Save frames as .bin files")
    save2Bin("data/argon_bubble/argon_128x128x256", datas)

    threshold = 0.09
    t_size, x_size, y_size, z_size = datas.shape
    print(t_size, x_size, y_size, z_size)

    # Find the boundary of the bounding box
    print("Find boundary")
    x_lowers, x_uppers, y_lowers, y_uppers, z_lowers, z_uppers, masks, masks_dialation = getSegmentFeatures(datas,
                                                                                                            threshold,
                                                                                                            t_size,
                                                                                                            x_size,
                                                                                                            y_size,
                                                                                                            z_size)
                                                                                                            
    print("bounding:", x_lowers, x_uppers)
    print("bounding:", y_lowers, y_uppers)
    print("bounding:", z_lowers, z_uppers)

    for t in range(t_size):
        x_lower = x_lowers[t]
        x_upper = x_uppers[t]
        y_lower = y_lowers[t]
        y_upper = y_uppers[t]
        z_lower = z_lowers[t]
        z_upper = z_uppers[t]
        mask = masks[t][x_lower:x_upper + 1, y_lower:y_upper + 1, z_lower:z_upper + 1]
        print("Bounding box size:", mask.shape)
    
    print("Temporal Fusion for FBB")
    yzs_global_list, xyzs_local_list, vs_list, x_lower_all, x_upper_all, y_lower_all, y_upper_all, z_lower_all, z_upper_all = getTrainingDataFormate2(datas,
                                                                                                                                                      x_lowers,
                                                                                                                                                      x_uppers,
                                                                                                                                                      y_lowers,
                                                                                                                                                      y_uppers,
                                                                                                                                                      z_lowers,
                                                                                                                                                      z_uppers,
                                                                                                                                                      128,
                                                                                                                                                      128,
                                                                                                                                                      256)
    print("FBB dimension:")
    print("x range: [", x_lower_all, ", ", x_upper_all, "] ", x_upper_all - x_lower_all + 1)
    print("y range: [", y_lower_all, ", ", y_upper_all, "] ", y_upper_all - y_lower_all + 1)
    print("z range: [", z_lower_all, ", ", z_upper_all, "] ", z_upper_all - z_lower_all + 1)

    # Save training data
    for t in range(t_size):
        xyzs = xyzs_local_list[t]
        vs = vs_list[t]
        # Save meta data of each key frame
        sample_size = xyzs.shape[0]
        print(sample_size)
        path = "data/argon_bubble/argon_128x128x256/feature_local/" + str(t) + ".txt"
        # Save training data of each key frame
        with open(path, "w") as f:
            f.write(str(sample_size))
        points = np.hstack((xyzs, vs))
        path = "data/argon_bubble/argon_128x128x256/feature_local/" + str(t) + ".bin"
        points.tofile(path)

    # Calculate the multi-resolution setting for F-Hash
    t_lower_all = 0
    t_upper_all = t_size - 1
    x_res, y_res, z_res, t_res = getMultiRes(x_lower_all, x_upper_all, y_lower_all, y_upper_all, z_lower_all, z_upper_all, t_lower_all, t_upper_all)
    # float to int
    x_res = [int(x) for x in x_res]
    y_res = [int(y) for y in y_res]
    z_res = [int(z) for z in z_res]
    t_res = [int(t) for t in t_res]
    # reverst the order
    x_res.reverse()
    y_res.reverse()
    z_res.reverse()
    t_res.reverse()
    print(x_res)
    print(y_res)
    print(z_res)
    print(t_res)
    # Save the multi-resolution setting for F-Hash
    resolutions = [x_res, y_res, z_res, t_res]
    with open("data/argon_bubble/argon_128x128x256/feature_local/resolutions.json", "w") as f:
        json.dump(resolutions, f)
    # Save the size of bounding box of each key frame
    bounding = [x_lowers, x_uppers, y_lowers, y_uppers, z_lowers, z_uppers]
    with open("data/argon_bubble/argon_128x128x256/feature_local/bounding.json", "w") as f:
        json.dump(bounding, f)
