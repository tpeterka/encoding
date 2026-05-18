"""
Author: Jianxin Sun
Email: sunjianxin66@gmail.com
Description:
    Test accuracy of the F-hash encoded INR
"""

import numpy as np
import json
import torch
import torch.nn as nn
import fhash
import time
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from utility import npArray2Vtk

class MyTrainDataset(Dataset):
    def __init__(self, path, t_size, time_step):
        # data loading 
        self.size = 0       
        path_meta = path + "/" + str(time_step) + ".txt"
        with open(path_meta, "r") as f:
            self.size = int(f.read())
        path_data = path + "/" + str(time_step) + ".bin"
        with open(path_data, "rb") as f:
            # Read the data as float32 values
            data_back = np.fromfile(f, dtype=np.float32).reshape(self.size, 5)
        
        self.x = torch.empty((self.size, 4), dtype=torch.float32)
        self.y = torch.empty((self.size, 1), dtype=torch.float32)
        self.x = torch.from_numpy(data_back[:, 0:4])
        self.y = torch.from_numpy(data_back[:, 4]).unsqueeze(1) # from (n,) to (n, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size

def getPredict(model_back, check_point_name, train_x, num_batch):
    '''
    Inference the INR for predicted value
    '''
    model_back.load_state_dict(torch.load(check_point_name, weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model_back.to(device)

    size = train_x.shape[0]
    batch_size = int(size/num_batch)
    predict_list = []
    for i in range(num_batch):
        start = batch_size*i
        end = batch_size*(i + 1)
        inputs = train_x[start:end, :]
        inputs = inputs.to(device)
        model_back.eval()
        with torch.no_grad():
            predict = model_back(inputs)
        predict_list.append(predict)
    # left over
    if (size - batch_size*num_batch > 0):
        start = batch_size*num_batch
        end = size
        inputs = train_x[start:end, :]
        inputs = inputs.to(device)
        model_back.eval()
        with torch.no_grad():
            predict = model_back(inputs)
        predict_list.append(predict)

    predict = predict_list[0]
    predict_list = predict_list[1:]
    for i in range(len(predict_list)):
        predict = torch.cat((predict, predict_list[i]), dim=0)

    model_back.to('cpu')
    return predict

def getAllFromLocal(data, x_lower, y_lower, z_lower, x_all_size, y_all_size, z_all_size):
    '''
    Construct the volume in original dimension from the prediction
    '''
    x_size, y_size, z_size = data.shape
    result = np.zeros((x_all_size, y_all_size, z_all_size), dtype=np.float32)
    x_start = x_lower
    x_end = x_lower + x_size
    y_start = y_lower
    y_end = y_lower + y_size
    z_start = z_lower
    z_end = z_lower + z_size
    result[x_start:x_end, y_start:y_end, z_start:z_end] = data
    return result

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("on:", device)

    # Load the training data
    t_size = 9
    time_step = 0
    train_dataset = MyTrainDataset("data/argon_bubble/argon_128x128x256/feature_local", t_size, time_step)
    print("training data size:", train_dataset.size)
    outputs = train_dataset.y
    outputs = outputs.to(device)

    # Load the multi-resolution setting for F-Hash
    with open("data/argon_bubble/argon_128x128x256/feature_local/resolutions.json") as f:
        x_res, y_res, z_res, t_res = json.load(f)
    # Load the size of bounding box of each key frame
    with open("data/argon_bubble/argon_128x128x256/feature_local/bounding.json") as f:
        x_lowers, x_uppers, y_lowers, y_uppers, z_lowers, z_uppers = json.load(f)
    print("bounding:", x_lowers, x_uppers)
    print("bounding:", y_lowers, y_uppers)
    print("bounding:", z_lowers, z_uppers)

    multi_res_level_num = len(x_res)
    print("multi resolution level:", multi_res_level_num)
    model_back = fhash.tesseract(num_levels=multi_res_level_num,
                                 time_resolutions = t_res, 
                                 x_resolutions = x_res,
                                 y_resolutions = y_res,
                                 z_resolutions = z_res)

    # Only check time step 0 here
    mses = []
    start_time = time.time()
    criterion = nn.MSELoss()
    spacing = 1
    for i in tqdm(range(120)):
        check_point_name = "models/argon_128x128x256/f_hash/" + str(i) + "_epoches.pt"
        predicted = getPredict(model_back, check_point_name, train_dataset.x, 2)
        mse = criterion(predicted, outputs).item()
        mses.append(mse)
        # save prediction while training as vtk file for evaluation
        x_size = x_uppers[time_step] - x_lowers[time_step] + 1
        y_size = y_uppers[time_step] - y_lowers[time_step] + 1
        z_size = z_uppers[time_step] - z_lowers[time_step] + 1
        predicted = predicted.cpu().numpy()
        predicted = predicted.reshape(x_size, y_size, z_size)
        data = getAllFromLocal(predicted, x_lowers[time_step], y_lowers[time_step], z_lowers[time_step], 128, 128, 256)
        path = "data/argon_bubble/argon_128x128x256_predict/f_hash/timestep_" + str(time_step) + "/" + str(i) + ".vtk"
        npArray2Vtk(data, path, spacing, spacing, spacing)
    end_time = time.time()
    execution_time = end_time - start_time  # Compute duration
    print(f"Execution time: {execution_time:.6f} seconds")
    print("MSE Loss:", mses)

    # Loss plot
    plt.figure()
    plt.plot(mses, label="F-Hash")
    plt.xlabel("Epoches")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout
    plt.legend(loc='upper right')
    plt.savefig("mse_loss.png", dpi=300)