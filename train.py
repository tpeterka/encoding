"""
Author: Jianxin Sun
Email: sunjianxin66@gmail.com
Description:
    Training the INR through F-Hash
"""

import numpy as np
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import fhash

class MyTrainDataset(Dataset):
    def __init__(self, path, t_size):
        # data loading
        self.feature_sizes = []
        datas_back = []
        for t in range(t_size):
            path_meta = path + "/" + str(t) + ".txt"
            with open(path_meta, "r") as f:
                feature_size = int(f.read())
                print(feature_size)
                self.feature_sizes.append(feature_size)
            path_data = path + "/" + str(t) + ".bin"
            with open(path_data, "rb") as f:
                # Read the data as float32 values
                data_back = np.fromfile(f, dtype=np.float32).reshape(feature_size, 5)
                datas_back.append(data_back)
        data_back_all = np.vstack(datas_back)
        
        self.size = sum(self.feature_sizes)
        self.x = torch.empty((self.size, 4), dtype=torch.float32)
        self.y = torch.empty((self.size, 1), dtype=torch.float32)
        self.x = torch.from_numpy(data_back_all[:, 0:4])
        self.y = torch.from_numpy(data_back_all[:, 4]).unsqueeze(1) # from (n,) to (n, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size

if __name__ == "__main__":

    # Load the training data
    t_size = 9
    train_dataset = MyTrainDataset("data/argon_bubble/argon_128x128x256/feature_local", t_size)
    print("training data size:", train_dataset.size)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 2000000, shuffle = True, num_workers=16, pin_memory=True)

    # Load the multi-resolution setting for F-Hash
    with open("data/argon_bubble/argon_128x128x256/feature_local/resolutions.json") as f:
        x_res, y_res, z_res, t_res = json.load(f)
    print(x_res)
    print(y_res)
    print(z_res)
    print(t_res)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("on:", device)

    multi_res_level_num = len(x_res)
    print("multi resolution level:", multi_res_level_num)
    model = fhash.tesseract(num_levels=multi_res_level_num,
                            time_resolutions = t_res, 
                            x_resolutions = x_res,
                            y_resolutions = y_res,
                            z_resolutions = z_res)
    model.to(device)
    criterion = nn.MSELoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # Network info printout
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data type: {param.dtype}")
    total_params = sum(p.numel() for p in model.parameters())
    # Display the number of parameters
    print(f"Total number of parameters: {total_params}")
    # Optional: Show trainable and non-trainable parameters separately
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    print(f"{'Layer':<20} {'Parameters':<15}")
    print("=" * 35)
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name = name.split('.')[0]  # Extract the layer name
            param_count = param.numel()
            print(f"{layer_name:<20} {param_count:<15}")
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 35)
    print(f"{'Total Parameters':<20} {total_params:<15}")

    # Training
    num_epochs = 120
    train_loss_list = []
    test_loss_list = []

    best_test_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10
    best_epoch = 0

    start_time_total = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        train_batch = 0
    #     total_test_loss = 0.0
    #     test_batch = 0
        for i, (inputs, outputs) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            # forward|
            predicted = model(inputs)
            loss = criterion(predicted, outputs)
            total_train_loss += loss.data.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_batch = i + 1
            # if (i%1) == 0:
            #     print(f"epoch {epoch + 1}/{num_epochs}, step {i + 1}/{4}, loss = {loss.item():.4f}")
        train_loss_list.append(total_train_loss/train_batch)
        if (epoch%1== 0):
            check_point_name = "models/argon_128x128x256/f_hash/" + str(epoch) + "_epoches.pt"
            torch.save(model.state_dict(), check_point_name)
            print(f'epoch: {epoch}, train_loss: {total_train_loss/train_batch}')
        end_time = time.time()
        execution_time = end_time - start_time  # Compute duration
        print(f"Execution time: {execution_time:.6f} seconds")
    end_time_total = time.time()
    execution_time_total = end_time_total - start_time_total  # Compute duration
    print(f"Execution time total: {execution_time_total:.6f} seconds")                                                                                  

