"""
Simplified F-Hash: Training
Trains a single-level 3D hash encoding + MLP on the full volume.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import SimpleHash
import argparse

class VolumeDataset(Dataset):
    """
    Loads training data from the binary file produced by prepare_data.py.
    Each sample is (xyz, value) where xyz is a 3D coordinate in [-1,1]^3 and value is a scalar in [0,1]
    """

    def __init__(self, data_dir="data", timestep=0):
        meta_path = os.path.join(data_dir, f"train_{timestep}.txt")
        bin_path = os.path.join(data_dir, f"train_{timestep}.bin")

        with open(meta_path, "r") as f:
            self.size = int(f.read())

        with open(bin_path, "rb") as f:
            data = np.fromfile(f, dtype=np.float32).reshape(self.size, 4)

        self.x = torch.from_numpy(data[:, 0:3])   # xyz coordinates
        self.y = torch.from_numpy(data[:, 3:4])    # scalar value

        print(f"Loaded {self.size} training samples")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size


if __name__ == "__main__":
    # --- Configuration ---
# size of feature grid for hash embedding (can be smaller than data, full size of data is 128x128x256
    res_x = 128         # grid resolution (x)
    res_y = 128         # grid resolution (y)
    res_z = 256         # grid resolution (z)
#     res_x = 64          # grid resolution (x)
#     res_y = 64          # grid resolution (y)
#     res_z = 128         # grid resolution (z)
    n_features = 2      # features per hash table entry
    hidden_dim = 64     # MLP hidden layer width
    batch_size = 2_000_000
    learning_rate = 0.01
    num_epochs = 60
    num_workers = 16
    checkpoint_dir = "models"
    output_dir = "output"
    timestep = 0

    # parse timestep (positional) argument
    parser=argparse.ArgumentParser(description="training script")
    parser.add_argument("timestep", help="timestep")
    args = parser.parse_args()
    timestep = args.timestep

    # --- Setup ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    dataset = VolumeDataset("data", timestep)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    # Create model
    model = SimpleHash(
        res_x=res_x, res_y=res_y, res_z=res_z,
        n_features=n_features, hidden_dim=hidden_dim
    )
    model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"{'Layer':<30} {'Parameters':<15}")
    print("=" * 45)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<30} {param.numel():<15}")
    print("=" * 45)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training ---
    print(f"\nTraining for {num_epochs} epochs...")
    start_time_total = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predicted = model(inputs)
            loss = criterion(predicted, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - start_time

        # Save checkpoint every epoch
#         ckpt_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
#         torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch:3d}/{num_epochs}, "
              f"loss={avg_loss:.8f}, "
              f"time={elapsed:.2f}s")

    total_time = time.time() - start_time_total
    print(f"\nTotal training time: {total_time:.2f}s")

    # checkpoint the model
    ckpt_path = os.path.join(checkpoint_dir, f"model_{timestep}.pt")
    torch.save(model.state_dict(), ckpt_path)

    # --- Save embedding table ---
    embedding_table = model.encoder.embedding.weight.data.cpu().numpy()
    embedding_path = os.path.join(output_dir, f"embedding_table_{timestep}.npy")
    np.save(embedding_path, embedding_table)

    print(f"\nEmbedding table shape: {embedding_table.shape}")
    print(f"  (table_size={res_x * res_y * res_z}, n_features={n_features})")
    print(f"  min={embedding_table.min():.6f}, max={embedding_table.max():.6f}")
    print(f"Saved embedding table to {embedding_path}")
