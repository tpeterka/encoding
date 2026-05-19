"""
Simplified F-Hash: Testing / Evaluation
Loads trained checkpoints, runs inference on the full volume,
computes MSE, saves VTK files, and plots the loss curve.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import SimpleHash
from utility import np_array_to_vtk
from prepare_data import load_argon_bubble_timestep


def predict(model, coords, device, batch_size=2_000_000):
    """
    Run batched inference on coordinates.

    model: trained SimpleHash model (already on device)
    coords: tensor of shape (N, 3)
    device: torch device
    batch_size: number of samples per batch

    Returns: tensor of shape (N, 1) on CPU
    """
    model.eval()
    n_samples = coords.shape[0]
    predictions = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            inputs = coords[start:end].to(device)
            pred = model(inputs)
            predictions.append(pred.cpu())

    return torch.cat(predictions, dim=0)


if __name__ == "__main__":
    # --- Configuration ---
# size of feature grid for hash embedding (can be smaller than data, full size of data is 128x128x256
#     res_x = 128
#     res_y = 128
#     res_z = 256
    res_x = 64
    res_y = 64
    res_z = 128
    n_features = 2
    hidden_dim = 64
    num_epochs = 120
    checkpoint_dir = "models"
    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load ground truth volume for MSE computation
    gt_volume = load_argon_bubble_timestep(timestep=0)

    # Build coordinate grid matching training data layout
    n_samples = res_x * res_y * res_z
    coords = np.zeros((n_samples, 3), dtype=np.float32)
    count = 0
    for x in range(res_x):
        for y in range(res_y):
            for z in range(res_z):
                coords[count, 0] = x / (res_x - 1) * 2 - 1
                coords[count, 1] = y / (res_y - 1) * 2 - 1
                coords[count, 2] = z / (res_z - 1) * 2 - 1
                count += 1
    coords_tensor = torch.from_numpy(coords)

    # Ground truth as a flat tensor for MSE computation
    gt_flat = torch.from_numpy(
        gt_volume.reshape(-1, 1).astype(np.float32)
    )  # must match the same (x, y, z) iteration order

    # Reorder ground truth to match the coordinate generation order
    # (x-major, then y, then z -- same as prepare_data.py)
    gt_ordered = np.zeros(n_samples, dtype=np.float32)
    count = 0
    for x in range(res_x):
        for y in range(res_y):
            for z in range(res_z):
                gt_ordered[count] = gt_volume[x, y, z]
                count += 1
    gt_tensor = torch.from_numpy(gt_ordered).unsqueeze(1)

    # Create model (weights will be loaded from checkpoints)
    model = SimpleHash(
        res_x=res_x, res_y=res_y, res_z=res_z,
        n_features=n_features, hidden_dim=hidden_dim
    )

    # Evaluate all checkpoints
    criterion = nn.MSELoss()
    mses = []
    spacing = 1

    print(f"\nEvaluating {num_epochs} checkpoints...")
    start_time = time.time()

    for epoch in tqdm(range(num_epochs)):
        ckpt_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint {ckpt_path} not found, skipping.")
            continue

        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.to(device)

        predicted = predict(model, coords_tensor, device)
        mse = criterion(predicted, gt_tensor).item()
        mses.append(mse)

        # Reshape prediction to volume and save as VTK
        pred_volume = predicted.numpy().reshape(res_x, res_y, res_z)
        vtk_path = os.path.join(output_dir, f"{epoch}.vtk")
        np_array_to_vtk(pred_volume, vtk_path, spacing, spacing, spacing)

        model.to('cpu')

    elapsed = time.time() - start_time
    print(f"\nEvaluation time: {elapsed:.2f}s")
    print(f"MSE values: {mses}")

    # Plot MSE convergence
    plt.figure()
    plt.plot(mses, label="Simple Hash")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Convergence")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_loss.png"), dpi=300)
    print(f"Saved MSE plot to {os.path.join(output_dir, 'mse_loss.png')}")
