"""
Simplified F-Hash: Data Preparation
Loads timestep 0 of the Argon Bubble dataset, normalizes it,
and saves the entire volume as training data (no coreset selection).
"""

import os
import numpy as np
from utility import save_training_data


def load_argon_bubble_timestep(timestep=0):
    """
    Load a single timestep of the Argon Bubble dataset.

    timestep: which timestep to load (0-8)
    Returns: 3D numpy array of shape (128, 128, 256), normalized to [0, 1]
    """
    x_size = 128
    y_size = 128
    z_size = 256

    file_path = os.path.join(
        "..", "data", "argon_bubble", f"rescaled2_{timestep}.dat"
    )
    print(f"Loading {file_path} ...")

    with open(file_path, "r") as f:
        data = np.fromfile(f, dtype=np.float32)
    data = data.reshape((x_size, y_size, z_size), order='F')

    # Normalize to [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)

    print(f"Loaded volume: shape={data.shape}, min={np.min(data):.4f}, "
          f"max={np.max(data):.4f}")

    return data


if __name__ == "__main__":
    # Configuration
    x_size = 128
    y_size = 128
    z_size = 256
    timesteps = 9

    # Create output directory
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # loop over all the timesteps
    for timestep in range(timesteps):

        # Load the volume
        data = load_argon_bubble_timestep(timestep)

        # Save training data as binary file
        bin_path = os.path.join(out_dir, "train_" + str(timestep) + ".bin")
        n_samples = save_training_data(data, bin_path, x_size, y_size, z_size)

        # Save metadata (sample count)
        meta_path = os.path.join(out_dir, "train_" + str(timestep) + ".txt")
        with open(meta_path, "w") as f:
            f.write(str(n_samples))

        print(f"Saved {n_samples} training samples to {bin_path}")
        print(f"Saved metadata to {meta_path}")
