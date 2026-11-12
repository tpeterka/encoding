"""
Diagnostic: print the table of per-tensor norms of patch embeddings for one timestep.

Reads the .npz metadata written by tensor_norms.py.
"""

import argparse
import os

import numpy as np


def load_tensor_norms(timestep, output_dir="output"):
    path = os.path.join(output_dir, f"mfa_adaptive_tensor_norms_{timestep}.npz")
    print(f"Loading {path} ...")
    return np.load(path)


def print_tensor_norms(timestep, output_dir="output"):
    data = load_tensor_norms(timestep, output_dir)
    tensor_info = data["tensor_info"]
    patch_counts = data["patch_counts"]
    frobenius_norms = data["frobenius_norms"]
    rms_patch_norms = data["rms_patch_norms"]

    print(f"\nPer-tensor latent norms: timestep {timestep}")
    print(f"| {'gid':>3} | {'tensor':>6} | {'ctrl pts':>12} | {'patches':>7} | {'Frobenius':>12} | {'Frobenius/sqrt(patches)':>23} |")
    print(f"|{'-' * 5}|{'-' * 8}|{'-' * 14}|{'-' * 9}|{'-' * 14}|{'-' * 25}|")
    for t in range(tensor_info.shape[0]):
        gid, tensor_idx, nx, ny, nz = (int(v) for v in tensor_info[t, :5])
        print(
            f"| {gid:>3} | {tensor_idx:>6} | {f'{nx}x{ny}x{nz}':>12} | {int(patch_counts[t]):>7} | "
            f"{frobenius_norms[t]:>12.4f} | {rms_patch_norms[t]:>23.4f} |"
        )

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print per-tensor norms of patch embeddings for one timestep")
    parser.add_argument("timestep", help="Timestep")
    parser.add_argument("--output-dir", default="output", help="Directory containing tensor_norms.py outputs (default: output)")
    args = parser.parse_args()

    print_tensor_norms(args.timestep, output_dir=args.output_dir)
