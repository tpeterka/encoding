"""
Combine per-tensor norms of patch embeddings into one norm per timestep, for
comparing timesteps.

Each tensor's Frobenius norm F_t = ||V_t||_F (written by tensor_norms.py) is
first normalized by the square root of its patch count n_t, giving the RMS patch
norm r_t = F_t / sqrt(n_t), which does not depend on how many patches the tensor
has. The normalized norms are then combined in quadrature with equal weight per
tensor:

    norm      = sqrt(mean_t r_t^2)    (primary; independent of the number of tensors)
    sum norm  = sqrt(sum_t r_t^2)     (= norm * sqrt(number of tensors))

Tensors with no patches (smaller than the patch size) are excluded.
Assumes tensor t corresponds to the same refinement level in every timestep.
"""

import argparse

import numpy as np

from print_tensor_norms import load_tensor_norms


def combine_tensor_norms(frobenius_norms, patch_counts):
    """
    Return (norm, sum_norm, n_tensors) from per-tensor Frobenius norms and patch counts,
    where norm = sqrt(mean_t r_t^2), sum_norm = sqrt(sum_t r_t^2), r_t = F_t / sqrt(n_t).
    """
    frobenius_norms = np.asarray(frobenius_norms, dtype=np.float64)
    patch_counts = np.asarray(patch_counts, dtype=np.float64)
    valid = patch_counts > 0
    n_tensors = int(np.count_nonzero(valid))
    if n_tensors == 0:
        raise ValueError("No tensors with patches")

    rms_patch_norms = frobenius_norms[valid] / np.sqrt(patch_counts[valid])
    sum_sq = float(np.sum(rms_patch_norms ** 2))
    return np.sqrt(sum_sq / n_tensors), np.sqrt(sum_sq), n_tensors


def compute_timestep_norms(timesteps, output_dir="output"):
    results = []
    for timestep in timesteps:
        data = load_tensor_norms(timestep, output_dir)
        norm, sum_norm, n_tensors = combine_tensor_norms(data["frobenius_norms"], data["patch_counts"])
        results.append((timestep, n_tensors, norm, sum_norm))

    print("\nPer-timestep norms, r_t = Frobenius_t / sqrt(patches_t)")
    print(f"| {'timestep':>8} | {'tensors':>7} | {'sqrt(mean r_t^2)':>16} | {'sqrt(sum r_t^2)':>15} |")
    print(f"|{'-' * 10}|{'-' * 9}|{'-' * 18}|{'-' * 17}|")
    for timestep, n_tensors, norm, sum_norm in results:
        print(f"| {timestep:>8} | {n_tensors:>7} | {norm:>16.4f} | {sum_norm:>15.4f} |")

    if len({n for _, n, _, _ in results}) > 1:
        print("Warning: timesteps have different numbers of tensors; compare sqrt(mean r_t^2)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine per-tensor norms into one norm per timestep: "
                    "sqrt(mean_t r_t^2), r_t = ||V_t||_F / sqrt(n_patches_t)"
    )
    parser.add_argument("timesteps", nargs="+", help="One or more timesteps")
    parser.add_argument("--output-dir", default="output", help="Directory containing tensor_norms.py outputs (default: output)")
    args = parser.parse_args()

    compute_timestep_norms(args.timesteps, output_dir=args.output_dir)
