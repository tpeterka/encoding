"""
Shared MFA control-point patch extraction utilities.
"""

import numpy as np


def extract_block_patches(block, cp, patch_x, patch_y, patch_z):
    model = block.mfa_model()
    var_model = model.var(0)
    if len(var_model.tmesh.tensor_prods) != 1:
        raise ValueError("Expected exactly one tensor product for fixed encoding")

    tensor = var_model.tmesh.tensor_prods[0]
    nctrl = np.asarray(tensor.nctrl_pts, dtype=np.int64)
    if nctrl.shape[0] != 3:
        raise ValueError(f"Expected 3D control-point lattice, got shape {nctrl.shape}")

    ctrl_pts = np.asarray(tensor.ctrl_pts, dtype=np.float32)
    if ctrl_pts.ndim != 2 or ctrl_pts.shape[1] != 1:
        raise ValueError(f"Expected scalar control points with shape (N, 1), got {ctrl_pts.shape}")

    expected_nctrl = int(np.prod(nctrl))
    if ctrl_pts.shape[0] != expected_nctrl:
        raise ValueError(
            f"Control point count mismatch: got {ctrl_pts.shape[0]}, expected {expected_nctrl}"
        )

    nx, ny, nz = (int(nctrl[0]), int(nctrl[1]), int(nctrl[2]))
    if nx < patch_x or ny < patch_y or nz < patch_z:
        raise ValueError(
            f"Control-point lattice ({nx}, {ny}, {nz}) smaller than patch size "
            f"({patch_x}, {patch_y}, {patch_z})"
        )

    # MFA linearizes control points with x fastest, then y, then z.
#     grid = ctrl_pts[:, 0].reshape((nx, ny, nz))
    grid = ctrl_pts[:, 0].reshape((nx, ny, nz), order="F")

    stride_x = patch_x - 1
    stride_y = patch_y - 1
    stride_z = patch_z - 1
    n_patches_x = (nx - patch_x) // stride_x + 1
    n_patches_y = (ny - patch_y) // stride_y + 1
    n_patches_z = (nz - patch_z) // stride_z + 1
    n_patches_total = n_patches_x * n_patches_y * n_patches_z
    patch_dim = patch_x * patch_y * patch_z

    gid = cp.gid()
    patch_vectors = np.empty((n_patches_total, patch_dim), dtype=np.float32)
    patch_bounds = np.empty((n_patches_total, 7), dtype=np.int64)

    patch_idx = 0
    for px in range(n_patches_x):
        ix_start = px * stride_x
        for py in range(n_patches_y):
            iy_start = py * stride_y
            for pz in range(n_patches_z):
                iz_start = pz * stride_z
                patch_vectors[patch_idx] = grid[
                    ix_start:ix_start + patch_x,
                    iy_start:iy_start + patch_y,
                    iz_start:iz_start + patch_z,
                ].ravel()
                patch_bounds[patch_idx] = [
                    gid,
                    ix_start,
                    iy_start,
                    iz_start,
                    ix_start + patch_x - 1,
                    iy_start + patch_y - 1,
                    iz_start + patch_z - 1,
                ]
                patch_idx += 1

    print(
        f"Block gid={gid}: nctrl_pts=({nx}, {ny}, {nz}), ctrl_pts shape={ctrl_pts.shape}, "
        f"grid shape={grid.shape}, patches per axis=({n_patches_x}, {n_patches_y}, {n_patches_z}), "
        f"total patches={n_patches_total}"
    )

    return patch_vectors, patch_bounds
