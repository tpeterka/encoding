"""
Control-point patch extraction for adaptively encoded MFA models.

An adaptive MFA model contains multiple hierarchical tensor products of
different sizes. Each tensor is patched independently, exactly as the single
tensor of a fixed encoding is patched in mfa_faiss/patch_data.py, and every
patch is tagged with its parent tensor index (position in tmesh.tensor_prods).
"""

import numpy as np

# columns of the per-patch metadata (patch bounds) array
PATCH_BOUNDS_COLUMNS = ("gid", "tensor_idx", "ix_min", "iy_min", "iz_min", "ix_max", "iy_max", "iz_max")

# columns of the per-tensor metadata array
TENSOR_INFO_COLUMNS = ("gid", "tensor_idx", "nx", "ny", "nz", "npx", "npy", "npz", "row_start", "row_count")


def extract_tensor_patches(tensor, gid, tensor_idx, patch_x, patch_y, patch_z):
    """
    Extract overlapping patches of control points from one tensor product.

    Returns (patch_vectors, patch_bounds, nctrl, n_patches) where nctrl and
    n_patches are (x, y, z) tuples. Returns empty arrays if the tensor is
    smaller than the patch size in any dimension.
    """
    nctrl = np.asarray(tensor.nctrl_pts, dtype=np.int64)
    if nctrl.shape[0] != 3:
        raise ValueError(f"Tensor {tensor_idx}: expected 3D control-point lattice, got shape {nctrl.shape}")

    ctrl_pts = np.asarray(tensor.ctrl_pts, dtype=np.float32)
    if ctrl_pts.ndim != 2 or ctrl_pts.shape[1] != 1:
        raise ValueError(
            f"Tensor {tensor_idx}: expected scalar control points with shape (N, 1), got {ctrl_pts.shape}"
        )

    expected_nctrl = int(np.prod(nctrl))
    if ctrl_pts.shape[0] != expected_nctrl:
        raise ValueError(
            f"Tensor {tensor_idx}: control point count mismatch: got {ctrl_pts.shape[0]}, expected {expected_nctrl}"
        )

    nx, ny, nz = (int(nctrl[0]), int(nctrl[1]), int(nctrl[2]))
    patch_dim = patch_x * patch_y * patch_z

    if nx < patch_x or ny < patch_y or nz < patch_z:
        print(
            f"Warning: block gid={gid} tensor {tensor_idx}: control-point lattice ({nx}, {ny}, {nz}) "
            f"smaller than patch size ({patch_x}, {patch_y}, {patch_z}); skipping"
        )
        return (
            np.empty((0, patch_dim), dtype=np.float32),
            np.empty((0, len(PATCH_BOUNDS_COLUMNS)), dtype=np.int64),
            (nx, ny, nz),
            (0, 0, 0),
        )

    # MFA linearizes control points with x fastest, then y, then z
    # this is Fortran order, hence order="F"
    grid = ctrl_pts[:, 0].reshape((nx, ny, nz), order="F")

    # adjacent patches share one layer of control points
    stride_x = patch_x - 1
    stride_y = patch_y - 1
    stride_z = patch_z - 1
    n_patches_x = (nx - patch_x) // stride_x + 1
    n_patches_y = (ny - patch_y) // stride_y + 1
    n_patches_z = (nz - patch_z) // stride_z + 1
    n_patches_total = n_patches_x * n_patches_y * n_patches_z

    patch_vectors = np.empty((n_patches_total, patch_dim), dtype=np.float32)
    patch_bounds = np.empty((n_patches_total, len(PATCH_BOUNDS_COLUMNS)), dtype=np.int64)

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
                    tensor_idx,
                    ix_start,
                    iy_start,
                    iz_start,
                    ix_start + patch_x - 1,
                    iy_start + patch_y - 1,
                    iz_start + patch_z - 1,
                ]
                patch_idx += 1

    return patch_vectors, patch_bounds, (nx, ny, nz), (n_patches_x, n_patches_y, n_patches_z)


def extract_block_patches(block, cp, patch_x, patch_y, patch_z, row_offset=0):
    """
    Extract patches from every tensor product of the science variable of one block.

    row_offset is the number of patch rows already collected from previous
    blocks, so that row_start in the tensor info refers to the global row in
    the stacked arrays / FAISS index.

    Returns (patch_vectors, patch_bounds, tensor_info).
    """
    model = block.mfa_model()
    var_model = model.var(0)
    tensors = var_model.tmesh.tensor_prods
    gid = cp.gid()

    all_vectors = []
    all_bounds = []
    tensor_info = np.empty((len(tensors), len(TENSOR_INFO_COLUMNS)), dtype=np.int64)

    row_start = row_offset
    for tensor_idx, tensor in enumerate(tensors):
        vectors, bounds, (nx, ny, nz), (npx, npy, npz) = extract_tensor_patches(
            tensor, gid, tensor_idx, patch_x, patch_y, patch_z
        )
        row_count = vectors.shape[0]
        tensor_info[tensor_idx] = [gid, tensor_idx, nx, ny, nz, npx, npy, npz, row_start, row_count]
        all_vectors.append(vectors)
        all_bounds.append(bounds)
        print(
            f"Block gid={gid} tensor {tensor_idx}: nctrl_pts=({nx}, {ny}, {nz}), "
            f"patches per axis=({npx}, {npy}, {npz}), patches={row_count}, rows=[{row_start}, {row_start + row_count})"
        )
        row_start += row_count

    patch_vectors = np.vstack(all_vectors)
    patch_bounds = np.vstack(all_bounds)
    print(f"Block gid={gid}: {len(tensors)} tensors, total patches={patch_vectors.shape[0]}")

    return patch_vectors, patch_bounds, tensor_info
