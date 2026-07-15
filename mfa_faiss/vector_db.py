"""
FAISS vector database for MFA control-point patches.

Loads one MFA file, extracts scalar control points from var(0) of the first
(and only) tensor product, forms overlapping 4x4x4 control-point patches, and
stores them in a FAISS vector database. Each patch vector contains one scalar
value per control point (64 elements for a 4x4x4 patch).

Patch placement uses stride = (patch_size - 1) along each axis so boundary
control points are shared between adjacent patches. Metadata is saved as
control-point lattice index bounds per patch:
    [gid, ix_min, iy_min, iz_min, ix_max, iy_max, iz_max]
"""

import argparse
import os

import diy
import faiss
import mfa
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
    grid = ctrl_pts[:, 0].reshape((nx, ny, nz))

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


def build_patch_database(infile, output_dir="output", patch_x=4, patch_y=4, patch_z=4):
    w = diy.mpi.MPIComm()
    master = diy.Master(w)
    assigner = diy.ContiguousAssigner(w.size, -1)

    print(f"Loading MFA blocks from {infile}")
    diy.read_blocks(infile, assigner, master, load=mfa.load_block)

    all_patch_vectors = []
    all_patch_bounds = []

    def collect_block_patches(block, cp):
        patch_vectors, patch_bounds = extract_block_patches(block, cp, patch_x, patch_y, patch_z)
        all_patch_vectors.append(patch_vectors)
        all_patch_bounds.append(patch_bounds)

    master.foreach(collect_block_patches)

    if not all_patch_vectors:
        raise ValueError(f"No blocks were loaded from {infile}")

    patch_vectors = np.vstack(all_patch_vectors).astype(np.float32, copy=False)
    patch_bounds = np.vstack(all_patch_bounds)
    patch_dim = patch_x * patch_y * patch_z

    print(
        f"\nPatch vectors: shape={patch_vectors.shape}, min={patch_vectors.min():.6f}, "
        f"max={patch_vectors.max():.6f}"
    )
    print(
        f"Patch bounds: shape={patch_bounds.shape} "
        f"(each row: [gid, ix_min, iy_min, iz_min, ix_max, iy_max, iz_max])"
    )

    index = faiss.IndexFlatL2(patch_dim)
    index.add(patch_vectors)
    print(f"FAISS index: {index.ntotal} vectors, dimension={index.d}")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(infile))[0]

    index_path = os.path.join(output_dir, f"mfa_patches_{stem}.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    bounds_path = os.path.join(output_dir, f"mfa_patch_ctrl_bounds_{stem}.npy")
    np.save(bounds_path, patch_bounds)
    print(f"Saved patch control-point bounds to {bounds_path}")

    return index, patch_bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index over MFA control-point patches")
    parser.add_argument("--infile", required=True, help="input MFA file")
    parser.add_argument("--output-dir", default="output", help="directory for FAISS index and metadata")
    parser.add_argument("--patch-x", type=int, default=4, help="patch size in x")
    parser.add_argument("--patch-y", type=int, default=4, help="patch size in y")
    parser.add_argument("--patch-z", type=int, default=4, help="patch size in z")
    args = parser.parse_args()

    build_patch_database(
        infile=args.infile,
        output_dir=args.output_dir,
        patch_x=args.patch_x,
        patch_y=args.patch_y,
        patch_z=args.patch_z,
    )
