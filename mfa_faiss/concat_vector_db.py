"""
FAISS vector database for MFA control-point patches.
Each vector is a concatenation of the control points in the patch.

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

from patch_data import extract_block_patches


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
