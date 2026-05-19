"""
Embedding-Only F-Hash: 3D Hash Encoding Model
Single-level, collision-free hash encoding with trilinear interpolation
and a minimal linear decoder for training. The goal is to learn the
embedding table only; the linear layer is discarded after training.
"""

import torch
import torch.nn as nn

# 8 corner offsets of a 3D cube (all combinations of {0,1}^3)
BOX_OFFSETS_3D = torch.tensor(
    [[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
    dtype=torch.int64
)  # shape: 1 x 8 x 3


def hash_linear_3d(coords, res_x, res_y, res_z):
    """
    Collision-free 3D linear hash function.
    Maps integer grid coordinates (x, y, z) to a unique linear index.

    coords: B x 8 x 3 (integer grid coordinates of cube corners)
    res_x, res_y, res_z: grid resolution per axis

    Returns: B x 8 (linear indices)
    """
    # index = x * (res_y * res_z) + y * res_z + z
    indices = (coords[..., 0] * (res_y * res_z)
               + coords[..., 1] * res_z
               + coords[..., 2])
    return indices.to(dtype=torch.int64)


def get_voxel_vertices_3d(xyz, bounding_box, res_x, res_y, res_z):
    """
    For a batch of 3D query points, find the enclosing grid cell and compute
    the 8 cube corner indices.

    xyz: B x 3 (query coordinates)
    bounding_box: (box_min, box_max), each of shape (3,)
    res_x, res_y, res_z: grid resolution per axis

    Returns:
        voxel_min_vertex: B x 3 (spatial position of cell min corner)
        voxel_max_vertex: B x 3 (spatial position of cell max corner)
        hashed_voxel_indices: B x 8 (linear indices of 8 corners)
    """
    box_min, box_max = bounding_box
    device = xyz.device
    box_min = box_min.to(device)
    box_max = box_max.to(device)

    # Clamp to bounding box
    xyz = torch.clamp(xyz, min=box_min, max=box_max)

    resolution = torch.tensor(
        [res_x - 1, res_y - 1, res_z - 1], dtype=torch.float32, device=device
    )
    grid_size = (box_max - box_min) / resolution  # size of one grid cell

    # Find the bottom-left grid index for each query point
    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()  # B x 3

    # Clamp indices at the boundary so bottom_left + 1 stays in bounds
    bottom_left_idx[:, 0] = torch.where(
        bottom_left_idx[:, 0] >= res_x - 1,
        res_x - 2, bottom_left_idx[:, 0]
    )
    bottom_left_idx[:, 1] = torch.where(
        bottom_left_idx[:, 1] >= res_y - 1,
        res_y - 2, bottom_left_idx[:, 1]
    )
    bottom_left_idx[:, 2] = torch.where(
        bottom_left_idx[:, 2] >= res_z - 1,
        res_z - 2, bottom_left_idx[:, 2]
    )

    # Spatial positions of the cell min/max corners
    voxel_min_vertex = bottom_left_idx.float() * grid_size + box_min  # B x 3
    voxel_max_vertex = voxel_min_vertex + grid_size  # B x 3

    # Compute 8 cube corner indices: B x 1 x 3 + 1 x 8 x 3 = B x 8 x 3
    offsets = BOX_OFFSETS_3D.to(device)
    voxel_indices = bottom_left_idx.unsqueeze(1) + offsets  # B x 8 x 3

    # Hash corners to linear indices
    hashed_voxel_indices = hash_linear_3d(voxel_indices, res_x, res_y, res_z)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices


class HashEmbedder3D(nn.Module):
    """
    Single-level 3D hash embedding with trilinear interpolation.
    The embedding table has res_x * res_y * res_z entries, each storing
    n_features learned feature values.
    """

    def __init__(self, bounding_box, res_x, res_y, res_z, n_features=2):
        super().__init__()
        self.bounding_box = bounding_box
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z
        self.n_features = n_features

        table_size = res_x * res_y * res_z
        self.embedding = nn.Embedding(table_size, n_features)
        nn.init.uniform_(self.embedding.weight, a=-0.0001, b=0.0001)

        print(f"HashEmbedder3D: table_size={table_size}, "
              f"grid=({res_x}, {res_y}, {res_z}), "
              f"n_features={n_features}")

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex,
                         voxel_embedds):
        """
        Trilinear interpolation across the 8 cube corners.

        x: B x 3 (query coordinates)
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x n_features

        The 8 corners are ordered by the BOX_OFFSETS_3D enumeration:
          index 0: (0,0,0), 1: (0,0,1), 2: (0,1,0), 3: (0,1,1)
          index 4: (1,0,0), 5: (1,0,1), 6: (1,1,0), 7: (1,1,1)

        Returns: B x n_features
        """
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)
        # weights: B x 3, values in [0, 1] for each axis

        wx = weights[:, 0:1]  # B x 1
        wy = weights[:, 1:2]
        wz = weights[:, 2:3]

        # Step 1: Interpolate along x-axis (8 -> 4 corners)
        c00 = voxel_embedds[:, 0] * (1 - wx) + voxel_embedds[:, 4] * wx
        c01 = voxel_embedds[:, 1] * (1 - wx) + voxel_embedds[:, 5] * wx
        c10 = voxel_embedds[:, 2] * (1 - wx) + voxel_embedds[:, 6] * wx
        c11 = voxel_embedds[:, 3] * (1 - wx) + voxel_embedds[:, 7] * wx

        # Step 2: Interpolate along y-axis (4 -> 2 corners)
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy

        # Step 3: Interpolate along z-axis (2 -> 1)
        c = c0 * (1 - wz) + c1 * wz

        return c

    def forward(self, x):
        """
        x: B x 3 (query coordinates in [-1, 1]^3)
        Returns: B x n_features
        """
        voxel_min, voxel_max, hashed_indices = get_voxel_vertices_3d(
            x, self.bounding_box, self.res_x, self.res_y, self.res_z
        )

        # Look up embeddings for all 8 corners: B x 8 x n_features
        voxel_embedds = self.embedding(hashed_indices)

        # Trilinear interpolation
        return self.trilinear_interp(x, voxel_min, voxel_max, voxel_embedds)


class EmbeddingOnlyModel(nn.Module):
    """
    Embedding-only model: 3D hash encoding + a single linear layer.

    The linear layer exists solely to map n_features to a scalar during
    training. After training, only the embedding table is saved; the
    linear layer is discarded.
    """

    def __init__(self, res_x=64, res_y=64, res_z=128, n_features=2):
        super().__init__()

        bounding_box = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0])
        )

        self.encoder = HashEmbedder3D(
            bounding_box, res_x, res_y, res_z, n_features
        )

        # Minimal linear layer for training (maps n_features -> 1 scalar)
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        """
        x: B x 3 (coordinates in [-1, 1]^3)
        Returns: B x 1 (predicted scalar value)
        """
        encoded = self.encoder(x)
        return self.linear(encoded)
