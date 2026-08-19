"""
Inspect metadata produced by space_patch_similarity.py.

Prints saved reference information, patch-grid dimensions, inferred stride, the
reference patch's patch-space location, and optional containment diagnostics for
a queried control-point coordinate.
"""

import argparse
import numpy as np


def parse_query_point(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--query-point must have exactly four comma-separated integers: gid,ix,iy,iz"
        )
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "--query-point must contain only integers: gid,ix,iy,iz"
        ) from exc


def format_scalar(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def infer_stride(axis_vals):
    if len(axis_vals) <= 1:
        return None
    diffs = np.diff(axis_vals)
    if np.all(diffs == diffs[0]):
        return int(diffs[0])
    return diffs.tolist()


def build_gid_grid(patch_bounds, gid):
    gid = int(gid)
    gid_mask = patch_bounds[:, 0] == gid
    if not np.any(gid_mask):
        available_gids = np.unique(patch_bounds[:, 0]).tolist()
        raise ValueError(
            f"No patches found for gid={gid}; available gids: {available_gids}"
        )

    gid_indices = np.flatnonzero(gid_mask)
    bounds_gid = patch_bounds[gid_mask]
    ix_vals = np.unique(bounds_gid[:, 1])
    iy_vals = np.unique(bounds_gid[:, 2])
    iz_vals = np.unique(bounds_gid[:, 3])

    shape = (len(ix_vals), len(iy_vals), len(iz_vals))
    expected_count = int(np.prod(shape))
    if expected_count != bounds_gid.shape[0]:
        raise ValueError(
            f"gid={gid} does not form a complete regular patch grid: expected {expected_count} "
            f"patches from shape {shape}, found {bounds_gid.shape[0]}"
        )

    x_index = {int(value): idx for idx, value in enumerate(ix_vals.tolist())}
    y_index = {int(value): idx for idx, value in enumerate(iy_vals.tolist())}
    z_index = {int(value): idx for idx, value in enumerate(iz_vals.tolist())}

    lower_bound_to_patch_index = {}
    lower_bound_to_flat_index = {}
    for flat_index, row in zip(gid_indices, bounds_gid):
        lower = (int(row[1]), int(row[2]), int(row[3]))
        patch_index = (x_index[lower[0]], y_index[lower[1]], z_index[lower[2]])
        if lower in lower_bound_to_patch_index:
            raise ValueError(f"Duplicate lower bound for gid={gid}: {lower}")
        lower_bound_to_patch_index[lower] = patch_index
        lower_bound_to_flat_index[lower] = int(flat_index)

    return {
        "gid": gid,
        "gid_indices": gid_indices,
        "bounds_gid": bounds_gid,
        "shape": shape,
        "ix_vals": ix_vals,
        "iy_vals": iy_vals,
        "iz_vals": iz_vals,
        "stride": (
            infer_stride(ix_vals),
            infer_stride(iy_vals),
            infer_stride(iz_vals),
        ),
        "lower_bound_to_patch_index": lower_bound_to_patch_index,
        "lower_bound_to_flat_index": lower_bound_to_flat_index,
    }


def find_containing_indices(patch_bounds, query_point):
    gid, ix, iy, iz = (int(value) for value in query_point)
    matches = (
        (patch_bounds[:, 0] == gid)
        & (patch_bounds[:, 1] <= ix)
        & (ix <= patch_bounds[:, 4])
        & (patch_bounds[:, 2] <= iy)
        & (iy <= patch_bounds[:, 5])
        & (patch_bounds[:, 3] <= iz)
        & (iz <= patch_bounds[:, 6])
    )
    return np.flatnonzero(matches)


# Keep this tie-break identical to space_patch_similarity.py.
def choose_reference_index(patch_bounds, query_point):
    match_indices = find_containing_indices(patch_bounds, query_point)
    if match_indices.size == 0:
        return None, match_indices
    if match_indices.size == 1:
        return int(match_indices[0]), match_indices

    candidate_bounds = patch_bounds[match_indices, 1:4]
    order = np.lexsort((candidate_bounds[:, 2], candidate_bounds[:, 1], candidate_bounds[:, 0]))
    return int(match_indices[order[0]]), match_indices


def print_metadata_summary(metadata):
    print("Metadata summary")
    print(f"  timestep: {format_scalar(metadata.get('timestep'))}")
    print(f"  resolution: {format_scalar(metadata.get('resolution'))}")
    print(f"  mode: {format_scalar(metadata.get('mode'))}")
    print(f"  distances shape: {metadata['distances'].shape}")
    print(f"  patch_bounds shape: {metadata['patch_bounds'].shape}")
    print(f"  available gids: {np.unique(metadata['patch_bounds'][:, 0]).tolist()}")
    if 'reference_point' in metadata:
        print(f"  saved reference_point: {tuple(int(v) for v in metadata['reference_point'].tolist())}")
    if 'reference_index' in metadata:
        print(f"  saved reference_index: {int(metadata['reference_index'])}")
    if 'reference_bounds_row' in metadata:
        print(f"  saved reference_bounds_row: {metadata['reference_bounds_row'].tolist()}")


def print_gid_summary(grid_info):
    ix_vals = grid_info["ix_vals"]
    iy_vals = grid_info["iy_vals"]
    iz_vals = grid_info["iz_vals"]
    stride_x, stride_y, stride_z = grid_info["stride"]

    print(f"\nPatch-grid summary for gid={grid_info['gid']}")
    print(f"  patch-space dimensions: {grid_info['shape']}")
    print(
        f"  lower-bound ranges: x=[{int(ix_vals[0])}, {int(ix_vals[-1])}], "
        f"y=[{int(iy_vals[0])}, {int(iy_vals[-1])}], "
        f"z=[{int(iz_vals[0])}, {int(iz_vals[-1])}]"
    )
    print(f"  inferred stride: x={stride_x}, y={stride_y}, z={stride_z}")


def print_reference_summary(metadata, grid_info):
    patch_bounds = metadata["patch_bounds"]
    distances = metadata["distances"]

    if 'reference_index' not in metadata or 'reference_bounds_row' not in metadata:
        print("\nReference summary\n  saved reference metadata not present")
        return

    reference_index = int(metadata['reference_index'])
    reference_bounds_row = metadata['reference_bounds_row']
    reference_gid = int(reference_bounds_row[0])
    reference_lower = tuple(int(v) for v in reference_bounds_row[1:4].tolist())

    print("\nReference summary")
    print(f"  flat index: {reference_index}")
    print(f"  bounds row: {reference_bounds_row.tolist()}")
    print(f"  saved distance value: {float(distances[reference_index]):.9g}")

    min_index = int(np.argmin(distances))
    print(f"  global minimum distance: {float(distances[min_index]):.9g} at flat index {min_index}")
    print(f"  global minimum bounds row: {patch_bounds[min_index].tolist()}")

    if reference_gid != grid_info['gid']:
        print(f"  reference gid {reference_gid} does not match selected gid {grid_info['gid']}")
        return

    patch_index = grid_info["lower_bound_to_patch_index"].get(reference_lower)
    if patch_index is None:
        print("  reference lower bound not found in selected gid grid")
        return

    print(f"  patch-space index: {patch_index}")


def print_query_summary(metadata, grid_info, query_point):
    patch_bounds = metadata["patch_bounds"]
    distances = metadata["distances"]
    selected_index, containing_indices = choose_reference_index(patch_bounds, query_point)

    print(f"\nQuery point: {query_point}")
    print(f"  containing patches: {containing_indices.tolist()}")
    if containing_indices.size == 0:
        return

    for flat_index in containing_indices.tolist():
        row = patch_bounds[flat_index]
        lower = tuple(int(v) for v in row[1:4].tolist())
        patch_index = None
        if int(row[0]) == grid_info['gid']:
            patch_index = grid_info['lower_bound_to_patch_index'].get(lower)
        print(
            f"  flat index {flat_index}: bounds={row.tolist()}, "
            f"patch-space index={patch_index}, distance={float(distances[flat_index]):.9g}"
        )

    if selected_index is None:
        return

    selected_row = patch_bounds[selected_index]
    selected_lower = tuple(int(v) for v in selected_row[1:4].tolist())
    selected_patch_index = None
    if int(selected_row[0]) == grid_info['gid']:
        selected_patch_index = grid_info['lower_bound_to_patch_index'].get(selected_lower)
    print(f"  selected by tie-break: flat index {selected_index}")
    print(f"  selected bounds row: {selected_row.tolist()}")
    print(f"  selected patch-space index: {selected_patch_index}")
    print(f"  selected distance value: {float(distances[selected_index]):.9g}")


def inspect_space_patch_similarity(metadata_file, gid=0, query_point=None):
    print(f"Loading {metadata_file} ...")
    with np.load(metadata_file) as npz:
        metadata = {key: npz[key] for key in npz.files}

    if 'distances' not in metadata or 'patch_bounds' not in metadata:
        raise ValueError(
            f"Expected 'distances' and 'patch_bounds' arrays in {metadata_file}"
        )

    distances = np.asarray(metadata['distances'], dtype=np.float32)
    patch_bounds = np.asarray(metadata['patch_bounds'], dtype=np.int64)
    if distances.ndim != 1:
        raise ValueError(f"Expected distances to be 1D, got shape {distances.shape}")
    if patch_bounds.ndim != 2 or patch_bounds.shape[1] != 7:
        raise ValueError(
            f"Expected patch_bounds with shape (n, 7), got {patch_bounds.shape}"
        )
    if distances.shape[0] != patch_bounds.shape[0]:
        raise ValueError(
            f"Distance count {distances.shape[0]} does not match patch_bounds row count {patch_bounds.shape[0]}"
        )

    metadata['distances'] = distances
    metadata['patch_bounds'] = patch_bounds

    print_metadata_summary(metadata)
    grid_info = build_gid_grid(patch_bounds, gid)
    print_gid_summary(grid_info)
    print_reference_summary(metadata, grid_info)
    if query_point is not None:
        print_query_summary(metadata, grid_info, query_point)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect metadata from space_patch_similarity.py"
    )
    parser.add_argument("metadata_file", help="input .npz metadata file")
    parser.add_argument("--gid", type=int, default=0, help="gid to inspect (default: 0)")
    parser.add_argument(
        "--query-point",
        help="optional control-point coordinate to inspect as gid,ix,iy,iz",
    )
    args = parser.parse_args()

    inspect_space_patch_similarity(
        args.metadata_file,
        gid=args.gid,
        query_point=parse_query_point(args.query_point) if args.query_point else None,
    )
