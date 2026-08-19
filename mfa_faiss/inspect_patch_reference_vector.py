"""
Inspect stored patch vectors relative to a selected reference patch.

Loads one FAISS patch database and matching bounds metadata, selects a reference
patch using the same containment/tie-break rule as space_patch_similarity.py,
and reports vector norms, nearest/farthest patches, and an optional direct
comparison patch.
"""

import argparse
import numpy as np
import faiss

from space_patch_similarity import build_input_paths, extract_vectors, find_reference_index, parse_reference_point


def summarize_vector(vector, prefix):
    print(f"{prefix} norm: {np.linalg.norm(vector):.9g}")
    print(
        f"{prefix} components: min={float(vector.min()):.9g}, "
        f"max={float(vector.max()):.9g}, mean={float(vector.mean()):.9g}, std={float(vector.std()):.9g}"
    )
    preview_count = min(12, vector.shape[0])
    print(f"{prefix} first {preview_count} components: {vector[:preview_count].tolist()}")


def print_patch_summary(bounds, distances, norms, index, label):
    row = bounds[index]
    print(f"{label} flat index: {int(index)}")
    print(f"{label} bounds row: {row.tolist()}")
    print(f"{label} vector norm: {float(norms[index]):.9g}")
    if distances is not None:
        print(f"{label} distance to reference: {float(distances[index]):.9g}")


def print_ranked_patches(bounds, distances, norms, indices, title):
    print(f"\n{title}")
    for rank, index in enumerate(indices, start=1):
        row = bounds[index]
        print(
            f"  {rank:2d}. index={int(index)}, distance={float(distances[index]):.9g}, "
            f"norm={float(norms[index]):.9g}, bounds={row.tolist()}"
        )


def inspect_patch_reference_vector(
    timestep,
    resolution,
    reference_point,
    mode="concat",
    output_dir="output",
    top_k=10,
    compare_point=None,
):
    index_path, bounds_path, representation, _ = build_input_paths(
        timestep, resolution, mode, output_dir
    )

    print(f"Loading {bounds_path} ...")
    bounds = np.load(bounds_path)
    if bounds.ndim != 2 or bounds.shape[1] != 7:
        raise ValueError(f"Expected patch bounds with shape (n, 7), got {bounds.shape}")

    print(f"Loading {index_path} ...")
    index = faiss.read_index(index_path)
    if bounds.shape[0] != index.ntotal:
        raise ValueError(
            f"Patch-bounds row count {bounds.shape[0]} does not match index size {index.ntotal}"
        )

    vectors = extract_vectors(index)
    norms = np.linalg.norm(vectors, axis=1)

    reference_index = find_reference_index(bounds, reference_point)
    reference_vector = vectors[reference_index]
    distances = np.linalg.norm(vectors - reference_vector, axis=1)

    print("Vector database summary")
    print(f"  timestep: {timestep}")
    print(f"  resolution: {resolution}")
    print(f"  mode: {mode}")
    print(f"  representation: {representation}")
    print(f"  vector array shape: {vectors.shape}")
    print(
        f"  vector norm stats: min={float(norms.min()):.9g}, max={float(norms.max()):.9g}, "
        f"mean={float(norms.mean()):.9g}, std={float(norms.std()):.9g}"
    )
    print(
        f"  distance-to-reference stats: min={float(distances.min()):.9g}, "
        f"max={float(distances.max()):.9g}, mean={float(distances.mean()):.9g}, "
        f"std={float(distances.std()):.9g}"
    )

    print("\nReference patch")
    print(f"  target point: {tuple(int(v) for v in reference_point)}")
    print_patch_summary(bounds, distances, norms, reference_index, "  selected")
    summarize_vector(reference_vector, "  selected reference vector")

    min_norm_index = int(np.argmin(norms))
    max_norm_index = int(np.argmax(norms))
    min_distance_index = int(np.argmin(distances))
    max_distance_index = int(np.argmax(distances))

    print("\nExtrema")
    print_patch_summary(bounds, None, norms, min_norm_index, "  min-norm patch")
    print_patch_summary(bounds, None, norms, max_norm_index, "  max-norm patch")
    print_patch_summary(bounds, distances, norms, min_distance_index, "  min-distance patch")
    print_patch_summary(bounds, distances, norms, max_distance_index, "  max-distance patch")

    nearest_indices = np.argsort(distances)[: max(1, top_k)]
    farthest_indices = np.argsort(distances)[-max(1, top_k) :][::-1]
    print_ranked_patches(bounds, distances, norms, nearest_indices, f"Nearest {len(nearest_indices)} patches")
    print_ranked_patches(bounds, distances, norms, farthest_indices, f"Farthest {len(farthest_indices)} patches")

    if compare_point is not None:
        compare_index = find_reference_index(bounds, compare_point)
        compare_vector = vectors[compare_index]
        print("\nComparison patch")
        print(f"  target point: {tuple(int(v) for v in compare_point)}")
        print_patch_summary(bounds, distances, norms, compare_index, "  selected")
        summarize_vector(compare_vector, "  comparison vector")
        compare_distance = float(np.linalg.norm(compare_vector - reference_vector))
        print(f"  direct distance to reference: {compare_distance:.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect stored patch vectors relative to a selected reference patch"
    )
    parser.add_argument("timestep", help="Timestep")
    parser.add_argument("--resolution", required=True, help='resolution (string), eg "64x64x128"')
    parser.add_argument(
        "--reference-point",
        required=True,
        help="reference control-point coordinate as gid,ix,iy,iz",
    )
    parser.add_argument(
        "--mode",
        choices=("concat", "embed"),
        default="concat",
        help="which index representation to inspect (default: concat)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing FAISS index files (default: output)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="number of nearest/farthest patches to print (default: 10)",
    )
    parser.add_argument(
        "--compare-point",
        help="optional comparison control-point coordinate as gid,ix,iy,iz",
    )
    args = parser.parse_args()

    inspect_patch_reference_vector(
        args.timestep,
        args.resolution,
        parse_reference_point(args.reference_point),
        mode=args.mode,
        output_dir=args.output_dir,
        top_k=args.top_k,
        compare_point=parse_reference_point(args.compare_point) if args.compare_point else None,
    )
