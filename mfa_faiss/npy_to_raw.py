"""
Convert a NumPy .npy array file to raw binary by stripping the .npy header.

This utility does not reshape or reinterpret the array. It writes the array data
exactly as stored, in C-order, so the resulting .raw/.bin file can be read by
ParaView's raw reader with manually supplied dimensions and data type.
"""

import os
import argparse
import numpy as np


def default_output_path(npy_file, extension):
    stem, _ = os.path.splitext(npy_file)
    return f"{stem}{extension}"


def convert_npy_to_raw(npy_file, output_path=None):
    print(f"Loading {npy_file} ...")
    array = np.load(npy_file)

    if output_path is None:
        output_path = default_output_path(npy_file, ".raw")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write only the array payload bytes, without NumPy's file header.
    array.tofile(output_path)

    byte_order = array.dtype.byteorder
    if byte_order == "=":
        endian = "little-endian" if np.little_endian else "big-endian"
    elif byte_order == "<":
        endian = "little-endian"
    elif byte_order == ">":
        endian = "big-endian"
    else:
        endian = "not applicable"

    print(f"Array shape: {array.shape}")
    print(f"Array dtype: {array.dtype}")
    print(f"Byte order: {endian}")
    print(f"Data order on disk: C-order")
    print(f"Saved raw binary to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a NumPy .npy array file to raw binary"
    )
    parser.add_argument("npy_file", help="input .npy file")
    parser.add_argument("--output", help="output .raw or .bin file path")
    args = parser.parse_args()

    convert_npy_to_raw(args.npy_file, output_path=args.output)
