## Embedding control points of an MFA model in a vector database

Patches of 4x4x4 control points are concatenated into a 64-element vector.
The number of patches and vectors depends on the number of control points.
The Argon bubble original data size is 128x128x256.
Two different resolution MFA models are generated.
The first is 32X compression ratio with 64x64x128 control points.
The second is 256X compression ratio with 32x42x64 control points.

### Dependencies

Requires MPI and MFA. The recommended installation is through Spack.
```bash
cd /path/to/encoding
source ./create-env.sh
source ./load-env.sh
```
Install additional dependencies in a virtual Python environment.

```
cd /path/to/encoding
python3 -m venv .venv               # first time only
source .venv/bin/activate
pip3 install torch tqdm faiss-cpu vtk   # first time only
```

### Encoding Argon bubble data in an MFA model

Example of `myconfig` script (tailor to your own machine and installation) for CMake:

```
#!/bin/bash

rm CMakeCache.txt

cmake .. \
-DCMAKE_INSTALL_PREFIX=/path/to/encoding/install \
-DCMAKE_CXX_COMPILER=mpicxx \
-DBUILD_SHARED_LIBS=false \
-DCMAKE_BUILD_TYPE=Release \
-Dmfa_thread=tbb \
-Deigen_thread=true \
-Dmfa_python=true \
-DCMAKE_CXX_FLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0" \

make -j8 install
```

Build:

```
cd /path/to/encoding
mkdir build
cd build
../myconfig
cd mfa
```

Run:

```
# 32X compression
./gridded_3d -f ../../data/argon_bubble/raw/0.bin -g 2 -v 64 -v 64 -v 128
mv approx.mfa /path/to/encoding/data/argon_bubble/mfa/0-64x64x128-approx.mfa

# 256X compression
./gridded_3d -f ../../data/argon_bubble/raw/0.bin -g 2 -v 32 -v 32 -v 64
mv approx.mfa /path/to/encoding/data/argon_bubble/mfa/0-32x32x64-approx.mfa
```

### Embedding MFA control points in a vector database using plain concatenation

```
cd /path/to/encoding/mfa_faiss
python3 concat_vector_db.py --infile ../data/argon_bubble/mfa/0-32x32x64-approx.mfa
python3 concat_vector_db.py --infile ../data/argon_bubble/mfa/0-64x64x128-approx.mfa
# similar for timesteps 0,1,119,120,239,240
```

Outputs of vector databases encoded from control points are in `path/to/encoding/mfa_faiss/output/mfa_patches_<timestep>-<ctrl_pt_resolution>-approx.index`

Additionally, the bounds of the patches are in `path/to/encoding/mfa_faiss/output/mfa_patch_ctrl_bounds_<timestep>-<ctrl_pt_resolution>-approx.npy`

### Training an autoencoder as the embedding model for MFA control points

```
cd /path/to/encoding/mfa_faiss
# single file
python3 train_patch_autoencoder.py --infile ../data/argon_bubble/mfa/0-32x32x64-approx.mfa
python3 train_patch_autoencoder.py --infile ../data/argon_bubble/mfa/0-32x32x64-approx.mfa
# multiple files
python3 train_patch_autoencoder.py --infile ../data/argon_bubble/mfa/0-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/1-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/119-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/120-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/239-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/240-64x64x128-approx.mfa
```

Output of the training is a model checkpoint file in `path/to/encoding/mfa_faiss/output/mfa_patch_autoencoder.pt

### Embedding MFA control points in a vector database using autoencoder as the embedding model

```
cd /path/to/encoding/mfa_faiss
python3 embed_vector_db.py \
  --infile ../data/argon_bubble/mfa/0-64x64x128-approx.mfa \
  --encoder-ckpt output/mfa_patch_autoencoder.pt
```

Outputs of vector databases encoded from control points are in `path/to/encoding/mfa_faiss/output/mfa_patch_latents_<timestep>-<ctrl_pt_resolution>-approx.index`

Additionally, the bounds of the patches are in `path/to/encoding/mfa_faiss/output/mfa_patch_ctrl_bounds_<timestep>-<ctrl_pt_resolution>-approx.npy`

# Computing patch similarity across two timesteps for control-point encoded vector embeddings

Computes pair-wise distance between corresponding patches in two timesteps and one aggregate L2 norm distance over those distances.

```
cd /path/to/encoding/mfa_faiss
python3 patch_similarity.py <timestep a> <timestep b> --resolution "64x64x128" --mode concat
python3 patch_similarity.py <timestep a> <timestep b> --resolution "64x64x128" --mode embed
```

# Computing patch similarity across patches within a single timestep for control-point encoded vector embeddings

Given a single patch as the reference, compute distances from all other patches in the timestep.

```
cd /path/to/encoding/mfa_faiss
python3 space_patch_similarity.py <timestep> --resolution "64x64x128" --reference-point <gid,ix,iy,iz> --mode concat
python3 space_patch_similarity.py <timestep> --resolution "64x64x128" --reference-point <gid,ix,iy,iz> --mode embed
```

Outputs of patch distances are in `path/to/encoding/mfa_faiss/output/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx.npy`

Additionally, the bounds and other metadata of the patches are in `path/to/encoding/mfa_faiss/output/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx.npz`

# Converting patch similarity distances across patches in a single timestep from numpy to VTK for visualization

```
cd /path/to/encoding/mfa_faiss
python3 space_patch_similarity_to_vtk.py <output_dir>/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx.npz

Output is saved to <output_dir>/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx_gid0.vtk as a structured grid of scalar values, one per patch.

# Diagnostic tools for debugging

## Computing patch norms of vectors across patches within a single timestep

Given a single timestep, compute norms of vector embeddings for all patches.

Generate norms:
```
cd /path/to/encoding/mfa_faiss
python3 patch_vector_norms.py <timestep> --resolution 64x64x128 --mode concat
python3 patch_vector_norms.py <timestep> --resolution 64x64x128 --mode embed
```

Output files are in output/mfa_patch_norms_<timestep>-64x64x128-approx.npy, .npz for concat mode and output/mfa_patch_latent_norms_<timestep>-64x64x128-approx.npy, .npz for embed mode.


Convert to VTK:
```
cd /path/to/encoding/mfa_faiss
python3 space_patch_similarity_to_vtk.py output/mfa_patch_norms_<timestep>-64x64x128-approx.npz
python3 space_patch_similarity_to_vtk.py output/mfa_patch_latent_norms_<timestep>-64x64x128-approx.npz
```

Output files are in output/mfa_patch_norms_0-64x64x128-approx_gid0.vtk for concat mode and in output/mfa_patch_latent_norms_0-64x64x128-approx_gid0.vtk for embed mode.

<!--
## Converting patch similarity distances in .npy to .raw for ParaView input

```
cd /path/to/encoding/mfa_faiss
python3 npy_to_raw.py <output_dir>/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx.npy
```

Output is saved to <output_dir>/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx.raw as a flat array of scalar values, one per patch.
Upon reading into ParaView, set the data type to float, the byte order to little endian (for macbook, eg.), and the dimensions to the number of patches in each dimension (eg., 21, 21, 42)

However, be aware that the image reader in ParaView for .raw files expects Fortran-order (x-fastest) while our .npy file is written in C-order (z-fastest), so the results will be transposed. The
dimensions in the file properties would need to be swapped, and a transpose filter would need to be applied in ParaView to view a correct image.
The .raw file converter is a diagnostic debugging tool only, not useful in most cases since its expected order does not align with the file order.
-->

## Converting patch similarity distances in .npy to .vtk for ParaView input

There is also a diagnostic script that converts .npy to .vtk with hard-coded data type, byte order, and patch dimensions:
```
cd /path/to/encoding/mfa_faiss
python3 mfa_faiss/npy_to_vtk.py <output_dir>/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx.npy
```

## Inspecting and querying .npz metadata file

```
python3 mfa_faiss/inspect_space_patch_similarity.py <output_dir>/mfa_space_patch_latent_distances_<timestep>-<ctrl_pt_resolution>-approx.npz --query-point <gid,ix,iy,iz> # optional query point
cd /path/to/encoding/mfa_faiss
```

## Creating synthetic .npy and .npz files
For shape 3,4,5, all values are 1 except a 0-value at index 1,2,3
```
python3 mfa_faiss/generate_synthetic_space_patch_data.py
cd /path/to/encoding/mfa_faiss
```
Or explicitly,
```
cd /path/to/encoding/mfa_faiss
python3 mfa_faiss/generate_synthetic_space_patch_data.py \
  --shape 3,4,5 \
  --zero-index 1,2,3 \
  --output-dir outpu
```

Output is saved in <output_dir>/synthetic_space_patch_<grid_dims>_zero_<zero_location>.npy, .npz

