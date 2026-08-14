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

### Embedding MFA control points in a vector database using plain concatenation as the embedding model

```
cd /path/to/encoding
source .venv/bin/activate
cd /path/to/encoding/mfa_faiss
python3 mfa/concat_vector_db.py --infile ../data/argon_bubble/mfa/0-32x32x64-approx.mfa
python3 mfa/concat_vector_db.py --infile ../data/argon_bubble/mfa/0-64x64x128-approx.mfa
# similar for timesteps 0,1,119,120,239,240
```

Outputs of vector databases encoded from control points are in `path/to/encoding/mfa_faiss/output/mfa_patches_<timestep>-<ctrl_pt_resolution>-approx.index`

Additionally, the bounds of the patches are in `path/to/encoding/mfa_faiss/output/mfa_patch_ctrl_bounds_<timestep>-<ctrl_pt_resolution>-approx.npy`

### Embedding MFA control points in a vector database using autoencoder as the embedding model

```
cd /path/to/encoding
source .venv/bin/activate
cd /path/to/encoding/mfa_faiss
python3 embed_vector_db.py \
  --infile ../data/argon_bubble/mfa/0-64x64x128-approx.mfa \
  --encoder-ckpt output/mfa_patch_autoencoder.pt
```

### Embedding MFA control points in a vector database using autoencoder as the embedding model

```
cd /path/to/encoding
source .venv/bin/activate
cd /path/to/encoding/mfa_faiss
# single file
python3 mfa_faiss/train_patch_autoencoder.py --infile ../data/argon_bubble/mfa/0-32x32x64-approx.mfa
python3 mfa_faiss/train_patch_autoencoder.py --infile ../data/argon_bubble/mfa/0-32x32x64-approx.mfa
# multiple files
python3 train_patch_autoencoder.py --infile ../data/argon_bubble/mfa/0-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/1-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/119-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/120-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/239-64x64x128-approx.mfa --infile ../data/argon_bubble/mfa/240-64x64x128-approx.mfa
```

Outputs of vector databases encoded from control points are in `path/to/encoding/mfa_faiss/output/mfa_patch_latents_<timestep>-<ctrl_pt_resolution>-approx.index`

Additionally, the bounds of the patches are in `path/to/encoding/mfa_faiss/output/mfa_patch_ctrl_bounds_<timestep>-<ctrl_pt_resolution>-approx.npy`

# Computing patch similarity for control-point encoded vector embeddings


```
cd /path/to/encoding
source .venv/bin/activate
cd /path/to/encoding/mfa_faiss
python3 patch_similarity.py <timestep a> <timestep b> --resolution "32x32x64" --mode concat
python3 patch_similarity.py <timestep a> <timestep b> --resolution "64x64x128" --mode concat
python3 patch_similarity.py <timestep a> <timestep b> --resolution "32x32x64" --mode embed
python3 patch_similarity.py <timestep a> <timestep b> --resolution "64x64x128" --mode embed
```

