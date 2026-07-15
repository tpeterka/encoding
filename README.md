## Encoding: Embedding various representations of a scientific dataset in a vector database

Feature grid hash encoding is a simplified single-resolution 3D version of
[F-Hash](https://github.com/sunjianxin/F-Hash)

### 1. Setup

It is recommended to install dependencies in a virtual Python environment.

```
cd /path/to/encoding
python3 -m venv .venv               # first time only
source .venv/bin/activate
pip3 install torch tqdm faiss-cpu vtk   # first time only
```

### 2. Create the `data/argon_bubble` directory
The following directories need to be manually created:
```
encoding
└── data
   └── argon_bubble  (Raw dataset: .dat)
```
### 3. Download the data
Download the Argon Bubble toy dataset, put all the extracted frames under encoding/data/argon_bubble folder in above directory tree.
```bash
python download_data.py
```
### 4. Run

There are different versions for creating a vector database (FAISS) from a feature grid, directly from the raw dataset input points, and from the control points of an MFA model of the dataset. See the
following subdirectories and follow the READMEs there:
- `simple_faiss`: FAISS vector database from a full-resolution feature grid
- `direct_faiss`: FAISS vector database direct from dataset raw input points
- `mfa_faiss:     FAISS vector database from control points of MFA model

In each case, the vectors are constructed from 4x4x4 patches of inputs (feature grid points, raw data points, control points), concatenating the 64 patch vertices into a 64-element vector. The number
of vectors depends on the number of patches, which differs depending on the number of inputs.

