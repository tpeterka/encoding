## Encoding: Embedding various representations of a scientific dataset in a vector database

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
   └── argon_bubble
      └── raw
```
### 3. Download the data

Download several timesteps of the Argon Bubble dataset under encoding/data/argon_bubble/raw folder in above directory tree.
For each timestep, at a minimum there must be a <timestep>.bin binary data file and a <timestep>.txt text file.
The text file only contains the total data size, which is 4194304.
All timesteps are the same size (128x128x256 = 4194304).
Optionally one may also download <timestep>.vtk files for visualization, although these are not used by any of the scripts in this repository.

### 4. Run

There are different versions for creating a vector database (FAISS) from a feature grid, directly from the raw dataset input points, and from the control points of an MFA model of the dataset. See the
following subdirectories and follow the READMEs there:
- `simple_faiss`: FAISS vector database from a full-resolution feature grid
- `direct_faiss`: FAISS vector database direct from dataset raw input points
- `mfa_faiss:     FAISS vector database from control points of MFA model

In each case, the vectors are constructed from 4x4x4 patches of inputs (feature grid points, raw data points, control points), concatenating the 64 patch vertices into a 64-element vector. The number
of vectors depends on the number of patches, which differs depending on the number of inputs.

