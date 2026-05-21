## Encoding: simplified single-resolution feature grid encoded in a vector database

Feature grid hash encoding is a simplified single-resolution 3D version of
[F-Hash](https://github.com/sunjianxin/F-Hash)

### 1. Setup

It is recommended to install dependencies in a virtual Python environment.

```
cd /path/to/encoding
python3 -m venv .venv               # first time only
source .venv/bin/activate
pip3 install torch tqdm faiss-cpu   # first time only
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
```bash
cd simple_faiss
python3 prepare_data.py
python3 train.py <timestep>
python3 vector_db.py <timestep>
python3 patch_similarity <timestep_a> <timestep_b>

```
