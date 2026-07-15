## Embedding dataset points directly in a vector database

Patches of 4x4x4 data points are concatenated into a 64-element vector. The number of patches and vectors depends on the data size. In the case of the Argon bubble, the original data size is
128x128x256.


```bash
cd direct_faiss
python3 vector_db.py <timestep>
python3 patch_similarity <timestep_a> <timestep_b>

```

