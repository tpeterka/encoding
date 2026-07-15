## Embedding a feature grid in a vector database

Feature grid hash encoding is a simplified single-resolution 3D version of
[F-Hash](https://github.com/sunjianxin/F-Hash)


```bash
cd simple_faiss
python3 train.py <timestep>
python3 vector_db.py <timestep>
python3 patch_similarity <timestep_a> <timestep_b>

```

