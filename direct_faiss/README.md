## Embedding dataset points directly in a vector database

Patches of 4x4x4 data points are concatenated into a 64-element vector. The number of patches and vectors depends on the data size. In the case of the Argon bubble, the original data size is
128x128x256.


```
cd direct_faiss
python3 vector_db.py <timestep>
python3 patch_similarity <timestep_a> <timestep_b>

```

# Diagnostic tools for debugging

## Computing patch norms of vectors across patches within a single timestep

Given a single timestep, compute norms of vector embeddings for all patches.

Generate norms:
```
python3 patch_vector_norms.py <timestep>
```

Output file is in output/patch_norms_<timestep>.npy, .npz


Convert to VTK:
```
python3 space_patch_similarity_to_vtk.py output/patch_norms_<timestep>.npz
```

Output file is in output/patch_norms_<timestep>.vtk


