# Embedding control points of an adaptive MFA model in a vector database

An adaptively encoded MFA model contains multiple hierarchical tensor products of different sizes.
Each tensor product is divided into adjacent patches of control points (default 4x4x4), and each patch is
embedded into a latent vector using the autoencoder embedding model trained in `mfa_faiss`.
Vectors are indexed by their parent tensor (the tensor's position in the MFA tensor product list) in addition to
their patch position within that tensor.

## Dependencies

Same as `mfa_faiss`; see `../mfa_faiss/README.md`. The encoder model and helpers are imported from `../mfa_faiss`.

```
cd /path/to/encoding
source ./load-env.sh
source .venv/bin/activate
```

## Encoding Argon bubble data in an adaptive MFA model

Built and run from `mfa_faiss`:

```
cd /path/to/encoding/build/mfa_faiss
./gridded_3d --adaptive 1 -e 1e-2 -f ../../data/argon_bubble/raw/0.bin -g 2 -v 10 -v 10 -v 10
mv approx.mfa ../../data/argon_bubble/mfa/0-adaptive-approx.mfa
```

## Embedding MFA control points of all tensors in a vector database

Uses an autoencoder checkpoint trained in `mfa_faiss` with a matching patch size.

```
cd /path/to/encoding/mfa_adaptive_faiss
python3 embed_vector_db.py \
  --infile ../data/argon_bubble/mfa/0-adaptive-approx.mfa \
  --encoder-ckpt ../mfa_faiss/output/mfa_patch_autoencoder_64x64x128_4x4x4_16.pt
```

Outputs, where `<timestep>` is the leading part of the input file name before the first `-`, e.g. `0` for
`0-adaptive-approx.mfa` (override with `--timestep <label>`):

- `output/mfa_adaptive_patch_latents_<timestep>.index`: FAISS index of all patch embeddings, rows grouped by tensor in tensor order
- `output/mfa_adaptive_patch_ctrl_bounds_<timestep>.npy`: one row per vector,
  `[gid, tensor_idx, ix_min, iy_min, iz_min, ix_max, iy_max, iz_max]`, control-point indices local to the parent tensor
- `output/mfa_adaptive_tensor_info_<timestep>.npy`: one row per tensor,
  `[gid, tensor_idx, nx, ny, nz, npx, npy, npz, row_start, row_count]`, where `nx, ny, nz` are the number of control points,
  `npx, npy, npz` the number of patches per axis, and `row_start, row_count` locate the tensor's vectors in the index

Tensors smaller than the patch size in any dimension are skipped with a warning (`row_count = 0`).

## Computing per-tensor norms of the patch embeddings in a single timestep

For each tensor t, the patch embeddings form a matrix V_t (one row per patch). Computes the Frobenius norm ||V_t||_F,
the RMS patch norm ||V_t||_F / sqrt(n_patches_t) (comparable across tensors with different patch counts), and the L2 norm
of every patch embedding.

```
cd /path/to/encoding/mfa_adaptive_faiss
python3 tensor_norms.py <timestep>
```

Outputs:

- `output/mfa_adaptive_tensor_norms_<timestep>.npy`: per-tensor Frobenius norms
- `output/mfa_adaptive_tensor_norms_<timestep>.npz`: `frobenius_norms`, `rms_patch_norms`, `patch_counts`,
  `patch_norms`, `patch_bounds`, `tensor_info`, `timestep`

## Combining per-tensor norms into one norm per timestep

Used for comparing timesteps. Each tensor's Frobenius norm F_t is first normalized by the square root of its patch count,
r_t = F_t / sqrt(n_t), so that it does not depend on how many patches the tensor has. The normalized norms are then
combined in quadrature with equal weight per tensor: N = sqrt(mean_t r_t^2), which does not depend on the number of
tensors. Also prints sqrt(sum_t r_t^2). Tensors with no patches are excluded. Assumes tensor t corresponds to the same
refinement level in every timestep. Requires `tensor_norms.py` to have been run for each timestep.

```
cd /path/to/encoding/mfa_adaptive_faiss
python3 timestep_norm.py <timestep> [<timestep> ...]
```

## Diagnostic tools

### Printing the table of per-tensor norms for a timestep

Requires `tensor_norms.py` to have been run for the timestep.

```
cd /path/to/encoding/mfa_adaptive_faiss
python3 print_tensor_norms.py <timestep>
```
