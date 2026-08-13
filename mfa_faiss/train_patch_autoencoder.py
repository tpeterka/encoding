import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
import diy
import mfa
from vector_db import extract_block_patches
from patch_model import PatchAutoencoder

def parse_args():
    """
    Parses command line arguments.
    """

    # parse args
    parser = argparse.ArgumentParser(description="Train autoencoder over MFA control-point patches")
    parser.add_argument("--infile", action="append", required=True, help="input MFA file")
    parser.add_argument("--output-dir", default="output", help="directory for autoencoder checkpoints")
    parser.add_argument("--patch-x", type=int, default=4, help="patch size in x")
    parser.add_argument("--patch-y", type=int, default=4, help="patch size in y")
    parser.add_argument("--patch-z", type=int, default=4, help="patch size in z")
    parser.add_argument("--latent-dim", type=int, default=16, help="latent dimension")
    parser.add_argument("--batch-size", type=int, default=4096, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="value fraction")
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()

    # validate args
    assert args.patch_x >= 2 and args.patch_y >=2 and args.patch_z >=2
    assert 0.0 < args.val_fraction and args.val_fraction < 1.0
    assert args.latent_dim >= 1
    assert args.batch_size >= 1
    assert args.epochs >= 1
    assert args.learning_rate > 0

    return args

def load_patches_from_file(infile, patch_x, patch_y, patch_z) -> np.ndarray:
    """
    Loads patches of control points from an MFA file and returns the patches
    """

    world       = diy.mpi.MPIComm()
    master      = diy.Master(world)
    assigner    = diy.ContiguousAssigner(world.size, -1)

    diy.read_blocks(infile, assigner, master, load = mfa.load_block)

    # get patch vectors from science variable control points
    # assumes only one tensor product (fixed, not adaptive encoding)
    all_patch_vectors = []

    def collect_block_patches(block, cp):
        patch_vectors, _ = extract_block_patches(block, cp, patch_x, patch_y, patch_z)
        all_patch_vectors.append(patch_vectors)

    master.foreach(collect_block_patches)
    assert all_patch_vectors            # sanity check, all_patch_vectors is not empty

    return np.vstack(all_patch_vectors).astype(np.float32, copy=False)

def collect_training_patches(infiles, patch_x, patch_y, patch_z) -> np.ndarray:
    """
    Collects patches for training
    """
    all_patches = []
    for f in infiles:
        patches = load_patches_from_file(f, patch_x, patch_y, patch_z)
        all_patches.append(patches)

    if not all_patches:
        raise ValueError("all_patches is empty")

    return np.vstack(all_patches).astype(np.float32, copy=False)

def normalize_patches(patches) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize patches, returning normalized patches, mean, standard deviation
    """
    mean = patches.mean(axis=0).astype(np.float32)
    std = patches.std(axis=0).astype(np.float32)
    # replace small std with 1.0 if it's too small
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    norm_patches = (patches - mean) / std

    return norm_patches.astype(np.float32, copy=False), mean, std

def build_datasets(norm_patches, val_fraction, seed):
    # get number of samples
    n_samples = norm_patches.shape[0]

    # shuffle indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    # compute validation size and split indices into validation and training
    if n_samples < 2:
        raise ValueError("n_samples is too small to split into training and validation")
    n_val = max(1, int(n_samples * val_fraction))
    if n_val >= n_samples:
        n_val = n_samples - 1
    val_indices     = indices[:n_val]
    train_indices   = indices[n_val:]
    assert val_indices.size > 0 and train_indices.size > 0        # sanity

    # build training and validation tensors
    train_tensor = torch.from_numpy(norm_patches[train_indices])
    val_tensor = torch.from_numpy(norm_patches[val_indices])

    return train_tensor, val_tensor

def build_dataloaders(train_tensor, val_tensor, batch_size, num_workers):
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_tensor)

    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def run_train_epoch(model, loader, optimizer, criterion, device) -> float:
    total_loss = 0.0
    n_batches = 0
    model.train()

    for (inputs,) in loader:
        inputs = inputs.to(device)
        predicted = model(inputs)
        loss = criterion(predicted, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        raise ValueError("n_batches = 0")
    avg_loss = total_loss / n_batches
    return avg_loss

def run_val_epoch(model, loader, criterion, device) -> float:
    total_loss = 0.0
    n_batches = 0
    model.eval()

    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            predicted = model(inputs)
            loss = criterion(predicted, inputs)

            total_loss += loss.item()
            n_batches += 1

    if n_batches == 0:
        raise ValueError("n_batches = 0")
    avg_loss = total_loss / n_batches
    return avg_loss

def main():
    # parse args
    args = parse_args()

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # collect training patches
    patches = collect_training_patches(args.infile, args.patch_x, args.patch_y, args.patch_z)
    input_dim = args.patch_x * args.patch_y * args.patch_z
    assert patches.shape[1] == input_dim        # sanity

    # normalize patches
    norm_patches, norm_mean, norm_std = normalize_patches(patches)

    # build train/val datasets and dataloaders
    train_tensor, val_tensor = build_datasets(norm_patches, args.val_fraction, args.seed)
    train_loader, val_loader = build_dataloaders(train_tensor, val_tensor, args.batch_size, args.num_workers)

    # instantiate model and move to devices
    model = PatchAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim)
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss criterion
    criterion = nn.MSELoss()

    # training loop

    print(f"\nTraining for {args.epochs} epochs...")
    start_time_total = time.time()

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()

        avg_train_loss = run_train_epoch(model, train_loader, optimizer, criterion, device)
        avg_val_loss = run_val_epoch(model, val_loader, criterion, device)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        elapsed = time.time() - start_time

        print(f"Epoch {epoch:3d}/{args.epochs}, "
              f"train_loss={avg_train_loss:.8f}, "
              f"val_loss={avg_val_loss:.8f}, "
              f"best_val_loss={best_val_loss:.8f}, "
              f"time={elapsed:.2f}s")

    total_time = time.time() - start_time_total
    print(f"\nTotal training time: {total_time:.2f}s")

    # checkpoint the model
    ckpt_dict = {
        "model_state_dict":     model.state_dict(),
        "input_dim":            input_dim,
        "latent_dim":           args.latent_dim,
        "patch_shape":          [args.patch_x, args.patch_y, args.patch_z],
        "normalization_mean":   norm_mean,
        "normalization_std":    norm_std,
        "train_loss_final":     avg_train_loss,
        "val_loss_final":       avg_val_loss,
        "best_val_loss":        best_val_loss,
        "infiles":              args.infile,
        "patch_x":              args.patch_x,
        "patch_y":              args.patch_y,
        "patch_z":              args.patch_z
    }
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"mfa_patch_autoencoder.pt")
    torch.save(ckpt_dict, ckpt_path)

    # sanity check: latent shape and reconstruction loss on one validation batch
    model.eval()
    with torch.no_grad():
        sample_batch = val_tensor[: min(8, len(val_tensor))].to(device)
        latent = model.encode(sample_batch)
        recon = model(sample_batch)
        sample_loss = criterion(recon, sample_batch).item()
    print(f"batch shape {sample_batch.shape}, "
        f"latent shape {latent.shape}, "
        f"sample loss {sample_loss:.8f}")

if __name__ == "__main__":
    main()

