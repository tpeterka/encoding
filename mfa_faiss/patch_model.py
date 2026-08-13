"""
Patch autoencoder model
"""

import torch.nn as nn

class PatchAutoencoder(nn.Module):
    """
    Patch autoencoder model
    """
    def __init__(self, input_dim=64, latent_dim=16, hidden_dims=(128, 64)):

        super().__init__()

        self.input_dim      = input_dim
        self.latent_dim     = latent_dim
        self.hidden_dims    = hidden_dims

        self.encoder        = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.latent_dim)
        )

        self.decoder        = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

