import math
import torch
import torch.nn as nn
import config


class PatchEmbedding(nn.Module):
    """Splits padded flux into patches and linearly projects each to d_model."""

    def __init__(self, patch_size=config.PATCH_SIZE, d_model=config.D_MODEL):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)

    def forward(self, flux):
        # flux: (B, PADDED_LEN) → (B, N_PATCHES, d_model)
        pass


def sinusoidal_positional_encoding(seq_len, d_model):
    # Returns (seq_len, d_model) fixed sinusoidal encoding
    pass
