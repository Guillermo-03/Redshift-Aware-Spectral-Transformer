import torch
import torch.nn as nn
from tokenizer import PatchEmbedding, sinusoidal_positional_encoding
import config


class SpectralTransformer(nn.Module):
    """
    BERT-style masked transformer for DESI spectra.
    Sequence: [CLS] [Z_MASK] patch_1 ... patch_253  (255 tokens total)
    """

    def __init__(
        self,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ffn=config.D_FFN,
        patch_size=config.PATCH_SIZE,
        n_patches=config.N_PATCHES,
        max_seq_len=config.MAX_SEQ_LEN,
    ):
        super().__init__()

        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.z_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, d_model)

        # Positional encoding (sinusoidal, not learned)
        self.register_buffer(
            "pos_enc",
            sinusoidal_positional_encoding(max_seq_len, d_model).unsqueeze(0),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm (more stable from scratch)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.reconstruction_head = nn.Linear(d_model, patch_size)
        self.redshift_head = nn.Sequential(
            nn.Linear(d_model, 192),
            nn.GELU(),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, flux, mask):
        # flux: (B, PADDED_LEN)
        # mask: (B, N_PATCHES) bool — True = this patch is masked
        # Returns: recon_patches (B, n_masked, PATCH_SIZE), z_pred (B, 1)
        pass
