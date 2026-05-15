import torch
import config

# Rest-frame wavelength grid for DESI (3600–9800 Å over 7081 pixels)
WAVE_MIN = 3600.0
WAVE_MAX = 9800.0

# Physically significant rest-frame wavelengths (Å)
LINES = {
    "CaHK":   3945.0,
    "break4k": 4000.0,
    "Hbeta":  4861.0,
    "OIII_1": 4959.0,
    "OIII_2": 5007.0,
    "Halpha": 6563.0,
}


def random_mask(n_patches, mask_ratio=config.MASK_RATIO):
    # Returns boolean mask of shape (n_patches,); True = masked
    pass


def domain_informed_mask(n_patches, redshift, mask_ratio=config.MASK_RATIO):
    # Biases masking toward patches containing known spectral lines shifted by redshift
    pass


def apply_mask(patch_tokens, mask, mask_embedding):
    # Replaces masked patch positions with learned mask_embedding
    # patch_tokens: (B, N_PATCHES, D_MODEL)
    # mask: (B, N_PATCHES) bool
    # Returns masked_tokens, mask
    pass
