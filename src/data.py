import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config


class DESISpectraDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        # Returns: flux tensor (PADDED_LEN,), redshift scalar
        pass

    def _normalize(self, flux):
        # Median/MAD normalization
        pass

    def _pad(self, flux):
        # Zero-pad from SPECTRUM_LEN to PADDED_LEN
        pass


def get_dataloaders(train_path, val_path, batch_size=config.BATCH_SIZE, max_samples=None):
    pass
