import torch
import numpy as np
from model import SpectralTransformer
from data import get_dataloaders
from masking import random_mask, apply_mask
import config


def evaluate(model, loader, device):
    """Returns dict with redshift_mae, recon_mse, and delta_z_outlier_rate."""
    pass


def load_and_evaluate(checkpoint_path, test_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SpectralTransformer().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(test_path, test_path, batch_size=256)
    metrics = evaluate(model, test_loader, device)

    print(f"Redshift MAE:       {metrics['redshift_mae']:.4f}")
    print(f"Reconstruction MSE: {metrics['recon_mse']:.4f}")
    print(f"Outlier rate (|Δz|>0.1): {metrics['outlier_rate']:.3f}")
    return metrics


if __name__ == "__main__":
    load_and_evaluate(
        checkpoint_path="checkpoints/best.pt",
        test_path="data/test.hdf5",
    )
