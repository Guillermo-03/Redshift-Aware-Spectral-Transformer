import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import SpectralTransformer
from data import get_dataloaders
from masking import random_mask, apply_mask
import config


def joint_loss(recon_pred, recon_target, z_pred, z_target, lambda_z=config.LAMBDA_REDSHIFT):
    # L_total = L_recon + lambda_z * L_redshift
    pass


def train_one_epoch(model, loader, optimizer, device):
    pass


def validate(model, loader, device):
    pass


def train(train_path, val_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(train_path, val_path)
    model = SpectralTransformer().to(device)

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{config.EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best.pt"))


if __name__ == "__main__":
    train("data/train.hdf5", "data/val.hdf5")
