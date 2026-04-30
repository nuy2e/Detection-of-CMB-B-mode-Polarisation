"""
CMB Foreground Removal using U-Net Convolutional Neural Networks

This module defines a PyTorch Dataset and a U-Net architecture to perform 
component separation on Cosmic Microwave Background (CMB) polarisation maps. 
It trains the network to map 6-channel contaminated inputs (Q and U maps across 
3 frequency bands: 90, 150, and 220 GHz) to 2-channel clean primordial maps (Q and U).

Author: Min Ki Hong
Date: April 2026
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# ============================================================
# Dataset Definition
# ============================================================

class CMBRemovalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading simulated CMB polarisation maps.

    Expects the input data to be stored in compressed .npz files. The input 
    (contaminated) maps should contain 6 channels, and the target (primordial) 
    maps should contain 2 channels (Q and U).

    Attributes:
        contam_dir (str): Directory path for contaminated maps.
        prim_dir (str): Directory path for primordial maps.
        contam_files (list): Sorted list of contaminated file names.
        prim_files (list): Sorted list of primordial file names.
    """

    def __init__(self, contam_dir, prim_dir):
        self.contam_dir = contam_dir
        self.prim_dir = prim_dir
        self.contam_files = sorted([f for f in os.listdir(contam_dir) if f.endswith('.npz')])
        self.prim_files = sorted([f for f in os.listdir(prim_dir) if f.endswith('.npz')])

        assert len(self.contam_files) == len(self.prim_files), "Mismatch in dataset sizes."

    def __len__(self):
        return len(self.contam_files)

    def __getitem__(self, idx):
        contam_path = os.path.join(self.contam_dir, self.contam_files[idx])
        prim_path = os.path.join(self.prim_dir, self.prim_files[idx])

        contam_data = np.load(contam_path)
        prim_data = np.load(prim_path)

        # Stack contaminated maps in fixed frequency order: 90, 150, 220 GHz
        contam = np.stack([
            contam_data["Q90"], contam_data["U90"],
            contam_data["Q150"], contam_data["U150"],
            contam_data["Q220"], contam_data["U220"],
        ], axis=0)  # Shape: (6, H, W)

        # Extract primordial Q and U maps
        if "Q" in prim_data.files and "U" in prim_data.files:
            prim = np.stack([prim_data["Q"], prim_data["U"]], axis=0)  # Shape: (2, H, W)
        else:
            first_two = prim_data.files[:2]
            prim = np.stack([prim_data[k] for k in first_two], axis=0)

        x = torch.tensor(contam, dtype=torch.float32)
        y = torch.tensor(prim, dtype=torch.float32)

        return x, y


# ============================================================
# Model Architecture
# ============================================================

class CMBRemovalUNet(nn.Module):
    """
    A lightweight U-Net architecture for image-to-image translation.

    Takes a 6-channel input (Q/U at 3 frequencies) and reconstructs a 
    2-channel output (clean primordial Q/U).
    """

    def __init__(self):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )

        # Decoder (Upsampling with skip connections)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)  # Output: 2 channels (Q, U)
        )

    def forward(self, x):
        """Forward pass through the U-Net."""
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))

        x_up = self.up2(xb)
        x_dec2 = self.dec2(torch.cat([x_up, x2], dim=1))

        x_up_final = self.up1(x_dec2)
        out = self.dec1(torch.cat([x_up_final, x1], dim=1))

        return out


# ============================================================
# Training & Evaluation Utilities
# ============================================================

def train_model(model, model_name, train_dl, val_dl, device, epochs, lr=1e-4):
    """
    Executes the training loop for the U-Net model.

    Args:
        model (nn.Module): The U-Net model.
        model_name (str): Identifier used for saving checkpoints.
        train_dl (DataLoader): Training data loader.
        val_dl (DataLoader): Validation data loader.
        device (torch.device): Device to run the training on (CPU or CUDA).
        epochs (int): Total number of training epochs.
        lr (float, optional): Learning rate. Defaults to 1e-4.

    Returns:
        tuple: (model, train_losses, val_losses)
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # L1 loss is robust to outliers

    # Ensure save directory exists
    save_dir = "model_saved"
    os.makedirs(save_dir, exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e}")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"cmb_cnn_{model_name}_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("Training complete.")
    return model, train_losses, val_losses


def plot_results(train_losses, val_losses, save_path=None):
    """Plots the training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L1 Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def show_example(model, dataset, device, idx=0):
    """
    Visualizes an example prediction against the ground truth.

    Args:
        model (nn.Module): Trained model.
        dataset (Dataset): The dataset to pull the sample from.
        device (torch.device): Compute device.
        idx (int, optional): Index of the sample to visualize. Defaults to 0.
    """
    model.eval()
    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)  # Shape: (2, H, W)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    stokes_labels = ["Q", "U"]

    for i, stokes in enumerate(stokes_labels):
        # Input (Averaged across all 6 channels for rough visual reference)
        ax[i, 0].imshow(np.mean(x.numpy(), axis=0), cmap='coolwarm')
        ax[i, 0].set_title(f"Input (Avg of 6 ch)")
        ax[i, 0].axis('off')

        # Ground Truth
        ax[i, 1].imshow(y.numpy()[i], cmap='coolwarm')
        ax[i, 1].set_title(f"True {stokes}")
        ax[i, 1].axis('off')

        # Prediction
        ax[i, 2].imshow(pred.numpy()[i], cmap='coolwarm')
        ax[i, 2].set_title(f"Predicted {stokes}")
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_dl, device):
    """Evaluates the model over the test dataset."""
    model.eval()
    loss_fn = nn.L1Loss()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()

    return total_loss / len(test_dl)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    # Dataset Configuration
    contam_dir = "../image_generation/image_data_dust_N70_r1000"
    prim_dir = "../image_generation/image_data_N70_r1000"

    dataset = CMBRemovalDataset(contam_dir, prim_dir)
    n_total = len(dataset)

    # Train/Val/Test Split (70% / 15% / 15%)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    model = CMBRemovalUNet()

    try:
        # Start Training
        model, train_losses, val_losses = train_model(
            model,
            model_name="N70_r1000",
            train_dl=train_dl,
            val_dl=val_dl,
            device=device,
            epochs=70
        )

        # Post-training analysis
        plot_results(train_losses, val_losses)

        test_loss = evaluate_model(model, test_dl, device)
        print(f"Final Test Loss: {test_loss:.4e}")

        show_example(model, dataset, device, idx=0)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
    except Exception as e:
        print(f"An error occurred: {e}")