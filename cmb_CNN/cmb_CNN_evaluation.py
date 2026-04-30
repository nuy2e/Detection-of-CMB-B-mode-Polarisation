"""
CMB CNN Evaluation and Power Spectra Recovery

This module evaluates the trained U-Net model on unseen test data. It predicts
the clean primordial Q and U polarisation maps from dust-contaminated inputs.
It then transforms these spatial maps into Fourier space to reconstruct the
E-mode and B-mode angular power spectra, comparing the CNN's predictions against
both the ground truth (pure CMB) and theoretical CLASS models.

Author: Min Ki Hong
Date: April 2026
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Model Architecture
# ============================================================

class CMBRemovalUNet(nn.Module):
    """
    U-Net architecture for CMB component separation.
    Maps 6-channel contaminated inputs (Q/U at 3 frequencies) to
    2-channel clean primordial outputs (Q/U).
    """

    def __init__(self):
        super().__init__()

        # Encoder
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

        # Decoder
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
        """Forward pass through the network."""
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        x = self.up2(xb)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return x


# ============================================================
# Inference & Spatial Plotting
# ============================================================

def predict_single(model, contam_path, prim_path=None, device="cpu"):
    """
    Runs inference on a single contaminated sample to predict primordial maps.

    Args:
        model (nn.Module): Trained U-Net model.
        contam_path (str): Path to the contaminated .npz file.
        prim_path (str, optional): Path to the true primordial .npz file.
        device (torch.device): Compute device.

    Returns:
        tuple: (y_pred, y_true) where both are numpy arrays of shape (2, H, W).
               y_true is None if prim_path is not provided.
    """
    d = np.load(contam_path)
    contam = np.stack([
        d["Q90"], d["U90"],
        d["Q150"], d["U150"],
        d["Q220"], d["U220"],
    ], axis=0)  # Shape: (6, H, W)

    x = torch.tensor(contam, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(x).cpu().squeeze(0).numpy()  # Shape: (2, H, W)

    y_true = None
    if prim_path is not None:
        prim_data = np.load(prim_path)
        if "Q" in prim_data.files and "U" in prim_data.files:
            y_true = np.stack([prim_data["Q"], prim_data["U"]], axis=0)
        else:
            keys = prim_data.files[:2]
            y_true = np.stack([prim_data[k] for k in keys], axis=0)

    return y_pred, y_true


def plot_space_results(contam_path, y_pred, y_true):
    """
    Plots the spatial Q and U maps comparing Contaminated, True, and Predicted fields.
    """
    fig, ax = plt.subplots(2, 3 if y_true is not None else 2, figsize=(12, 7))
    titles = ["Contaminated (mean of 6 ch)", "True", "Predicted"]

    # Calculate input mean for visualization purposes
    d = np.load(contam_path)
    example_input = np.mean(np.stack([
        d["Q90"], d["U90"],
        d["Q150"], d["U150"],
        d["Q220"], d["U220"],
    ]), axis=0)

    stokes_labels = ["Q", "U"]
    for i, stokes in enumerate(stokes_labels):
        im0 = ax[i, 0].imshow(example_input, cmap="coolwarm")
        ax[i, 0].set_title(f"{titles[0]}")
        fig.colorbar(im0, ax=ax[i, 0], fraction=0.046, pad=0.04)

        if y_true is not None:
            im1 = ax[i, 1].imshow(y_true[i], cmap="coolwarm")
            ax[i, 1].set_title(f"True {stokes}")
            fig.colorbar(im1, ax=ax[i, 1], fraction=0.046, pad=0.04)

            im2 = ax[i, 2].imshow(y_pred[i], cmap="coolwarm")
            ax[i, 2].set_title(f"Predicted {stokes}")
            fig.colorbar(im2, ax=ax[i, 2], fraction=0.046, pad=0.04)
        else:
            im1 = ax[i, 1].imshow(y_pred[i], cmap="coolwarm")
            ax[i, 1].set_title(f"Predicted {stokes}")
            fig.colorbar(im1, ax=ax[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# ============================================================
# Power Spectra Reconstruction
# ============================================================

def deg2rad(x):
    """Converts degrees to radians."""
    return x * np.pi / 180.0


def radial_average_2d(power2d, ell, nbins=40, ell_max=None):
    """
    Azimuthally averages a 2D power map into 1D angular power spectrum (C_ell).
    """
    if ell_max is None:
        ell_max = np.nanmax(ell)

    mask = (ell > 0) & (ell <= ell_max) & np.isfinite(power2d)
    ell_flat = ell[mask].ravel()
    p_flat = power2d[mask].ravel()

    bins = np.logspace(np.log10(ell_flat.min()), np.log10(ell_flat.max()), nbins + 1)
    which = np.digitize(ell_flat, bins) - 1

    Cl = np.array([p_flat[which == i].mean() if np.any(which == i) else np.nan for i in range(nbins)])
    ellc = np.sqrt(bins[:-1] * bins[1:])  # geometric bin centers

    return ellc, Cl


def calculate_power_spectra(Q_map, U_map, patch_size_deg=10.0, N_pixels=256):
    """
    Calculates the E-mode and B-mode power spectra from flat-sky Q and U maps.

    Args:
        Q_map (np.ndarray): 2D Q polarisation map.
        U_map (np.ndarray): 2D U polarisation map.
        patch_size_deg (float): Size of the patch in degrees.
        N_pixels (int): Number of pixels per side.

    Returns:
        tuple: (ellc, Cl_EE, Cl_BB) containing bin centers and reconstructed spectra.
    """
    L = deg2rad(patch_size_deg)
    dx = L / N_pixels

    fx = np.fft.fftfreq(N_pixels, d=dx)
    fy = np.fft.fftfreq(N_pixels, d=dx)
    kx = 2 * np.pi * np.tile(fx, (N_pixels, 1))
    ky = 2 * np.pi * np.tile(fy[:, None], (1, N_pixels))

    ell_2d = np.sqrt(kx ** 2 + ky ** 2)
    phi = np.arctan2(ky, kx)

    sin2 = np.sin(2 * phi)
    cos2 = np.cos(2 * phi)

    # FFT to Fourier space
    Qk = np.fft.fft2(Q_map) / (N_pixels * dx) ** 2
    Uk = np.fft.fft2(U_map) / (N_pixels * dx) ** 2

    # E/B separation
    Ek = Qk * cos2 + Uk * sin2
    Bk = -Qk * sin2 + Uk * cos2

    # Calculate 2D power
    P_BB_2d = np.abs(Bk) ** 2
    P_EE_2d = np.abs(Ek) ** 2

    # Azimuthal averaging
    ellc, Cl_BB = radial_average_2d(P_BB_2d, ell_2d, nbins=40)
    _, Cl_EE = radial_average_2d(P_EE_2d, ell_2d, nbins=40)

    return ellc, Cl_EE, Cl_BB


def plot_powerspectra(Cl_prim_est, Cl_pred_est, Cl_theory, ellc):
    """
    Plots the theoretical, true estimated, and CNN-predicted angular power spectra.
    """
    ell, Cl_EE_theory, Cl_BB_theory = Cl_theory[:, 0], Cl_theory[:, 1], Cl_theory[:, 2]

    # Scale C_l to D_l
    Dl_BB_theory = ell * (ell + 1) * Cl_BB_theory / (2 * np.pi)
    Dl_EE_theory = ell * (ell + 1) * Cl_EE_theory / (2 * np.pi)

    Cl_EE_prim_est, Cl_BB_prim_est = Cl_prim_est
    Dl_EE_prim_est = ellc * (ellc + 1) * Cl_EE_prim_est / (2 * np.pi)
    Dl_BB_prim_est = ellc * (ellc + 1) * Cl_BB_prim_est / (2 * np.pi)

    Cl_EE_pred_est, Cl_BB_pred_est = Cl_pred_est
    Dl_EE_pred_est = ellc * (ellc + 1) * Cl_EE_pred_est / (2 * np.pi)
    Dl_BB_pred_est = ellc * (ellc + 1) * Cl_BB_pred_est / (2 * np.pi)

    # Plot constraints
    mask_ellc = (ellc > 50) & (ellc < 1500)
    mask_ell = (ell > 50) & (ell < 1500)

    plt.figure(figsize=(17, 11))

    # --- B-Mode Spectra ---
    plt.loglog(ellc[mask_ellc], Dl_BB_prim_est[mask_ellc], color='#87CEEB', lw=2.2,
               alpha=0.7, ls="--", label='True Primordial BB (Estimated)')
    plt.loglog(ellc[mask_ellc], Dl_BB_pred_est[mask_ellc], color='#20B2AA', lw=2.0,
               label='CNN Predicted BB')
    plt.loglog(ell[mask_ell], Dl_BB_theory[mask_ell], color="b", lw=2.5,
               label="Theoretical BB (CLASS, r=1000)")

    # --- E-Mode Spectra ---
    plt.loglog(ellc[mask_ellc], Dl_EE_prim_est[mask_ellc], color='#FFA07A', lw=2.2,
               alpha=0.7, ls="--", label='True Primordial EE (Estimated)')
    plt.loglog(ellc[mask_ellc], Dl_EE_pred_est[mask_ellc], color='#FFD700', lw=2.0,
               label='CNN Predicted EE')
    plt.loglog(ell[mask_ell], Dl_EE_theory[mask_ell], color="r", lw=2.5,
               label="Theoretical EE (CLASS)")

    # --- Formatting ---
    plt.xlabel(r'$\ell$', fontsize=35)
    plt.ylabel(r'$\mathcal{D}_\ell \equiv \frac{\ell(\ell +1)}{2 \pi} C_\ell$ [$\mu$K$^2$]', fontsize=35)
    plt.legend(fontsize=18, loc='lower left')
    plt.title("Angular Power Spectra: Theory vs. Ground Truth vs. CNN Prediction", fontsize=35, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=25)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main Evaluation Wrapper
# ============================================================

def evaluate_and_plot(model, device, contam_test_path, prim_test_path, theory_path):
    """
    Orchestrates the evaluation pipeline: loads theory, runs model inference,
    plots spatial maps, calculates power spectra, and plots Fourier results.
    """
    model.eval()

    # Load Theoretical Spectra
    try:
        Cl_theory = np.loadtxt(theory_path, comments="#")
    except FileNotFoundError:
        print(f"Warning: Theoretical spectra not found at {theory_path}.")
        return

    # Spatial Inference
    y_pred, y_true = predict_single(model, contam_test_path, prim_test_path, device=device)
    plot_space_results(contam_test_path, y_pred, y_true)

    Q_prim, U_prim = y_true[0], y_true[1]
    Q_pred, U_pred = y_pred[0], y_pred[1]

    # Calculate Power Spectra
    ellc, Cl_EE_prim_est, Cl_BB_prim_est = calculate_power_spectra(Q_prim, U_prim)
    _, Cl_EE_pred_est, Cl_BB_pred_est = calculate_power_spectra(Q_pred, U_pred)

    Cl_prim_est = (Cl_EE_prim_est, Cl_BB_prim_est)
    Cl_pred_est = (Cl_EE_pred_est, Cl_BB_pred_est)

    # Plot Spectra Comparison
    plot_powerspectra(Cl_prim_est, Cl_pred_est, Cl_theory, ellc)


# ============================================================
# Execution Entry Point
# ============================================================

if __name__ == "__main__":

    # Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load Model Weights
    model = CMBRemovalUNet().to(device)
    weights_path = "model_saved/cmb_cnn_N7_r1000_54.pth"

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find model weights at {weights_path}.")
        exit(1)

    # Define File Paths
    contam_test_path = "../image_generation/image_data_test/sample001_contam.npz"
    prim_test_path = "../image_generation/image_data_test/sample001_prim.npz"
    theory_path = "../simulation/output/Cl_theory_1000.txt"

    # Run Evaluation
    evaluate_and_plot(model, device, contam_test_path, prim_test_path, theory_path)