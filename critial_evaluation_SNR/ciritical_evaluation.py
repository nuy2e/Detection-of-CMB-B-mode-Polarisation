"""
CMB B-Mode Signal-to-Noise Ratio (SNR) Forecasting

This module evaluates the detectability of primordial CMB B-modes (parameterized
by the tensor-to-scalar ratio, r) against thermal dust foregrounds. It computes
the theoretical SNR for different CMB experiments (e.g., LiteBIRD, CMB-S4) under
various foreground cleaning efficiencies.

Author: Min Ki Hong
Date: April 2026
"""

import numpy as np
import healpy as hp
from classy import Class
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt


# ============================================================
# Primordial CMB Theory
# ============================================================

def get_primordial_BB(lmax, r_target=0.01):
    """
    Computes the primordial B-mode angular power spectrum using CLASS.
    Calculates the spectrum for r=1.0 and scales it linearly to the target r.

    Args:
        lmax (int): Maximum multipole moment to compute.
        r_target (float, optional): Target tensor-to-scalar ratio. Defaults to 0.01.

    Returns:
        tuple: (ell, Cl_BB) containing multipoles and the B-mode power spectrum in μK^2.
    """
    cosmo = Class()
    cosmo.set({
        'output': 'tCl,pCl',
        'modes': 's,t',
        'r': 1.0,
        'A_s': 2.1e-9,
        'n_s': 0.965,
        'h': 0.674,
        'omega_b': 0.0224,
        'omega_cdm': 0.120,
        'l_max_scalars': lmax,
        'l_max_tensors': lmax,
        'lensing': 'no'
    })
    cosmo.compute()

    cls = cosmo.raw_cl(lmax)
    ell = cls['ell']
    cl_bb_base = cls['bb']

    cosmo.struct_cleanup()
    cosmo.empty()

    # Convert from K^2 to μK^2 and scale to desired r
    cl_bb_uk2 = cl_bb_base * 1e12 * r_target

    return ell, cl_bb_uk2


# ============================================================
# Galactic Foregrounds (Thermal Dust)
# ============================================================

def latitude_mask(nside, b_cut_deg):
    """
    Creates a binary healpix mask excluding the galactic plane.

    Args:
        nside (int): Healpix resolution parameter.
        b_cut_deg (float): Galactic latitude cut in degrees (e.g., 20 means mask |b| < 20).

    Returns:
        np.ndarray: Binary mask array of size npix.
    """
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    b = np.pi / 2.0 - theta

    mask = np.ones(npix)
    mask[np.abs(b) < np.deg2rad(b_cut_deg)] = 0
    return mask


def get_dust_BB(lmax, nside=512, freq_ghz=150.0, dust_preset="d1"):
    """
    Generates a thermal dust map using PySM3, masks the galactic plane,
    and computes the pseudo-Cl B-mode power spectrum.

    Args:
        lmax (int): Maximum multipole moment for the power spectrum.
        nside (int, optional): Healpix resolution. Defaults to 512.
        freq_ghz (float, optional): Evaluation frequency in GHz. Defaults to 150.0.
        dust_preset (str, optional): PySM3 model preset. Defaults to "d1".

    Returns:
        tuple: (ell, Cl_BB_dust) containing multipoles and the pseudo power spectrum.
    """
    sky = pysm3.Sky(nside=nside, preset_strings=[dust_preset])
    nu = freq_ghz * u.GHz

    I_map, Q_map, U_map = sky.components[0].get_emission(nu)

    # Mask the brightest galactic emission
    mask = latitude_mask(nside, b_cut_deg=20)
    I_map *= mask
    Q_map *= mask
    U_map *= mask

    maps = [I_map, Q_map, U_map]

    # anafast returns TT, EE, BB, TE, EB, TB
    cls_all = hp.anafast(maps, pol=True, lmax=lmax)
    cl_bb = cls_all[2]

    ell = np.arange(len(cl_bb))
    return ell, cl_bb


# ============================================================
# Statistical Forecasting
# ============================================================

def compute_snr(ell, Cl_prim, Cl_dust, f_fg, f_sky, ell_min, ell_max):
    """
    Computes the total Signal-to-Noise Ratio (SNR) for the primordial B-mode.

    Uses the Knox formula, assuming the total variance is dominated by the
    residual foregrounds (ignoring instrumental noise and lensing for this proxy).

    Args:
        ell (np.ndarray): Multipole array.
        Cl_prim (np.ndarray): Primordial B-mode power spectrum.
        Cl_dust (np.ndarray): Dust B-mode power spectrum.
        f_fg (float): Fraction of dust power remaining after component separation (e.g., 0.01 for 1% remaining).
        f_sky (float): Fraction of the sky observed by the experiment.
        ell_min (int): Minimum multipole accessible to the experiment.
        ell_max (int): Maximum multipole accessible to the experiment.

    Returns:
        float: Calculated SNR.
    """
    # Total observed spectrum: primordial + residual dust
    Cl_obs = Cl_prim + (f_fg * Cl_dust)

    # Restrict to experiment's spatial sensitivity range
    mask = (ell >= ell_min) & (ell <= ell_max) & (ell > 0)
    ell_sel = ell[mask]
    Cl_prim_sel = Cl_prim[mask]
    Cl_obs_sel = Cl_obs[mask]

    # Prevent division by zero
    valid = Cl_obs_sel > 0
    ell_sel = ell_sel[valid]
    Cl_prim_sel = Cl_prim_sel[valid]
    Cl_obs_sel = Cl_obs_sel[valid]

    # Calculate cumulative SNR across available multipoles
    snr2 = np.sum((2.0 * ell_sel + 1.0) * (f_sky / 2.0) * (Cl_prim_sel / Cl_obs_sel) ** 2)

    return np.sqrt(snr2)


# ============================================================
# Visualization
# ============================================================

def plot_foreground_residuals(ell, Cl_prim, Cl_dust, f_fg_list):
    """
    Plots the D_l power spectra for the primordial signal alongside various
    levels of dust residuals.
    """
    # Convert C_l to D_l [μK^2]
    Dl_prim = ell * (ell + 1) * Cl_prim / (2.0 * np.pi)

    plt.figure(figsize=(9, 6))

    # Plot Primordial (ignoring ell=0, 1)
    plt.loglog(ell[2:], Dl_prim[2:], label=r"Primordial BB ($r = 0.01$)", color='black', linewidth=2.5)

    # Plot Dust Residuals
    for f_fg in f_fg_list:
        Dl_dust = ell * (ell + 1) * (f_fg * Cl_dust) / (2.0 * np.pi)
        plt.loglog(ell[2:], Dl_dust[2:], linestyle="--", label=rf"Residual Dust ($f_{{fg}} = {f_fg:g}$)")

    plt.xlabel(r"Multipole $\ell$", fontsize=14)
    plt.ylabel(r"$\mathcal{D}_\ell^{BB} \equiv \ell(\ell+1)C_\ell^{BB}/2\pi\ \, [\mu\mathrm{K}^2]$", fontsize=14)
    plt.title("Primordial vs. Residual Dust B-Mode Power Spectra (150 GHz)", fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    # 1. Configuration
    nside = 512
    lmax = 3 * nside - 1
    freq_ghz = 150.0
    r_target = 0.01

    print("Generating theoretical models...")

    # 2. Compute Base Spectra
    ell_prim, Cl_prim = get_primordial_BB(lmax=lmax, r_target=r_target)
    ell_dust, Cl_dust = get_dust_BB(lmax=lmax, nside=nside, freq_ghz=freq_ghz)

    # Align multipole arrays
    ell = np.minimum(ell_prim, ell_dust)
    Cl_prim = Cl_prim[:len(ell)]
    Cl_dust = Cl_dust[:len(ell)]

    # 3. Define Experimental Configurations
    experiments = {
        "LiteBIRD-like (Satellite)": {
            "f_sky": 0.7,
            "ell_min": 30,  # Recombination bump sensitivity
            "ell_max": 150,
        },
        "CMB-S4-like (Ground)": {
            "f_sky": 0.04,
            "ell_min": 80,  # Missing lowest multipoles due to atmosphere/ground limits
            "ell_max": 150,
        },
    }

    # 4. Evaluate SNR for different cleaning efficiencies
    # f_fg = 1.0 (raw sky), 0.1 (90% cleaned), 0.01 (99% cleaned), etc.
    f_fg_list = [1.0, 0.1, 0.07, 0.05, 0.03, 0.01]

    for exp_name, cfg in experiments.items():
        print(f"\n=== {exp_name} ===")
        print(f"Sky Fraction: {cfg['f_sky']} | ell range: [{cfg['ell_min']}, {cfg['ell_max']}]")
        print("-" * 45)

        for f_fg in f_fg_list:
            snr = compute_snr(
                ell=ell,
                Cl_prim=Cl_prim,
                Cl_dust=Cl_dust,
                f_fg=f_fg,
                f_sky=cfg["f_sky"],
                ell_min=cfg["ell_min"],
                ell_max=cfg["ell_max"],
            )
            print(f"Residual Dust Fraction = {f_fg:6.3g}  ->  Estimated SNR = {snr:6.3f}")

    # 5. Visualize Results
    plot_foreground_residuals(ell, Cl_prim, Cl_dust, f_fg_list)