"""
CMB Polarisation Simulation and Power Spectra Reconstruction

This module simulates Cosmic Microwave Background (CMB) E-mode and B-mode 
polarisation on a flat-sky patch using theoretical power spectra from CLASS. 
It incorporates galactic dust emission models using PySM3 and Healpy, 
and validates the simulation by reconstructing the angular power spectra 
from the generated real-space maps via Fourier transforms.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import healpy as hp
import pysm3
import pysm3.units as u
from classy import Class

# ============================================================
# Configuration & Environment Setup
# ============================================================

os.environ["QT_QPA_PLATFORM"] = "xcb"
# Print thread configuration for debugging/logging
print(f"OMP_NUM_THREADS = {os.getenv('OMP_NUM_THREADS')}")
print(f"NUMEXPR_MAX_THREADS = {os.getenv('NUMEXPR_MAX_THREADS')}")


# ============================================================
# Mathematical Utilities
# ============================================================

def deg2rad(x):
    """
    Converts angles from degrees to radians.

    Args:
        x (float or np.ndarray): Angle in degrees.

    Returns:
        float or np.ndarray: Angle in radians.
    """
    return x * np.pi / 180.0


def radial_average_2d(power2d, ell, nbins=40, ell_max=None):
    """
    Azimuthally averages a 2D Fourier power map into a 1D angular power spectrum (C_ell).

    Args:
        power2d (np.ndarray): 2D array of power in Fourier space.
        ell (np.ndarray): 2D array of multipole (ell) values corresponding to the grid.
        nbins (int, optional): Number of logarithmic bins. Defaults to 40.
        ell_max (float, optional): Maximum multipole to consider. Defaults to None.

    Returns:
        tuple: (ellc, Cl) where ellc are the geometric bin centers and Cl is the averaged power.
    """
    if ell_max is None:
        ell_max = np.nanmax(ell)

    mask = (ell > 0) & (ell <= ell_max) & np.isfinite(power2d)
    ell_flat = ell[mask].ravel()
    p_flat = power2d[mask].ravel()

    bins = np.logspace(np.log10(ell_flat.min()), np.log10(ell_flat.max()), nbins + 1)
    which = np.digitize(ell_flat, bins) - 1

    Cl = np.array([p_flat[which == i].mean() if np.any(which == i) else np.nan for i in range(nbins)])
    ellc = np.sqrt(bins[:-1] * bins[1:])  # Geometric bin centers

    return ellc, Cl


# ============================================================
# Physics & Simulation Modules
# ============================================================

def get_theory_spectra(lmax=3000, r_scale=0.01):
    """
    Computes the theoretical CMB angular power spectra using CLASS.
    The tensor-to-scalar ratio (r) is scaled manually after generating a base r=1 model.

    Args:
        lmax (int, optional): Maximum multipole moment. Defaults to 3000.
        r_scale (float, optional): Target tensor-to-scalar ratio. Defaults to 0.01.

    Returns:
        tuple: (ell, Cl_EE, Cl_BB) arrays representing the multipoles and theoretical spectra in uK^2.
    """
    cosmo = Class()
    cosmo.set({
        'output': 'tCl,pCl',
        'modes': 's,t',  # Include tensors
        'r': 1,  # Set to 1 as a base for easy scaling
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

    # Scale and convert to uK^2
    Cl_BB = cls['bb'] * 1e12 * r_scale
    Cl_EE = cls['ee'] * 1e12

    return ell, Cl_EE, Cl_BB


def get_dust_emission(patch_size=10.0, xsize=256, freq_ghz=150):
    """
    Generates dust emission maps using PySM3 and projects them onto a flat-sky patch.

    Args:
        patch_size (float, optional): Size of the patch in degrees. Defaults to 10.0.
        xsize (int, optional): Number of pixels along one side. Defaults to 256.
        freq_ghz (int, optional): Observation frequency in GHz. Defaults to 150.

    Returns:
        tuple: (Q_dust, U_dust, Cl_dust, elld) containing the projected maps, 
               the full-sky power spectra, and the multipole array.
    """
    sky = pysm3.Sky(nside=512, preset_strings=["d1"], output_unit=u.uK_CMB)
    IQU = sky.get_emission(freq_ghz * u.GHz)

    Q_dust_hp = IQU[1].value
    U_dust_hp = IQU[2].value

    reso = (patch_size * 60) / xsize  # arcmin per pixel
    lon0, lat0 = 0.0, 0.0  # Patch center

    Q_dust = hp.gnomview(Q_dust_hp, rot=[lon0, lat0], xsize=xsize, reso=reso,
                         return_projected_map=True, no_plot=True)
    U_dust = hp.gnomview(U_dust_hp, rot=[lon0, lat0], xsize=xsize, reso=reso,
                         return_projected_map=True, no_plot=True)

    # Compute full-sky angular power spectra for the dust
    Cl_dust_all = hp.sphtfunc.anafast(IQU, pol=True, lmax=3 * 512, alm=False)
    _, Cl_dEE, Cl_dBB, _, _, _ = Cl_dust_all

    elld = np.arange(len(Cl_dBB))

    return Q_dust, U_dust, (Cl_dEE, Cl_dBB), elld


def simulate_flat_sky_cmb(ell, Cl_EE, Cl_BB, patch_size_deg=10.0, N=256):
    """
    Simulates CMB Q and U polarisation maps on a flat-sky patch using Fourier generation.

    Args:
        ell (np.ndarray): Array of multipoles from theory.
        Cl_EE (np.ndarray): Theoretical EE power spectrum.
        Cl_BB (np.ndarray): Theoretical BB power spectrum.
        patch_size_deg (float, optional): Size of the patch in degrees. Defaults to 10.0.
        N (int, optional): Number of pixels per side. Defaults to 256.

    Returns:
        tuple: (Q, U, ell_2d, phi, dx) containing the real-space Q/U maps, 
               the 2D multipole grid, the phase grid, and the pixel resolution.
    """
    L = deg2rad(patch_size_deg)
    dx = L / N

    # Construct Fourier grid
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    kx = 2 * np.pi * np.tile(fx, (N, 1))
    ky = 2 * np.pi * np.tile(fy[:, None], (1, N))

    ell_2d = np.sqrt(kx ** 2 + ky ** 2)
    phi = np.arctan2(ky, kx)

    # Interpolate 1D spectra onto 2D grid
    Cl_interp_BB = interp1d(ell, Cl_BB, bounds_error=False, fill_value=0.0)
    Cl_interp_EE = interp1d(ell, Cl_EE, bounds_error=False, fill_value=0.0)
    Cl_BB_2d = Cl_interp_BB(ell_2d)
    Cl_EE_2d = Cl_interp_EE(ell_2d)

    # Draw random Fourier modes
    rng = np.random.default_rng()
    gauss_cplx_B = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)
    gauss_cplx_E = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)

    amp_B = np.sqrt(np.maximum(Cl_BB_2d, 0.0))
    amp_E = np.sqrt(np.maximum(Cl_EE_2d, 0.0))
    Bk = gauss_cplx_B * amp_B
    Ek = gauss_cplx_E * amp_E

    # Enforce Hermitian symmetry for a real-valued spatial map
    for i in range(N):
        for j in range(N):
            i_sym, j_sym = (-i) % N, (-j) % N
            Bk[i_sym, j_sym] = np.conj(Bk[i, j])
            Ek[i_sym, j_sym] = np.conj(Ek[i, j])

    Bk[0, 0] = 0;
    Bk[np.isnan(Bk)] = 0.0
    Ek[0, 0] = 0;
    Ek[np.isnan(Ek)] = 0.0

    # Convert E/B to Q/U in Fourier space
    sin2 = np.sin(2 * phi)
    cos2 = np.cos(2 * phi)
    Qk = -Bk * sin2 + Ek * cos2
    Uk = Bk * cos2 + Ek * sin2

    # Inverse FFT to real space
    Q = np.fft.ifft2(Qk).real * (N * dx) ** 2
    U = np.fft.ifft2(Uk).real * (N * dx) ** 2

    return Q, U, ell_2d, phi, dx


def reconstruct_spectra(Q, U, dx, N, ell_2d, phi):
    """
    Transforms real-space Q and U maps back to Fourier space to estimate the power spectra.

    Args:
        Q (np.ndarray): Real-space Q polarisation map.
        U (np.ndarray): Real-space U polarisation map.
        dx (float): Pixel resolution in radians.
        N (int): Number of pixels per side.
        ell_2d (np.ndarray): 2D multipole grid.
        phi (np.ndarray): 2D phase angle grid.

    Returns:
        tuple: (ellc, Cl_est) containing the binned multipole centers and estimated (EE, BB) spectra.
    """
    Qk_est = np.fft.fft2(Q) / (N * dx) ** 2
    Uk_est = np.fft.fft2(U) / (N * dx) ** 2

    sin2 = np.sin(2 * phi)
    cos2 = np.cos(2 * phi)

    Ek_est = Qk_est * cos2 + Uk_est * sin2
    Bk_est = -Qk_est * sin2 + Uk_est * cos2

    P_BB_2d = np.abs(Bk_est) ** 2
    P_EE_2d = np.abs(Ek_est) ** 2

    ellc, Cl_BB_est = radial_average_2d(P_BB_2d, ell_2d, nbins=40)
    _, Cl_EE_est = radial_average_2d(P_EE_2d, ell_2d, nbins=40)

    return ellc, (Cl_EE_est, Cl_BB_est)


# ============================================================
# Plotting Implementations
# ============================================================

def plt_powerspectra(Cl_theory, Cl_est, Cl_dust, ellc, ell, elld):
    """
    Plots the theoretical, estimated, and dust angular power spectra (D_ell).
    """
    Cl_EE, Cl_BB = Cl_theory
    Cl_EE_est, Cl_BB_est = Cl_est
    Cl_dEE, Cl_dBB = Cl_dust

    Dl_EE = ell * (ell + 1) * Cl_EE / (2 * np.pi)
    Dl_EE_est = ellc * (ellc + 1) * Cl_EE_est / (2 * np.pi)
    Dl_dEE = elld * (elld + 1) * Cl_dEE / (2 * np.pi)

    Dl_BB = ell * (ell + 1) * Cl_BB / (2 * np.pi)
    Dl_BB_est = ellc * (ellc + 1) * Cl_BB_est / (2 * np.pi)
    Dl_dBB = elld * (elld + 1) * Cl_dBB / (2 * np.pi)

    mask_ell = ell > 0
    mask_ellc = ellc > 0
    mask_elld = elld > 0

    plt.figure(figsize=(15, 8))

    # BB Spectra
    plt.loglog(ell[mask_ell], Dl_BB[mask_ell], label='Unlensed BB Theory (r=0.01)', lw=2)
    plt.loglog(ellc[mask_ellc], Dl_BB_est[mask_ellc], label='Unlensed BB Estimated (r=0.01)', lw=2)
    plt.loglog(elld[mask_elld], Dl_dBB[mask_elld], label='Dust BB', lw=2)

    # EE Spectra
    plt.loglog(ell[mask_ell], Dl_EE[mask_ell], label='Unlensed EE Theory', lw=2)
    plt.loglog(ellc[mask_ellc], Dl_EE_est[mask_ellc], label='Unlensed EE Estimated', lw=2)
    plt.loglog(elld[mask_elld], Dl_dEE[mask_elld], label='Dust EE', lw=2)

    plt.xlabel(r'$\ell$', fontsize=16)
    plt.ylabel(r'$\mathcal{D}_\ell$ [$\mu$K$^2$]', fontsize=16)
    plt.legend(fontsize=14)

    plt.xlim(2, 3000)
    plt.ylim(1e-10, 1e+2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.show()


def plot_polarisation_maps(Q_map, U_map, title_suffix):
    """
    Generates a 3-panel plot for Q, U, and Polarisation Amplitude.
    """
    plt.rcParams.update({'axes.titlesize': 16, 'font.size': 14})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    P_map = np.hypot(Q_map, U_map)
    vmax_qu = max(np.max(np.abs(Q_map)), np.max(np.abs(U_map)))

    im0 = axs[0].imshow(Q_map, origin='lower', cmap='RdBu_r', vmin=-vmax_qu, vmax=vmax_qu)
    axs[0].set_title(f'Q {title_suffix}')
    fig.colorbar(im0, ax=axs[0], shrink=0.85, aspect=20)

    im1 = axs[1].imshow(U_map, origin='lower', cmap='RdBu_r', vmin=-vmax_qu, vmax=vmax_qu)
    axs[1].set_title(f'U {title_suffix}')
    fig.colorbar(im1, ax=axs[1], shrink=0.85, aspect=20)

    im2 = axs[2].imshow(P_map, origin='lower', cmap='viridis')
    axs[2].set_title(f'Pol. Amplitude {title_suffix}')
    fig.colorbar(im2, ax=axs[2], shrink=0.85, aspect=20)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # 1. Generate Theoretical Spectra
    ell, Cl_EE, Cl_BB = get_theory_spectra(lmax=3000, r_scale=0.01)

    # Save theory to disk (using local directory to ensure portability)
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(f"{output_dir}/Cl_theory.txt", np.column_stack([ell, Cl_EE, Cl_BB]),
               header="ell   Cl_EE[μK^2]   Cl_BB[μK^2]")

    # 2. Simulate Primordial Maps
    patch_size = 10.0
    N_pixels = 256
    Q, U, ell_2d, phi, dx = simulate_flat_sky_cmb(ell, Cl_EE, Cl_BB, patch_size_deg=patch_size, N=N_pixels)

    # 3. Generate Dust Maps
    Q_dust, U_dust, Cl_dust, elld = get_dust_emission(patch_size=patch_size, xsize=N_pixels, freq_ghz=150)

    # 4. Combine Signals
    Q_total = Q + Q_dust
    U_total = U + U_dust

    # 5. Reconstruct Spectra (From primordial only, as per original logic)
    ellc, Cl_est = reconstruct_spectra(Q, U, dx, N_pixels, ell_2d, phi)

    # 6. Visualizations
    plt_powerspectra((Cl_EE, Cl_BB), Cl_est, Cl_dust, ellc, ell, elld)

    plot_polarisation_maps(Q, U, "(Primordial B&E-only)")
    plot_polarisation_maps(Q_total, U_total, "(Total = CMB + Dust)")