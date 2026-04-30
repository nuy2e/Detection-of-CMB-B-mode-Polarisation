from classy import Class
import numpy as np
from scipy.interpolate import interp1d
import random, healpy as hp, pysm3
import pysm3.units as u
import os


# Utility
# ============================================================
def deg2rad(x):
    return x * np.pi / 180.0

def save_sample_npz(path, Q_maps, U_maps, freqs):
    """
    Save one sample (all frequencies) into .npz
    Assumes freqs == [90,150,220] and Q_maps/U_maps are len 3 in that order.
    """
    assert len(freqs) == 3, "save_sample_npz is hard-coded for 3 freqs."
    np.savez_compressed(
        path,
        freqs=np.array(freqs),
        Q90=Q_maps[0],  U90=U_maps[0],
        Q150=Q_maps[1], U150=U_maps[1],
        Q220=Q_maps[2], U220=U_maps[2],
    )

def load_dust_models():
    print("Loading all dust models ...")
    # typical PySM3 presets are d0..d9
    sky_models = {
        f"d{i}": pysm3.Sky(nside=512, preset_strings=[f"d{i}"], output_unit=u.uK_CMB)
        for i in range(0, 10)
    }
    print("Finished loading dust models.")
    return sky_models

def cosmo_compute(r):
    cosmo = Class()
    cosmo.set({
        "output": "tCl,pCl",
        "modes": "s,t",
        "r": r,
        "A_s": 2.1e-9,
        "n_s": 0.965,
        "h": 0.674,
        "omega_b": 0.0224,
        "omega_cdm": 0.120,
        "l_max_scalars": 3000,
        "l_max_tensors": 3000,
        "lensing": "no",
    })
    cosmo.compute()
    cls = cosmo.raw_cl(3000)  # dict with keys 'ell','tt','ee','bb','te'
    return cls

def QU_dust_maps(sky, freqs, N_pix, centre_loc, rot_angle):
    lon0, lat0 = centre_loc
    patch_size = 10.0  # degrees
    reso = (patch_size * 60) / N_pix  # arcmin/pixel as expected by gnomview
    Q_list, U_list = [], []
    for nu in freqs:
        IQU = sky.get_emission(nu * u.GHz)
        Q_hp, U_hp = IQU[1].value, IQU[2].value
        Q_proj = hp.gnomview(Q_hp, rot=[lon0, lat0, rot_angle],
                             xsize=N_pix, reso=reso,
                             return_projected_map=True, no_plot=True)
        U_proj = hp.gnomview(U_hp, rot=[lon0, lat0, rot_angle],
                             xsize=N_pix, reso=reso,
                             return_projected_map=True, no_plot=True)
        Q_list.append(Q_proj.astype(np.float32))
        U_list.append(U_proj.astype(np.float32))
    return Q_list, U_list

def QU_prim_maps(cls, r_val, N_pix):
    ell = cls['ell']
    Cl_BB = cls['bb'] * 1e12 * r_val
    Cl_EE = cls['ee'] * 1e12

    L_deg = 10.0
    L = deg2rad(L_deg)
    N = N_pix
    dx = L / N  # radians per pixel

    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    kx = 2 * np.pi * np.tile(fx, (N, 1))
    ky = 2 * np.pi * np.tile(fy[:, None], (1, N))
    ell_2d = np.sqrt(kx**2 + ky**2)  # flat-sky: |k| ~ ell
    phi = np.arctan2(ky, kx)

    Cl_interp_BB = interp1d(ell, Cl_BB, bounds_error=False, fill_value=0.0)
    Cl_interp_EE = interp1d(ell, Cl_EE, bounds_error=False, fill_value=0.0)

    Cl_BB_2d = Cl_interp_BB(ell_2d)
    Cl_EE_2d = Cl_interp_EE(ell_2d)

    rng = np.random.default_rng()
    gauss_cplx_B = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)
    gauss_cplx_E = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)

    amp_B = np.sqrt(np.maximum(Cl_BB_2d, 0.0))
    amp_E = np.sqrt(np.maximum(Cl_EE_2d, 0.0))
    Bk = gauss_cplx_B * amp_B
    Ek = gauss_cplx_E * amp_E

    # enforce Hermitian symmetry (ensures real maps)
    for i in range(N):
        for j in range(N):
            i_sym = (-i) % N
            j_sym = (-j) % N
            Bk[i_sym, j_sym] = np.conj(Bk[i, j])
    Bk[0, 0] = 0
    Bk[np.isnan(Bk)] = 0.0

    for i in range(N):
        for j in range(N):
            i_sym = (-i) % N
            j_sym = (-j) % N
            Ek[i_sym, j_sym] = np.conj(Ek[i, j])
    Ek[0, 0] = 0
    Ek[np.isnan(Ek)] = 0.0

    # convert to Q_k, U_k in Fourier space
    sin2 = np.sin(2 * phi);
    cos2 = np.cos(2 * phi)
    Qk = -Bk * sin2 + Ek * cos2
    Uk = Bk * cos2 + Ek * sin2

    Q = np.fft.ifft2(Qk).real * (N * dx) ** 2
    U = np.fft.ifft2(Uk).real * (N * dx) ** 2
    return Q.astype(np.float32), U.astype(np.float32)


# ============================================================
# Main
# ============================================================
r = 0.01
total_samples = 1

cls = cosmo_compute(r)
sky_models = load_dust_models()  # always load for contamination generation


save_dir_prim = f"image_data_N{total_samples}_r{r}"
save_dir_dust = f"image_data_dust_N{total_samples}_r{r}"
os.makedirs(save_dir_prim, exist_ok=True)
os.makedirs(save_dir_dust, exist_ok=True)

freq_list = [90, 150, 220]
N_pix = 256
rand = False

for i in range(total_samples):
    if rand:
        r_val = 10 ** np.random.uniform(-3, -1)
        centre_loc = (np.random.uniform(0, 360),
                      np.degrees(np.arcsin(np.random.uniform(-1, 1))))
        rot_angle = np.random.uniform(0, 360)
        dust_model = f"d{random.randint(0, 9)}"
    else:
        r_val = 1000
        centre_loc = (0, 0)
        rot_angle = 0
        dust_model = f"d1"


    # ===== Primordial maps (no frequency stacking) =====
    Q_prim, U_prim = QU_prim_maps(cls, r_val, N_pix)
    prim_path = os.path.join(save_dir_prim, f"sample{i+1:03d}_prim.npz")
    np.savez_compressed(prim_path, Q=Q_prim, U=U_prim)

    # ===== Dust-contaminated maps (with 3 frequencies) =====

    sky = sky_models[dust_model]
    Q_dust_list, U_dust_list = QU_dust_maps(sky, freq_list, N_pix, centre_loc, rot_angle)

    Q_maps_dust, U_maps_dust = [], []
    for j in range(len(freq_list)):
        Q_total = Q_prim + Q_dust_list[j]
        U_total = U_prim + U_dust_list[j]
        Q_maps_dust.append(Q_total.astype(np.float16))
        U_maps_dust.append(U_total.astype(np.float16))

    contam_path = os.path.join(save_dir_dust, f"sample{i+1:03d}_contam.npz")
    save_sample_npz(contam_path, Q_maps_dust, U_maps_dust, freq_list)

    print(f"\rSaved pair {i+1}/{total_samples}", end="", flush=True)

print(f"\nAll {total_samples} primordial and contaminated samples saved.")
