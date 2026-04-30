# CMB B-mode Polarisation & CNN Component Separation

## Overview
This repository contains the computational pipeline developed as part of a Third Year BSc Dissertation on the detection of Cosmic Microwave Background (CMB) B-mode polarisation. 

The codebase explores the feasibility of using deep learning, specifically a U-Net Convolutional Neural Network (CNN), to perform component separation, isolating the faint primordial B-mode signal from overwhelming Galactic thermal dust foregrounds. It also includes theoretical Signal-to-Noise Ratio (SNR) forecasting for upcoming CMB missions.

## Executive Summary & Key Results
For those short on time, the primary conclusions of this computational project are:

* **The Challenge of Dynamic Range in CNNs:** The U-Net toy model successfully reproduced overall map amplitudes but failed to recover the detailed spatial statistics (i.e., acoustic peaks in the EE and B-mode power spectra). This highlights a major hurdle in machine learning for cosmology: the network underfits due to the extreme dynamic range difference between the faint primordial signal ($\sim 10^{-6}\mu\text{K}$) and dominant dust foregrounds ($\sim 10\mu\text{K}$). 
* **Foreground Suppression Requirements:** Analytical SNR forecasting demonstrates that for a LiteBIRD-like satellite mission (70% sky coverage), polarised thermal dust emission must be suppressed by **$\ge 97\\%$** to achieve a meaningful $3\sigma$ detection of the primordial recombination bump ($30 < \ell < 150$).
* **Future Outlook:** While CNNs offer promising non-linear modelling capabilities, extracting precise cosmological power spectra requires deeper architectures, non-Euclidean (spherical) implementations, and rigorous uncertainty quantification before they can rival established semi-blind algorithms like SMICA or Commander.

## Repository Structure

The pipeline is modularized into four distinct stages:

### 1. `simulation/`
* **`cmb_simulation.py`**: Computes the baseline theoretical CMB angular power spectra using the `CLASS` cosmology code (unlensed, including tensors at $r=0.01$).
* **`output/Cl_theory.txt`**: The saved theoretical power spectra used as the ground truth reference.

### 2. `image_generation/`
* **`cmb_sim_image_generation.py`**: Simulates flat-sky $10^\circ \times 10^\circ$ patches of the CMB using Fourier space generation. It combines these pure primordial maps with highly detailed thermal dust foregrounds generated via `PySM3` (d1 model) across three frequency channels (90, 150, 220 GHz).
* **`image files/`**: Output directories containing the `.npz` arrays of the clean and contaminated training/testing datasets.

### 3. `cmb_CNN/`
* **`cmb_CNN_train.py`**: Defines and trains a PyTorch U-Net architecture. It learns to map 6-channel contaminated inputs (Q and U at 3 frequencies) back to the 2-channel pure primordial CMB maps (Q and U).
* **`cmb_CNN_evaluation.py`**: Evaluates the trained model on unseen test maps. It converts the CNN's spatial predictions back into Fourier space to reconstruct the $\mathcal{D}_\ell$ power spectra and compares them against the `CLASS` theory.
* **`model_saved/`**: Contains the saved `.pth` weights from the trained neural network.

### 4. `crtical_evaluation_SNR/`
* **`ciritical_evaluation.py`**: An analytical script calculating the expected Signal-to-Noise Ratio for detecting the primordial B-mode signal under various levels of foreground residual contamination. It evaluates targets for both LiteBIRD-like (satellite) and CMB-S4-like (ground) experimental configurations.

## Requirements & Environment
**Operating System:** This pipeline strongly requires a Unix-based environment (**Linux** or **macOS**). Windows users must use **WSL (Windows Subsystem for Linux)**. This constraint is due to the `classy` (Cosmic Linear Anisotropy Solving System) and `healpy` packages, which do not compile natively on Windows.

To run this pipeline, you need the following scientific libraries:
* `torch` (PyTorch)
* `numpy`
* `matplotlib`
* `scipy`
* `healpy` (for spherical harmonics and PySM3 projection)
* `pysm3` (for Galactic dust modelling)
* `classy` (Cosmic Linear Anisotropy Solving System)

## Author
* **Min Ki Hong**
* **Institution:** School of Physics and Astronomy, University of Manchester (2025/2026)
* **Note:** This codebase represents the computational appendix of the dissertation: *"Detection of CMB B-mode Polarisation: Experimental Challenges and Computational Methods."*
