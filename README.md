# spectra

A tool for simulating the Schumann–Runge bands of molecular oxygen written in Python. Built using NumPy, Polars, PySide6, PyQtGraph, and SciPy, `spectra` is designed to be easily understood and modified.

The capabilities of `spectra` are briefly summarized below. More detailed theory and notation are explained in the included document.

## Background

### Rotational Hamiltonian

The rotational Hamiltonian used for both the $B^3\Sigma_u^-$ and $X^3\Sigma_g^-$ states is

$$
H = H_{r} + H_{ss} + H_{sr},
$$

where

$$
\begin{aligned}
    H_{r}  &= B\mathbf{N}^{2} - D\mathbf{N}^{4} \\
    H_{ss} &= \frac{2}{3}\lambda(3S_{z}^{2} - \mathbf{S}^{2}) \\
    H_{sr} &= \gamma\mathbf{N}\cdot\mathbf{S}.
\end{aligned}
$$

Upper state constants are taken from [*Molecular Spectroscopic Constants of O2: The Upper State of the Schumann–Runge Bands*](https://doi.org/10.1016/0022-2852(86)90196-7) by Cheung et al. (1986). Ground state constants are taken from [*High Resolution Spectral Analysis of Oxygen: IV*](https://doi.org/10.1063/1.4900510) by Yu et al. (2014).

### Spectral Broadening

Convolutions include the effects of both Gaussian and Lorentzian broadening mechanisms. Each type of broadening can be toggled on or off individually.

#### Gaussian

- Thermal Doppler broadening
- Instrument broadening

#### Lorentzian

- Pressure broadening
- Natural broadening
- Predissociation broadening

### Rotational Lines

By default, 40 rotational lines are simulated. Currently, predissociation factors are computed for each rotational line using a polynomial fit that is valid up to $v = 21$ and $J = 40$. Therefore, it is recommended not to exceed $J = 40$ by a large margin if accuracy is to be preserved.

### Equilibrium & Non-equilibrium

In general, the electronic, vibrational, and rotational Boltzmann partition functions are computed assuming Boltzmann population distributions. For equilibrium simulations, the input temperature for all states (translational, electronic, vibrational, and rotational) is the same. Non-equilibrium simulations have different temperatures specified for each state, but the population distributions within each state remain Boltzmann.

### Plot Types

Four plot types are currently implemented:

- Line
  - Each rotational line is plotted at its exact wavenumber.
- Line Info
  - Information is printed above each line for easier identification.
- Convolve Separate
  - The rotational lines within a single vibrational band are convolved.
- Convolve All
  - All vibrational bands are convolved together.

## Example Spectrum

An equilibrium simulation of $\text{O}_2^{16}$ using the following parameters is shown below.

- $(v', v'') = (2, 0)$ and $(4, 1)$
- $N_\text{max} = 35$
- $T=300$ $\text{K}$
- $p=101325$ $\text{Pa}$

![Example Spectrum](img/example.webp)

All user-accessible options are visible in the GUI, including vibrational band selection, broadening toggles, simulation parameters, and plotting options. Each vibrational band has an associated table containing information about all the simulated rotational lines within that band.

## Installation

This repository uses the [uv](https://github.com/astral-sh/uv) package manager for Python. After installing uv, the necessary packages can be installed in a virtual environment by navigating to the repository's root directory and installing the dependencies:

```bash
uv sync
```

Then, navigate to the `src/` directory and run the GUI:

```bash
uv run ./gui.py
```

## Roadmap

### GUI Functionality

- [x] Switch to PyQtGraph instead of Matplotlib for improved plot performance
- [x] Add the ability to export rotational line data from the built-in spreadsheet
- [ ] Build a more intuitive interface for adding/removing simulated and experimental data, especially once multiple molecules are added
- [ ] Design and implement a GUI for LIF computations, including estimated fluorescence yield and the ability to search for rotational line overlaps

### Physics

- [ ] Add support for more diatomic molecules, starting with $\text{NO}$
- [ ] Implement electronic spectra for atomic species
- [ ] Include the ability to view and edit the rotational Hamiltonian on a per-term basis

### Other

- [ ] Package a pre-compiled binary to improve user experience
