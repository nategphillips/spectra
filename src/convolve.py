# module convolve
"""Contains functions used for convolution."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz

from line import Line


def broadening_fn(
    wavenumbers_conv: NDArray[np.float64],
    line: Line,
    fwhm_selections: dict[str, bool],
    inst_broadening_wl: float,
) -> NDArray[np.float64]:
    """Return the contribution of a single rotational line to the total spectra.

    Uses a Voigt probability density function.

    Args:
        wavenumbers_conv (NDArray[np.float64]): A continuous array of wavenumbers.
        line (Line): A rotational `Line` object.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        inst_broadening_wl (float): Instrument broadening FWHM in [nm].

    Returns:
        NDArray[np.float64]: The Voigt probability density function for a single rotational line.
    """
    # Instrument broadening in [1/cm] is added to thermal broadening to get the full Gaussian FWHM.
    # Note that Gaussian FWHMs must be summed in quadrature: see "Hypersonic Nonequilibrium Flows:
    # Fundamentals and Recent Advances" p. 361.
    fwhm_gaussian: float = np.sqrt(
        line.fwhm_instrument(fwhm_selections["instrument"], inst_broadening_wl) ** 2
        + line.fwhm_doppler(fwhm_selections["doppler"]) ** 2
    )

    # NOTE: 24/10/25 - Since predissociating repulsive states have no interfering absorption, the
    #       broadened absorption lines will be Lorentzian in shape. See Julienne, 1975.

    # Add the effects of natural, collisional, and predissociation broadening to get the full
    # Lorentzian FWHM. Lorentzian FHWMs are summed linearly: see "Hypersonic Nonequilibrium Flows:
    # Fundamentals and Recent Advances" p. 361.
    fwhm_lorentzian: float = (
        line.fwhm_natural(fwhm_selections["natural"])
        + line.fwhm_collisional(fwhm_selections["collisional"])
        + line.fwhm_predissociation(fwhm_selections["predissociation"])
    )

    # If only Gaussian FWHM parameters are present, then return a Gaussian profile.
    if (fwhm_gaussian > 0.0) and (fwhm_lorentzian == 0.0):
        return np.exp(-((wavenumbers_conv - line.wavenumber) ** 2) / (2 * fwhm_gaussian**2)) / (
            fwhm_gaussian * np.sqrt(2 * np.pi)
        )

    # Similarly, if only Lorentzian FWHM parameters exist, then return a Lorentzian profile.
    if (fwhm_gaussian == 0.0) and (fwhm_lorentzian > 0.0):
        return np.divide(
            fwhm_lorentzian,
            (np.pi * ((wavenumbers_conv - line.wavenumber) ** 2 + fwhm_lorentzian**2)),
        )

    # TODO: 25/02/14 - Should check if both Gaussian and Lorentzian FWHM params are zero here and
    #       return an error if so.

    # Otherwise, compute the argument of the complex Faddeeva function and return a Voigt profile.
    z: NDArray[np.float64] = ((wavenumbers_conv - line.wavenumber) + 1j * fwhm_lorentzian) / (
        fwhm_gaussian * np.sqrt(2)
    )

    # The probability density function for the Voigt profile.
    return np.real(wofz(z)) / (fwhm_gaussian * np.sqrt(2 * np.pi))


def convolve(
    lines: list[Line],
    wavenumbers_conv: NDArray[np.float64],
    fwhm_selections: dict[str, bool],
    inst_broadening_wl: float,
) -> NDArray[np.float64]:
    """Convolve a discrete number of spectral lines into a continuous spectra.

    Args:
        lines (list[Line]): A list of rotational `Line` objects.
        wavenumbers_conv (NDArray[np.float64]): A continuous array of wavenumbers.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        inst_broadening_wl (float): Instrument broadening FWHM in [nm].

    Returns:
        NDArray[np.float64]: The total intensity spectrum with contributions from all lines.
    """
    intensities_conv: NDArray[np.float64] = np.zeros_like(wavenumbers_conv)

    # TODO: 25/02/12 - See if switching to scipy's convolve method improves the speed of this,
    #       especially with a large number of bands or points.

    # Add the effects of each line to the continuous spectra by computing its broadening function
    # multiplied by its intensity and adding it to the total intensity.
    for line in lines:
        intensities_conv += line.intensity * broadening_fn(
            wavenumbers_conv, line, fwhm_selections, inst_broadening_wl
        )

    return intensities_conv
