# module utils
"""Contains useful utility functions."""

from typing import overload

import numpy as np
from numpy.typing import NDArray


def n_to_j(n_qn: int, branch_idx: int) -> int:
    """Convert the rotational quantum number from N to J.

    Args:
        n_qn (int): Rotational quantum number N.
        branch_idx (int): Branch index. The total number of branches (and therefore the conversion
            from N to J) is dependent on the spin multiplicity of the molecule.

    Raises:
        ValueError: If the branch index cannot be found.

    Returns:
        int: The rotational quantum number J.
    """
    # For Hund's case (b), spin multiplicity 3.
    match branch_idx:
        case 1:
            # F1: J = N + 1
            return n_qn + 1
        case 2:
            # F2: J = N
            return n_qn
        case 3:
            # F3: J = N - 1
            return n_qn - 1
        case _:
            raise ValueError(f"Unknown branch index: {branch_idx}.")


@overload
def wavenum_to_wavelen(wavenumber: float) -> float: ...


@overload
def wavenum_to_wavelen(wavenumber: NDArray[np.float64]) -> NDArray[np.float64]: ...


def wavenum_to_wavelen(wavenumber: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Convert wavenumbers to wavelengths.

    Args:
        wavenumber (float | NDArray[np.float64]): Wavenumber(s) in [1/cm].

    Returns:
        float | NDArray[np.float64]: The corresponding wavelength(s) in [nm].
    """
    return 1.0 / wavenumber * 1e7


def bandwidth_wavelen_to_wavenum(center_wl: float, fwhm_wl: float) -> float:
    """Convert a FWHM bandwidth from [nm] to [1/cm] given a center wavelength.

    Note that this is not a linear approximation, so it is accurate for large FWHM parameters. See
    https://toolbox.lightcon.com/tools/bandwidthconverter for details.

    Args:
        center_wl (float): Center wavelength in [nm] around which the bandwidth is defined.
        fwhm_wl (float): FWHM bandwidth in [nm].

    Returns:
        float: The FWHM bandwidth in [1/cm].
    """
    return 1e7 * fwhm_wl / (center_wl**2 - fwhm_wl**2 / 4)
