# module main
"""A simulation of the Schumann-Runge bands of molecular oxygen written in Python."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from atom import Atom
from molecule import Molecule
from sim import Sim
from simtype import SimType
from state import State

if TYPE_CHECKING:
    from numpy.typing import NDArray


def main() -> None:
    """Entry point."""
    molecule: Molecule = Molecule(name="O2", atom_1=Atom("O"), atom_2=Atom("O"))

    state_up: State = State(name="B3Su-", spin_multiplicity=3, molecule=molecule)
    state_lo: State = State(name="X3Sg-", spin_multiplicity=3, molecule=molecule)

    bands: list[tuple[int, int]] = [(2, 0), (4, 1)]

    # TODO: 24/10/25 - Implement an option for switching between equilibrium and nonequilibrium
    #       simulations.

    temp: float = 300.0

    sim: Sim = Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        rot_lvls=np.arange(0, 40),
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=101325.0,
        bands=bands,
    )

    sample: NDArray[np.float64] = np.genfromtxt(
        fname=Path("..", "data", "samples", "harvard_20.csv"), delimiter=",", skip_header=1
    )
    wns_samp: NDArray[np.float64] = sample[:, 0]
    ins_samp: NDArray[np.float64] = sample[:, 1] / sample[:, 1].max()

    plt.plot(wns_samp, ins_samp, label="sample")

    inst_broadening_wl: float = 0.0
    granularity: int = int(1e4)
    fwhm_selections: dict[str, bool] = {
        "instrument": True,
        "doppler": True,
        "natural": True,
        "collisional": True,
        "predissociation": True,
    }

    # Find the max intensity in all the bands.
    max_intensity: float = max(
        band.intensities_conv(
            fwhm_selections,
            inst_broadening_wl,
            band.wavenumbers_conv(inst_broadening_wl, granularity),
        ).max()
        for band in sim.bands
    )

    # Plot all bands normalized to one while conserving the relative intensities between bands.
    for band in sim.bands:
        plt.plot(
            band.wavenumbers_conv(inst_broadening_wl, granularity),
            band.intensities_conv(
                fwhm_selections,
                inst_broadening_wl,
                band.wavenumbers_conv(inst_broadening_wl, granularity),
            )
            / max_intensity,
            label=f"band: {band.v_qn_up, band.v_qn_lo}",
        )

    # Convolve all bands together and normalize to one.
    wns, ins = sim.all_conv_data(fwhm_selections, inst_broadening_wl, granularity)
    ins /= ins.max()

    plt.plot(wns, ins, label="all convolved")

    # Interpolate simulated data to have the same number of points as the experimental data and
    # compute the residual.
    ins_inrp: NDArray[np.float64] = np.interp(sample[:, 0], wns, ins)
    residual: NDArray[np.float64] = np.abs(ins_samp - ins_inrp)

    # Show residual below the main data for clarity.
    plt.plot(wns_samp, -residual, label="residual")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
