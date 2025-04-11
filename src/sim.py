# module sim
"""Contains the implementation of the Sim class."""

import numpy as np
import polars as pl
from numpy.typing import NDArray

import constants
import terms
import utils
from band import Band
from molecule import Molecule
from simtype import SimType
from state import State


class Sim:
    """Simulate the spectra of a particular molecule."""

    def __init__(
        self,
        sim_type: SimType,
        molecule: Molecule,
        state_up: State,
        state_lo: State,
        rot_lvls: NDArray[np.int64],
        temp_trn: float,
        temp_elc: float,
        temp_vib: float,
        temp_rot: float,
        pressure: float,
        bands: list[tuple[int, int]],
    ) -> None:
        """Initialize class variables.

        Args:
            sim_type (SimType): The type of simulation to perform.
            molecule (Molecule): Which molecule to simulate.
            state_up (State): Upper electronic state.
            state_lo (State): Lower electronic state.
            rot_lvls (NDArray[np.int64]): Which rotational levels to simulate.
            temp_trn (float): Translational temperature.
            temp_elc (float): Electronic temperature.
            temp_vib (float): Vibrational temperature.
            temp_rot (float): Rotational temperature.
            pressure (float): Pressure.
            bands (list[tuple[int, int]]): Which vibrational bands to simulate.
        """
        self.sim_type: SimType = sim_type
        self.molecule: Molecule = molecule
        self.state_up: State = state_up
        self.state_lo: State = state_lo
        self.rot_lvls: NDArray[np.int64] = rot_lvls
        self.temp_trn: float = temp_trn
        self.temp_elc: float = temp_elc
        self.temp_vib: float = temp_vib
        self.temp_rot: float = temp_rot
        self.pressure: float = pressure
        self.elc_part: float = self.get_elc_partition_fn()
        self.vib_part: float = self.get_vib_partition_fn()
        self.elc_boltz_frac: float = self.get_elc_boltz_frac()
        self.franck_condon: NDArray[np.float64] = self.get_franck_condon()
        self.einstein: NDArray[np.float64] = self.get_einstein()
        self.predissociation: dict[str, list[float]] = self.get_predissociation()
        self.bands: list[Band] = self.get_bands(bands)

    def all_line_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Combine the line data for all vibrational bands."""
        wavenumbers_line: NDArray[np.float64] = np.array([])
        intensities_line: NDArray[np.float64] = np.array([])

        for band in self.bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, band.intensities_line()))

        return wavenumbers_line, intensities_line

    def all_conv_data(
        self, fwhm_selections: dict[str, bool], inst_broadening_wl: float, granularity: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create common axes for superimposing the convolved data of all vibrational bands."""
        # NOTE: 25/02/12 - In the case of overlapping lines, the overall absorption coefficient is
        # expressed as a sum over the individual line absorption coefficients. See "Analysis of
        # Collision-Broadened and Overlapped Spectral Lines to Obtain Individual Line Parameters" by
        # BelBruno (1981).

        # The total span of wavenumbers from all bands.
        wavenumbers_line: NDArray[np.float64] = np.concatenate(
            [band.wavenumbers_line() for band in self.bands]
        )

        # A qualitative amount of padding added to either side of the x-axis limits. Ensures that
        # spectral features at either extreme are not clipped when the FWHM parameters are large.
        # The first line's Doppler FWHM is chosen as an arbitrary reference to keep things simple.
        # The minimum Gaussian FWHM allowed is 2 to ensure that no clipping is encountered.
        padding: float = 10.0 * max(
            self.bands[0].lines[0].fwhm_instrument(True, inst_broadening_wl), 2
        )

        grid_min: float = wavenumbers_line.min() - padding
        grid_max: float = wavenumbers_line.max() + padding

        # Create common wavenumber and intensity grids to hold all of the vibrational band data.
        wavenumbers_conv: NDArray[np.float64] = np.linspace(
            grid_min, grid_max, granularity, dtype=np.float64
        )
        intensities_conv: NDArray[np.float64] = np.zeros_like(wavenumbers_conv)

        # The wavelength axis is common to all vibrational bands so that their contributions to the
        # spectra can be summed.
        for band in self.bands:
            intensities_conv += band.intensities_conv(
                fwhm_selections, inst_broadening_wl, wavenumbers_conv
            )

        return wavenumbers_conv, intensities_conv

    def get_predissociation(self) -> dict[str, list[float]]:
        """Return polynomial coefficients for computing predissociation linewidths."""
        return pl.read_csv(
            utils.get_data_path("data", self.molecule.name, "predissociation", "lewis_coeffs.csv")
        ).to_dict(as_series=False)

    def get_einstein(self) -> NDArray[np.float64]:
        """Return a table of Einstein coefficients for spontaneous emission: A_{v'v''}.

        Rows correspond to the upper state vibrational quantum number (v'), while columns correspond
        to the lower state vibrational quantum number (v'').
        """
        return np.loadtxt(
            utils.get_data_path(
                "data",
                self.molecule.name,
                "einstein",
                f"{self.state_up.name}_to_{self.state_lo.name}_laux.csv",
            ),
            delimiter=",",
        )

    def get_franck_condon(self) -> NDArray[np.float64]:
        """Return a table of Franck-Condon factors for the associated electronic transition.

        Rows correspond to the upper state vibrational quantum number (v'), while columns correspond
        to the lower state vibrational quantum number (v'').
        """
        return np.loadtxt(
            utils.get_data_path(
                "data",
                self.molecule.name,
                "franck-condon",
                f"{self.state_up.name}_to_{self.state_lo.name}_cheung.csv",
            ),
            delimiter=",",
        )

    def get_bands(self, bands: list[tuple[int, int]]):
        """Return the selected vibrational bands within the simulation."""
        return [Band(sim=self, v_qn_up=band[0], v_qn_lo=band[1]) for band in bands]

    def get_vib_partition_fn(self) -> float:
        """Return the vibrational partition function."""
        # NOTE: 24/10/22 - The maximum vibrational quantum number is dictated by the tabulated data
        #       available.
        match self.sim_type:
            case SimType.EMISSION:
                state = self.state_up
                v_qn_max = len(self.state_up.constants["G"])
            case SimType.ABSORPTION:
                state = self.state_lo
                v_qn_max = len(self.state_lo.constants["G"])

        q_v: float = 0.0

        # NOTE: 24/10/22 - The vibrational partition function is always computed using a set number
        #       of vibrational bands to ensure an accurate estimate of the state sum is obtained.
        #       This approach is used to ensure the sum is calculated correctly regardless of the
        #       number of vibrational bands simulated by the user.
        for v_qn in range(0, v_qn_max):
            # NOTE: 24/10/25 - The zero-point vibrational energy is used as a reference to which all
            #       other vibrational energies are measured. This ensures the state sum begins at a
            #       value of 1 when v = 0.
            q_v += np.exp(
                -(terms.vibrational_term(state, v_qn) - terms.vibrational_term(state, 0))
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.temp_vib)
            )

        return q_v

    def get_elc_partition_fn(self) -> float:
        """Return the electronic partition function."""
        energies: list[float] = list(constants.ELECTRONIC_ENERGIES[self.molecule.name].values())
        degeneracies: list[int] = list(
            constants.ELECTRONIC_DEGENERACIES[self.molecule.name].values()
        )

        q_e: float = 0.0

        # NOTE: 24/10/25 - This sum is basically unnecessary since the energies of electronic states
        #       above the ground state are so high. This means that any contribution to the
        #       electronic partition function from anything other than the ground state is
        #       negligible.
        for e, g in zip(energies, degeneracies):
            q_e += g * np.exp(
                -e * constants.PLANC * constants.LIGHT / (constants.BOLTZ * self.temp_elc)
            )

        return q_e

    def get_elc_boltz_frac(self) -> float:
        """Return the electronic Boltzmann fraction N_e / N."""
        match self.sim_type:
            case SimType.EMISSION:
                state = self.state_up.name
            case SimType.ABSORPTION:
                state = self.state_lo.name

        energy: float = constants.ELECTRONIC_ENERGIES[self.molecule.name][state]
        degeneracy: int = constants.ELECTRONIC_DEGENERACIES[self.molecule.name][state]

        return (
            degeneracy
            * np.exp(
                -energy * constants.PLANC * constants.LIGHT / (constants.BOLTZ * self.temp_elc)
            )
            / self.elc_part
        )
