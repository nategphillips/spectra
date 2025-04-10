# module line
"""Contains the implementation of the Line class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import constants
import terms
import utils
from simtype import SimType

if TYPE_CHECKING:
    from band import Band
    from sim import Sim


class Line:
    """Represents a rotational line within a vibrational band."""

    def __init__(
        self,
        sim: Sim,
        band: Band,
        n_qn_up: int,
        n_qn_lo: int,
        j_qn_up: int,
        j_qn_lo: int,
        branch_idx_up: int,
        branch_idx_lo: int,
        branch_name: str,
        is_satellite: bool,
    ) -> None:
        """Initialize class variables.

        Args:
            sim (Sim): Parent simulation.
            band (Band): Vibrational band.
            n_qn_up (int): Upper state rotational quantum number N'.
            n_qn_lo (int): Lower state rotational quantum number N''.
            j_qn_up (int): Upper state rotational quantum number J'.
            j_qn_lo (int): Lower state rotational quantum number J''.
            branch_idx_up (int): Upper branch index.
            branch_idx_lo (int): Lower branch index.
            branch_name (str): Branch name.
            is_satellite (bool): Whether or not the line is a satellite line.
        """
        self.sim: Sim = sim
        self.band: Band = band
        self.n_qn_up: int = n_qn_up
        self.n_qn_lo: int = n_qn_lo
        self.j_qn_up: int = j_qn_up
        self.j_qn_lo: int = j_qn_lo
        self.branch_idx_up: int = branch_idx_up
        self.branch_idx_lo: int = branch_idx_lo
        self.branch_name: str = branch_name
        self.is_satellite: bool = is_satellite
        self.wavenumber: float = self.get_wavenumber()
        self.honl_london_factor: float = self.get_honl_london_factor()
        self.rot_boltz_frac: float = self.get_rot_boltz_frac()
        self.intensity: float = self.get_intensity()

    def fwhm_predissociation(self, is_selected: bool) -> float:
        """Return the predissociation broadening FWHM in [1/cm].

        Args:
            is_selected (bool): True if predissociation broadening should be simulated.

        Returns:
            float: The predissociation broadening FWHM in [1/cm].
        """
        if is_selected:
            # TODO: 24/10/25 - Using the polynomial fit and coefficients described by Lewis, 1986
            #       for the predissociation of all bands for now. The goal is to use experimental
            #       values when available, and use this fit otherwise. The fit is good up to J = 40
            #       and v = 21. Check this to make sure v' and J' should be used even in absorption.

            # FIXME: 25/02/12 - This will break the simulation for some vibrational bands if J
            #        exceeds 40 by too large of a margin.

            a1: float = self.sim.predissociation["a1"][self.band.v_qn_up]
            a2: float = self.sim.predissociation["a2"][self.band.v_qn_up]
            a3: float = self.sim.predissociation["a3"][self.band.v_qn_up]
            a4: float = self.sim.predissociation["a4"][self.band.v_qn_up]
            a5: float = self.sim.predissociation["a5"][self.band.v_qn_up]
            x: int = self.j_qn_up * (self.j_qn_up + 1)

            return a1 + a2 * x + a3 * x**2 + a4 * x**3 + a5 * x**4

        return 0.0

    def fwhm_natural(self, is_selected: bool) -> float:
        """Return the natural broadening FWHM in [1/cm].

        Args:
            is_selected (bool): True if natural broadening should be simulated.

        Returns:
            float: The natural broadening FWHM in [1/cm].
        """
        if is_selected:
            # TODO: 24/10/21 - Look over this, seems weird still.

            # The sum of the Einstein A coefficients for all downward transitions from the two
            # levels of the transitions i and j.
            i: int = self.band.v_qn_up
            a_ik: float = 0.0
            for k in range(0, i):
                a_ik += self.sim.einstein[i][k]

            j: int = self.band.v_qn_lo
            a_jk: float = 0.0
            for k in range(0, j):
                a_jk += self.sim.einstein[j][k]

            # Natural broadening in [1/cm].
            return ((a_ik + a_jk) / (2 * np.pi)) / constants.LIGHT

        return 0.0

    def fwhm_collisional(self, is_selected: bool) -> float:
        """Return the collisional broadening FWHM in [1/cm].

        Args:
            is_selected (bool): True if collisional broadening should be simulated.

        Returns:
            float: The collisional broadening FWHM in [1/cm].
        """
        if is_selected:
            # NOTE: 24/11/05 - In most cases, the amount of electronically excited molecules in the
            #       gas is essentially zero, meaning that most molecules are in the ground state.
            #       Therefore, the ground state radius is used to compute the cross-section. An even
            #       more accurate approach would be to multiply the radius in each state by its
            #       Boltzmann fraction and add them together.
            cross_section: float = (
                np.pi
                * (
                    2
                    * constants.INTERNUCLEAR_DISTANCE[self.sim.molecule.name][
                        self.sim.state_lo.name
                    ]
                )
                ** 2
            )

            # NOTE: 24/10/22 - Both the cross-section and reduced mass refer to the interactions
            #       between two molecules, not the two atoms that compose a molecule. For now, only
            #       homogeneous gases are considered, so the diameter and masses of the two
            #       molecules are identical. The internuclear distance is being used as the
            #       effective radius of the molecule. For homogeneous gases, the reduced mass is
            #       just half the molecular mass (remember, this is for molecule-molecule
            #       interactions).
            reduced_mass: float = self.sim.molecule.mass / 2

            # NOTE: 24/11/05 - The translational tempearature is used for collisional and Doppler
            #       broadening since both effects are direct consequences of the thermal velocity of
            #       molecules.

            # Collisional (pressure) broadening in [1/cm].
            return (
                self.sim.pressure
                * cross_section
                * np.sqrt(8 / (np.pi * reduced_mass * constants.BOLTZ * self.sim.temp_trn))
                / np.pi
            ) / constants.LIGHT

        return 0.0

    def fwhm_doppler(self, is_selected: bool) -> float:
        """Return the Doppler broadening FWHM in [1/cm].

        Args:
            is_selected (bool): True if Doppler broadening should be simulated.

        Returns:
            float: The Doppler broadening FWHM in [1/cm].
        """
        if is_selected:
            # Doppler (thermal) broadening in [1/cm]. Note that the speed of light is converted from
            # [cm/s] to [m/s] to ensure that the units work out correctly.
            return self.wavenumber * np.sqrt(
                8
                * constants.BOLTZ
                * self.sim.temp_trn
                * np.log(2)
                / (self.sim.molecule.mass * (constants.LIGHT / 1e2) ** 2)
            )

        return 0.0

    def fwhm_instrument(self, is_selected: bool, inst_broadening_wl: float) -> float:
        """Return the instrument broadening FWHM in [1/cm].

        Args:
            is_selected (bool): True if instrument broadening should be simulated.
            inst_broadening_wl (float): Instrument broadening FWHM in [nm].

        Returns:
            float: The instrument broadening FWHM in [1/cm].
        """
        if is_selected:
            # NOTE: 25/02/12 - Instrument broadening is passed into this function with units [nm],
            #       so we must convert it to [1/cm]. Note that the FWHM is a bandwidth, so we cannot
            #       simply convert [nm] to [1/cm] in the normal sense - there must be a central
            #       wavelength to expand about.
            return utils.bandwidth_wavelen_to_wavenum(
                utils.wavenum_to_wavelen(self.wavenumber), inst_broadening_wl
            )

        return 0.0

    def get_wavenumber(self) -> float:
        """Return the wavenumber in [1/cm].

        Returns:
            float: The wavenumber of the rotational line in [1/cm].
        """
        # NOTE: 24/10/18 - Make sure to understand transition structure: Herzberg pp. 149-152, and
        #       pp. 168-169.

        # Herzberg p. 168, eq. (IV, 24)
        return (
            self.band.band_origin
            + terms.rotational_term(
                self.sim.state_up, self.band.v_qn_up, self.j_qn_up, self.branch_idx_up
            )
            - terms.rotational_term(
                self.sim.state_lo, self.band.v_qn_lo, self.j_qn_lo, self.branch_idx_lo
            )
        )

    def get_intensity(self) -> float:
        """Return the intensity.

        Returns:
            float: The intensity of the rotational line.
        """
        # NOTE: 24/10/18 - Before going any further make sure to read Herzberg pp. 20-21,
        #       pp. 126-127, pp. 200-201, and pp. 382-383.

        match self.sim.sim_type:
            case SimType.EMISSION:
                j_qn = self.j_qn_up
                wavenumber_factor = self.wavenumber**4
            case SimType.ABSORPTION:
                j_qn = self.j_qn_lo
                wavenumber_factor = self.wavenumber

        return (
            wavenumber_factor
            * self.rot_boltz_frac
            * self.band.vib_boltz_frac
            * self.sim.elc_boltz_frac
            * self.honl_london_factor
            / (2 * j_qn + 1)
            * self.sim.franck_condon[self.band.v_qn_up][self.band.v_qn_lo]
        )

    def get_rot_boltz_frac(self) -> float:
        """Return the rotational Boltzmann fraction, N_J / N.

        Returns:
            float: The vibrational Boltzmann fraction, N_J / N.
        """
        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                v_qn = self.band.v_qn_up
                j_qn = self.j_qn_up
                branch_idx = self.branch_idx_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.band.v_qn_lo
                j_qn = self.j_qn_lo
                branch_idx = self.branch_idx_lo

        return (
            (2 * j_qn + 1)
            * np.exp(
                -terms.rotational_term(state, v_qn, j_qn, branch_idx)
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.sim.temp_rot)
            )
            / self.band.rot_part
        )

    def get_honl_london_factor(self) -> float:
        """Return the Hönl-London (line strength) factor.

        Returns:
            float: The Hönl-London (line strength) factor.
        """
        # For emission, the relevant rotational quantum number is N'; for absorption, it's N''.
        match self.sim.sim_type:
            case SimType.EMISSION:
                n_qn = self.n_qn_up
            case SimType.ABSORPTION:
                n_qn = self.n_qn_lo

        # Convert the properties of the current rotational line into a useful key.
        if self.is_satellite:
            key = f"{self.branch_name}Q{self.branch_idx_up}{self.branch_idx_lo}"
        else:
            # For main branches, the upper and lower branches indicies are the same, so it doesn't
            # matter which one is used here.
            key = f"{self.branch_name}{self.branch_idx_up}"

        # These factors are from Tatum - 1966: Hönl-London Factors for 3Σ±-3Σ± Transitions.
        factors: dict[SimType, dict[str, float]] = {
            SimType.EMISSION: {
                "P1": ((n_qn + 1) * (2 * n_qn + 5)) / (2 * n_qn + 3),
                "R1": (n_qn * (2 * n_qn + 3)) / (2 * n_qn + 1),
                "P2": (n_qn * (n_qn + 2)) / (n_qn + 1),
                "R2": ((n_qn - 1) * (n_qn + 1)) / n_qn,
                "P3": ((n_qn + 1) * (2 * n_qn - 1)) / (2 * n_qn + 1),
                "R3": (n_qn * (2 * n_qn - 3)) / (2 * n_qn - 1),
                "PQ12": 1 / (n_qn + 1),
                "RQ21": 1 / n_qn,
                "PQ13": 1 / ((n_qn + 1) * (2 * n_qn + 1) * (2 * n_qn + 3)),
                "RQ31": 1 / (n_qn * (2 * n_qn - 1) * (2 * n_qn + 1)),
                "PQ23": 1 / (n_qn + 1),
                "RQ32": 1 / n_qn,
            },
            SimType.ABSORPTION: {
                "P1": (n_qn * (2 * n_qn + 3)) / (2 * n_qn + 1),
                "R1": ((n_qn + 1) * (2 * n_qn + 5)) / (2 * n_qn + 3),
                "P2": ((n_qn - 1) * (n_qn + 1)) / n_qn,
                "R2": (n_qn * (n_qn + 2)) / (n_qn + 1),
                "P3": (n_qn * (2 * n_qn - 3)) / (2 * n_qn - 1),
                "R3": ((n_qn + 1) * (2 * n_qn - 1)) / (2 * n_qn + 1),
                "PQ12": 1 / n_qn,
                "RQ21": 1 / (n_qn + 1),
                "PQ13": 1 / (n_qn * (2 * n_qn - 1) * (2 * n_qn + 1)),
                "RQ31": 1 / ((n_qn + 1) * (2 * n_qn + 1) * (2 * n_qn + 3)),
                "PQ23": 1 / n_qn,
                "RQ32": 1 / (n_qn + 1),
            },
        }

        return factors[self.sim.sim_type][key]
