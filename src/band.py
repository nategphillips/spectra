# module band
"""Contains the implementation of the Band class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import constants
import convolve
import terms
import utils
from line import Line
from simtype import SimType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sim import Sim


class Band:
    """Represents a vibrational band of a particular molecule."""

    def __init__(self, sim: Sim, v_qn_up: int, v_qn_lo: int) -> None:
        """Initialize class variables.

        Args:
            sim (Sim): Parent simulation.
            v_qn_up (int): Upper vibrational quantum number v'.
            v_qn_lo (int): Lower vibrational quantum number v''.
        """
        self.sim: Sim = sim
        self.v_qn_up: int = v_qn_up
        self.v_qn_lo: int = v_qn_lo
        self.band_origin: float = self.get_band_origin()
        self.rot_part: float = self.get_rot_partition_fn()
        self.vib_boltz_frac: float = self.get_vib_boltz_frac()
        self.lines: list[Line] = self.get_lines()

    def wavenumbers_line(self) -> NDArray[np.float64]:
        """Return an array of wavenumbers, one for each line.

        Returns:
            NDArray[np.float64]: All discrete rotational line wavenumbers belonging to the
                vibrational band.
        """
        return np.array([line.wavenumber for line in self.lines])

    def intensities_line(self) -> NDArray[np.float64]:
        """Return an array of intensities, one for each line.

        Returns:
            NDArray[np.float64]: All rotational line intensities belonging to the vibrational band.
        """
        return np.array([line.intensity for line in self.lines])

    def wavenumbers_conv(self, inst_broadening_wl: float, granularity: int) -> NDArray[np.float64]:
        """Return an array of convolved wavenumbers.

        Args:
            inst_broadening_wl (float): Instrument broadening FWHM in [nm].
            granularity (int): Number of points on the wavenumber axis.

        Returns:
            NDArray[np.float64]: A continuous range of wavenumbers.
        """
        # A qualitative amount of padding added to either side of the x-axis limits. Ensures that
        # spectral features at either extreme are not clipped when the FWHM parameters are large.
        # The first line's instrument FWHM is chosen as an arbitrary reference to keep things
        # simple. The minimum Gaussian FWHM allowed is 2 to ensure that no clipping is encountered.
        padding: float = 10.0 * max(self.lines[0].fwhm_instrument(True, inst_broadening_wl), 2)

        # The individual line wavenumbers are only used to find the minimum and maximum bounds of
        # the spectrum since the spectrum itself is no longer quantized.
        wns_line: NDArray[np.float64] = self.wavenumbers_line()

        # Generate a fine-grained x-axis using existing wavenumber data.
        return np.linspace(wns_line.min() - padding, wns_line.max() + padding, granularity)

    def intensities_conv(
        self,
        fwhm_selections: dict[str, bool],
        inst_broadening_wl: float,
        wavenumbers_conv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return an array of convolved intensities.

        Args:
            fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
            inst_broadening_wl (float): Instrument broadening FWHM in [nm].
            wavenumbers_conv (NDArray[np.float64]): The convolved wavelengths to use.

        Returns:
            NDArray[np.float64]: A continuous range of intensities.
        """
        return convolve.convolve(
            self.lines,
            wavenumbers_conv,
            fwhm_selections,
            inst_broadening_wl,
        )

    def get_vib_boltz_frac(self) -> float:
        """Return the vibrational Boltzmann fraction N_v / N.

        Returns:
            float: The vibrational Boltzmann fraction, N_v / N.
        """
        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.v_qn_lo

        # NOTE: 24/10/25 - Calculates the vibrational Boltzmann fraction with respect to the
        #       zero-point vibrational energy to match the vibrational partition function.
        return (
            np.exp(
                -(terms.vibrational_term(state, v_qn) - terms.vibrational_term(state, 0))
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.sim.temp_vib)
            )
            / self.sim.vib_part
        )

    def get_band_origin(self) -> float:
        """Return the band origin in [1/cm].

        Returns:
            float: The band origin in [1/cm].
        """
        # Herzberg p. 168, eq. (IV, 24)

        upper_state: dict[str, list[float]] = self.sim.state_up.constants
        lower_state: dict[str, list[float]] = self.sim.state_lo.constants

        # NOTE: 24/11/05 - In the Cheung paper, the electronic energy is defined differently than in
        #       Herzberg's book. The conversion specified by Cheung on p. 5 is
        #       nu_0 = T + 2 / 3 * lamda - gamma.
        energy_offset: float = (
            2 / 3 * upper_state["lamda"][self.v_qn_up] - upper_state["gamma"][self.v_qn_up]
        )

        # NOTE: 24/11/05 - The band origin as defined by Herzberg is nu_0 = nu_e + nu_v, and is
        #       different for each vibrational transition. The T values in Cheung include the
        #       vibrational term for each level, i.e. T = T_e + G. The ground state has no
        #       electronic energy, so it is not subtracted. In Cheung's data, the term values
        #       provided are measured above the zeroth vibrational level of the ground state. This
        #       means that the lower state zero-point vibrational energy must be used.
        return (
            upper_state["T"][self.v_qn_up]
            + energy_offset
            - (lower_state["G"][self.v_qn_lo] - lower_state["G"][0])
        )

    def get_rot_partition_fn(self) -> float:
        """Return the rotational partition function, Q_r.

        Returns:
            float: The rotational partition function, Q_r.
        """
        # TODO: 24/10/25 - Add nuclear effects to make this the effective rotational partition
        #       function.

        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.v_qn_lo

        q_r: float = 0.0

        # NOTE: 24/10/22 - The rotational partition function is always computed using the same
        #       number of lines. At reasonable temperatures (~300 K), only around 50 rotational
        #       lines contribute to the state sum. However, at high temperatures (~3000 K), at least
        #       100 lines need to be considered to obtain an accurate estimate of the state sum.
        #       This approach is used to ensure the sum is calculated correctly regardless of the
        #       number of rotational lines simulated by the user.
        for j_qn in range(201):
            # TODO: 24/10/22 - Not sure which branch index should be used here. The triplet energies
            #       are all close together, so it shouldn't matter too much. Averaging could work,
            #       but I'm not sure if this is necessary.
            q_r += (2 * j_qn + 1) * np.exp(
                -terms.rotational_term(state, v_qn, j_qn, 2)
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.sim.temp_rot)
            )

        # NOTE: 24/10/22 - Alternatively, the high-temperature approximation can be used instead of
        #       the direct sum approach. This also works well.

        # q_r = (
        #     constants.BOLTZ
        #     * self.sim.temp_rot
        #     / (constants.PLANC * constants.LIGHT * state.constants["B"][v_qn])
        # )

        # The state sum must be divided by the symmetry parameter to account for identical
        # rotational orientations in space.
        return q_r / self.sim.molecule.symmetry_param

    def get_lines(self) -> list[Line]:
        """Return a list of all allowed rotational lines.

        Returns:
            list[Line]: A list of all allowed `Line` objects for the given selection rules.
        """
        lines: list[Line] = []

        for n_qn_up in self.sim.rot_lvls:
            for n_qn_lo in self.sim.rot_lvls:
                # Ensure the rotational selection rules corresponding to each electronic state are
                # properly followed.
                if self.sim.state_up.is_allowed(n_qn_up) & self.sim.state_lo.is_allowed(n_qn_lo):
                    lines.extend(self.allowed_branches(n_qn_up, n_qn_lo))

        return lines

    def allowed_branches(self, n_qn_up: int, n_qn_lo: int) -> list[Line]:
        """Determine the selection rules for Hund's case (b).

        Args:
            n_qn_up (int): Upper state rotational quantum number N'.
            n_qn_lo (int): Lower state rotational quantum number N''.

        Raises:
            ValueError: If the spin multiplicity of the two electronic states do not match.

        Returns:
            list[Line]: A list of `Line` objects for all allowed branches.
        """
        # For Σ-Σ transitions, the rotational selection rules are ∆N = ±1, ∆N ≠ 0.
        # Herzberg p. 244, eq. (V, 44)

        lines: list[Line] = []

        # Determine how many lines should be present in the fine structure of the molecule due to
        # the effects of spin multiplicity.
        if self.sim.state_up.spin_multiplicity == self.sim.state_lo.spin_multiplicity:
            branch_range: range = range(1, self.sim.state_up.spin_multiplicity + 1)
        else:
            raise ValueError("Spin multiplicity of the two electronic states do not match.")

        delta_n_qn: int = n_qn_up - n_qn_lo

        # R branch
        if delta_n_qn == 1:
            lines.extend(self.branch_index(n_qn_up, n_qn_lo, branch_range, "R"))
        # Q branch
        if delta_n_qn == 0:
            # Note that the Q branch doesn't exist for the Schumann-Runge bands of O2.
            lines.extend(self.branch_index(n_qn_up, n_qn_lo, branch_range, "Q"))
        # P branch
        elif delta_n_qn == -1:
            lines.extend(self.branch_index(n_qn_up, n_qn_lo, branch_range, "P"))

        return lines

    def branch_index(
        self, n_qn_up: int, n_qn_lo: int, branch_range: range, branch_name: str
    ) -> list[Line]:
        """Return the rotational lines within a given branch.

        Args:
            n_qn_up (int): Upper state rotational quantum number N'.
            n_qn_lo (int): Lower state rotational quantum number N''.
            branch_range (range): Range of branches corresponding to the spin multiplicity.
            branch_name (str): The name of the branch, e.g. R, Q, or P.

        Returns:
            list[Line]: A list of `Line` objects within a given branch.
        """

        def add_line(branch_idx_up: int, branch_idx_lo: int, is_satellite: bool) -> None:
            """Create and append a rotational line."""
            lines.append(
                Line(
                    sim=self.sim,
                    band=self,
                    n_qn_up=n_qn_up,
                    n_qn_lo=n_qn_lo,
                    j_qn_up=utils.n_to_j(n_qn_up, branch_idx_up),
                    j_qn_lo=utils.n_to_j(n_qn_lo, branch_idx_lo),
                    branch_idx_up=branch_idx_up,
                    branch_idx_lo=branch_idx_lo,
                    branch_name=branch_name,
                    is_satellite=is_satellite,
                )
            )

        # Herzberg pp. 249-251, eqs. (V, 48-53)

        # NOTE: 24/10/16 - Every transition has 6 total lines (3 main + 3 satellite) except for the
        #       N' = 0 to N'' = 1 transition, which has 3 total lines (1 main + 2 satellite).

        lines: list[Line] = []

        # Handle the special case where N' = 0 (only the P1, PQ12, and PQ13 lines exist).
        if n_qn_up == 0:
            if branch_name == "P":
                add_line(1, 1, False)
            for branch_idx_lo in (2, 3):
                add_line(1, branch_idx_lo, True)

            return lines

        # Handle regular cases for other N'.
        for branch_idx_up in branch_range:
            for branch_idx_lo in branch_range:
                # Main branches: R1, R2, R3, P1, P2, P3
                if branch_idx_up == branch_idx_lo:
                    add_line(branch_idx_up, branch_idx_lo, False)
                # Satellite branches: RQ31, RQ32, RQ21
                elif (
                    (branch_name == "R")
                    and (branch_idx_up > branch_idx_lo)
                    or (branch_name == "P")
                    and (branch_idx_up < branch_idx_lo)
                ):
                    add_line(branch_idx_up, branch_idx_lo, True)

        return lines
