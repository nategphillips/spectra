# module lif
"""A three-level LIF model for the Schumann-Runge bands of molecular oxygen."""

from dataclasses import dataclass
from pathlib import Path
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import scipy as sy
from numpy.typing import NDArray

import constants
from atom import Atom
from line import Line
from molecule import Molecule
from sim import Sim
from simtype import SimType
from state import State

MIN_TIME: float = 0.0
MAX_TIME: float = 60e-9
N_TIME: int = 1000
N_FLUENCE: int = 100


@dataclass
class RateParams:
    """Holds parameters related to the rate equations.

    Attributes:
        a_21 (float): Einstein coefficient for spontaneous emission.
        b_12 (float): Einstein coefficient for photon absorption.
        b_21 (float): Einstein coefficient for stimulated emission.
        w_c (float): Collisional refilling rate.
        w_d (float): Predissociation rate.
        w_f (float): Fluorescent radiative decay rate.
        w_q (float): Collisional quenching rate.
    """

    a_21: float
    b_12: float
    b_21: float
    w_c: float
    w_d: float
    w_f: float
    w_q: float


@dataclass
class LaserParams:
    """Holds parameters related to the laser.

    Attributes:
        pulse_center (float): Center of the laser pulse in [s].
        pulse_width (float): Width of the laser pulse in [s].
        fluence (float): Laser energy per unit area in [J/cm^2].
    """

    pulse_center: float
    pulse_width: float
    fluence: float


@overload
def laser_intensity(t: float, laser_params: LaserParams) -> float: ...


@overload
def laser_intensity(t: NDArray[np.float64], laser_params: LaserParams) -> NDArray[np.float64]: ...


def laser_intensity(
    t: float | NDArray[np.float64], laser_params: LaserParams
) -> float | NDArray[np.float64]:
    """Return the laser intensity for a given time point or array of time points.

    Calculates the laser intensity based on provided laser parameters. When given a scalar time
    input, it returns a scalar intensity value. When given an array of time points, it returns an
    array of intensity values.

    Args:
        t (float | NDArray[np.float64]): Time point(s) at which to calculate the intensity.
        laser_params (LaserParams): Parameters defining the laser beam properties.

    Returns:
        float | NDArray[np.float64]: Laser intensity at the specified time point(s).
    """
    return (
        laser_params.fluence
        / laser_params.pulse_width
        * np.sqrt(4 * np.log(2) / np.pi)
        * np.exp(-4 * np.log(2) * ((t - laser_params.pulse_center) / laser_params.pulse_width) ** 2)
    )


def rate_equations(
    n: list[float],
    t: float,
    rate_params: RateParams,
    laser_params: LaserParams,
    line: Line,
) -> list[float]:
    """Return the rate equations governing the three-level LIF system.

    Args:
        n (list[float]): Nondimensional population density of N1, N2, and N3 at a point in time.
        t (float): Current time in [s].
        rate_params (RateParams): Rate parameters and Einstein coefficients for the system.
        laser_params (LaserParams): Laser parameters.
        line (Line): The rotational `Line` object of interest.

    Returns:
        list[float]: Differential equations dN1/dt, dN2/dt, and dN3/dt.
    """
    n1, n2, n3 = n

    # TODO: 24/10/29 - Implement the overlap integral between the transition and laser lineshapes.
    overlap_integral: float = 1.5  # [cm]

    f_b: float = line.rot_boltz_frac

    i_l: float = laser_intensity(t, laser_params)
    w_la: float = i_l * rate_params.b_12 * overlap_integral / constants.LIGHT
    w_le: float = i_l * rate_params.b_21 * overlap_integral / constants.LIGHT

    dn1_dt: float = -w_la * n1 + n2 * (w_le + rate_params.a_21) + rate_params.w_c * (n3 - n1)
    dn2_dt: float = w_la * n1 - n2 * (
        w_le + rate_params.w_d + rate_params.a_21 + rate_params.w_f + rate_params.w_q
    )
    dn3_dt: float = -rate_params.w_c * f_b / (1 - f_b) * (n3 - n1)

    return [dn1_dt, dn2_dt, dn3_dt]


def simulate(
    t: NDArray[np.float64], rate_params: RateParams, laser_params: LaserParams, line: Line
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the population densities of the three states as functions of time.

    Args:
        t (NDArray[np.float64]): The time domain to simulate over in [s].
        rate_params (RateParams): Rate parameters and Einstein coefficients for the system.
        laser_params (LaserParams): Laser parameters.
        line (Line): The rotational `Line` object of interest.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: The normalized
            population densities N1, N2, and N3 as functions of time.
    """
    n: list[float] = [1.0, 0.0, 1.0]

    solution: NDArray[np.float64] = sy.integrate.odeint(
        rate_equations, n, t, args=(rate_params, laser_params, line)
    )

    n1: NDArray[np.float64] = solution[:, 0]
    n2: NDArray[np.float64] = solution[:, 1]
    n3: NDArray[np.float64] = solution[:, 2]

    return n1, n2, n3


def get_signal(
    t: NDArray[np.float64], n2: NDArray[np.float64], rate_params: RateParams
) -> NDArray[np.float64]:
    """Return the LIF signal as a function of time.

    Args:
        t (NDArray[np.float64]): The time domain to simulate over in [s].
        n2 (NDArray[np.float64]): Normalized population density of state 2.
        rate_params (RateParams): Rate parameters and Einstein coefficients for the system.

    Returns:
        NDArray[np.float64]: The total integrated LIF signal from state 2 as a function of time.
    """
    return rate_params.w_f * sy.integrate.cumulative_trapezoid(n2, t, initial=0)


def get_sim(
    molecule: Molecule,
    state_up: State,
    state_lo: State,
    temp: float,
    pres: float,
    v_qn_up: int,
    v_qn_lo: int,
) -> Sim:
    """Return a simulation object with the desired parameters.

    Args:
        molecule (Molecule): Molecule of interest.
        state_up (State): Upper electronic state.
        state_lo (State): Lower electronic state.
        temp (float): Equilibrium temperature.
        pres (float): Pressure.
        v_qn_up (int): Upper state vibrational quantum number v'.
        v_qn_lo (int): Lower state vibrational quantum number v''.

    Returns:
        Sim: A `Sim` object with the desired parameters.
    """
    bands: list[tuple[int, int]] = [(v_qn_up, v_qn_lo)]

    return Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        rot_lvls=np.arange(0, 40),
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=pres,
        bands=bands,
    )


def get_line(sim: Sim, branch_name: str, branch_idx_lo: int, n_qn_lo: int) -> Line:
    """Return a rotational line with the desired parameters.

    Args:
        sim (Sim): The parent simulation.
        branch_name (str): Branch name, e.g. R, Q, or P.
        branch_idx_lo (int): Lower state branch index.
        n_qn_lo (int): Lower state rotational quantum number N''.

    Raises:
        ValueError: If the requested rotational line does not exist within the simulation.

    Returns:
        Line: The rotational line matching the input parameters.
    """
    for line in sim.bands[0].lines:
        if (
            line.branch_name == branch_name
            and line.branch_idx_lo == branch_idx_lo
            and line.n_qn_lo == n_qn_lo
            and not line.is_satellite
        ):
            return line

    raise ValueError("No matching rotational line found.")


def get_rates(sim: Sim, line: Line) -> RateParams:
    """Return the rate parameters.

    Args:
        sim (Sim): The parent simulation.
        line (Line): The desired rotational line.

    Returns:
        RateParams: Parameters related to the rate equations for the selected rotational line.
    """
    g_u: int = constants.ELECTRONIC_DEGENERACIES[sim.molecule.name][sim.state_up.name]
    g_l: int = constants.ELECTRONIC_DEGENERACIES[sim.molecule.name][sim.state_lo.name]

    a21_coeffs: NDArray[np.float64] = np.loadtxt(
        fname=Path(
            "..",
            "data",
            sim.molecule.name,
            "einstein",
            f"{sim.state_up.name}_to_{sim.state_lo.name}_allison.csv",
        ),
        delimiter=",",
    )

    # Only a single vibrational band will be simulated at a time.
    v_qn_up: int = sim.bands[0].v_qn_up
    v_qn_lo: int = sim.bands[0].v_qn_lo

    j_qn: int = line.j_qn_lo
    s_j: float = line.honl_london_factor
    nu_d: float = line.fwhm_predissociation(True)  # [1/cm]
    w_d: float = 2 * np.pi * constants.LIGHT * nu_d  # [1/s]
    a_21: float = a21_coeffs[v_qn_up][v_qn_lo] * s_j / (2 * j_qn + 1)  # [1/s]
    w_f: float = np.sum(a21_coeffs[v_qn_up]) * s_j / (2 * j_qn + 1)  # [1/s]
    nu: float = line.wavenumber  # [1/cm]
    b_12: float = (
        a_21 / (8 * np.pi * constants.PLANC * constants.LIGHT * nu**3) * g_u / g_l
    )  # [cm/J]
    b_21: float = b_12 * g_l / g_u  # [cm/J]

    # These two use pressure in [atm]. Additionally, use rotational temperature here since that's
    # what's measured with LIF.
    w_c: float = 7.78e9 * (sim.pressure / 101325) * np.sqrt(300 / sim.temp_rot)  # [1/s]
    w_q: float = 7.8e9 * (sim.pressure / 101325) * np.sqrt(300 / sim.temp_rot)  # [1/s]

    return RateParams(a_21, b_12, b_21, w_c, w_d, w_f, w_q)


def run_simulation(
    molecule: Molecule,
    state_up: State,
    state_lo: State,
    temp: float,
    pres: float,
    v_qn_up: int,
    v_qn_lo: int,
    branch_name: str,
    branch_idx_lo: int,
    n_qn_lo: int,
    pulse_center: float,
    pulse_width: float,
    fluence: float,
) -> None:
    """Plot the population densities, signal, and laser intensity as functions of time.

    Args:
        molecule (Molecule): Molecule of interest.
        state_up (State): Upper electronic state.
        state_lo (State): Lower electronic state.
        temp (float): Equilibrium temperature.
        pres (float): Pressure.
        v_qn_up (int): Upper state vibrational quantum number v'.
        v_qn_lo (int): Lower state vibrational quantum number v''.
        branch_name (str): Branch name.
        branch_idx_lo (int): Lower state branch index.
        n_qn_lo (int): Lower state rotational quantum number N''.
        pulse_center (float): Center of the laser pulse in [s].
        pulse_width (float): Width of the laser pulse in [s].
        fluence (float): Laser energy per unit area in [J/cm^2].
    """
    sim: Sim = get_sim(molecule, state_up, state_lo, temp, pres, v_qn_up, v_qn_lo)
    line: Line = get_line(sim, branch_name, branch_idx_lo, n_qn_lo)
    rate_params: RateParams = get_rates(sim, line)
    laser_params: LaserParams = LaserParams(pulse_center, pulse_width, fluence)
    t: NDArray[np.float64] = np.linspace(MIN_TIME, MAX_TIME, N_TIME, dtype=np.float64)

    n1, n2, n3 = simulate(t, rate_params, laser_params, line)

    # Normalize signal w.r.t N2.
    sf: NDArray[np.float64] = get_signal(t, n2, rate_params)
    sf /= n2.max()

    # Normalize laser w.r.t. itself.
    il: NDArray[np.float64] = laser_intensity(t, laser_params)
    il /= il.max()

    _, ax1 = plt.subplots()
    ax1.set_xlabel("Time, $t$ [s]")
    ax1.set_ylabel("$N_{1}$, $N_{3}$, $I_{L}$, Normalized")
    ax1.plot(t, n1, label="$N_{1}$")
    ax1.plot(t, n3, label="$N_{3}$")
    ax1.plot(t, il, label="$I_{L}$")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel("$N_{2}$, $S_{f}$, Normalized")
    ax2.plot(t, n2, label="$N_{2}$", linestyle="-.")
    ax2.plot(t, sf, label="$S_{f}$", linestyle="-.")
    ax2.legend()

    plt.show()


def scan_fluences(
    molecule: Molecule,
    state_up: State,
    state_lo: State,
    temp: float,
    pres: float,
    v_qn_up: int,
    v_qn_lo: int,
    branch_name: str,
    branch_idx_lo: int,
    n_qn_lo: int,
    pulse_center: float,
    pulse_width: float,
    max_fluence: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scan over fluence values and return the resulting singals.

    Args:
        molecule (Molecule): Molecule of interest.
        state_up (State): Upper electronic state.
        state_lo (State): Lower electronic state.
        temp (float): Equilibrium temperature.
        pres (float): Pressure.
        v_qn_up (int): Upper state vibrational quantum number v'.
        v_qn_lo (int): Lower state vibrational quantum number v''.
        branch_name (str): Branch name.
        branch_idx_lo (int): Lower state branch index.
        n_qn_lo (int): Lower state rotational quantum number N''.
        pulse_center (float): Center of the laser pulse in [s].
        pulse_width (float): Width of the laser pulse in [s].
        max_fluence (float): Maximum value of laser energy per unit area in [J/cm^2].

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The laser fluence values and their
            corresponding normalized signals.
    """
    sim: Sim = get_sim(molecule, state_up, state_lo, temp, pres, v_qn_up, v_qn_lo)
    line: Line = get_line(sim, branch_name, branch_idx_lo, n_qn_lo)
    rate_params: RateParams = get_rates(sim, line)
    t: NDArray[np.float64] = np.linspace(MIN_TIME, MAX_TIME, N_TIME, dtype=np.float64)

    fluences: NDArray[np.float64] = np.linspace(0.0, max_fluence, N_FLUENCE, dtype=np.float64)
    signals: NDArray[np.float64] = np.zeros_like(fluences)

    for idx, fluence in enumerate(fluences):
        laser_params = LaserParams(pulse_center, pulse_width, fluence)

        _, n2, _ = simulate(t, rate_params, laser_params, line)

        signal: NDArray[np.float64] = get_signal(t, n2, rate_params)
        signals[idx] = signal.max()

    return fluences, signals / signals.max()


def n2_vs_time_and_fluence(
    molecule: Molecule,
    state_up: State,
    state_lo: State,
    temp: float,
    pres: float,
    v_qn_up: int,
    v_qn_lo: int,
    branch_name: str,
    branch_idx_lo: int,
    n_qn_lo: int,
    pulse_center: float,
    pulse_width: float,
    max_fluence: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scan over fluence values and return the time-dependent populations.

    Args:
    molecule (Molecule): Molecule of interest.
        state_up (State): Upper electronic state.
        state_lo (State): Lower electronic state.
        temp (float): Equilibrium temperature.
        pres (float): Pressure.
        v_qn_up (int): Upper state vibrational quantum number v'.
        v_qn_lo (int): Lower state vibrational quantum number v''.
        branch_name (str): Branch name.
        branch_idx_lo (int): Lower state branch index.
        n_qn_lo (int): Lower state rotational quantum number N''.
        pulse_center (float): Center of the laser pulse in [s].
        pulse_width (float): Width of the laser pulse in [s].
        max_fluence (float): Maximum value of laser energy per unit area in [J/cm^2].

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: The laser fluences,
            corresponding time, and normalized N2 population density.
    """
    sim: Sim = get_sim(molecule, state_up, state_lo, temp, pres, v_qn_up, v_qn_lo)
    line: Line = get_line(sim, branch_name, branch_idx_lo, n_qn_lo)
    rate_params: RateParams = get_rates(sim, line)
    t: NDArray[np.float64] = np.linspace(MIN_TIME, MAX_TIME, N_TIME, dtype=np.float64)

    fluences: NDArray[np.float64] = np.linspace(0.0, max_fluence, N_FLUENCE, dtype=np.float64)
    n2_populations: NDArray[np.float64] = np.zeros((len(fluences), len(t)))

    for idx, fluence in enumerate(fluences):
        laser_params: LaserParams = LaserParams(pulse_center, pulse_width, fluence)
        _, n2, _ = simulate(t, rate_params, laser_params, line)
        n2_populations[idx, :] = n2

    return fluences, t, n2_populations


def plot_n2_vs_time_and_fluence(
    fluences: NDArray[np.float64], t: NDArray[np.float64], n2_populations: NDArray[np.float64]
) -> None:
    """Plot a heatmap showing the population of state N2 as a function of time and fluence.

    Args:
        fluences (NDArray[np.float64]): Laser fluence values in [J/cm^2].
        t (NDArray[np.float64]): The time domain in [s].
        n2_populations (NDArray[np.float64]): Normalized population density of state 2.
    """
    t, f = np.meshgrid(t, fluences)

    contour = plt.contourf(t, f, n2_populations, levels=50, cmap="magma")

    cbar = plt.colorbar(contour)
    cbar.set_label("$N_{2}$")

    plt.xlabel("Time, $t$ [s]")
    plt.ylabel("Laser Fluence, $\\Phi$ [J/cm$^{2}$]")

    plt.show()


def main() -> None:
    """Entry point."""
    molecule: Molecule = Molecule("O2", Atom("O"), Atom("O"))
    state_up: State = State("B3Su-", 3, molecule)
    state_lo: State = State("X3Sg-", 3, molecule)

    # NOTE: 24/10/29 - For now, laser fluence should be specified in [J/cm^2].

    run_simulation(
        molecule, state_up, state_lo, 300, 101325, 15, 3, "R", 1, 11, 30e-9, 20e-9, 25e-3
    )
    run_simulation(
        molecule, state_up, state_lo, 300, 101325, 15, 3, "R", 1, 11, 30e-9, 20e-9, 1000e-3
    )

    jay_27_p9x: NDArray[np.float64] = np.array([0, 1.8, 3.6, 6, 12, 24, 42.5]) / 1e3
    jay_27_p9y: NDArray[np.float64] = np.array([0, 0.08, 0.15, 0.27, 0.47, 0.7, 1])
    plt.scatter(jay_27_p9x, jay_27_p9y)
    f, sf = scan_fluences(
        molecule, state_up, state_lo, 1800, 101325, 2, 7, "P", 1, 9, 30e-9, 20e-9, 42.5e-3
    )
    plt.plot(f, sf, label="(2, 7)")

    jay_06_r17x: NDArray[np.float64] = np.array([0, 2, 3.8, 7, 12.1, 23, 43]) / 1e3
    jay_06_r17y: NDArray[np.float64] = np.array([0, 0.025, 0.06, 0.12, 0.27, 0.55, 1])
    plt.scatter(jay_06_r17x, jay_06_r17y)
    f, sf = scan_fluences(
        molecule, state_up, state_lo, 1800, 101325, 0, 6, "R", 1, 17, 30e-9, 20e-9, 43e-3
    )
    plt.plot(f, sf, label="(0, 6)")

    plt.xlabel("Laser Fluence, $\\Phi$ [J/cm$^{2}$]")
    plt.ylabel("Signal, $S_{f}$ [a.u.]")
    plt.legend()
    plt.show()

    fluences, t, n2_populations = n2_vs_time_and_fluence(
        molecule, state_up, state_lo, 300, 101325, 15, 3, "R", 1, 11, 30e-9, 20e-9, 1000e-3
    )
    plot_n2_vs_time_and_fluence(fluences, t, n2_populations)


if __name__ == "__main__":
    main()
