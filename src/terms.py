# module terms
"""Contains functions used for vibrational and rotational term calculations."""

from typing import TYPE_CHECKING

import numpy as np

from state import State

if TYPE_CHECKING:
    from numpy.typing import NDArray


def vibrational_term(state: State, v_qn: int) -> float:
    """Return the vibrational term value in [1/cm].

    Args:
        state (State): Electronic `State` object.
        v_qn (int): Vibrational quantum number v.

    Returns:
        float: The vibrational term value in [1/cm].
    """
    return state.constants["G"][v_qn]


def rotational_term(state: State, v_qn: int, j_qn: int, branch_idx: int) -> float:
    """Return the rotational term value in [1/cm].

    Args:
        state (State): Electronic `State` object.
        v_qn (int): Vibrational quantum number v.
        j_qn (int): Rotational quantum number J.
        branch_idx (int): Branch index.

    Raises:
        ValueError: If the branch index cannot be found.

    Returns:
        float: The rotational term value in [1/cm].
    """
    lookup_table: dict[str, list[float]] = state.constants

    b: float = lookup_table["B"][v_qn]
    d: float = lookup_table["D"][v_qn]
    l: float = lookup_table["lamda"][v_qn]
    g: float = lookup_table["gamma"][v_qn]
    ld: float = lookup_table["lamda_D"][v_qn]
    gd: float = lookup_table["gamma_D"][v_qn]

    # NOTE: 24/11/05 - The Hamiltonians in Cheung and Yu are defined slightly differently, which
    #       leads to some constants having different values. Since the Cheung Hamiltonian matrix
    #       elements are used to solve for the energy eigenvalues, the constants from Yu are changed
    #       to fit the convention used by Cheung. See the table below for details.
    #
    #       Cheung  | Yu
    #       --------|------------
    #       D       | -D
    #       lamda_D | 2 * lamda_D
    #       gamma_D | 2 * gamma_D

    if state.name == "X3Sg-":
        d *= -1
        ld *= 2
        gd *= 2

    # The Hamiltonian from Cheung is written in Hund's case (a) representation, so J is used instead
    # of N.
    x: int = j_qn * (j_qn + 1)

    # The four Hamiltonian matrix elements given in Cheung.
    h11: float = (
        b * (x + 2)
        - d * (x**2 + 8 * x + 4)
        - 4 / 3 * l
        - 2 * g
        - 4 / 3 * ld * (x + 2)
        - 4 * gd * (x + 1)
    )
    h12: float = -2 * np.sqrt(x) * (b - 2 * d * (x + 1) - g / 2 - 2 / 3 * ld - gd / 2 * (x + 4))
    h21: float = h12
    h22: float = b * x - d * (x**2 + 4 * x) + 2 / 3 * l - g + 2 / 3 * x * ld - 3 * x * gd

    hamiltonian: NDArray[np.float64] = np.array([[h11, h12], [h21, h22]])
    f1, f3 = np.linalg.eigvals(hamiltonian)

    match branch_idx:
        case 1:
            return f1
        case 2:
            return b * x - d * x**2 + 2 / 3 * l - g + 2 / 3 * x * ld - x * gd
        case 3:
            return f3
        case _:
            raise ValueError(f"Invalid branch index: {branch_idx}")
