# module energy
'''
Keeps up with energy.
'''

import numpy as np

from state import State

def get_band_origin(grnd_state: 'State', exct_state: 'State') -> float:
    elc_energy = exct_state.electronic_term() - grnd_state.electronic_term()
    vib_energy = exct_state.vibrational_term() - grnd_state.vibrational_term()

    return elc_energy + vib_energy

def rotational_term(rot_qn: int, state: 'State', branch_idx: int) -> float:
    first_term = state.rotational_terms()[0] * rot_qn * (rot_qn + 1) - \
                 state.rotational_terms()[1] * rot_qn**2 * (rot_qn + 1)**2 + \
                 state.rotational_terms()[2] * rot_qn**3 * (rot_qn + 1)**3

    # See footnote 2 on pg. 223 of Herzberg
    # For N = 1, the sign in front of the square root must be inverted
    if rot_qn == 1:
        sqrt_sign = -1
    else:
        sqrt_sign = 1

    # NOTE: reminder that the sign in front of state.spn_const[0] was changed from
    #       a - to a + on 8/7/23
    if branch_idx == 1:
        return first_term + (2 * rot_qn + 3) * state.rotational_terms()[0] + \
               state.spn_const[0] - sqrt_sign * np.sqrt((2 * rot_qn + 3)**2 * \
               state.rotational_terms()[0]**2 + state.spn_const[0]**2 - 2 * \
               state.spn_const[0] * state.rotational_terms()[0]) + \
               state.spn_const[1] * (rot_qn + 1)

    if branch_idx == 2:
        return first_term

    return first_term - (2 * rot_qn - 1) * state.rotational_terms()[0] - \
           state.spn_const[0] + sqrt_sign * np.sqrt((2 * rot_qn - 1)**2 * \
           state.rotational_terms()[0]**2 + state.spn_const[0]**2 - 2 * \
           state.spn_const[0] * state.rotational_terms()[0]) - \
           state.spn_const[1] * rot_qn
