# module simtype
"""Contains the implementation of the SimType class."""

from enum import Enum


class SimType(Enum):
    """Defines the type of simulation to be performed."""

    ABSORPTION = 1
    EMISSION = 2
