# module constants
"""Provides physical and molecular constants."""

# Avodagro constant [1/mol]
AVOGD: float = 6.02214076e23
# Boltzmann constant [J/K]
BOLTZ: float = 1.380649e-23
# Speed of light [cm/s]
LIGHT: float = 2.99792458e10
# Planck constant [J*s]
PLANC: float = 6.62607015e-34

# Atomic masses [g/mol]
ATOMIC_MASSES: dict[str, float] = {"O": 15.999}

# Internuclear distance [m]
# Data from NIST Chemistry WebBook
INTERNUCLEAR_DISTANCE: dict[str, dict[str, float]] = {
    "O2": {"X3Sg-": 1.20752e-10, "B3Su-": 1.6042e-10}
}

# Electronic energies [1/cm]
# Data from NIST Chemistry WebBook
ELECTRONIC_ENERGIES: dict[str, dict[str, float]] = {
    "O2": {
        "X3Sg-": 0.0,
        "a1Pg": 7918.1,
        "b1Sg+": 13195.1,
        "c1Su-": 33057.0,
        "A3Pu": 34690.0,
        "A3Su+": 35397.8,
        "B3Su-": 49793.28,
    }
}

# Electronic degeneracies [-]
# Data from Park, 1990
ELECTRONIC_DEGENERACIES: dict[str, dict[str, int]] = {
    "O2": {"X3Sg-": 3, "a1Pg": 2, "b1Sg+": 1, "c1Su-": 1, "A3Pu": 6, "A3Su+": 3, "B3Su-": 3}
}
