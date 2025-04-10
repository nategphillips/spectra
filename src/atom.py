# module atom
"""Contains the implementation of the Atom class."""

import constants


class Atom:
    """Represents an atom with a name and mass."""

    def __init__(self, name: str) -> None:
        """Initialize class variables.

        Args:
            name (str): Molecule name.
        """
        self.name: str = name
        self.mass: float = self.get_mass(name) / constants.AVOGD / 1e3

    @staticmethod
    def get_mass(name: str) -> float:
        """Return the atomic mass in [g/mol].

        Args:
            name (str): Name of the atom.

        Raises:
            ValueError: If the selected atom is not supported.

        Returns:
            float: The atomic mass in [g/mol].
        """
        if name not in constants.ATOMIC_MASSES:
            raise ValueError(f"Atom `{name}` not supported.")

        return constants.ATOMIC_MASSES[name]
