from abc import ABC, abstractmethod

from hess.utilities.rdpattern import get_canonical_smiles


class BaseReactionFilter(ABC):
    """
    Abstract base class for all reaction event filters.
    Subclasses define different methodologies to detect if a product has formed.
    """

    @abstractmethod
    def extractProductSmiles(self, reactant, current_frame):
        """
        Evaluates if the current_frame is a product and returns its SMILES.

        :param reactant: Initial state structure.
        :type reactant: ase.Atoms
        :param current_frame: Current trajectory frame structure.
        :type current_frame: ase.Atoms
        :return: True if a reaction has occurred, False otherwise.
        :rtype: str or None
        """
        pass


class SmilesReactionFilter(BaseReactionFilter):
    """
    Detects reactions by converting 3D coordinates to molecular graphs (SMILES).
    Uses RDKit to perceive covalent bonds from XYZ coordinates.
    """

    def __init__(self, charge=0):
        """
        Initialize the SMILES reaction filter.

        :param charge: Total formal charge of the system.
        :type charge: int
        """
        self._initial_smiles = None
        self.charge = charge

    def extractProductSmiles(self, reactant, current_frame):
        """
        Evaluates if the current_frame is a product and returns its SMILES.
        Returns None if it is identical to the reactant.

        :param reactant: Initial state structure.
        :type reactant: ase.Atoms
        :param current_frame: Current trajectory frame structure.
        :type current_frame: ase.Atoms
        :return: True if the SMILES string has changed, indicating a reaction.
        :rtype: str or None
        """
        if self._initial_smiles is None:
            self._initial_smiles = get_canonical_smiles(
                ase_st=reactant, charge=self.charge
            )

        try:
            current_smiles = get_canonical_smiles(
                ase_st=current_frame, charge=self.charge
            )
        except ValueError:
            # RDKit failed to parse the hot MD frame. Frame is non-physical.
            return None
        if current_smiles != self._initial_smiles:
            return current_smiles
        return None
