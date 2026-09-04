"""
Reaction Event Filters.
Contains mathematical methods to differentiate the structures of two frames,
used to detect physical state transitions in MD trajectories.
"""

from abc import ABC, abstractmethod

from hess.utilities.rdpattern import get_canonical_smiles


class BaseReactionFilter(ABC):
    """
    Abstract base class for all reaction event filters.
    """

    @abstractmethod
    def reset(self, initial_frame):
        """
        Initialize or reset the filter's internal state tracker.

        :param initial_frame: 3D atomic coordinates of the starting state
        :type initial_frame: ase.Atoms
        :return: Unique string/hash identifying the chemical state
        :rtype: str
        """
        pass

    @abstractmethod
    def processFrame(self, current_frame):
        """
        Evaluate current frame against the internally tracked last structure.
        If a structural transition is detected, update the internal state.

        :param current_frame: 3D atomic coordinates of the current frame
        :type current_frame: ase.Atoms
        :return: Tuple of (transition_detected: bool, state_id: str/None)
        :rtype: tuple(bool, str or None)
        """
        pass


class SmilesReactionFilter(BaseReactionFilter):
    """
    Differentiates structures by perceiving covalent bonds from XYZ coordinates
    and generating a canonical RDKit SMILES string.
    """

    def __init__(self, charge=0):
        self.charge = charge
        self.last_frame = None
        self.last_state_id = None

    def reset(self, initial_frame):
        self.last_frame = initial_frame
        self.last_state_id = get_canonical_smiles(
            ase_st=initial_frame, charge=self.charge
        )
        return self.last_state_id

    def processFrame(self, current_frame):
        try:
            curr_state = get_canonical_smiles(
                ase_st=current_frame, charge=self.charge
            )
        except ValueError:
            # Frame is non-physical or fragmented poorly
            return False, None

        if curr_state != self.last_state_id:
            # Transition detected! Update internal tracker.
            self.last_frame = current_frame
            self.last_state_id = curr_state
            return True, curr_state

        return False, curr_state


def get_reaction_filter(method_name, options):
    """
    Factory function to instantiate the correct reaction filter.

    :param method_name: String identifier (e.g., 'smiles').
    :type method_name: str
    :param options: Command-line options namespace.
    :type options: argparse.Namespace
    :return: Instantiated reaction filter object.
    :rtype: BaseReactionFilter
    """
    name = method_name.lower()
    if name == "smiles":
        return SmilesReactionFilter(charge=options.charge)
    else:
        raise ValueError(f"Reaction filter '{method_name}' is not supported.")
