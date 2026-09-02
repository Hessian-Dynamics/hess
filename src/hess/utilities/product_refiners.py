"""
Product Refinement Engines for Hilbert.
Provides abstract interfaces and concrete implementations for refining
approximate (xTB) geometries into high-accuracy DFT/MLIP states.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

from ase.optimize import BFGS

from hess.utilities import logger


log = logger.TextLogger("refiner")


class BaseProductRefiner(ABC):
    """
    Abstract Base Class for all structure refinement engines.
    Enforces a standardized protocol for Single-Point (SCF) and Relaxations.
    """

    def __init__(self, **kwargs):
        """
        Initialize the refiner with necessary engine parameters.
        """
        self.kwargs = kwargs

    @abstractmethod
    def runSCF(self, atoms):
        """
        Execute a single-point energy and gradient calculation.

        :param atoms: Input atomic structure.
        :type atoms: ase.Atoms
        :return: A copy of the structure containing updated calculator results.
        :rtype: ase.Atoms
        """
        pass

    @abstractmethod
    def runOptimization(self, atoms, fmax=0.01, steps=250):
        """
        Execute a ground-state geometry relaxation to a local minimum.

        :param atoms: Input atomic structure.
        :type atoms: ase.Atoms
        :param fmax: Maximum force convergence threshold in eV/Å.
        :type fmax: float
        :param steps: Maximum number of optimization steps.
        :type steps: int
        :return: A copy of the relaxed structure.
        :rtype: ase.Atoms
        """
        pass


class SevenNetRefiner(BaseProductRefiner):
    """
    Refinement engine utilizing the SevenNet Universal MLIP.
    Achieves near-DFT accuracy at a fraction of the computational cost.
    """

    def __init__(self, model="7net-0", **kwargs):
        """
        Initialize SevenNet ASE calculator.

        :param model: SevenNet model name or path to .pth checkpoint.
        :type model: str
        """
        super().__init__(**kwargs)
        self.model = model

        try:
            from sevenn.calculator import SevenNetCalculator
        except ImportError:
            raise ImportError(
                "SevenNet not installed. Run: pip install sevenn"
            ) from None

        self.calc = SevenNetCalculator(model=self.model)

    def runSCF(self, atoms):
        """
        Execute SevenNet single-point calculation.
        """
        work_atoms = deepcopy(atoms)
        work_atoms.calc = self.calc

        # Trigger calculation
        work_atoms.get_potential_energy()
        work_atoms.get_forces()

        return work_atoms

    def runOptimization(self, atoms, fmax=0.01, steps=250):
        """
        Relax geometry using ASE's BFGS combined with SevenNet.
        """
        work_atoms = deepcopy(atoms)
        work_atoms.calc = self.calc

        log(f"Starting SevenNet ({self.model}) relaxation (fmax={fmax})...")
        # logfile=None suppresses standard ASE text spam
        dyn = BFGS(work_atoms, logfile=None)
        dyn.run(fmax=fmax, steps=steps)

        energy = work_atoms.get_potential_energy()
        log(f"SevenNet relaxation complete. Final Energy: {energy:.4f} eV")

        return work_atoms


def get_refiner(engine_name, **kwargs):
    """
    Factory function to instantiate the correct refiner.

    :param engine_name: String identifier (e.g., 'sevennet', 'orca').
    :type engine_name: str
    :return: Instantiated refiner object.
    :rtype: BaseProductRefiner
    """
    name = engine_name.lower()
    if name == "sevennet":
        return SevenNetRefiner(**kwargs)
    else:
        raise ValueError(f"Refinement engine '{engine_name}' is not supported.")
