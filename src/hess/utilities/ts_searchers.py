"""
Transition State (TS) Search Engines for Hilbert.
Provides abstract interfaces and concrete implementations for finding
saddle points connecting reactants and products on potential energy surfaces.
"""

from abc import ABC, abstractmethod

from ase.mep.neb import NEB, idpp_interpolate
from ase.optimize import FIRE
from sella import Sella

from hess.utilities import logger
from hess.utilities.product_refiners import get_refiner


log = logger.TextLogger("ts_searcher")


class BaseTSSearcher(ABC):
    """
    Abstract Base Class for all Transition State search algorithms.
    """

    def __init__(self, options):
        """
        Initialize the TS Searcher with parsed command-line options.

        :param options: ArgumentNamespace containing CLI parameters.
        :type options: argparse.Namespace
        """
        self.options = options

    @abstractmethod
    def findTransitionState(self, reactant_atoms, product_atoms):
        """
        Execute the TS search to find the rigorous saddle point.

        :param reactant_atoms: Input atomic structure of the reactant.
        :type reactant_atoms: ase.Atoms
        :param product_atoms: Input atomic structure of the product.
        :type product_atoms: ase.Atoms
        :return: Tuple of (Optimized TS ase.Atoms, Activation Energy in eV)
        :rtype: tuple(ase.Atoms, float)
        """
        pass


class SellaNEBSearcher(BaseTSSearcher):
    """
    Combines ASE Nudged Elastic Band (NEB) with the Sella optimizer to map
    the Minimum Energy Path and climb to the exact saddle point.
    """

    def findTransitionState(self, reactant_atoms, product_atoms):
        """
        Orchestrate IDPP interpolation -> NEB -> Sella optimization.

        :param reactant_atoms: Input atomic structure of the reactant.
        :type reactant_atoms: ase.Atoms
        :param product_atoms: Input atomic structure of the product.
        :type product_atoms: ase.Atoms
        :return: Tuple of (Optimized TS structure, Activation Energy in eV).
        :rtype: tuple[ase.Atoms, float]
        """
        log(f"Initializing NEB with engine: {self.options.neb_engine}")
        neb_refiner = get_refiner(self.options.neb_engine)
        ts_refiner = get_refiner(self.options.ts_engine)

        reactant, product = self.prepareCalculators(
            reactant_atoms, product_atoms, neb_refiner
        )
        reactant_energy = reactant.get_potential_energy()

        images = self.buildNebImages(reactant, product, neb_refiner)
        ts_guess = self.optimizeNebPath(images, reactant_energy)

        return self.refineSaddlePoint(ts_guess, ts_refiner, reactant_energy)

    def prepareCalculators(self, r_atoms, p_atoms, neb_refiner):
        """
        Prepare atomic structures with required ASE calculators.

        :param r_atoms: Reactant atomic coordinates.
        :type r_atoms: ase.Atoms
        :param p_atoms: Product atomic coordinates.
        :type p_atoms: ase.Atoms
        :param neb_refiner: Refinement engine providing the ASE calculator.
        :type neb_refiner: BaseProductRefiner
        :return: Tuple of prepared (reactant, product) atomic structures.
        :rtype: tuple[ase.Atoms, ase.Atoms]
        """
        reactant = r_atoms.copy()
        product = p_atoms.copy()
        reactant.calc = neb_refiner.calc
        product.calc = neb_refiner.calc
        return reactant, product

    def buildNebImages(self, reactant, product, neb_refiner):
        """
        Construct IDPP interpolated NEB band between reactant and product.

        :param reactant: Initial reactant state structure.
        :type reactant: ase.Atoms
        :param product: Target product state structure.
        :type product: ase.Atoms
        :param neb_refiner: Refinement engine providing the ASE calculator.
        :type neb_refiner: BaseProductRefiner
        :return: Interpolated list of atomic structures representing NEB band.
        :rtype: list[ase.Atoms]
        """
        images = [reactant]
        for _ in range(self.options.nimages - 2):
            images.append(reactant.copy())
        images.append(product)

        log(f"Running IDPP interpolation ({self.options.nimages} images)...")
        idpp_interpolate(images)

        for img in images[1:-1]:
            img.calc = neb_refiner.calc
        return images

    def optimizeNebPath(self, images, reactant_energy):
        """
        Execute NEB path optimization and extract the highest energy TS guess.

        :param images: Interpolated NEB image band.
        :type images: list[ase.Atoms]
        :param reactant_energy: Reference potential energy of reactant in eV.
        :type reactant_energy: float
        :return: Copy of atomic structure for the maximum energy image.
        :rtype: ase.Atoms
        """
        log("Optimizing Minimum Energy Path (NEB) using FIRE...")
        neb = NEB(images, climb=True, allow_shared_calculator=True)
        optimizer = FIRE(neb, logfile=None)
        optimizer.run(fmax=0.1, steps=200)

        energies = [img.get_potential_energy() for img in images]
        max_idx = energies[1:-1].index(max(energies[1:-1])) + 1
        ts_guess = images[max_idx].copy()

        rough_ea = energies[max_idx] - reactant_energy
        log(f"NEB complete. Rough Activation Energy: {rough_ea:.4f} eV")
        return ts_guess

    def refineSaddlePoint(self, ts_guess, ts_refiner, reactant_energy):
        """
        Run Sella Eigenvector Following to climb to the exact saddle point.

        :param ts_guess: Initial transition state guess from the NEB band.
        :type ts_guess: ase.Atoms
        :param ts_refiner: Refinement engine for saddle optimization.
        :type ts_refiner: BaseProductRefiner
        :param reactant_energy: Reference potential energy of reactant in eV.
        :type reactant_energy: float
        :return: Tuple of (Optimized TS structure, Activation Energy in eV).
        :rtype: tuple[ase.Atoms, float]
        """
        log(f"Refining Saddle Point with {self.options.ts_engine}...")
        ts_guess.calc = ts_refiner.calc

        sella_opt = Sella(ts_guess, logfile=None)
        sella_opt.run(fmax=0.01, steps=250)

        final_energy = ts_guess.get_potential_energy()
        activation_energy = final_energy - reactant_energy
        log(f"Sella refinement complete. Final Ea: {activation_energy:.4f} eV")

        return ts_guess, activation_energy


def get_ts_searcher(method_name, options):
    """
    Factory function to instantiate the correct TS searcher.

    :param method_name: String identifier (e.g., 'sella_neb').
    :type method_name: str
    :param options: Command-line options namespace.
    :type options: argparse.Namespace
    :return: Instantiated TS searcher object.
    :rtype: BaseTSSearcher
    """
    name = method_name.lower()
    if name == "sella_neb":
        return SellaNEBSearcher(options)
    elif name == "pygsm":
        return GSMSearcher(options)
    else:
        raise ValueError(f"TS Search method '{method_name}' is not supported.")


class GSMSearcher(BaseTSSearcher):
    """
    Transition State search utilizing the pyGSM (Growing String Method) engine.
    pyGSM has its own internal Eigenvector-following optimizers, so it operates
    independently of Sella and ASE-NEB.
    """

    def findTransitionState(self, reactant_atoms, product_atoms):
        """
        Implementation of the Double-Ended Growing String Method.
        """
        raise NotImplementedError(
            "pyGSM integration is scaffolded but not fully implemented yet."
        )
