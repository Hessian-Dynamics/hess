"""
xTB Nanoreactor Discovery Driver.
Layer 2 workflow orchestrator implementing parallel XtbMDDriver replicates,
SMILES-based product filtering, and organized artifact generation.
"""

import csv
import io
import multiprocessing
import os
import sys
from pathlib import Path

import ase.io

from hess.analysis.rxn_event_filters import SmilesReactionFilter
from hess.scripts import xtbmd_driver
from hess.utilities import fileutils, logger, xtb_core
from hess.utilities.product_refiners import get_refiner
from hess.utilities.rdpattern import (
    atoms_to_rdkit_mol,
    write_sdf,
)


log = logger.TextLogger("xtb_nanoreactor")


PROGRAM_NAME = "xTB Nanoreactor Discovery"
JOBNAME = "xtb_nanoreactor"

# Flag constants for Layer 2 Orchestrator
FLAG_INPUT = "-i"
FLAG_JOBNAME = "-JOBNAME"
FLAG_NREPLICATES = "-nreplicates"
FLAG_WORKERS = "-workers"
FLAG_REFINE = "-refine"

# Sorting Flags
FLAG_SORT_DISCOVERY = "-sort_by_discovery"
FLAG_SORT_FREQUENCY = "-sort_by_frequency"
FLAG_SORT_ENERGY = "-sort_by_energy"


class XtbNanoreactorDriver:
    """
    Orchestrates replicate MD workflows to explore reaction networks,
    filters unique products via SMILES, and exports sorted datasets.
    """

    def __init__(self, options, jobname):
        """
        Initialize XtbNanoreactorDriver instance.

        :param options: Parsed command-line argument namespace.
        :type options: argparse.Namespace
        :param jobname: Job name identifier.
        :type jobname: str
        """
        self.options = options
        self.jobname = jobname
        self.options.input = str(Path(self.options.input).resolve())
        self.input_file = self.options.input

        self.reference_structure = None
        self.all_frames = []
        self.unique_products = {}
        self.sorted_product_keys = []
        self.rxn_filter = SmilesReactionFilter(charge=self.options.charge)

    def run(self):
        """
        Lifecycle execution sequence for nanoreactor replicates.
        """
        self.initVariables()
        self.runReactantOpt()
        self.runReplicates()
        self.processReactionProducts()
        if self.options.sort_by_energy:
            self.evaluateProductEnergies()
        if self.options.refine:
            self.refineProducts()

        self.sortProducts()
        self.exportData()

    def initVariables(self):
        """
        Verify inputs and initialize replicate parallelization parameters.
        """
        log(f"Loading initial nanoreactor from: {self.input_file}")
        num_reps = self.options.nreplicates
        log(f"Initializing {num_reps} replicate worker(s)...")

    def runReactantOpt(self):
        """
        Optimize the initial reactant to establish a ground-truth reference.
        """
        log("Optimizing starting reactant for reference state...")
        with fileutils.chdir("reference_opt", create=True):
            engine = xtb_core.XTBCoreEngine(self.options, "ref_opt")
            out = engine.runOpt(self.input_file)

            if not out.success or not out.opt_fpath:
                raise RuntimeError("Failed to optimize reference structure.")

            self.reference_structure = ase.io.read(out.opt_fpath)
            log("Reference reactant optimized successfully.")

    def _runReplicateWorker(self, rep_idx):
        """
        Worker function to dispatch a single XtbMDDriver instance.

        :param rep_idx: Replicate index.
        :type rep_idx: int
        :return: Dictionary containing replicate index and trajectory frames.
        :rtype: dict
        """
        rep_jobname = f"{self.jobname}_rep_{rep_idx}"
        with fileutils.chdir(f"rep_{rep_idx}", create=True):
            md_worker = xtbmd_driver.XtbMDDriver(self.options, rep_jobname)
            md_worker.run()
            return {"idx": rep_idx, "frames": md_worker.frames}

    def runReplicates(self):
        """
        Execute configured MD replicate simulations in parallel.
        """
        reps = range(1, self.options.nreplicates + 1)
        workers = min(self.options.workers, len(reps))

        log(f"Dispatching {len(reps)} replicates on {workers} worker(s)...")

        if workers == 1:
            for r in reps:
                res = self._runReplicateWorker(r)
                self.all_frames.extend(res["frames"])
        else:
            with multiprocessing.Pool(workers) as pool:
                res_list = pool.map(self._runReplicateWorker, reps)
                for res in res_list:
                    self.all_frames.extend(res["frames"])

        log(f"Aggregation complete. Total frames: {len(self.all_frames)}")

    def _strToAtoms(self, xyz_str):
        """
        Convert an XYZ string into an ASE Atoms object.

        :param xyz_str: Multi-line XYZ format string.
        :type xyz_str: str
        :return: Parsed atomic structure.
        :rtype: ase.Atoms
        """
        buffer = io.StringIO(xyz_str)
        return ase.io.read(buffer, format="xyz")

    def binSingleFrame(self, idx, xyz_str):
        """
        Filter a single trajectory frame and register unique products.

        :param idx: Frame index in the aggregated trajectory.
        :type idx: int
        :param xyz_str: XYZ coordinate string for the frame.
        :type xyz_str: str
        """
        frame_struct = self._strToAtoms(xyz_str)

        smiles = self.rxn_filter.extractProductSmiles(
            self.reference_structure, frame_struct
        )
        if smiles:
            if smiles not in self.unique_products:
                self.unique_products[smiles] = {
                    "smiles": smiles,
                    "first_frame_idx": idx,
                    "frequency": 1,
                    "structure": xyz_str,
                    "energy": 0.0,
                }
            else:
                self.unique_products[smiles]["frequency"] += 1

    def processReactionProducts(self):
        """
        Iterate through all trajectory frames to discover distinct products.
        """
        log("Filtering trajectory frames for unique reaction products...")

        for idx, xyz_str in enumerate(self.all_frames):
            self.binSingleFrame(idx, xyz_str)

        num_prods = len(self.unique_products)
        log(f"Product binning complete. Found {num_prods} unique products.")

    def _evaluateSingleEnergy(self, smiles, metadata):
        """
        Run a single-point energy calculation on a discovered product.

        :param smiles: The SMILES string of the product.
        :type smiles: str
        :param metadata: Product dictionary containing geometry.
        :type metadata: dict
        """
        with fileutils.chdir(f"eval_{smiles[:8]}", create=True):
            temp_xyz = Path("temp.xyz")
            with open(temp_xyz, "w", encoding="utf-8") as f:
                f.write(metadata["structure"])

            engine = xtb_core.XTBCoreEngine(self.options, "eval")
            out = engine.runOpt(str(temp_xyz), ohess=False)

            if out.energy_eh is not None:
                metadata["energy"] = out.energy_eh
                if out.charges:
                    metadata["charges"] = ",".join(
                        f"{c:.4f}" for c in out.charges
                    )
                if out.wbo:
                    wbo_str = ";".join(
                        f"{idx1}-{idx2}:{val:.2f}"
                        for idx1, idx2, val in out.wbo
                    )
                    metadata["wbo"] = wbo_str

    def evaluateProductEnergies(self):
        """
        Calculate ground-state electronic energies for all unique products.
        """
        log("Evaluating ground-state energies for unique products...")
        with fileutils.chdir("product_energies", create=True):
            for smiles, metadata in self.unique_products.items():
                self._evaluateSingleEnergy(smiles, metadata)

    def sortProducts(self):
        """
        Sort discovered products based on the user-selected CLI flag.
        """
        items = list(self.unique_products.values())

        if self.options.sort_by_frequency:
            items.sort(key=lambda x: x["frequency"], reverse=True)
            log("Sorted products by discovery frequency (descending).")
        elif self.options.sort_by_energy:
            items.sort(key=lambda x: x["energy"])
            log("Sorted products by ground-state energy (ascending).")
        else:
            items.sort(key=lambda x: x["first_frame_idx"])
            log("Sorted products by first appearance chronological order.")

        self.sorted_product_keys = [item["smiles"] for item in items]

    def refineProducts(self):
        """
        Refine geometries and energies of all unique products in memory.
        """
        log(
            f"Starting in-memory refinement using engine: {self.options.refine}"
        )
        refiner = get_refiner(self.options.refine)

        for _smiles, meta in self.unique_products.items():
            struct = self._strToAtoms(meta["structure"])
            refined_atoms = refiner.runOptimization(struct)

            meta["energy"] = refined_atoms.get_potential_energy()

            with io.StringIO() as buffer:
                ase.io.write(buffer, refined_atoms, format="xyz")
                meta["structure"] = buffer.getvalue()

    def exportData(self):
        """
        Generate final output artifacts: XYZ, SDF, and CSV in a single pass.
        """
        if not self.sorted_product_keys:
            log("No reaction products discovered during simulations.")
            return

        xyz_fpath = f"{self.jobname}_products.xyz"
        sdf_fpath = f"{self.jobname}_products.sdf"
        csv_fpath = f"{self.jobname}_summary.csv"

        mols = []
        props_list = []

        with (
            open(xyz_fpath, "w", encoding="utf-8") as f_xyz,
            open(csv_fpath, "w", newline="", encoding="utf-8") as f_csv,
        ):
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(
                ["Rank", "SMILES", "FirstFrame", "Freq", "Energy"]
            )

            for rank, smiles in enumerate(self.sorted_product_keys, 1):
                meta = self.unique_products[smiles]
                energy_str = f"{meta.get('energy', 0.0):.6f}"
                freq = meta["frequency"]
                frame_idx = meta["first_frame_idx"]

                # 1. Export CSV Row
                csv_writer.writerow([rank, smiles, frame_idx, freq, energy_str])

                # 2. Export XYZ Block
                lines = meta["structure"].strip().split("\n")
                lines[1] = f"SMILES:{smiles} Freq:{freq} Energy:{energy_str}Eh"
                f_xyz.write("\n".join(lines) + "\n")

                # 3. Build SDF Mol
                struct = self._strToAtoms(meta["structure"])
                rdmol = atoms_to_rdkit_mol(struct, charge=self.options.charge)
                props = {
                    "SMILES": smiles,
                    "Frequency": freq,
                    "Energy_Eh": energy_str,
                    "FirstFrame": frame_idx,
                    "Charges_Mulliken": meta.get("charges", ""),
                    "Wiberg_Bond_Orders": meta.get("wbo", ""),
                }
                mols.append(rdmol)
                props_list.append(props)

        write_sdf(mols, sdf_fpath, properties_list=props_list)
        log(f"Exported data to: {xyz_fpath}, {sdf_fpath}, {csv_fpath}")


def add_sorting_arguments(parser):
    """
    Append mutually exclusive sorting flags to the argument parser.

    :param parser: Configured ArgumentParser.
    :type parser: argparse.ArgumentParser
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        FLAG_SORT_DISCOVERY,
        dest="sort_by_discovery",
        action="store_true",
        default=True,
        help="Sort chronologically by first appearance (default)",
    )
    group.add_argument(
        FLAG_SORT_FREQUENCY,
        dest="sort_by_frequency",
        action="store_true",
        help="Sort by descending observation frequency",
    )
    group.add_argument(
        FLAG_SORT_ENERGY,
        dest="sort_by_energy",
        action="store_true",
        help="Sort by ascending ground-state electronic energy",
    )


def get_parser():
    """
    Build argument parser composing MD flags, Orchestrator, and Sorting flags.

    :return: Configured argparse.ArgumentParser object.
    :rtype: argparse.ArgumentParser
    """
    parser = xtbmd_driver.get_parser()

    parser.add_argument(
        FLAG_NREPLICATES,
        dest="nreplicates",
        type=int,
        default=5,
        help="Number of independent nanoreactor replicates (default: 5)",
    )
    parser.add_argument(
        FLAG_WORKERS,
        dest="workers",
        type=int,
        default=None,
        help="Number of parallel replicate workers (default: 1)",
    )

    parser.add_argument(
        FLAG_REFINE,
        dest="refine",
        type=str,
        default=None,
        help="Refine output SDF using specified engine (e.g., sevennet)",
    )
    add_sorting_arguments(parser)
    return parser


def validate_options(options, parser):
    """
    Validate argument values for the orchestrator layer.

    :param options: Parsed options namespace.
    :type options: argparse.Namespace
    :param parser: Argument parser instance.
    :type parser: argparse.ArgumentParser
    """
    xtbmd_driver.validate_options(options, parser)

    if options.workers is None:
        env_cores = os.environ.get("JOBSERVER_ALLOCATED_CORES")
        if env_cores and env_cores.isdigit():
            options.workers = int(env_cores)
        else:
            import multiprocessing

            options.workers = multiprocessing.cpu_count() or 1

    # Prevent xTB Thread Oversubscription
    # If orchestrator launches multiple workers without a thread limit,
    # lock xTB to 1 thread per worker to prevent CPU thrashing.
    if options.workers > 1 and options.parallel is None:
        options.parallel = 1

    if options.nreplicates < 1:
        parser.error("Number of replicates must be at least 1.")

    # Mutually exclusive flags mean if one is true, others are false
    # Explicitly handle default boolean overlap.
    # argparse mutually exclusive group handles the enforcement, but with
    # store_true and a default=True, the default might override.
    # We explicitly handle the logic:
    if options.sort_by_frequency or options.sort_by_energy:
        options.sort_by_discovery = False


def main(args=None):
    """
    CLI entrypoint for xTB Nanoreactor Discovery driver.

    :param args: Optional argument list.
    :type args: list[str]
    """
    if args is None:
        args = sys.argv[1:]

    if os.environ.get("_JOBSERVER_SANDBOX") != "1":
        from jobserver import JobDispatcher

        dispatcher = JobDispatcher()
        dispatcher.launch("hess-nanoreactor", args)
        return

    parser = get_parser()
    options = parser.parse_args(args)
    jobname = options.jobname if options.jobname else JOBNAME

    global log
    validate_options(options, parser)

    driver = XtbNanoreactorDriver(options, jobname)
    driver.run()


if __name__ == "__main__":
    main()
