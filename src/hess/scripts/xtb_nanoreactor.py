"""
xTB Nanoreactor Pipeline.
Executes parallel MD replicates and orchestrates Reaction Network Mapping.
"""

import copy
import io
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass, field

import ase.io
import networkx as nx
from networkx.readwrite import json_graph

from hess.analysis.rxn_event_filters import get_reaction_filter
from hess.scripts import xtbmd_driver
from hess.utilities import fileutils, logger
from hess.utilities.product_refiners import get_refiner
from hess.utilities.rdpattern import atoms_to_rdkit_mol, write_sdf
from hess.utilities.ts_searchers import get_ts_searcher


FLAG_NREPLICATES = "-nreplicates"
FLAG_WORKERS = "-workers"

FLAG_SORT_DISCOVERY = "-sort_by_discovery"
FLAG_SORT_FREQUENCY = "-sort_by_frequency"
FLAG_SORT_ENERGY = "-sort_by_energy"

log = logger.TextLogger("nanoreactor")
log_error = log.error


@dataclass
class ReactionNode:
    node_type: str
    conformers: list[ase.Atoms] = field(default_factory=list)
    geometry: ase.Atoms | None = None
    energy: float = 0.0


@dataclass
class ReactionEdge:
    ea: float | None = None
    ts_structure: str | None = None


class ReactionNetworkBuilder:
    """
    Scans MD trajectories frame-by-frame, identifies state transitions (edges),
    and aggregates them into a master Directed Graph.
    """

    def __init__(self, options):
        self.options = options
        self.master_graph = nx.DiGraph()
        self.rxn_filter = get_reaction_filter(options.rxn_filter, options)

    def buildFromTrajectories(self, reactant_atoms, trajectories):
        """Parse multiple trajectories to construct the reaction network."""
        start_state = self.rxn_filter.reset(reactant_atoms)
        if not start_state:
            raise ValueError("Failed to resolve state ID for reactant.")

        self.master_graph.add_node(
            start_state,
            data=ReactionNode(
                node_type="reactant", conformers=[reactant_atoms]
            ),
        )
        log(f"Root Reactant Node: {start_state}")

        total_edges = 0
        for i, traj in enumerate(trajectories):
            prev_state = self.rxn_filter.reset(reactant_atoms)
            for frame in traj:
                has_changed, curr_state = self.rxn_filter.processFrame(frame)

                if not has_changed or curr_state is None:
                    continue

                if not self.master_graph.has_edge(prev_state, curr_state):
                    self.master_graph.add_edge(
                        prev_state, curr_state, data=ReactionEdge()
                    )
                    log(f"Rep {i + 1}: {prev_state} -> {curr_state}")
                    total_edges += 1

                if "data" not in self.master_graph.nodes[curr_state]:
                    self.master_graph.nodes[curr_state]["data"] = ReactionNode(
                        node_type="product"
                    )

                self.master_graph.nodes[curr_state]["data"].conformers.append(
                    frame
                )
                prev_state = curr_state

        log(
            f"Graph constructed: {self.master_graph.number_of_nodes()} "
            f"unique nodes, {total_edges} physical edges observed."
        )
        return self.master_graph

    def mergeGraphs(self, local_graphs):
        """Merge returned local graphs from workers into the Master Graph."""
        log("Merging local sub-graphs into Master Reaction Network...")

        valid_graphs = [g for g in local_graphs if g is not None]
        if not valid_graphs:
            log("No valid trajectories to build network from.")
            return self.master_graph

        self.master_graph = copy.deepcopy(valid_graphs[0])

        for g in valid_graphs[1:]:
            for node, n_dict in g.nodes(data=True):
                incoming_node = n_dict["data"]
                if not self.master_graph.has_node(node):
                    self.master_graph.add_node(
                        node, data=copy.deepcopy(incoming_node)
                    )
                else:
                    self.master_graph.nodes[node]["data"].conformers.extend(
                        incoming_node.conformers
                    )

            for u, v, e_dict in g.edges(data=True):
                if not self.master_graph.has_edge(u, v):
                    edge_data = e_dict.get("data", ReactionEdge())
                    self.master_graph.add_edge(
                        u, v, data=copy.deepcopy(edge_data)
                    )

        log(
            f"Master Graph finalized: {self.master_graph.number_of_nodes()} "
            f"nodes, {self.master_graph.number_of_edges()} edges."
        )
        return self.master_graph


class XtbNanoreactorDriver:
    """
    Orchestrates replicate MD workflows to explore reaction networks,
    refines products, identifies intermediates, and tracks unique states.
    """

    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.master_graph = None
        self.sorted_nodes = []
        self.reference_structure = None

    def run(self):
        """Lifecycle execution sequence."""
        self.initVariables()
        self.dispatchReplicates()

        if not self.master_graph or self.master_graph.number_of_nodes() <= 1:
            log("No reaction products discovered during simulations.")
            return

        self.refineNodes()
        self.sortNodes()

        if self.options.ts_search:
            self.searchTransitionStates()

        self.exportData()

    def initVariables(self):
        """Verify input files, validate tools, and optimize starting state."""
        log(f"Loading initial nanoreactor from: {self.options.input}")

        md_driver = xtbmd_driver.XtbMDDriver(self.options, "ref_opt")
        md_driver.verifyXTBInstallation()

        log("Optimizing starting reactant for reference state...")
        out = md_driver.runOpt(self.options.input, extra_options="--opt tight")
        if not out.success or not out.opt_fpath:
            raise RuntimeError(
                "Failed to geometrically optimize the reactant structure. "
                "Check reference log."
            )

        self.reference_structure = ase.io.read(out.opt_fpath)
        log("Reference reactant optimized successfully.")

    @staticmethod
    def _runSingleReplicate(args):
        """Static method to execute a single MD worker in isolation."""
        idx, rep_dir, opts, struct = args

        with fileutils.chdir(rep_dir, create=True):
            input_xyz = os.path.abspath(f"rep_{idx}_input.xyz")
            ase.io.write(input_xyz, struct)

            rep_opts = copy.deepcopy(opts)
            rep_opts.input = input_xyz
            rep_opts.opt_frames = True  # Force in-memory optimization

            driver = xtbmd_driver.XtbMDDriver(rep_opts, f"rep_{idx}")
            driver.struct = struct.copy()
            driver.run()

            # Parse optimized frames entirely in memory
            frames = []
            for xyz_str in driver.frames:
                frame_atoms = ase.io.read(io.StringIO(xyz_str), format="xyz")
                frames.append(frame_atoms)

            builder = ReactionNetworkBuilder(rep_opts)
            return builder.buildFromTrajectories(struct, [frames])

    def dispatchReplicates(self):
        """Spawn independent xTB MD replicates using parallel Pool."""
        worker_args = []
        for i in range(self.options.nreplicates):
            rep_dir = os.path.abspath(f"rep_{i + 1}")
            worker_args.append(
                (i + 1, rep_dir, self.options, self.reference_structure)
            )

        log(f"Initializing {self.options.nreplicates} replicate worker(s)...")
        log(
            f"Dispatching {self.options.nreplicates} replicates "
            f"on {self.options.workers} worker(s)..."
        )

        with multiprocessing.Pool(processes=self.options.workers) as pool:
            results = pool.map(self._runSingleReplicate, worker_args)

        builder = ReactionNetworkBuilder(self.options)
        self.master_graph = builder.mergeGraphs(results)

    def refineNodes(self):
        """Refine node geometries and select lowest energy conformer."""
        nodes = list(self.master_graph.nodes(data=True))

        if not self.options.refine:
            for _node_id, n_data in nodes:
                node_obj = n_data["data"]
                if node_obj.conformers:
                    node_obj.geometry = node_obj.conformers[0]
                node_obj.energy = 0.0
            return

        log(f"Refining with engine: {self.options.refine}")
        refiner = get_refiner(self.options.refine)
        for _node_id, n_data in nodes:
            node_obj = n_data["data"]
            best_energy = float("inf")
            best_conformer = None

            for conformer in node_obj.conformers:
                struct = conformer.copy()
                try:
                    struct = refiner.runOptimization(struct)
                    energy = struct.get_potential_energy()
                except Exception as e:
                    log(f"Refinement failed for conformer of {_node_id}: {e}")
                    continue

                if energy < best_energy:
                    best_energy = energy
                    best_conformer = struct

            if best_conformer is None and node_obj.conformers:
                best_conformer = node_obj.conformers[0]
                best_energy = 0.0

            node_obj.geometry = best_conformer
            node_obj.energy = best_energy

    def sortNodes(self):
        """Sort the graph nodes based on CLI flags."""
        self.sorted_nodes = list(self.master_graph.nodes())

        if self.options.sort_by_energy and self.options.refine:
            self.sorted_nodes.sort(
                key=lambda n: self.master_graph.nodes[n]["data"].energy
            )
            log("Sorted nodes by refined ground-state energy.")
        elif self.options.sort_by_frequency:
            self.sorted_nodes.sort(
                key=lambda n: len(
                    self.master_graph.nodes[n]["data"].conformers
                ),
                reverse=True,
            )
            log("Sorted nodes by path traversal frequency.")
        elif self.options.sort_by_discovery:
            log("Sorted nodes by chronological discovery order.")
        else:
            log("No sorting flag provided. Using default discovery order.")

    def searchTransitionStates(self):
        """Compute Activation Energies (Ea) for Graph Edges."""
        log(f"Starting TS Search using method: {self.options.ts_search}")
        searcher = get_ts_searcher(self.options.ts_search, self.options)

        for source, target in self.master_graph.edges():
            log(f"Searching TS for Edge: {source} -> {target}")

            source_atoms = self.master_graph.nodes[source]["data"].geometry
            target_atoms = self.master_graph.nodes[target]["data"].geometry

            if source_atoms is None or target_atoms is None:
                log(f"Skipping TS for {source} -> {target}: Missing geometry.")
                continue

            try:
                ts_atoms, ea = searcher.findTransitionState(
                    source_atoms, target_atoms
                )
                edge_obj = self.master_graph.edges[source, target]["data"]
                edge_obj.ea = ea

                with io.StringIO() as buffer:
                    ase.io.write(buffer, ts_atoms, format="xyz")
                    edge_obj.ts_structure = buffer.getvalue()

            except Exception as e:
                log(f"TS Search failed for {source} -> {target}: {str(e)}")

    def exportData(self):
        """Generate final output artifacts: JSON network, XYZ, and SDF."""
        if not self.sorted_nodes:
            log("No reaction network to export.")
            return

        json_fpath = f"{self.jobname}_network.json"
        xyz_fpath = f"{self.jobname}_nodes.xyz"
        sdf_fpath = f"{self.jobname}_nodes.sdf"

        export_graph = nx.DiGraph()
        mols = []
        props_list = []

        with open(xyz_fpath, "w", encoding="utf-8") as f_xyz:
            for node in self.sorted_nodes:
                node_obj = self.master_graph.nodes[node]["data"]
                energy = node_obj.energy
                struct = node_obj.geometry

                with io.StringIO() as buffer:
                    ase.io.write(buffer, struct, format="xyz")
                    xyz_str = buffer.getvalue()

                export_graph.add_node(node, energy=energy, structure=xyz_str)

                lines = xyz_str.strip().split("\n")
                lines[1] = f"SMILES:{node} Energy:{energy:.6f}eV"
                f_xyz.write("\n".join(lines) + "\n")

                try:
                    rdmol = atoms_to_rdkit_mol(
                        struct, charge=self.options.charge
                    )
                    mols.append(rdmol)
                    props_list.append(
                        {
                            "SMILES": node,
                            "Energy_eV": f"{energy:.6f}",
                            "Frequency": len(node_obj.conformers),
                        }
                    )
                except Exception as e:
                    log(f"Warning: Failed to convert {node} to SDF Mol: {e}")

            for u, v, e_data in self.master_graph.edges(data=True):
                edge_obj = e_data["data"]
                export_graph.add_edge(
                    u,
                    v,
                    Ea=edge_obj.ea,
                    ts_structure=edge_obj.ts_structure,
                )

        if mols:
            write_sdf(mols, sdf_fpath, properties_list=props_list)

        with open(json_fpath, "w", encoding="utf-8") as f_json:
            json.dump(json_graph.node_link_data(export_graph), f_json, indent=2)

        log(
            f"Exported Reaction Network: {json_fpath}, {xyz_fpath}, {sdf_fpath}"
        )


def add_sorting_arguments(parser):
    """Register conformer and graph sorting arguments."""
    parser.add_argument(
        FLAG_SORT_DISCOVERY,
        dest="sort_by_discovery",
        action="store_true",
        help="Sort exported graph by chronological discovery time",
    )
    parser.add_argument(
        FLAG_SORT_FREQUENCY,
        dest="sort_by_frequency",
        action="store_true",
        help="Sort exported graph by path traversal frequency",
    )
    parser.add_argument(
        FLAG_SORT_ENERGY,
        dest="sort_by_energy",
        action="store_true",
        help="Sort exported graph nodes by refined ground-state energy",
    )


def get_parser():
    """Build argument parser composing MD, Orchestrator, and Sorting flags."""
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
        "-refine",
        dest="refine",
        type=str,
        default=None,
        help="Refine node geometries using specified engine (e.g., sevennet)",
    )
    parser.add_argument(
        "-ts_search",
        dest="ts_search",
        type=str,
        default=None,
        help="Search for TS using specified method (e.g., sella_neb)",
    )
    parser.add_argument(
        "-ts_engine",
        dest="ts_engine",
        type=str,
        default="sevennet",
        help="Engine for Sella TS optimization (default: sevennet)",
    )
    parser.add_argument(
        "-neb_engine",
        dest="neb_engine",
        type=str,
        default="sevennet",
        help="Engine for NEB Path optimization (default: sevennet)",
    )
    parser.add_argument(
        "-nimages",
        dest="nimages",
        type=int,
        default=10,
        help="Number of images for NEB path (default: 10)",
    )
    parser.add_argument(
        "-rxn_filter",
        dest="rxn_filter",
        type=str,
        default="smiles",
        help="Method to detect state changes (default: smiles)",
    )
    add_sorting_arguments(parser)
    return parser


def validate_options(options, parser):
    """Validate argument values for the orchestrator layer."""
    xtbmd_driver.validate_options(options, parser)

    if options.workers is None:
        env_cores = os.environ.get("JOBSERVER_ALLOCATED_CORES")
        if env_cores and env_cores.isdigit():
            options.workers = int(env_cores)
        else:
            options.workers = multiprocessing.cpu_count() or 1

    if options.nimages < 3:
        parser.error("-nimages must be at least 3 for NEB path.")

    if options.workers > 1 and options.parallel is None:
        options.parallel = 1


def main(args=None):
    """Entrypoint for the nanoreactor orchestrator."""
    if args is None:
        args = sys.argv[1:]

    if os.environ.get("_JOBSERVER_SANDBOX") != "1":
        from jobserver import JobDispatcher

        dispatcher = JobDispatcher()
        dispatcher.launch("hess-nanoreactor", args)
        return

    parser = get_parser()
    options = parser.parse_args(args)

    if not options.input:
        parser.error("-i input coordinate file must be provided.")

    jobname = options.jobname if options.jobname else "nanoreactor"
    global log, log_error
    log = logger.TextLogger(jobname)
    log_error = log.error

    validate_options(options, parser)

    driver = XtbNanoreactorDriver(options, jobname)
    driver.run()


if __name__ == "__main__":
    main()
