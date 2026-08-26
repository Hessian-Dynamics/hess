"""
MLFF Geometry Optimization Driver using ASE and MACE-OFF23.
"""

import argparse
import os
import sys

from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp

from hilbert.utilities import logger


PROGRAM_NAME = "MLFF Geometry Optimization"
JOBNAME = "mlff_geo_opt"
FLAG_JOBNAME = "-JOBNAME"


class MLFFOptimizationDriver:
    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.struct = None
        self.calc = None
        self.optimizer = None

    def run(self):
        """
        Run the workflow
        """
        self.initVariables()
        self.optimize()
        self.exportData()

    def initVariables(self):
        """
        Load molecule file, initialize MACE MLFF calculator,
        and set up optimizer.
        """

        log(f"Loading molecule from: {self.options.input}")
        self.struct = read(self.options.input)

        log(f"Initializing MACE ({self.options.model}) MLFF...")

        # MACE-OFF23 is pretrained on organic molecules:
        # (H, C, N, O, F, P, S, Cl, Br, I)
        self.calc = mace_mp(
            model=self.options.model, device=self.options.device
        )
        self.struct.calc = self.calc

    def optimize(self):
        """
        Run BFGS geometry optimization until force threshold is met.
        """

        log(
            f"Starting relaxation (fmax threshold = "
            f"{self.options.fmax} eV/Å)..."
        )
        logfn = logger.get_logfile_name(self.jobname)
        traj_req = self.options.traj if self.options.traj else None
        self.optimizer = BFGS(self.struct, trajectory=traj_req, logfile=logfn)
        self.optimizer.run(fmax=self.options.fmax, steps=self.options.steps)
        log(f"Optimization finished in {self.optimizer.nsteps} steps.")

    def exportData(self):
        """
        Export optimized geometry and log final energetic summary.
        """

        output_file = f"{self.options.jobname}_optimized.xyz"
        write(output_file, self.struct)
        energy = self.struct.get_potential_energy()
        log(f"Final Potential Energy: {energy:.6f} eV")
        log(f"Optimized structure saved to: {output_file}")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, prefix_chars="-")

    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        required=True,
        help="Path to input molecule file (e.g., .xyz, .sdf, .pdb, .cif)",
    )
    parser.add_argument(
        "-fmax",
        dest="fmax",
        type=float,
        default=0.01,
        help="Maximum force convergence criterion in eV/Å (default: 0.01)",
    )
    parser.add_argument(
        "-steps",
        dest="steps",
        type=int,
        default=500,
        help="Maximum optimization steps allowed (default: 500)",
    )
    parser.add_argument(
        "-model",
        dest="model",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="MACE model size scale (default: medium)",
    )
    parser.add_argument(
        "-device",
        dest="device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "-traj",
        dest="traj",
        type=str,
        default=None,
        help=(
            "Path to save output .traj optimization trajectory file (optional)"
        ),
    )
    parser.add_argument(
        FLAG_JOBNAME, dest="jobname", type=str, default=JOBNAME, help="Jobname"
    )
    return parser


def validate_options(options, parser):
    if not os.path.exists(options.input):
        parser.error(f"Input file '{options.input}' does not exist.")


log = logger.TextLogger(JOBNAME)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    global log
    parser = get_parser()
    options = parser.parse_args(args)
    jobname = options.jobname if options.jobname else JOBNAME
    validate_options(options, parser)

    log = logger.TextLogger(jobname)
    driver = MLFFOptimizationDriver(options, jobname)
    driver.run()


if __name__ == "__main__":
    main()
