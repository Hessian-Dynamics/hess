"""
xTB Molecular Dynamics and MetaDynamics Driver.
Inherits from XTBCoreEngine following the Hilbert driver architecture.
"""

import os
import sys
from pathlib import Path

from hilbert.utilities import fileutils, logger, xtb_core


PROGRAM_NAME = "xTB Molecular Dynamics"
JOBNAME = "xtb_md"
FLAG_JOBNAME = "-JOBNAME"

# Flag constants
FLAG_INPUT = "-i"
FLAG_TIME = "-time"
FLAG_STEP = "-step"
FLAG_TEMP = "-temp"
FLAG_DUMP = "-dump"
FLAG_NVT = "-nvt"
FLAG_HMASS = "-hmass"
FLAG_SHAKE = "-shake"
FLAG_SCCACC = "-sccacc"
FLAG_VELO = "-velo"
FLAG_KPUSH = "-kpush"
FLAG_ALP = "-alp"
FLAG_MTD_SAVE = "-mtd_save"
FLAG_SKIP_MD = "-skip_md"
FLAG_SKIP_MTD = "-skip_mtd"
FLAG_OPT_FRAMES = "-opt_frames"
FLAG_SAMPLE_STRIDE = "-sample_stride"

# Default constants
DEFAULT_JOBNAME = "xtb_md"
DEFAULT_TIME = 10.0
DEFAULT_STEP = 1.0
DEFAULT_TEMP = 300.0
DEFAULT_DUMP = 50.0
DEFAULT_NVT = 1
DEFAULT_HMASS = 1
DEFAULT_SHAKE = 0
DEFAULT_SCCACC = 2.0
DEFAULT_VELO = 0
DEFAULT_KPUSH = 0.1
DEFAULT_ALP = 0.6
DEFAULT_MTD_SAVE = 50
DEFAULT_SAMPLE_STRIDE = 1


class XtbMDDriver(xtb_core.XTBCoreEngine):
    """
    Driver managing standard MD, MetaDynamics, and frame optimization.
    """

    def __init__(self, options, jobname):
        """
        Initialize XtbMDDriver instance.

        :param options: Parsed command-line argument namespace.
        :param jobname: Job name identifier.
        """
        super().__init__(options, jobname)
        self.input_file = str(Path(self.options.input).resolve())
        self.md_out = None
        self.mtd_out = None
        self.frames = []
        self.opt_frames_list = []

    def run(self):
        """
        Lifecycle execution sequence.
        """
        self.initVariables()

        md_out = None
        if not self.options.skip_md:
            with fileutils.chdir("md", create=True):
                md_out = self.runMD()

        mtd_out = None
        if not self.options.skip_mtd:
            with fileutils.chdir("mtd", create=True):
                mtd_out = self.runMTD()

        self.frames = self.getCombinedFrames(md_out, mtd_out)
        if self.options.opt_frames and self.frames:
            with fileutils.chdir("opt_frames", create=True):
                self.frames = self.optFrames(self.frames)

        self.writeFrames()
        self.exportData()

    def initVariables(self):
        """
        Verify input files and log initialization.
        """
        log(f"Loading initial structure from: {self.input_file}")
        self.verifyXTBInstallation()

    def createMDControl(self, control_path, is_mtd=False):
        """
        Format and write xTB MD/MTD control deck.

        :param control_path: Target path for control file.
        :param is_mtd: True to include MetaDynamics block.
        """
        extra_options = ""
        md_block = xtb_core.XTB_MD_CONTROL_TEMPLATE.format(
            time=self.options.time,
            step=self.options.step,
            temp=self.options.temp,
            dump=self.options.dump,
            nvt=self.options.nvt,
            hmass=self.options.hmass,
            shake=self.options.shake,
            sccacc=self.options.sccacc,
            velo=self.options.velo,
            extra_options=extra_options,
        ).strip()

        blocks = [md_block]
        if is_mtd:
            mtd_block = xtb_core.XTB_METADYN_TEMPLATE.format(
                save=self.options.mtd_save,
                kpush=self.options.kpush,
                alp=self.options.alp,
                extra_options="",
            ).strip()
            blocks.append(mtd_block)

        control_content = "\n\n".join(blocks) + "\n"
        with open(control_path, "w", encoding="utf-8") as f:
            f.write(control_content)

    def runMD(self):
        """
        Execute unbiased MD simulation.

        :return: XTBOutput object for the MD run or None.
        """
        if self.options.skip_md:
            return None

        log("Starting unbiased Molecular Dynamics simulation...")
        control_path = "control_md.txt"
        self.createMDControl(control_path, is_mtd=False)

        cmd = self.getXTBCommand(self.input_file)
        cmd.extend(["--md", "--input", control_path])
        self.md_out = self.runOneXTB(cmd, logfn=f"{self.jobname}_md.log")
        num_frames = len(self.md_out.frames)
        log(f"Unbiased MD complete. Frames collected: {num_frames}")
        return self.md_out

    def runMTD(self):
        """
        Execute biased MetaDynamics simulation.

        :return: XTBOutput object for the MTD run or None.
        """
        if self.options.skip_mtd:
            return None

        log("Starting MetaDynamics (MTD) simulation...")
        control_path = "control_mtd.txt"
        self.createMDControl(control_path, is_mtd=True)

        cmd = self.getXTBCommand(self.input_file)
        cmd.extend(["--md", "--input", control_path])
        self.mtd_out = self.runOneXTB(cmd, logfn=f"{self.jobname}_mtd.log")
        num_frames = len(self.mtd_out.frames)
        log(f"MetaDynamics complete. Frames collected: {num_frames}")
        return self.mtd_out

    def getCombinedFrames(self, md_out, mtd_out):
        """
        Merge frames from MD and MetaDynamics stages.

        :param md_out: XTBOutput from unbiased MD.
        :param mtd_out: XTBOutput from MetaDynamics.
        :return: List of all combined XYZ frame strings.
        """
        combined = []
        if md_out and md_out.frames:
            combined.extend(md_out.frames)
        if mtd_out and mtd_out.frames:
            combined.extend(mtd_out.frames)
        return combined

    def optFrames(self, frames):
        """
        Relax sampled trajectory frames into local minima conformers.

        :param frames: List of XYZ frame strings.
        :return: List of relaxed XYZ frame strings.
        """
        stride = max(1, self.options.sample_stride)
        sampled_frames = frames[::stride]
        num_sampled = len(sampled_frames)
        log(f"Optimizing {num_sampled} sampled trajectory frames...")

        opt_list = []
        temp_input = Path("temp_frame_opt.xyz")

        for frame in sampled_frames:
            with open(temp_input, "w", encoding="utf-8") as f:
                f.write(frame)

            opt_out = self.runOpt(
                str(temp_input),
                optlevel=self.options.optlevel,
                maxcycle=self.options.maxcycle,
            )
            if opt_out.success and opt_out.opt_fpath:
                with open(opt_out.opt_fpath, encoding="utf-8") as f:
                    opt_list.append(f.read())

        if temp_input.exists():
            temp_input.unlink()

        num_opt = len(opt_list)
        log(f"Frame optimization finished. Relaxed conformers: {num_opt}")
        return opt_list

    def writeFrames(self):
        """
        Save all output frames to a single <jobname>_frames.trj file.
        """
        if self.frames:
            trj_file = f"{self.jobname}_frames.trj"
            with open(trj_file, "w", encoding="utf-8") as f:
                f.write("".join(self.frames))
            log(f"Frames saved to: {trj_file}")

    def exportData(self):
        """
        Log final summary of outputs and artifacts.
        """
        num_frames = len(self.frames)
        log(f"Total output frames: {num_frames}")
        logfn = logger.get_logfile_name(self.jobname)
        log(f"Detailed run log saved to: {logfn}")


def get_parser():
    """
    Build parser extending core electronic options with MD flags.

    :return: Configured argparse.ArgumentParser object.
    """
    parser = xtb_core.get_parser()

    # Input & Identification
    parser.add_argument(
        FLAG_INPUT,
        dest="input",
        type=str,
        required=True,
        help="Path to input structure file (.xyz, .mol, .sdf, .pdb)",
    )
    parser.add_argument(
        FLAG_JOBNAME,
        dest="jobname",
        type=str,
        default=DEFAULT_JOBNAME,
        help="Job name identifier (default: xtb_md)",
    )

    # MD Parameters
    parser.add_argument(
        FLAG_TIME,
        dest="time",
        type=float,
        default=DEFAULT_TIME,
        help="MD simulation duration in ps (default: 10.0)",
    )
    parser.add_argument(
        FLAG_STEP,
        dest="step",
        type=float,
        default=DEFAULT_STEP,
        help="MD integration time step in fs (default: 1.0)",
    )
    parser.add_argument(
        FLAG_TEMP,
        dest="temp",
        type=float,
        default=DEFAULT_TEMP,
        help="MD thermostat temperature in K (default: 300.0)",
    )
    parser.add_argument(
        FLAG_DUMP,
        dest="dump",
        type=float,
        default=DEFAULT_DUMP,
        help="Trajectory snapshot dump interval in fs (default: 50.0)",
    )
    parser.add_argument(
        FLAG_NVT,
        dest="nvt",
        type=int,
        choices=[0, 1],
        default=DEFAULT_NVT,
        help="Thermostat ensemble: 1 for NVT (default), 0 for NVE",
    )
    parser.add_argument(
        FLAG_HMASS,
        dest="hmass",
        type=int,
        default=DEFAULT_HMASS,
        help="Hydrogen mass repartitioning in amu (default: 1)",
    )
    parser.add_argument(
        FLAG_SHAKE,
        dest="shake",
        type=int,
        choices=[0, 1, 2],
        default=DEFAULT_SHAKE,
        help="SHAKE constraints: 0=off, 1=X-H, 2=all bonds (default: 0)",
    )
    parser.add_argument(
        FLAG_SCCACC,
        dest="sccacc",
        type=float,
        default=DEFAULT_SCCACC,
        help="SCC accuracy level in MD (default: 2.0)",
    )
    parser.add_argument(
        FLAG_VELO,
        dest="velo",
        type=int,
        choices=[0, 1],
        default=DEFAULT_VELO,
        help="1 to include velocities in trajectory dumps (default: 0)",
    )

    # MetaDynamics Parameters
    parser.add_argument(
        FLAG_KPUSH,
        dest="kpush",
        type=float,
        default=DEFAULT_KPUSH,
        help="Pushing force constant in au (default: 0.1)",
    )
    parser.add_argument(
        FLAG_ALP,
        dest="alp",
        type=float,
        default=DEFAULT_ALP,
        help="Gaussian width parameter in Å^-2 (default: 0.6)",
    )
    parser.add_argument(
        FLAG_MTD_SAVE,
        dest="mtd_save",
        type=int,
        default=DEFAULT_MTD_SAVE,
        help="Maximum saved structures for RMSD bias potential (default: 50)",
    )

    # Workflow Control Flags
    parser.add_argument(
        FLAG_SKIP_MD,
        dest="skip_md",
        action="store_true",
        help="Skip unbiased MD simulation stage",
    )
    parser.add_argument(
        FLAG_SKIP_MTD,
        dest="skip_mtd",
        action="store_true",
        help="Skip MetaDynamics simulation stage",
    )
    parser.add_argument(
        FLAG_OPT_FRAMES,
        dest="opt_frames",
        action="store_true",
        help="Optimize sampled trajectory frames into conformers",
    )
    parser.add_argument(
        FLAG_SAMPLE_STRIDE,
        dest="sample_stride",
        type=int,
        default=DEFAULT_SAMPLE_STRIDE,
        help="Stride interval for sampling frames to optimize (default: 1)",
    )

    return parser


def validate_options(options, parser):
    """
    Validate argument values and file paths.

    :param options: Parsed options namespace.
    :param parser: Argument parser instance.
    """
    if not os.path.exists(options.input):
        parser.error(f"Input file '{options.input}' does not exist.")


def main(args=None):
    """
    CLI entrypoint for xTB MD driver.

    :param args: Optional argument list.
    """
    if args is None:
        args = sys.argv[1:]

    # If launched from user shell, dispatch via JobServer infrastructure
    if os.environ.get("_JOBSERVER_SANDBOX") != "1":
        from jobserver import JobDispatcher

        dispatcher = JobDispatcher()
        dispatcher.launch("hilbert-xtbmd", args)
        return

    parser = get_parser()
    options = parser.parse_args(args)
    jobname = options.jobname if options.jobname else JOBNAME

    global log
    log = logger.TextLogger(jobname)
    validate_options(options, parser)

    driver = XtbMDDriver(options, jobname)
    driver.run()


if __name__ == "__main__":
    main()
