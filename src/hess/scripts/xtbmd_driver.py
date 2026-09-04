"""
xTB Molecular Dynamics and MetaDynamics Driver.
Inherits from XTBCoreEngine following the Hilbert driver architecture.
"""

import os
import sys
from pathlib import Path

from hess.utilities import fileutils, logger, xtb_core


log = logger.TextLogger("xtb_md")


PROGRAM_NAME = "xTB Molecular Dynamics"
JOBNAME = "xtb_md"
FLAG_JOBNAME = "-JOBNAME"

# Flag constants
FLAG_INPUT = "-i"
FLAG_SKIP_MD = "-skip_md"
FLAG_SKIP_MTD = "-skip_mtd"
FLAG_OPT_FRAMES = "-opt_frames"

# Default constants
DEFAULT_JOBNAME = "xtb_md"


class XtbMDDriver(xtb_core.XTBCoreEngine):
    """
    Driver managing standard MD, MetaDynamics, and frame optimization.
    """

    def __init__(self, options, jobname):
        """
        Initialize XtbMDDriver instance.

        :param options: Parsed command-line argument namespace.
        :type options: argparse.Namespace
        :param jobname: Job name identifier.
        :type jobname: str
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

        if not self.options.skip_md:
            with fileutils.chdir("md", create=True):
                self.doUnbiasedMD()

        if not self.options.skip_mtd:
            with fileutils.chdir("mtd", create=True):
                self.doMetaDynamics()

        self.frames = self.getCombinedFrames(self.md_out, self.mtd_out)
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

    def doUnbiasedMD(self):
        """
        Execute unbiased MD simulation.

        :return: XTBOutput object for the MD run or None.
        :rtype: XTBOutput or None
        """
        if self.options.skip_md:
            return None

        log("Starting unbiased Molecular Dynamics simulation...")
        self.md_out = super().runMD(
            self.input_file,
            is_mtd=False,
            logfn=f"{self.jobname}_md.log",
        )
        num_frames = len(self.md_out.frames)
        log(f"Unbiased MD complete. Frames collected: {num_frames}")
        return self.md_out

    def doMetaDynamics(self):
        """
        Execute biased MetaDynamics simulation.

        :return: XTBOutput object for the MTD run or None.
        :rtype: XTBOutput or None
        """
        if self.options.skip_mtd:
            return None

        log("Starting MetaDynamics (MTD) simulation...")
        self.mtd_out = super().runMTD(
            self.input_file,
            logfn=f"{self.jobname}_mtd.log",
        )
        num_frames = len(self.mtd_out.frames)
        log(f"MetaDynamics complete. Frames collected: {num_frames}")
        return self.mtd_out

    def getCombinedFrames(self, md_out, mtd_out):
        """
        Merge frames from MD and MetaDynamics stages.

        :param md_out: XTBOutput from unbiased MD.
        :type md_out: XTBOutput
        :param mtd_out: XTBOutput from MetaDynamics.
        :type mtd_out: XTBOutput
        :return: List of all combined XYZ frame strings.
        :rtype: list[str]
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
        :type frames: list[str]
        :return: List of relaxed XYZ frame strings.
        :rtype: list[str]
        """
        stride = max(1, self.options.sample_stride)
        sampled_frames = frames[::stride]
        num_sampled = len(sampled_frames)
        log(f"Optimizing {num_sampled} sampled trajectory frames...")

        opt_list = []
        temp_input = Path("temp_frame_opt.xyz")

        for frame in sampled_frames:
            with open(temp_input, "w", encoding="utf-8") as file_handle:
                file_handle.write(frame)

            opt_out = self.runOpt(str(temp_input))
            if opt_out.success and opt_out.opt_fpath:
                with open(opt_out.opt_fpath, encoding="utf-8") as file_handle:
                    opt_list.append(file_handle.read())

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
            with open(trj_file, "w", encoding="utf-8") as file_handle:
                file_handle.write("".join(self.frames))
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
    :rtype: argparse.ArgumentParser
    """
    parser = xtb_core.get_md_parser()

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

    return parser


def validate_options(options, parser):
    """
    Validate argument values and file paths.

    :param options: Parsed options namespace.
    :type options: argparse.Namespace
    :param parser: Argument parser instance.
    :type parser: argparse.ArgumentParser
    """
    if not os.path.exists(options.input):
        parser.error(f"Input file '{options.input}' does not exist.")


def main(args=None):
    """
    CLI entrypoint for xTB MD driver.

    :param args: Optional argument list.
    :type args: list[str]
    """
    if args is None:
        args = sys.argv[1:]

    # If launched from user shell, dispatch via JobServer infrastructure
    if os.environ.get("_JOBSERVER_SANDBOX") != "1":
        from jobserver import JobDispatcher

        dispatcher = JobDispatcher()
        dispatcher.launch("hess-xtbmd", args)
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
