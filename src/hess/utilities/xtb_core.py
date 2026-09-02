"""
Core xTB execution engine and output parser for Hilbert workflows.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from hess.utilities import logger


HARTREE_TO_EV = 27.211386245988

# Flag constants - Electronic & Optimization
FLAG_CHARGE = "-charge"
FLAG_UNP_ELE = "-unp_ele"
FLAG_SOLVENT = "-solvent"
FLAG_ETEMP = "-etemp"
FLAG_ACC = "-acc"
FLAG_GFN = "-gfn"
FLAG_GFNFF = "-gfnff"
FLAG_OPTLEVEL = "-optlevel"
FLAG_MAXCYCLE = "-maxcycle"
FLAG_PARALLEL = "-parallel"

# Flag constants - Molecular Dynamics & MetaDynamics
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
FLAG_SAMPLE_STRIDE = "-sample_stride"

# Default constants - Electronic & Optimization
DEFAULT_CHARGE = 0
DEFAULT_UNP_ELE = 0
DEFAULT_SOLVENT = None
DEFAULT_ETEMP = 300.0
DEFAULT_ACC = 1.0
DEFAULT_GFN = 2
DEFAULT_OPTLEVEL = "normal"
DEFAULT_MAXCYCLE = 250
DEFAULT_MICROCYCLE = 25
DEFAULT_ENGINE = "lbfgs"
DEFAULT_FORCE_CONSTANT = 0.5

# Default constants - MD & MetaDynamics
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

# Control deck templates
XTB_OPT_CONTROL_TEMPLATE = """$opt
   engine={engine}
   maxcycle={maxcycle}
   microcycle={microcycle}{extra_options}
$end
"""

XTB_FIX_CONTROL_TEMPLATE = """$fix
{fix_directives}
$end
"""

XTB_CONSTRAIN_CONTROL_TEMPLATE = """$constrain
   force constant={force_constant}{constrain_directives}
$end
"""

XTB_MD_CONTROL_TEMPLATE = """$md
   time={time}
   step={step}
   temp={temp}
   dump={dump}
   nvt={nvt}
   hmass={hmass}
   shake={shake}
   sccacc={sccacc}
   velo={velo}{extra_options}
$end
"""

XTB_METADYN_TEMPLATE = """$metadyn
   save={save}
   kpush={kpush}
   alp={alp}{extra_options}
$end
"""


class XTBOutput:
    """
    Parses and stores all artifacts and properties from an xTB run directory.
    """

    def __init__(self, run_dir=None, log_file=None):
        """
        Initialize XTBOutput paths and data containers.

        :param run_dir: Directory containing xTB calculation output files.
        :type run_dir: str or pathlib.Path
        :param log_file: Specific log file path to parse.
        :type log_file: str or pathlib.Path
        """
        self.run_dir = Path(run_dir).resolve() if run_dir else Path.cwd()

        # File paths
        self.opt_fpath = self.resolveFile("xtbopt.xyz")
        self.trj_fpath = self.resolveFile("xtb.trj")
        self.charges_fpath = self.resolveFile("charges")
        self.mol_fpath = self.resolveFile("xtbtopo.mol")
        self.vib_fpath = self.resolveFile("vibspectrum")
        self.wbo_fpath = self.resolveFile("wbo")
        self.hessian_fpath = self.resolveFile("hessian")
        md_restart = self.resolveFile("mdrestart")
        xtb_restart = self.resolveFile("xtbrestart")
        self.restart_fpath = xtb_restart if xtb_restart else md_restart

        # Locate log file
        if log_file and Path(log_file).is_file():
            self.log_fpath = Path(log_file).resolve()
        else:
            log_files = sorted(
                self.run_dir.glob("*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            self.log_fpath = log_files[0] if log_files else None

        # Status markers
        self.success = (self.run_dir / ".xtboptok").exists() or (
            self.run_dir / "xtbmdok"
        ).exists()

        # Energetics and properties
        self.energy_eh = None
        self.energy_ev = None
        self.enthalpy_eh = None
        self.free_energy_eh = None
        self.gnorm = None
        self.gap_ev = None
        self.dipole_debye = None
        self.charges = []
        self.wbo = []
        self.frames = []
        self.frame_timestamps = []
        self.frequencies = []

        self.parseAll()

    def resolveFile(self, filename):
        """
        Resolve file path if it exists in the run directory.

        :param filename: Name of target file.
        :type filename: str
        :return: Path object if file exists, else None.
        :rtype: pathlib.Path or None
        """
        fpath = self.run_dir / filename
        return fpath if fpath.is_file() else None

    def parseAll(self):
        """
        Parse all detected output files in the run directory.
        """
        if self.log_fpath:
            self.parseLog()
        if self.charges_fpath:
            self.parseCharges()
        if self.trj_fpath:
            self.parseTrajectory()
        if self.vib_fpath:
            self.parseVibspectrum()
        if self.wbo_fpath:
            self.parseWBO()

    def parseWBO(self):
        """
        Parse Wiberg bond orders from wbo output file.
        Returns a list of tuples: (atom1_idx, atom2_idx, wbo_value)
        Indices are 1-based as per xTB output.
        """
        if not self.wbo_fpath or not self.wbo_fpath.exists():
            return

        wbo_list = []
        with open(self.wbo_fpath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        idx1 = int(parts[0])
                        idx2 = int(parts[1])
                        val = float(parts[2])
                        wbo_list.append((idx1, idx2, val))
                    except ValueError:
                        continue
        self.wbo = wbo_list

    def parseLog(self):
        """
        Extract energies, gradient norm, gap, and dipole from log file.
        """
        if not self.log_fpath or not self.log_fpath.exists():
            return

        with open(self.log_fpath, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Total Energy (Eh)
        e_match = re.findall(
            r"TOTAL ENERGY\s+([\-\d\.]+)\s+Eh", content, re.IGNORECASE
        )
        if e_match:
            self.energy_eh = float(e_match[-1])
            self.energy_ev = self.energy_eh * HARTREE_TO_EV

        # Total Enthalpy (Eh)
        h_match = re.findall(r"TOTAL ENTHALPY\s+([\-\d\.]+)\s+Eh", content)
        if h_match:
            self.enthalpy_eh = float(h_match[-1])

        # Total Free Energy (Eh)
        g_match = re.findall(r"TOTAL FREE ENERGY\s+([\-\d\.]+)\s+Eh", content)
        if g_match:
            self.free_energy_eh = float(g_match[-1])

        # Gradient Norm
        gnorm_match = re.findall(r"GRADIENT NORM\s+([\-\d\.]+)\s+Eh", content)
        if gnorm_match:
            self.gnorm = float(gnorm_match[-1])

        # HOMO-LUMO Gap
        gap_match = re.findall(r"HOMO-LUMO GAP\s+([\-\d\.]+)\s+eV", content)
        if gap_match:
            self.gap_ev = float(gap_match[-1])

        # Dipole Moment
        dipole_match = re.findall(
            r"full:\s+[\-\d\.]+\s+[\-\d\.]+\s+[\-\d\.]+\s+([\d\.]+)",
            content,
        )
        if dipole_match:
            self.dipole_debye = float(dipole_match[-1])

    def parseCharges(self):
        """
        Parse atomic partial charges from charges output file.
        """
        if not self.charges_fpath or not self.charges_fpath.exists():
            return

        with open(self.charges_fpath, encoding="utf-8") as f:
            self.charges = [float(line.strip()) for line in f if line.strip()]

    def parseTrajectory(self):
        """
        Split multi-frame xtb.trj trajectory into separate XYZ frame strings
        and extract frame timestamps in ps if present.
        """
        if not self.trj_fpath or not self.trj_fpath.exists():
            return

        with open(self.trj_fpath, encoding="utf-8") as f:
            lines = f.readlines()

        frames = []
        timestamps = []
        i = 0
        frame_idx = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            num_atoms = int(line)
            comment = lines[i + 1] if i + 1 < len(lines) else ""
            frame_lines = lines[i : i + num_atoms + 2]
            frames.append("".join(frame_lines))

            time_ps = 0.0
            if "time:" in comment:
                parts = comment.split("time:")[1].strip().split()
                if parts:
                    time_ps = float(parts[0])
            else:
                time_ps = frame_idx * (DEFAULT_DUMP / 1000.0)

            timestamps.append(time_ps)
            frame_idx += 1
            i += num_atoms + 2

        self.frames = frames
        self.frame_timestamps = timestamps

    def parseVibspectrum(self):
        """
        Parse vibrational wavenumbers (cm^-1) from vibspectrum file.
        """
        if not self.vib_fpath or not self.vib_fpath.exists():
            return

        freqs = []
        with open(self.vib_fpath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        freqs.append(
                            float(
                                parts[2]
                                if not parts[1]
                                .replace(".", "")
                                .replace("-", "")
                                .isdigit()
                                else parts[1]
                            )
                        )
                    except (IndexError, ValueError):
                        pass
        self.frequencies = freqs

    def cleanStructure(self):
        """
        Placeholder for post-processing and sanitizing structures.
        """
        pass

    def fixChargeBonds(self):
        """
        Placeholder for topology and charge-bond corrections.
        """
        pass


class XTBCoreEngine:
    """
    Base execution engine providing verification, command generation,
    and subprocess execution for xTB calculations.
    """

    def __init__(self, options, jobname):
        """
        Initialize base xTB calculation engine.

        :param options: Parsed command-line argument namespace.
        :type options: argparse.Namespace
        :param jobname: Job name identifier.
        :type jobname: str
        """
        self.options = options
        self.jobname = jobname
        self.xtb_path = None
        self.xtb_version = None

    def verifyXTBInstallation(self):
        """
        Verify that xTB binary is installed and executable in system PATH.

        :return: Path to verified xTB executable.
        :rtype: str
        """
        xtb_path = shutil.which("xtb")
        if not xtb_path:
            raise RuntimeError(
                "xTB executable 'xtb' not found in system PATH.\n"
                "Please install xTB."
            )

        res = subprocess.run(
            [xtb_path, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version_match = re.search(r"xtb version ([\d\.]+)", res.stdout)
        self.xtb_version = (
            version_match.group(1) if version_match else "unknown"
        )
        self.xtb_path = xtb_path
        return self.xtb_path

    def getXTBCommand(self, input_structure):
        """
        Assemble standard xTB command arguments with electronic flags.

        :param input_structure: Path to input molecular coordinate file.
        :type input_structure: str
        :return: List of command-line arguments for subprocess.
        :rtype: list[str]
        """
        if not self.xtb_path:
            self.verifyXTBInstallation()

        cmd = [self.xtb_path, str(input_structure)]

        if self.options.charge is not None:
            cmd.extend(["--chrg", str(self.options.charge)])
        if self.options.unp_ele is not None:
            cmd.extend(["--uhf", str(self.options.unp_ele)])
        if self.options.solvent:
            cmd.extend(["--alpb", str(self.options.solvent)])

        if self.options.gfnff:
            cmd.append("--gfnff")
        elif self.options.gfn is not None:
            cmd.extend(["--gfn", str(self.options.gfn)])

        if self.options.acc is not None:
            cmd.extend(["--acc", str(self.options.acc)])
        if self.options.etemp is not None:
            cmd.extend(["--etemp", str(self.options.etemp)])
        if self.options.parallel is not None:
            cmd.extend(["--parallel", str(self.options.parallel)])

        return cmd

    def createOptControlFile(
        self,
        control_path="control.txt",
        engine=DEFAULT_ENGINE,
        maxcycle=DEFAULT_MAXCYCLE,
        microcycle=DEFAULT_MICROCYCLE,
        extra_options="",
    ):
        """
        Write xTB optimization control file.

        :param control_path: Target path for control file.
        :type control_path: str or pathlib.Path
        :param engine: Optimizer engine (default: lbfgs).
        :type engine: str
        :param maxcycle: Maximum optimization cycles.
        :type maxcycle: int
        :param microcycle: Steps before re-generating internal coordinates.
        :type microcycle: int
        :param extra_options: Additional directives for $opt block.
        :type extra_options: str
        :return: Path to created control file.
        :rtype: pathlib.Path
        """
        control_content = XTB_OPT_CONTROL_TEMPLATE.format(
            engine=engine,
            maxcycle=maxcycle,
            microcycle=microcycle,
            extra_options=extra_options,
        )
        cpath = Path(control_path)
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(control_content)
        return cpath

    def createMDControlFile(
        self,
        control_path="control.txt",
        is_mtd=False,
    ):
        """
        Format and write xTB MD/MTD control file directly using self.options.

        :param control_path: Target path for control file.
        :type control_path: str or pathlib.Path
        :param is_mtd: True to include MetaDynamics block.
        :type is_mtd: bool
        :return: Path to created control file.
        :rtype: pathlib.Path
        """
        opts = self.options

        md_block = XTB_MD_CONTROL_TEMPLATE.format(
            time=opts.time,
            step=opts.step,
            temp=opts.temp,
            dump=opts.dump,
            nvt=opts.nvt,
            hmass=opts.hmass,
            shake=opts.shake,
            sccacc=opts.sccacc,
            velo=opts.velo,
            extra_options="",
        ).strip()

        blocks = [md_block]
        if is_mtd:
            mtd_block = XTB_METADYN_TEMPLATE.format(
                save=opts.mtd_save,
                kpush=opts.kpush,
                alp=opts.alp,
                extra_options="",
            ).strip()
            blocks.append(mtd_block)

        cpath = Path(control_path)
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(blocks) + "\n")
        return cpath

    def runOneXTB(self, cmd, workdir=None, logfn=None, env=None):
        """
        Execute an xTB subprocess command and return parsed XTBOutput.

        :param cmd: Complete command argument list.
        :type cmd: list[str]
        :param workdir: Execution working directory.
        :type workdir: str or pathlib.Path
        :param logfn: Optional output log filename.
        :type logfn: str
        :param env: Optional environment variables dictionary.
        :type env: dict
        :return: XTBOutput object containing parsed results.
        :rtype: XTBOutput
        """
        cwd = Path(workdir).resolve() if workdir else Path.cwd()
        logfile = (
            cwd / logfn
            if logfn
            else cwd / logger.get_logfile_name(self.jobname)
        )

        with open(logfile, "w", encoding="utf-8") as out_f:
            subprocess.run(
                cmd,
                stdout=out_f,
                stderr=subprocess.STDOUT,
                cwd=str(cwd),
                env=env,
                text=True,
                check=False,
            )

        return XTBOutput(run_dir=cwd, log_file=logfile)

    def runOpt(
        self,
        input_structure,
        workdir=None,
        extra_options="",
        ohess=True,
    ):
        """
        Run xTB geometry optimization with optional frequency calculation.

        :param input_structure: Path to input molecular coordinate file.
        :type input_structure: str
        :param workdir: Execution directory.
        :type workdir: str or pathlib.Path
        :param extra_options: Additional directives for $opt block.
        :type extra_options: str
        :param ohess: True to run --ohess, False for --opt.
        :type ohess: bool
        :return: XTBOutput object containing parsed results.
        :rtype: XTBOutput
        """
        cwd = Path(workdir).resolve() if workdir else Path.cwd()
        control_path = cwd / "control.txt"

        if not control_path.exists():
            self.createOptControlFile(
                control_path=control_path,
                engine=DEFAULT_ENGINE,
                maxcycle=self.options.maxcycle,
                microcycle=DEFAULT_MICROCYCLE,
                extra_options=extra_options,
            )

        cmd = self.getXTBCommand(input_structure)
        opt_flag = "--ohess" if ohess else "--opt"
        cmd.extend(
            [opt_flag, self.options.optlevel, "--input", str(control_path.name)]
        )
        return self.runOneXTB(cmd, workdir=cwd)

    def runMD(
        self,
        input_structure,
        is_mtd=False,
        workdir=None,
        logfn=None,
        env=None,
    ):
        """
        Run xTB Molecular Dynamics or MetaDynamics simulation.

        :param input_structure: Path to input molecular coordinate file.
        :type input_structure: str
        :param is_mtd: True to include MetaDynamics bias potential.
        :type is_mtd: bool
        :param workdir: Execution working directory.
        :type workdir: str or pathlib.Path
        :param logfn: Optional output log filename.
        :type logfn: str
        :param env: Optional environment variable dictionary.
        :type env: dict
        :return: XTBOutput object containing parsed trajectory and properties.
        :rtype: XTBOutput
        """
        cwd = Path(workdir).resolve() if workdir else Path.cwd()
        cwd.mkdir(parents=True, exist_ok=True)
        control_path = cwd / "control.txt"

        self.createMDControlFile(
            control_path=control_path,
            is_mtd=is_mtd,
        )

        cmd = self.getXTBCommand(input_structure)
        cmd.extend(["--md", "--input", str(control_path.name)])
        return self.runOneXTB(cmd, workdir=cwd, logfn=logfn, env=env)

    def runMTD(
        self,
        input_structure,
        workdir=None,
        logfn=None,
        env=None,
    ):
        """
        Run xTB MetaDynamics (MTD) simulation.

        :param input_structure: Path to input molecular coordinate file.
        :type input_structure: str
        :param workdir: Execution working directory.
        :type workdir: str or pathlib.Path
        :param logfn: Optional output log filename.
        :type logfn: str
        :param env: Optional environment variable dictionary.
        :type env: dict
        :return: XTBOutput object containing parsed trajectory and properties.
        :rtype: XTBOutput
        """
        return self.runMD(
            input_structure,
            is_mtd=True,
            workdir=workdir,
            logfn=logfn,
            env=env,
        )


def add_md_arguments(parser):
    """
    Add standard Molecular Dynamics and MetaDynamics CLI arguments to parser.

    :param parser: Target ArgumentParser instance.
    :type parser: argparse.ArgumentParser
    :return: The updated ArgumentParser instance.
    :rtype: argparse.ArgumentParser
    """
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
    parser.add_argument(
        FLAG_SAMPLE_STRIDE,
        dest="sample_stride",
        type=int,
        default=DEFAULT_SAMPLE_STRIDE,
        help="Stride interval for sampling trajectory frames (default: 1)",
    )
    return parser


def get_parser():
    """
    Build argument parser for xTB electronic and convergence options.

    :return: Configured argparse.ArgumentParser object.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        prefix_chars="-",
    )
    # Electronic & Solvation
    parser.add_argument(
        FLAG_CHARGE,
        dest="charge",
        type=int,
        default=DEFAULT_CHARGE,
        help="Total molecular charge (default: 0)",
    )
    parser.add_argument(
        FLAG_UNP_ELE,
        dest="unp_ele",
        type=int,
        default=DEFAULT_UNP_ELE,
        help="Number of unpaired electrons (default: 0)",
    )
    parser.add_argument(
        FLAG_SOLVENT,
        dest="solvent",
        type=str,
        default=DEFAULT_SOLVENT,
        help="ALPB implicit solvent (e.g. water, acetone, toluene, thf)",
    )
    # Convergence & Acceleration Controls
    parser.add_argument(
        FLAG_ETEMP,
        dest="etemp",
        type=float,
        default=DEFAULT_ETEMP,
        help="Electronic temperature in K for Fermi smearing (default: 300.0)",
    )
    parser.add_argument(
        FLAG_ACC,
        dest="acc",
        type=float,
        default=DEFAULT_ACC,
        help="Calculation accuracy (lower is tighter, default: 1.0)",
    )
    parser.add_argument(
        FLAG_GFN,
        dest="gfn",
        type=int,
        choices=[0, 1, 2],
        default=DEFAULT_GFN,
        help="GFN-xTB version: 0, 1, or 2 (default: 2)",
    )
    parser.add_argument(
        FLAG_GFNFF,
        dest="gfnff",
        action="store_true",
        help="Use GFN-FF force field parametrisation",
    )
    parser.add_argument(
        FLAG_OPTLEVEL,
        dest="optlevel",
        type=str,
        choices=[
            "crude",
            "sloppy",
            "loose",
            "normal",
            "tight",
            "verytight",
            "extreme",
        ],
        default=DEFAULT_OPTLEVEL,
        help="Optimization convergence tightness (default: normal)",
    )
    parser.add_argument(
        FLAG_MAXCYCLE,
        dest="maxcycle",
        type=int,
        default=DEFAULT_MAXCYCLE,
        help="Maximum optimization cycles allowed (default: 250)",
    )
    parser.add_argument(
        FLAG_PARALLEL,
        dest="parallel",
        type=int,
        default=None,
        help="Number of parallel CPU threads",
    )
    return parser


def get_md_parser():
    """
    Build argument parser containing electronic, optimization, and MD options.

    :return: Configured argparse.ArgumentParser object.
    :rtype: argparse.ArgumentParser
    """
    parser = get_parser()
    add_md_arguments(parser)
    return parser
