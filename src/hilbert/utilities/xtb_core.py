"""
Core xTB execution engine and output parser for Hilbert workflows.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from hilbert.utilities import logger


HARTREE_TO_EV = 27.211386245988

# Flag constants
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

# Default constants
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
        :param log_file: Specific log file path to parse.
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
        self.restart_fpath = self.resolveFile("xtbrestart") or self.resolveFile(
            "mdrestart"
        )

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
        self.frames = []
        self.frequencies = []

        self.parseAll()

    def resolveFile(self, filename):
        """
        Resolve file path if it exists in the run directory.

        :param filename: Name of target file.
        :return: Path object if file exists, else None.
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
        Split multi-frame xtb.trj trajectory into separate XYZ frame strings.
        """
        if not self.trj_fpath or not self.trj_fpath.exists():
            return

        with open(self.trj_fpath, encoding="utf-8") as f:
            lines = f.readlines()

        frames = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                num_atoms = int(line)
                frame_lines = lines[i : i + num_atoms + 2]
                frames.append("".join(frame_lines))
                i += num_atoms + 2
            except ValueError:
                i += 1

        self.frames = frames

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
                        freqs.append(float(parts[1]))
                    except ValueError:
                        continue
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
        :param jobname: Job name identifier.
        """
        self.options = options
        self.jobname = jobname
        self.xtb_path = None
        self.xtb_version = None

    def verifyXTBInstallation(self):
        """
        Verify that xTB binary is installed and executable in system PATH.

        :return: Path to verified xTB executable.
        """
        xtb_path = shutil.which("xtb")
        if not xtb_path:
            raise RuntimeError(
                "xTB executable 'xtb' not found in system PATH.\n"
                "Please install xTB."
            )

        try:
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
        except Exception as e:
            raise RuntimeError(
                f"Found xTB at '{xtb_path}', but failed to execute: {e}"
            ) from e

        self.xtb_path = xtb_path
        return self.xtb_path

    def getXTBCommand(self, input_structure):
        """
        Assemble standard xTB command arguments with electronic flags.

        :param input_structure: Path to input molecular coordinate file.
        :return: List of command-line arguments for subprocess.
        """
        if not self.xtb_path:
            self.verifyXTBInstallation()

        cmd = [self.xtb_path, input_structure]

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
        :param engine: Optimizer engine (default: lbfgs).
        :param maxcycle: Maximum optimization cycles.
        :param microcycle: Steps before re-generating internal coordinates.
        :param extra_options: Additional directives for $opt block.
        :return: Path to created control file.
        """
        control_content = XTB_OPT_CONTROL_TEMPLATE.format(
            engine=engine,
            maxcycle=maxcycle,
            microcycle=microcycle,
            extra_options=extra_options,
        )
        with open(control_path, "w", encoding="utf-8") as f:
            f.write(control_content)
        return control_path

    def runOneXTB(self, cmd, workdir=None, logfn=None):
        """
        Execute an xTB subprocess command and return parsed XTBOutput.

        :param cmd: Complete command argument list.
        :param workdir: Execution working directory.
        :param logfn: Optional output log filename.
        :return: XTBOutput object containing parsed results.
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
                text=True,
                check=False,
            )

        return XTBOutput(run_dir=cwd, log_file=logfile)

    def runOpt(
        self,
        input_structure,
        optlevel=None,
        engine=DEFAULT_ENGINE,
        maxcycle=None,
        microcycle=DEFAULT_MICROCYCLE,
        workdir=None,
    ):
        """
        Run xTB geometry optimization with frequency calculation.

        :param input_structure: Path to input molecular coordinate file.
        :param optlevel: Tightness level for optimization convergence.
        :param engine: Optimizer engine (default: lbfgs).
        :param maxcycle: Maximum optimization cycles.
        :param microcycle: Steps before re-generating coordinates.
        :param workdir: Execution directory.
        :return: XTBOutput object containing parsed results.
        """
        cwd = Path(workdir).resolve() if workdir else Path.cwd()
        control_path = cwd / "control.txt"

        chosen_optlevel = (
            optlevel if optlevel is not None else self.options.optlevel
        )
        chosen_maxcycle = (
            maxcycle if maxcycle is not None else self.options.maxcycle
        )

        if not control_path.exists():
            self.createOptControlFile(
                control_path=control_path,
                engine=engine,
                maxcycle=chosen_maxcycle,
                microcycle=microcycle,
            )

        cmd = self.getXTBCommand(input_structure)
        cmd.extend(
            ["--ohess", chosen_optlevel, "--input", str(control_path.name)]
        )
        return self.runOneXTB(cmd, workdir=cwd)


def get_parser():
    """
    Build argument parser for xTB electronic and convergence options.

    :return: Configured argparse.ArgumentParser object.
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
