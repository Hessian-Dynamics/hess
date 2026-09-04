"""
Permanent Integration Testing & Benchmarking Suite for Hilbert Nanoreactor.
Executes 3 real-world benchmark molecules in parallel, measures wall-clock time,
conducts scientific verification, and appends to a persistent CSV.
"""

import concurrent.futures
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import ase.io
from rdkit import Chem


ROOT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT_DIR / "benchmarks"
HISTORY_CSV = BENCHMARK_DIR / "integration_test_history.csv"
PYTHON_EXEC = sys.executable or str(ROOT_DIR / ".venv" / "bin" / "python")
SCRIPT_PATH = ROOT_DIR / "src" / "hess" / "scripts" / "xtb_nanoreactor.py"

# Standard Benchmark Suite Definition
BENCHMARK_MOLECULES = [
    {
        "name": "Cyclobutene",
        "file": ROOT_DIR / "cyclobutene.xyz",
        "domain": "Pericyclic / Electrocyclic Opening",
        "expected_product_smarts": "C=CC=C",  # 1,3-Butadiene
        "charge": 0,
    },
    {
        "name": "Quadricyclane",
        "file": ROOT_DIR / "quadricyclane.xyz",
        "domain": "Solar Thermal Energy Storage",
        "expected_product_smarts": "C=C",  # Valence ring opening
        "charge": 0,
    },
    {
        "name": "Formamide",
        "file": ROOT_DIR / "formamide.xyz",
        "domain": "Prebiotic / Peptide Synthesis",
        "expected_product_smarts": "N=CO",  # Formimidic acid tautomerism
        "charge": 0,
    },
]

# Optimal Execution Parameters
DEFAULT_PARAMS = {
    "time": "2.0",  # 2.0 ps trajectory per replicate
    "temp": "2500",  # 2500 K thermal excitation
    "dump": "10",  # Snapshot every 10 fs (200 frames/rep)
    "nreplicates": "4",  # 4 independent MD runs
    "workers": "2",  # 2 worker processes per molecule
    "parallel": "1",  # 1 OpenMP thread per worker
    "nimages": "6",  # 6 NEB images for TS path
    "refine": "sevennet",  # SevenNet Universal MLIP for ground-state relaxation
    "ts_search": "sella_neb",  # ASE NEB + Sella Eigenvector Following
}


def get_git_commit():
    """Retrieve current short git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT_DIR)
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def run_single_benchmark(mol_meta, run_dir):
    """
    Execute a single molecule nanoreactor benchmark in its isolated sandbox.
    """
    name = mol_meta["name"]
    mol_dir = run_dir / name.lower()
    mol_dir.mkdir(parents=True, exist_ok=True)

    jobname = f"bench_{name.lower()}"
    cmd = [
        "env",
        "_JOBSERVER_SANDBOX=1",
        f"PYTHONPATH={ROOT_DIR / 'src'}",
        PYTHON_EXEC,
        str(SCRIPT_PATH),
        "-i",
        str(mol_meta["file"]),
        "-JOBNAME",
        jobname,
        "-charge",
        str(mol_meta["charge"]),
        "-time",
        DEFAULT_PARAMS["time"],
        "-temp",
        DEFAULT_PARAMS["temp"],
        "-dump",
        DEFAULT_PARAMS["dump"],
        "-nreplicates",
        DEFAULT_PARAMS["nreplicates"],
        "-workers",
        DEFAULT_PARAMS["workers"],
        "-parallel",
        DEFAULT_PARAMS["parallel"],
        "-nimages",
        DEFAULT_PARAMS["nimages"],
        "-refine",
        DEFAULT_PARAMS["refine"],
        "-ts_search",
        DEFAULT_PARAMS["ts_search"],
        "-sort_by_energy",
    ]

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"Started {name} in {mol_dir.name}"
    )
    start_t = time.perf_counter()

    res = subprocess.run(
        cmd,
        cwd=str(mol_dir),
        capture_output=True,
        text=True,
    )
    wall_t = time.perf_counter() - start_t
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"Finished {name} in {wall_t:.1f}s (code {res.returncode})"
    )

    return {
        "meta": mol_meta,
        "wall_time": wall_t,
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "dir": mol_dir,
        "jobname": jobname,
    }


def analyze_scientific_results(result):
    """Perform scientific checks: charge, thermodynamics, and pathways."""
    mol_dir = result["dir"]
    jobname = result["jobname"]
    meta = result["meta"]

    json_path = mol_dir / f"{jobname}_network.json"
    sdf_path = mol_dir / f"{jobname}_nodes.sdf"

    ref_charge = meta["charge"]

    analysis = {
        "num_nodes": 0,
        "num_products": 0,
        "num_edges": 0,
        "num_ts": 0,
        "min_ea": None,
        "max_ea": None,
        "charge_consistent": True,
        "detailed_balance_err": 0.0,
        "key_pathway_found": False,
        "status": "Success" if result["returncode"] == 0 else "Failed",
    }

    if not json_path.exists():
        analysis["status"] = "Failed"
        return analysis

    try:
        with open(json_path, encoding="utf-8") as f:
            net_data = json.load(f)

        nodes = net_data.get("nodes", [])
        edges = net_data.get("edges", [])

        analysis["num_nodes"] = len(nodes)
        analysis["num_products"] = max(0, len(nodes) - 1)
        analysis["num_edges"] = len(edges)

        # 1. Transition State & Activation Energy Analysis
        eas = [e["Ea"] for e in edges if e.get("Ea") is not None]
        analysis["num_ts"] = len(eas)
        if eas:
            analysis["min_ea"] = min(eas)
            analysis["max_ea"] = max(eas)

        # 2. Detailed Balance Thermodynamic Verification
        # For reversible edge pairs (u -> v) and (v -> u):
        # Verification: |(Ea_fwd - Ea_rev) - (E_prod - E_react)| == 0
        node_energies = {n["id"]: n.get("energy", 0.0) for n in nodes}
        edge_dict = {(e["source"], e["target"]): e.get("Ea") for e in edges}
        db_errors = []

        for (u, v), ea_uv in edge_dict.items():
            if (
                (v, u) in edge_dict
                and ea_uv is not None
                and edge_dict[(v, u)] is not None
            ):
                ea_vu = edge_dict[(v, u)]
                delta_e_rxn = node_energies.get(v, 0.0) - node_energies.get(
                    u, 0.0
                )
                delta_ea = ea_uv - ea_vu
                db_errors.append(abs(delta_ea - delta_e_rxn))

        if db_errors:
            analysis["detailed_balance_err"] = sum(db_errors) / len(db_errors)

        # 3. Charge & Valence Consistency via SDF
        if sdf_path.exists():
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
            for mol in suppl:
                if mol is not None:
                    mol_charge = Chem.GetFormalCharge(mol)
                    if mol_charge != ref_charge:
                        analysis["charge_consistent"] = False

        # 4. Key Chemical Pathway Verification
        expected_smarts = meta.get("expected_product_smarts")
        if expected_smarts:
            query = Chem.MolFromSmarts(expected_smarts)
            for n in nodes:
                mol = Chem.MolFromSmiles(n["id"])
                if mol and query and mol.HasSubstructMatch(query):
                    analysis["key_pathway_found"] = True
                    break

    except Exception as e:
        print(f"Error during scientific analysis of {meta['name']}: {e}")
        analysis["status"] = "Analysis Error"

    return analysis


def append_history_csv(records):
    """
    Append benchmark records to the persistent master CSV file.
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_CSV.exists()

    headers = [
        "Timestamp",
        "Git_Commit",
        "Molecule",
        "Formula",
        "Domain",
        "Wall_Time_s",
        "Num_Nodes",
        "Num_Products",
        "Num_Edges",
        "Num_TS",
        "Min_Ea_eV",
        "Max_Ea_eV",
        "Detailed_Balance_Err_eV",
        "Charge_Consistent",
        "Key_Pathway_Found",
        "Status",
    ]

    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"\n[Registry] Saved {len(records)} row(s) to: {HISTORY_CSV.name}")


def generate_markdown_report(run_dir, timestamp, git_commit, records):
    """Generate comprehensive Markdown report with verification metrics."""
    report_file = run_dir / "benchmark_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Hilbert Nanoreactor Integration Benchmark Report\n\n")
        f.write(f"- **Execution Timestamp**: `{timestamp}`\n")
        f.write(f"- **Git Commit**: `{git_commit}`\n")
        f.write(f"- **Sandbox Directory**: `{run_dir}`\n\n")

        f.write("## Standard Execution Parameters\n")
        for k, v in DEFAULT_PARAMS.items():
            f.write(f"- **`{k}`**: `{v}`\n")
        f.write("\n---\n\n")

        f.write("## Performance & Scientific Verification Results\n\n")
        f.write(
            "| Molecule | Formula | Time | Nodes | Edges | TS | "
            "Ea Range | DB Err | Charge | Key Rxn | Status |\n"
        )
        f.write(
            "| :--- | :--- | :---: | :---: | :---: | :---: | "
            ":---: | :---: | :---: | :---: | :--- |\n"
        )

        for r in records:
            ea_str = (
                f"{r['Min_Ea_eV']:.2f} - {r['Max_Ea_eV']:.2f}"
                if r["Min_Ea_eV"] is not None
                else "N/A"
            )
            chrg_icon = "✅" if r["Charge_Consistent"] else "❌"
            rxn_icon = "✅" if r["Key_Pathway_Found"] else "⚠️"
            f.write(
                f"| **{r['Molecule']}** | `{r['Formula']}` | "
                f"{r['Wall_Time_s']:.1f}s | "
                f"{r['Num_Nodes']} | {r['Num_Edges']} | {r['Num_TS']} | "
                f"{ea_str} | {r['Detailed_Balance_Err_eV']:.4f} | "
                f"{chrg_icon} | {rxn_icon} | **{r['Status']}** |\n"
            )

        f.write("\n### Scientific Verification Notes\n")
        f.write(
            "1. **Detailed Balance**: Measures $|(E_a^{\\text{fwd}} - "
            "E_a^{\\text{rev}}) - \\Delta E_{\\text{rxn}}|$.\n"
        )
        f.write(
            "2. **Charge Conservation**: Checks formal charge neutrality "
            "across all 3D SDF molecules.\n"
        )
        f.write(
            "3. **Key Reaction**: Confirms known transformations "
            "(Butadiene, Norbornadiene, Cracking).\n"
        )

    print(f"[Report] Generated detailed markdown report at: {report_file}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()

    run_dir = BENCHMARK_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("HILBERT NANOREACTOR INTEGRATION BENCHMARK SUITE")
    print(f"Timestamp: {timestamp} | Commit: {git_commit}")
    print("Executing 3 Real-World Molecules Concurrently (ProcessPoolExecutor)")
    print(f"Sandbox: {run_dir}")
    print("=" * 78)

    # Launch all 3 molecules concurrently
    total_start = time.perf_counter()
    raw_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_single_benchmark, mol, run_dir)
            for mol in BENCHMARK_MOLECULES
        ]
        for fut in concurrent.futures.as_completed(futures):
            raw_results.append(fut.result())

    total_wall_time = time.perf_counter() - total_start
    print("=" * 78)
    print(f"All runs finished! Total Wall Time: {total_wall_time:.2f}s")
    print("Analyzing scientific consistency...")

    # Sort results in consistent order
    name_order = [m["name"] for m in BENCHMARK_MOLECULES]
    raw_results.sort(key=lambda r: name_order.index(r["meta"]["name"]))

    csv_records = []
    for res in raw_results:
        sci = analyze_scientific_results(res)
        ref_atoms = ase.io.read(str(res["meta"]["file"]))

        rec = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Git_Commit": git_commit,
            "Molecule": res["meta"]["name"],
            "Formula": ref_atoms.get_chemical_formula(),
            "Domain": res["meta"]["domain"],
            "Wall_Time_s": round(res["wall_time"], 2),
            "Num_Nodes": sci["num_nodes"],
            "Num_Products": sci["num_products"],
            "Num_Edges": sci["num_edges"],
            "Num_TS": sci["num_ts"],
            "Min_Ea_eV": round(sci["min_ea"], 4)
            if sci["min_ea"] is not None
            else None,
            "Max_Ea_eV": round(sci["max_ea"], 4)
            if sci["max_ea"] is not None
            else None,
            "Detailed_Balance_Err_eV": round(sci["detailed_balance_err"], 4),
            "Charge_Consistent": sci["charge_consistent"],
            "Key_Pathway_Found": sci["key_pathway_found"],
            "Status": sci["status"],
        }
        csv_records.append(rec)

    # Append to Master History CSV
    append_history_csv(csv_records)

    # Generate Markdown Report
    generate_markdown_report(run_dir, timestamp, git_commit, csv_records)

    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY")
    print("=" * 78)
    for r in csv_records:
        print(
            f"• {r['Molecule']} ({r['Formula']}): {r['Wall_Time_s']}s | "
            f"Nodes: {r['Num_Nodes']} | Edges: {r['Num_Edges']} | "
            f"TS: {r['Num_TS']} | DB Err: {r['Detailed_Balance_Err_eV']}eV | "
            f"Chrg: {r['Charge_Consistent']} | Rxn: {r['Key_Pathway_Found']} | "
            f"{r['Status']}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
