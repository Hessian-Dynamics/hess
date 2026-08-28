# Example Geometry Optimization & Thermochemistry Run

This directory contains a complete example of an xTB geometry optimization with analytical implicit solvation (ALPB water) and analytical frequency/thermochemistry calculation (`--ohess`).

---

## 1. Molecule
* **Compound:** Ethanol ($CH_3CH_2OH$)
* **Input File:** `input.xyz` (Initial strained geometry)

---

## 2. Command Executed
```bash
uv run hilbert-xtbopt -i input.xyz -ohess -alpb water -JOBNAME ethanol_opt
```

---

## 3. Calculation & Convergence Summary
* **Hamiltonian:** GFN2-xTB
* **Solvation Model:** ALPB (`water`, $\varepsilon = 78.4$)
* **Optimizer Engine:** L-ANCopt (`engine=lbfgs`)
* **Convergence Threshold:** Normal ($\Delta E < 5 \times 10^{-6}\text{ }E_h$, $\|\nabla E\| < 1 \times 10^{-3}\text{ }E_h/a_0$)
* **Initial Energy:** `-11.391672 Eh`
* **Final Potential Energy:** `-11.399161 Eh` (`-310.186975 eV`)
* **Final Total Enthalpy ($H$):** `-11.316324 Eh`
* **Final Gibbs Free Energy ($G$):** `-11.347183 Eh`
* **Final Gradient Norm:** `0.000793 Eh/a0`
* **HOMO-LUMO Gap:** `12.4945 eV`

---

## 4. Output Artifacts
* `xtbopt.xyz` — Final relaxed 3D molecular coordinates.
* `control.txt` — Generated xTB control parameter deck (`$opt`).
* `ethanol_opt.log` — Full step-by-step optimization and thermochemistry log.
* `vibspectrum` — Calculated vibrational frequencies and intensities.
* `g98.out` / `hessian` — Gaussian-formatted vibrational modes and Hessian matrix.
* `charges` / `wbo` — Atomic partial charges and Wiberg bond orders.
* `xtbtopo.mol` — Final molecular topology.
* `.xtboptok` — Normal completion confirmation flag.
