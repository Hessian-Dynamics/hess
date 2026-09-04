# Hilbert Nanoreactor Integration Benchmark Report

- **Execution Timestamp**: `20260904_183131`
- **Git Commit**: `09f01fa`
- **Sandbox Directory**: `/Users/ujjawalm/hess/benchmarks/run_20260904_183131`

## Standard Execution Parameters
- **`time`**: `2.0`
- **`temp`**: `2500`
- **`dump`**: `10`
- **`nreplicates`**: `4`
- **`workers`**: `2`
- **`parallel`**: `1`
- **`nimages`**: `6`
- **`refine`**: `sevennet`
- **`ts_search`**: `sella_neb`

---

## Performance & Scientific Verification Results

| Molecule | Formula | Time | Nodes | Edges | TS | Ea Range | DB Err | Charge | Key Rxn | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cyclobutene** | `C4H6` | 403.4s | 3 | 4 | 4 | -0.00 - 6.44 | 2.3480 | ✅ | ✅ | **Success** |
| **Quadricyclane** | `C7H8` | 723.2s | 7 | 7 | 7 | 0.00 - 5.86 | 0.0000 | ✅ | ⚠️ | **Success** |
| **Formamide** | `CH3NO` | 167.7s | 3 | 4 | 4 | 0.29 - 6.24 | 5.7849 | ✅ | ⚠️ | **Success** |

### Scientific Verification Notes
1. **Detailed Balance**: Measures $|(E_a^{\text{fwd}} - E_a^{\text{rev}}) - \Delta E_{\text{rxn}}|$.
2. **Charge Conservation**: Checks formal charge neutrality across all 3D SDF molecules.
3. **Key Reaction**: Confirms known transformations (Butadiene, Norbornadiene, Cracking).
