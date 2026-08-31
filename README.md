# hess

**The core physics and computational chemistry engine for Hessian Dynamics.**

`hess` is a unified, high-performance orchestration layer designed to execute molecular simulations across diverse computational backends (Semi-empirical, Machine Learning Force Fields, DFT, etc.). 

It standardizes inputs, execution lifecycles, and outputs, allowing complex workflows to run agnostically of the underlying math engine.

## Core Philosophy
1. **Engine Agnostic:** Whether the forces are computed via xTB, MACE (PyTorch), or classical force fields, the CLI and output trajectories remain perfectly standardized.
2. **Hardware Optimized:** Designed to pair natively with `jobserver` for strict physical core pinning (OpenMP/MKL affinity) and background queue management.
3. **Strict Constraints:** Zero magic strings, zero double-hyphen flags, and strict architectural enforcement.

## Available Drivers

* **`hess-xtbmd`** — Unbiased Molecular Dynamics using the xTB semi-empirical engine.
* **`hess-xtbopt`** — Geometry optimization and ground-state relaxation (xTB).
* *(In Development)* **`hess-mlffopt`** — Neural Network Potential (MACE) optimizations.

## Quick Start

Calculations are launched via single-hyphen flags and seamlessly hand off to the background job scheduler.

```bash
# Run a 10ps Molecular Dynamics simulation on 4 physical cores
hess-xtbmd -i molecule.xyz -JOBNAME sim_01 -time 10.0 -HOST localhost:4

# Poll the live status of the running calculation
jobserver poll sim_01
```
