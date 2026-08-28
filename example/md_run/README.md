# xTB Molecular Dynamics & MetaDynamics Example Run

This directory demonstrates an end-to-end xTB Molecular Dynamics (MD) and MetaDynamics (MTD) workflow using `hilbert-xtbmd`.

---

## 1. Execution Command

```bash
hilbert-xtbmd \
  -i input.xyz \
  -time 0.1 \
  -step 1.0 \
  -temp 350.0 \
  -dump 20.0 \
  -kpush 0.1 \
  -alp 0.6 \
  -opt_frames \
  -sample_stride 2 \
  -JOBNAME ethanol_md
```

---

## 2. Directory Layout & Artifacts

```
example/md_run/
├── input.xyz                        # Initial ethanol coordinate file
├── ethanol_md.log                   # Hilbert driver run log
├── ethanol_md_frames.trj            # Unified frames file (relaxed conformers if -opt_frames, else raw trajectory)
│
├── md/                              # Unbiased Molecular Dynamics subfolder
│   ├── control_md.txt               # xTB $md block control file
│   ├── ethanol_md_md.log            # Full raw xTB MD stdout log
│   ├── xtb.trj                      # MD trajectory coordinates
│   └── mdrestart                    # MD restart state
│
├── mtd/                             # Biased MetaDynamics subfolder
│   ├── control_mtd.txt              # xTB $md + $metadyn block control file
│   ├── ethanol_md_mtd.log           # Full raw xTB MTD stdout log
│   └── xtb.trj                      # MetaDynamics trajectory coordinates
│
└── opt_frames/                      # Frame optimization subfolder
    ├── control.txt                  # L-ANCopt ($opt engine=lbfgs) control deck
    └── xtbopt.xyz                   # Minimized coordinates
```

---

## 3. Output Behavior

* **Default (Without `-opt_frames`):** Writes the concatenated raw MD and MetaDynamics trajectory snapshots directly into `<jobname>_frames.trj`.
* **With `-opt_frames`:** Optimizes the sampled trajectory snapshots into energy-minimized local minima conformers and writes them into `<jobname>_frames.trj`.
