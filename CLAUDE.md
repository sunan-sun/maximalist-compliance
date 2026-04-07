# Project: Maximalist Compliance

## Overview
Regressor-based robot dynamic identification pipeline with momentum observer for collision/contact detection.

## Project Structure
```
maxcomp/                  # Python package
  core.py                 # RobotIdentifier, BaseParams
  identification.py       # Full identification pipeline + plotting
  regressors.py           # Sub-regressor helpers + ground truth functions
  excitation.py           # Fourier trajectory optimizer
  verification.py         # Dynamic term verification vs Pinocchio
  observer.py             # Momentum-based disturbance observer
  control.py              # Computed torque controller
  viewer.py               # MuJoCo interactive viewer
scripts/                  # CLI entry points
  identify.py             # python scripts/identify.py <urdf>
  verify.py               # python scripts/verify.py <urdf>
  optimize_excitation.py  # python scripts/optimize_excitation.py <urdf>
  observe.py              # python scripts/observe.py <urdf>
  control.py              # python scripts/control.py <urdf>
  view.py                 # python scripts/view.py <model.xml>
robots/                   # URDF/MJCF robot models
output/                   # Generated plots (gitignored)
```

### Pipeline stages
1. **Identification** (`maxcomp/identification.py`) — find identifiable base parameters via rank-revealing QR on the joint torque regressor, estimate them from Fourier excitation trajectories
2. **Verification** (`maxcomp/verification.py`) — reconstruct individual dynamic terms (M, G, C·q̇, p) via sub-regressors and compare against Pinocchio ground truth
3. **Observer** (`maxcomp/observer.py`) — momentum-based disturbance observer using only identified parameters

## Obsidian Vault
Research notes are in `research-note/` within the same parent directory as this repo (e.g. `/home/sunan/research-note/`). Always search the parent directory for `research-note/` rather than hardcoding a path.

Key files in `projects/maximalist-compliance/`:
- `motivation.md` — project goals, core claim, and research context

## On Session Start
1. Locate the vault by finding `research-note/` in the parent directory of this repo
2. Read `motivation.md` from `<vault>/projects/maximalist-compliance/`
3. Check for any recent notes relevant to the current task

## On Technical Q&A
After any exchange that does one or more of the following:
- Answers a conceptual or theoretical question about dynamics, identification, or observers
- Generates a new formulation, critique, or insight
- Surfaces an open problem or reframes an existing one

> Prompt the user whether to append an entry to `<vault>/projects/maximalist-compliance/technical-qa.md` (resolve vault path per machine)

**Do not append for:** file operations, memory saves, formatting-only requests, or purely administrative tasks.

### Entry format (prepend — newest at top):
```
## YYYY-MM-DD — <short topic title>

**Q:** <one-sentence question summary>

**A:**
- <key point>
- <key point>
- ...

**Open questions raised:**
- ...

---
```

## Key Concepts
- **Base parameters**: the identifiable linear combinations of standard inertial parameters (10 per body). Found via column-pivoted QR on stacked regressors.
- **Sub-regressor trick**: isolate dynamic terms by zeroing specific inputs to the joint torque regressor (e.g. set ddq=0, g=0 to get C·q̇).
- **Cᵀ vs C**: the momentum observer needs Cᵀ·q̇, not C·q̇. These differ because C is not symmetric. Cᵀ·q̇ = Ṁ·q̇ − C·q̇ and cannot be obtained from the regressor alone — requires `pin.computeCoriolisMatrix`.

## Robot Models
- `robots/pendulum.urdf` — 1 DOF
- `robots/planar_2dof.urdf` — 2 DOF
- `robots/sixdof_arm.urdf` — 6 DOF (UR5-inspired, default)
- `robots/my-robot/` — CAD-generated MuJoCo model

## Dependencies
- **pinocchio** (robotics library, install via conda-forge — the pip `pinocchio` package is a different, unrelated nose plugin)
- numpy, scipy, matplotlib
