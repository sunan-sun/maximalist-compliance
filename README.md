# Robot Dynamic Identification & Momentum Observer

Plug-and-play pipeline for identifying robot inertial parameters from a URDF
and running a data-driven collision/contact detection observer.

## Dependencies

```
pip install pinocchio numpy scipy matplotlib
```

---

## Files

| File | Purpose |
|------|---------|
| `robot_identification.py` | Identifies base inertial parameters from URDF |
| `dynamic_terms_verification.py` | Verifies M, G, C, p estimates against Pinocchio ground truth |
| `momentum_observer_identified.py` | Runs momentum observer using identified parameters |
| `robots/` | URDF files: `pendulum.urdf`, `planar_2dof.urdf`, `sixdof_arm.urdf` |

The files form a pipeline — each one imports from the previous. You can run
any of them standalone; they call the upstream stages automatically.

---

## 1. `robot_identification.py`

Loads a URDF, generates a Fourier-series excitation trajectory, simulates
joint torques via RNEA, and estimates the **base inertial parameters** π̂_b
by least squares on the base regressor Y_b.

```bash
python robot_identification.py robots/planar_2dof.urdf
python robot_identification.py robots/sixdof_arm.urdf
python robot_identification.py robots/pendulum.urdf --T 60 --noise 0.05
```

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `urdf` | `robots/planar_2dof.urdf` | Path to robot URDF |
| `--T` | `30.0` | Identification trajectory duration [s] |
| `--dt` | `1e-3` | Timestep [s] |
| `--noise` | `0.0` | Gaussian noise added to torque measurements [N·m] |
| `--n_samples` | `5000` | Random configs used to find base parameter structure |

**Output**

- Console: base parameter estimates vs true values, relative errors, held-out RMSE
- Plot (saved as PNG): excitation trajectory, torques, base parameter bar chart, absolute errors

---

## 2. `dynamic_terms_verification.py`

Uses π̂_b from identification to reconstruct individual dynamic terms via
**sub-regressors**, then compares against Pinocchio ground truth on a
held-out trajectory.

```bash
python dynamic_terms_verification.py robots/planar_2dof.urdf
python dynamic_terms_verification.py robots/sixdof_arm.urdf
python dynamic_terms_verification.py robots/pendulum.urdf --T 60
```

**Options** — same as `robot_identification.py` plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--T` | `30.0` | Identification duration [s] |

**Sub-regressor trick** (how each term is isolated)

| Term | How |
|------|-----|
| `M(q)q̈` | Evaluate Y at (q, 0, q̈) with gravity off |
| `G(q)` | Evaluate Y at (q, 0, 0) with gravity on |
| `C(q,q̇)q̇` | Evaluate Y at (q, q̇, 0) with gravity off |
| `p = M(q)q̇` | Evaluate Y at (q, 0, q̇) with gravity off — substitute q̈ → q̇ |
| `Cᵀ(q,q̇)q̇` | **Not estimable** from identification alone — requires `pin.computeCoriolisMatrix` directly |

**Output**

- Console: RMSE for each dynamic term
- Plot (saved as PNG): 4 panels overlaying estimated vs true for M·q̈, G, p, and C·q̇ vs Cᵀ·q̇

---

## 3. `momentum_observer_identified.py`

Runs a PD-controlled simulation with a step external torque disturbance,
and estimates the disturbance using a **momentum observer** (De Luca & Mattone, 2005)
driven by identified parameters. Two observers run in parallel:

- **Full**: uses Cᵀ·q̇ from Pinocchio (requires model)
- **Approx**: drops Coriolis term (β ≈ −G only) — valid when robot moves slowly

```bash
python momentum_observer_identified.py robots/planar_2dof.urdf
python momentum_observer_identified.py robots/sixdof_arm.urdf
python momentum_observer_identified.py robots/sixdof_arm.urdf --K_O 30 --T_id 60
```

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `urdf` | `robots/planar_2dof.urdf` | Path to robot URDF |
| `--T_id` | `30.0` | Identification duration [s] |
| `--T_sim` | `6.0` | Simulation duration [s] |
| `--dt` | `1e-3` | Timestep [s] |
| `--K_O` | `10.0` | Observer gain (higher = faster response, more noise sensitivity) |
| `--Kp` | `50.0` | PD proportional gain (scaled per-joint by inertia automatically) |
| `--Kd` | `8.0` | PD derivative gain (scaled per-joint by inertia automatically) |
| `--tau_max` | `500.0` | Torque saturation limit [N·m] |
| `--noise` | `0.0` | Torque noise during identification [N·m] |

**External torque schedule (joint 1)**

```
0 – 2 s   : no contact
2 – 4 s   : +5 N·m step disturbance
4 – T s   : released
```

**Observer convergence time** ≈ `1/K_O` seconds. With `K_O=10` expect ~0.1 s lag.
Stability condition: `K_O × dt < 2` (e.g. K_O=10, dt=1e-3 → 0.01, well within limit).

**Output**

- Console: RMSE of observer residual vs true τ_ext during contact window
- Plot (saved as PNG): 4 panels — joint positions, momentum (true vs estimated),
  residual r vs τ_ext (true/full/approx), G(q) vs Cᵀ·q̇ magnitudes

---

## Recommended run order

```bash
# 1. Verify identification works
python robot_identification.py robots/planar_2dof.urdf

# 2. Verify individual dynamic terms
python dynamic_terms_verification.py robots/planar_2dof.urdf

# 3. Run the observer
python momentum_observer_identified.py robots/planar_2dof.urdf

# 4. Try the 6-DOF arm (longer identification recommended)
python momentum_observer_identified.py robots/sixdof_arm.urdf --T_id 60 --K_O 20
```

## Adding your own robot

Drop a URDF into `robots/` and pass its path as the first argument:

```bash
python momentum_observer_identified.py robots/my_robot.urdf
```

Requirements:
- All joints must be `revolute` (prismatic not yet supported)
- The URDF must have `<inertial>` blocks on every non-fixed link
- Pinocchio must be able to parse it (`pin.buildModelFromUrdf` is used)
