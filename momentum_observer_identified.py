"""
Data-Driven Momentum Observer
==============================
Replaces the hardcoded dynamic model of momentum_observer.py with the
identified base parameters Φ*_0 from robot_identification.py.

Pipeline
--------
  [Offline]  robot_identification  →  Φ*_0, bp
                                        ↓
  [Online]   momentum observer using sub-regressors:

    p(q,q̇)  = Y_M_b(q, 0, q̇) · Φ*_0          ← generalized momentum
    G(q)    = Y_G_b(q, 0,  0) · Φ*_0          ← gravity
    Cᵀ·q̇   = pin.computeCoriolisMatrix.T @ q̇  ← NOT estimable from regressor
                                                   (requires pinocchio model)

Observer dynamics  (De Luca & Mattone, 2005):
    ṗ_obs = τ  +  β  +  r
    ṙ     = K_O · (p − p_obs)
w
    β_full   = Cᵀ(q,q̇)·q̇  −  G(q)    (complete)
    β_approx =              −  G(q)    (paper approximation, Coriolis ignored)

Two observers are run in parallel so you can see how much the Coriolis
approximation matters for your robot and operating conditions.

Plant simulation uses pin.aba (Articulated Body Algorithm, forward dynamics).

Usage
-----
    python momentum_observer_identified.py robots/planar_2dof.urdf
    python momentum_observer_identified.py robots/sixdof_arm.urdf --K_O 20
"""

import argparse
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from robot_identification import RobotIdentifier, BaseParams
from dynamic_terms_verification import (
    sub_regressor_inertia,
    sub_regressor_gravity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Plant simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(
    model,
    data,
    bp:        BaseParams,
    pi_b_est:  np.ndarray,
    K_O:       float = 10.0,
    dt:        float = 1e-3,
    T:         float = 6.0,
    Kp:        float = 50.0,
    Kd:        float = 8.0,
    q_ref:     np.ndarray = None,
    tau_max:   float = 500.0,
):
    """
    Simulate the robot plant + two momentum observers in parallel.

    External torque schedule (joint 0):
        0   →  2 s :  no contact
        2   →  4 s :  step external torque  τ_ext[0] = +5 N·m
        4   →  T   :  released

    Returns a dict of logged signals.
    """
    nv = model.nv
    nq = model.nq

    if q_ref is None:
        q_ref = np.zeros(nv)

    # ── Per-joint gain scaling ────────────────────────────────────────────────
    # Scalar Kp/Kd applied uniformly causes Euler instability on low-inertia
    # joints (wrist).  Euler stability requires Kd < 2·m_eff/dt; for a wrist
    # with m_eff~0.0003 kg·m² and dt=1e-3, that limit is Kd < 0.6 — far below
    # a typical scalar Kd.
    #
    # Fix: scale gains proportional to the diagonal of M(q_neutral).
    # Every joint then has the same natural frequency ω = sqrt(Kp / m_ref)
    # and the same damping ratio, while respecting the per-joint stability limit.
    M0     = pin.crba(model, data, pin.neutral(model))
    m_diag = np.diag(M0)
    m_ref  = m_diag.mean()
    Kp_vec = Kp * (m_diag / m_ref)   # [N·m/rad]   per joint
    Kd_vec = Kd * (m_diag / m_ref)   # [N·m·s/rad] per joint
    print(f"  Inertia diagonal (neutral):  min={m_diag.min():.4f}  max={m_diag.max():.4f}  mean={m_ref:.4f} kg·m²")
    print(f"  Per-joint Kp range: [{Kp_vec.min():.2f}, {Kp_vec.max():.2f}]  "
          f"Kd range: [{Kd_vec.min():.3f}, {Kd_vec.max():.3f}]")

    # ── Initial conditions ───────────────────────────────────────────────────
    q  = pin.neutral(model)
    dq = np.zeros(nv)

    # Observer state: only p_obs is integrated; r is computed algebraically
    p_obs_full   = np.zeros(nv)   # with Coriolis
    p_obs_approx = np.zeros(nv)   # without Coriolis (paper approximation)

    t_arr = np.arange(0.0, T, dt)
    N     = len(t_arr)

    log = {
        "t":          t_arr,
        "q":          np.zeros((N, nv)),
        "dq":         np.zeros((N, nv)),
        "tau_ctrl":   np.zeros((N, nv)),
        "tau_ext":    np.zeros((N, nv)),
        "r_full":     np.zeros((N, nv)),
        "r_approx":   np.zeros((N, nv)),
        "p":          np.zeros((N, nv)),   # true momentum (from pinocchio M)
        "p_est":      np.zeros((N, nv)),   # estimated momentum (from Φ*_0)
        "G_est":      np.zeros((N, nv)),   # estimated gravity
        "CT_dq":      np.zeros((N, nv)),   # Cᵀ·q̇  (pinocchio)
    }

    for i, t in enumerate(t_arr):

        # ── External torque schedule ─────────────────────────────────────────
        tau_ext = np.zeros(nv)
        if 2.0 <= t < 4.0:
            tau_ext[0] = 2.0    # step disturbance on joint 0

        # ── All dynamic terms at current (q, dq) — consistent time step ─────
        q_r    = q if nq == nv else q[:nv]

        Y_G_b  = sub_regressor_gravity(model, data, q_r, bp)
        G_est  = Y_G_b @ pi_b_est

        Y_p_b  = sub_regressor_inertia(model, data, q_r, dq, bp)
        p_est  = Y_p_b @ pi_b_est

        C_mat  = pin.computeCoriolisMatrix(model, data, q_r, dq)
        CT_dq  = C_mat.T @ dq

        M_true = pin.crba(model, data, q_r)
        p_true = M_true @ dq

        # ── PD control law with gravity compensation (per-joint gains) ───────
        tau  = Kp_vec * (q_ref - q_r) - Kd_vec * dq + G_est
        tau  = np.clip(tau, -tau_max, tau_max)

        # ── β terms ──────────────────────────────────────────────────────────
        beta_full   = CT_dq - G_est   # complete
        beta_approx =       - G_est   # Coriolis ignored

        # ── Observer update — r is algebraic, only p_obs is integrated ────────
        #   Continuous:  ṗ_obs = τ + β + r,   r = K_O·(p − p_obs)
        #   Discrete:    p_obs[k+1] = p_obs[k] + (τ + β + K_O·(p−p_obs))·dt
        #   Eigenvalue:  λ = 1 − K_O·dt  → stable iff K_O·dt < 2
        #
        #   Integrating r separately (ṙ = K_O·ė) gives purely-imaginary
        #   eigenvalues ±i√K_O which Euler always makes diverge.

        r_full   = K_O * (p_est - p_obs_full)
        r_approx = K_O * (p_est - p_obs_approx)

        p_obs_full   += (tau + beta_full   + r_full)   * dt
        p_obs_approx += (tau + beta_approx + r_approx) * dt

        # ── Plant: forward dynamics via ABA ──────────────────────────────────
        ddq = pin.aba(model, data, q, dq, tau + tau_ext)
        if not np.isfinite(ddq).all():
            print(f"\n  ⚠  Simulation diverged at t={t:.3f} s — stopping early.")
            for k in log:
                if hasattr(log[k], '__len__') and k != "t":
                    log[k] = log[k][:i]
            log["t"] = log["t"][:i]
            break
        q  = pin.integrate(model, q, dq * dt)
        dq = dq + ddq * dt

        # ── Log ──────────────────────────────────────────────────────────────
        log["q"][i]        = q_r
        log["dq"][i]       = dq
        log["tau_ctrl"][i] = tau
        log["tau_ext"][i]  = tau_ext
        log["r_full"][i]   = r_full
        log["r_approx"][i] = r_approx
        log["p"][i]        = p_true
        log["p_est"][i]    = p_est
        log["G_est"][i]    = G_est
        log["CT_dq"][i]    = CT_dq

    return log


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot(log, robot_name, contact_window=(2.0, 4.0)):
    t  = log["t"]
    nv = log["q"].shape[1]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    t0, t1 = contact_window

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"Data-Driven Momentum Observer — {robot_name}\n"
        f"Full observer (with Cᵀ·q̇)  vs  Approx (Coriolis ignored)  |  "
        f"Shaded region = contact",
        fontsize=11, fontweight="bold",
    )

    # ── Top-left: joint trajectories ─────────────────────────────────────────
    ax = axes[0, 0]
    for j in range(nv):
        ax.plot(t, log["q"][:, j], color=colors[j % len(colors)],
                lw=1.3, label=f"q{j+1}")
    ax.axvspan(t0, t1, alpha=0.1, color="red", label="contact")
    ax.set_title("Joint positions  q(t)")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # ── Top-right: momentum  p true vs estimated ──────────────────────────────
    ax = axes[0, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["p"][:, j],     color=c, lw=1.5, alpha=0.6,
                label=f"j{j+1} true")
        ax.plot(t, log["p_est"][:, j], color=c, lw=2.0, ls="--",
                zorder=3, label=f"j{j+1} est")
    ax.axvspan(t0, t1, alpha=0.1, color="red")
    ax.set_title("Generalised momentum  p = M(q)·q̇\ntrue (—) vs identified (--)")
    ax.set_ylabel("kg·m²/s"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # ── Bottom-left: residual r per joint — full vs approx ───────────────────
    ax = axes[1, 0]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["tau_ext"][:, j], color=c, lw=2.0, alpha=0.5,
                label=f"j{j+1} τ_ext true")
        ax.plot(t, log["r_full"][:, j],  color=c, lw=1.8, ls="--",
                zorder=3, label=f"j{j+1} r full")
        ax.plot(t, log["r_approx"][:, j], color=c, lw=5, ls=":",
                zorder=2, label=f"j{j+1} r approx")
    ax.axvspan(t0, t1, alpha=0.1, color="red")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title("Observer residual  r  vs true  τ_ext\n"
                 "true (—)  full (--) with Cᵀ·q̇   approx (·) without Coriolis")
    ax.set_ylabel("N·m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    # ── Bottom-right: Cᵀ·q̇ vs G(q)  (relative magnitudes) ───────────────────
    ax = axes[1, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["G_est"][:, j],  color=c, lw=1.4,
                label=f"j{j+1} G(q)")
        ax.plot(t, log["CT_dq"][:, j],  color=c, lw=1.4, ls="--", alpha=0.8,
                label=f"j{j+1} Cᵀ·q̇")
    ax.axvspan(t0, t1, alpha=0.1, color="red")
    ax.axhline(0, color="k", lw=0.5, ls="--")

    G_rms    = np.sqrt(np.mean(log["G_est"] ** 2))
    CT_rms   = np.sqrt(np.mean(log["CT_dq"] ** 2))
    ax.set_title(
        f"β terms: G(q) (—) vs Cᵀ·q̇ (--)\n"
        f"G RMS={G_rms:.3f} N·m   Cᵀ·q̇ RMS={CT_rms:.3f} N·m "
        f"({100*CT_rms/max(G_rms,1e-9):.1f}% of G)"
    )
    ax.set_ylabel("N·m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"momentum_observer_identified_{robot_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("urdf",       nargs="?", default="robots/sixdof_arm.urdf")
    p.add_argument("--T_id",     type=float, default=30.0,  help="Identification duration [s]")
    p.add_argument("--T_sim",    type=float, default=6.0,   help="Simulation duration [s]")
    p.add_argument("--dt",       type=float, default=1e-3,  help="Simulation timestep [s]")
    p.add_argument("--K_O",      type=float, default=10.0,  help="Observer gain")
    p.add_argument("--Kp",       type=float, default=50.0,  help="PD proportional gain")
    p.add_argument("--Kd",       type=float, default=8.0,   help="PD derivative gain")
    p.add_argument("--noise",    type=float, default=0.0,   help="Identification torque noise [N·m]")
    p.add_argument("--tau_max",  type=float, default=500.0, help="Torque saturation limit [N·m]")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Offline: identification ───────────────────────────────────────────────
    print("=" * 60)
    print(" OFFLINE — Dynamic Parameter Identification")
    print("=" * 60)
    ident = RobotIdentifier(args.urdf, verbose=True)
    pi_b_est, pi_b_true, bp = ident.run(
        T=args.T_id, noise_std=args.noise, plot=False,
    )

    model = ident.model
    data  = ident.data
    nv    = ident.nv

    # ── Online: simulation + observer ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" ONLINE — Momentum Observer Simulation")
    print("=" * 60)
    print(f"  K_O={args.K_O},  Kp={args.Kp},  Kd={args.Kd},  dt={args.dt}")
    print(f"  External torque on joint 1:  +5 N·m from t=2s to t=4s")

    # Hold at neutral — gains are per-joint scaled so no need for a non-trivial pose.
    # The external torque step on joint 1 is the only excitation needed.
    q_ref = pin.neutral(model)[:nv]

    log = simulate(
        model, data, bp, pi_b_est,
        K_O     = args.K_O,
        dt      = args.dt,
        T       = args.T_sim,
        Kp      = args.Kp,
        Kd      = args.Kd,
        q_ref   = q_ref,
        tau_max = args.tau_max,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  Observer residual vs true external torque (contact window 2–4 s):")
    contact = (log["t"] >= 2.0) & (log["t"] < 4.0)
    for label, r_key in [("Full (with Cᵀ·q̇)", "r_full"),
                          ("Approx (no Coriolis)", "r_approx")]:
        err = log[r_key][contact] - log["tau_ext"][contact]
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"    {label:<25}  RMSE = {rmse:.4e} N·m")

    plot(log, model.name)
    print("\nDone.")
