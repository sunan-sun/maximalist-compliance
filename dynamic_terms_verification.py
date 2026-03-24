"""
Dynamic Term Verification
=========================
Uses the identified base parameters Φ*_0 (from robot_identification.py)
to reconstruct individual dynamic terms via sub-regressors, then compares
against Pinocchio's ground-truth computations.

Terms reconstructed:
  M(q)·q̈    ←  Y_M(q, 0, q̈)|_{g=0}  · Φ*_0
  G(q)       ←  Y_G(q, 0,  0)|_{g≠0}  · Φ*_0
  C(q,q̇)·q̇ ←  Y_C(q, q̇, 0)|_{g=0}  · Φ*_0   ← NEW
  p(q,q̇)    ←  Y_M(q, 0, q̇)|_{g=0}  · Φ*_0   (momentum: substitute q̈→q̇)

Note on Coriolis:
  The sub-regressor gives  C(q,q̇)·q̇  (regressor evaluated at ddq=0, g=0).
  The momentum observer β term needs  Cᵀ(q,q̇)·q̇  =  Ṁ(q)·q̇ − C(q,q̇)·q̇.
  Cᵀ·q̇ is computed here via pin.computeCoriolisMatrix (ground truth only).

Ground truth (Pinocchio):
  M(q)·q̈    ←  pin.crba(q) @ q̈
  G(q)       ←  pin.computeGeneralizedGravity(q)
  C(q,q̇)·q̇ ←  RNEA(q, q̇, 0) − G(q)
  Cᵀ(q,q̇)·q̇←  pin.computeCoriolisMatrix(q, q̇).T @ q̇
  p(q,q̇)    ←  pin.crba(q) @ q̇
"""

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import argparse

from robot_identification import RobotIdentifier, BaseParams


# ─────────────────────────────────────────────────────────────────────────────
# Sub-regressor helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_cols(Y_full: np.ndarray, bp: BaseParams) -> np.ndarray:
    """Select base columns from full regressor:  Y_b = Y_full[:, P[:r]]"""
    return Y_full[:, bp.P[:bp.r]]


def sub_regressor_inertia(
    model, data, q: np.ndarray, ddq: np.ndarray, bp: BaseParams
) -> np.ndarray:
    """
    Y_M_b(q, 0, q̈)  with gravity zeroed.
    Y_M_b · Φ*_0  =  M(q)·q̈
    """
    g_saved = model.gravity.linear.copy()
    model.gravity.linear[:] = 0.0

    Y_M = pin.computeJointTorqueRegressor(
        model, data, q, np.zeros(model.nv), ddq
    )
    Y_M_b = _base_cols(Y_M, bp)

    model.gravity.linear[:] = g_saved
    return Y_M_b


def sub_regressor_gravity(
    model, data, q: np.ndarray, bp: BaseParams
) -> np.ndarray:
    """
    Y_G_b(q, 0, 0)  with gravity active.
    Y_G_b · Φ*_0  =  G(q)
    """
    Y_G = pin.computeJointTorqueRegressor(
        model, data, q, np.zeros(model.nv), np.zeros(model.nv)
    )
    return _base_cols(Y_G, bp)


def sub_regressor_coriolis(
    model, data, q: np.ndarray, dq: np.ndarray, bp: BaseParams
) -> np.ndarray:
    """
    Y_C_b(q, q̇, 0)  with gravity zeroed.
    Y_C_b · Φ*_0  =  C(q,q̇)·q̇

    Derivation: Y(q, dq, ddq)·π = M·ddq + C·dq + G
      → set ddq=0, g=0  →  Y(q, dq, 0)|_{g=0}·π = C(q,dq)·dq  ✓

    Note: this gives  C·q̇,  NOT  Cᵀ·q̇.
    The momentum observer β needs Cᵀ·q̇ = Ṁ·q̇ − C·q̇,
    which is not separable from the standard regressor — use
    ground_truth_coriolis_transpose_term() for that (requires model access).
    """
    g_saved = model.gravity.linear.copy()
    model.gravity.linear[:] = 0.0

    Y_C = pin.computeJointTorqueRegressor(
        model, data, q, dq, np.zeros(model.nv)
    )
    Y_C_b = _base_cols(Y_C, bp)

    model.gravity.linear[:] = g_saved
    return Y_C_b


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth via Pinocchio
# ─────────────────────────────────────────────────────────────────────────────

def ground_truth_inertia_term(model, data, q, ddq):
    """M(q)·q̈  via Composite Rigid Body Algorithm."""
    M = pin.crba(model, data, q)
    return M @ ddq


def ground_truth_gravity(model, data, q):
    """G(q)  via Pinocchio's generalised gravity."""
    return pin.computeGeneralizedGravity(model, data, q)


def ground_truth_coriolis_term(model, data, q, dq):
    """
    C(q,q̇)·q̇  =  RNEA(q, q̇, 0) − G(q)
    """
    tau_noa = pin.rnea(model, data, q, dq, np.zeros(model.nv))
    G       = pin.computeGeneralizedGravity(model, data, q)
    return tau_noa - G


def ground_truth_coriolis_transpose_term(model, data, q, dq):
    """
    Cᵀ(q,q̇)·q̇  via pin.computeCoriolisMatrix.

    This is the term that appears in the momentum observer β:
        β = Cᵀ·q̇ − G(q)

    Why Cᵀ ≠ C:  the Christoffel-symbol Coriolis matrix satisfies
        Ṁ = C + Cᵀ   (Ṁ symmetric, C generally not)
    so  Cᵀ·q̇ = Ṁ·q̇ − C·q̇,  which cannot be obtained by zeroing
    regressor inputs — it requires differentiating M w.r.t. q.
    """
    C = pin.computeCoriolisMatrix(model, data, q, dq)
    return C.T @ dq


def ground_truth_momentum(model, data, q, dq):
    """p = M(q)·q̇"""
    M = pin.crba(model, data, q)
    return M @ dq


# ─────────────────────────────────────────────────────────────────────────────
# Main verification
# ─────────────────────────────────────────────────────────────────────────────

def verify(
    urdf:        str,
    T:           float = 20.0,
    dt:          float = 1e-2,
    omega_f:     float = 0.5,
    n_harmonics: int   = 5,
    amplitude:   float = 0.6,
    noise_std:   float = 0.0,
    id_seed:     int   = 42,    # seed used during identification
    test_seed:   int   = 7,     # different seed for verification trajectory
):
    # ── Step 1: run identification ────────────────────────────────────────────
    print("=" * 60)
    print(" STEP 1 — Parameter Identification")
    print("=" * 60)
    ident = RobotIdentifier(urdf, verbose=True)
    pi_b_est, pi_b_true, bp = ident.run(
        T=T, dt=dt, omega_f=omega_f, n_harmonics=n_harmonics,
        amplitude=amplitude, noise_std=noise_std, plot=False,
    )

    model = ident.model
    data  = ident.data
    nv    = ident.nv

    # ── Step 2: generate a separate verification trajectory ──────────────────
    print("\n" + "=" * 60)
    print(" STEP 2 — Dynamic Term Verification (held-out trajectory)")
    print("=" * 60)
    t_arr, q_traj, dq_traj, ddq_traj = ident.fourier_trajectory(
        T=T, dt=dt, omega_f=omega_f, n_harmonics=n_harmonics,
        amplitude=amplitude, seed=test_seed,
    )
    N = len(t_arr)
    print(f"  Verification trajectory: {N} points, seed={test_seed}")

    # ── Step 3: compute all terms along the trajectory ───────────────────────
    Mddq_est  = np.zeros((N, nv))   # estimated  M(q)·q̈
    G_est     = np.zeros((N, nv))   # estimated  G(q)
    Cqdq_est  = np.zeros((N, nv))   # estimated  C(q,q̇)·q̇
    p_est     = np.zeros((N, nv))   # estimated  p = M(q)·q̇

    Mddq_true  = np.zeros((N, nv))  # ground truth  M(q)·q̈
    G_true     = np.zeros((N, nv))  # ground truth  G(q)
    Cqdq_true  = np.zeros((N, nv))  # ground truth  C(q,q̇)·q̇
    CTqdq_true = np.zeros((N, nv))  # ground truth  Cᵀ(q,q̇)·q̇  (for observer β)
    p_true     = np.zeros((N, nv))  # ground truth  p = M(q)·q̇

    for i in range(N):
        q   = q_traj[i]
        dq  = dq_traj[i]
        ddq = ddq_traj[i]

        # ── Estimated (via sub-regressors + Φ*_0) ──
        Y_M_b = sub_regressor_inertia(model, data, q, ddq, bp)
        Y_G_b = sub_regressor_gravity(model, data, q, bp)
        Y_C_b = sub_regressor_coriolis(model, data, q, dq, bp)
        Y_p_b = sub_regressor_inertia(model, data, q, dq, bp)   # ddq → dq

        Mddq_est[i] = Y_M_b @ pi_b_est
        G_est[i]    = Y_G_b @ pi_b_est
        Cqdq_est[i] = Y_C_b @ pi_b_est
        p_est[i]    = Y_p_b @ pi_b_est

        # ── Ground truth (Pinocchio) ──
        Mddq_true[i]  = ground_truth_inertia_term(model, data, q, ddq)
        G_true[i]     = ground_truth_gravity(model, data, q)
        Cqdq_true[i]  = ground_truth_coriolis_term(model, data, q, dq)
        CTqdq_true[i] = ground_truth_coriolis_transpose_term(model, data, q, dq)
        p_true[i]     = ground_truth_momentum(model, data, q, dq)

    # ── Step 4: RMSE summary ─────────────────────────────────────────────────
    def nrmse(est, true):
        rms_err  = np.sqrt(np.mean((est - true) ** 2))
        rms_true = np.sqrt(np.mean(true ** 2))
        return rms_err, rms_err / max(rms_true, 1e-12)

    print("\n  RMSE summary:")
    print(f"  {'term':<25}  {'RMSE':>12}  {'norm RMSE':>12}")
    print("  " + "-" * 54)
    for label, est, true in [
        ("M(q)·q̈",      Mddq_est,  Mddq_true),
        ("G(q)",          G_est,     G_true),
        ("C(q,q̇)·q̇",   Cqdq_est,  Cqdq_true),
        ("p = M(q)·q̇",  p_est,     p_true),
    ]:
        rmse, nrm = nrmse(est, true)
        print(f"  {label:<25}  {rmse:>12.4e}  {nrm:>12.4e}")

    # Cᵀ·q̇ vs C·q̇ difference (shows how much the transpose matters)
    Mddq_rms  = np.sqrt(np.mean(Mddq_true ** 2))
    Cqdq_rms  = np.sqrt(np.mean(Cqdq_true ** 2))
    CTqdq_rms = np.sqrt(np.mean(CTqdq_true ** 2))
    diff_rms  = np.sqrt(np.mean((CTqdq_true - Cqdq_true) ** 2))
    print(f"\n  Coriolis breakdown:")
    print(f"    C·q̇   RMS  : {Cqdq_rms:.4e}  ({100*Cqdq_rms/max(Mddq_rms,1e-12):.1f}% of inertia)")
    print(f"    Cᵀ·q̇  RMS  : {CTqdq_rms:.4e}  (needed for observer β)")
    print(f"    |Cᵀ−C|·q̇   : {diff_rms:.4e}  (error from using C instead of Cᵀ)")

    # ── Step 5: plot ─────────────────────────────────────────────────────────
    _plot(t_arr, Mddq_est, Mddq_true,
          G_est, G_true,
          Cqdq_est, Cqdq_true, CTqdq_true,
          p_est, p_true,
          model.name)

    return pi_b_est, bp, {"Cqdq_est": Cqdq_est, "Cqdq_true": Cqdq_true, "CTqdq_true": CTqdq_true}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(t, Mddq_est, Mddq_true,
          G_est, G_true,
          Cqdq_est, Cqdq_true, CTqdq_true,
          p_est, p_true,
          robot_name):
    nv     = Mddq_true.shape[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"Dynamic Term Verification — {robot_name}\n"
        f"Estimated (sub-regressor · Φ*_0)  vs  Ground Truth (Pinocchio)",
        fontsize=12, fontweight="bold",
    )

    def overlay(ax, true, est, title, ylabel):
        for j in range(nv):
            c = colors[j % len(colors)]
            ax.plot(t, true[:, j], color=c, lw=1.5, alpha=0.6,
                    label=f"j{j+1} true")
            ax.plot(t, est[:, j],  color=c, lw=2.5, ls="--", alpha=1.0,
                    zorder=3, label=f"j{j+1} est")
        ax.set_title(title)
        ax.set_ylabel(ylabel); ax.set_xlabel("t [s]")
        ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    overlay(axes[0, 0], Mddq_true, Mddq_est, "Inertia term  M(q)·q̈",         "N·m")
    overlay(axes[0, 1], G_true,    G_est,    "Gravity term  G(q)",              "N·m")
    overlay(axes[1, 0], p_true,    p_est,    "Generalised momentum  p=M(q)·q̇", "kg·m²/s")

    # ── Bottom-right: Coriolis  C·q̇  (estimated vs true) + Cᵀ·q̇ ─────────────
    ax = axes[1, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, Cqdq_true[:, j],  color=c, lw=1.5, alpha=0.6,
                label=f"j{j+1} C·q̇ true")
        ax.plot(t, Cqdq_est[:, j],   color=c, lw=2.5, ls="--", alpha=1.0,
                zorder=3, label=f"j{j+1} C·q̇ est")
        ax.plot(t, CTqdq_true[:, j], color=c, lw=1.2, ls=":",  alpha=0.8,
                label=f"j{j+1} Cᵀ·q̇ true")

    Mddq_rms = np.sqrt(np.mean(Mddq_true ** 2))
    Cqdq_rms = np.sqrt(np.mean(Cqdq_true ** 2))
    ax.set_title(
        f"Coriolis  C(q,q̇)·q̇  est (--) vs true (—)  |  Cᵀ·q̇ for β (·)\n"
        f"C·q̇ RMS = {Cqdq_rms:.3e} N·m  "
        f"({100 * Cqdq_rms / max(Mddq_rms, 1e-12):.1f}% of inertia RMS)"
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("N·m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"dynamic_terms_{robot_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("urdf", nargs="?", default="robots/sixdof_arm.urdf")
    p.add_argument("--T",         type=float, default=30.0)
    p.add_argument("--dt",        type=float, default=1e-2)
    p.add_argument("--omega",     type=float, default=0.5)
    p.add_argument("--harmonics", type=int,   default=5)
    p.add_argument("--amplitude", type=float, default=0.8)
    p.add_argument("--noise",     type=float, default=0.0)
    args = p.parse_args()

    verify(
        urdf        = args.urdf,
        T           = args.T,
        dt          = args.dt,
        omega_f     = args.omega,
        n_harmonics = args.harmonics,
        amplitude   = args.amplitude,
        noise_std   = args.noise,
    )
