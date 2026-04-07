"""
Dynamic term verification.

Reconstructs individual dynamic terms (M, G, C*dq, p) via sub-regressors
and compares against Pinocchio ground truth.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from maxcomp.core import RobotIdentifier, BaseParams
from maxcomp.regressors import (
    sub_regressor_inertia,
    sub_regressor_gravity,
    sub_regressor_mass_matrix,
    sub_regressor_coriolis,
    ground_truth_mass_matrix,
    ground_truth_inertia_term,
    ground_truth_gravity,
    ground_truth_coriolis_term,
    ground_truth_coriolis_transpose_term,
    ground_truth_momentum,
)
from maxcomp.identification import run_identification


def verify(
    urdf:        str,
    T:           float = 20.0,
    dt:          float = 1e-2,
    omega_f:     float = 0.5,
    n_harmonics: int   = 5,
    amplitude:   float = 0.6,
    noise_std:   float = 0.0,
    id_seed:     int   = 42,
    test_seed:   int   = 7,
):
    # Step 1: run identification
    print("=" * 60)
    print(" STEP 1 -- Parameter Identification")
    print("=" * 60)
    ident = RobotIdentifier(urdf, verbose=True)
    pi_b_est, pi_b_true, bp = run_identification(
        ident,
        T=T, dt=dt, omega_f=omega_f, n_harmonics=n_harmonics,
        amplitude=amplitude, noise_std=noise_std, plot=False,
    )

    model = ident.model
    data  = ident.data
    nv    = ident.nv

    # Step 2: generate a separate verification trajectory
    print("\n" + "=" * 60)
    print(" STEP 2 -- Dynamic Term Verification (held-out trajectory)")
    print("=" * 60)
    t_arr, q_traj, dq_traj, ddq_traj = ident.fourier_trajectory(
        T=T, dt=dt, omega_f=omega_f, n_harmonics=n_harmonics,
        amplitude=amplitude, seed=test_seed,
    )
    N = len(t_arr)
    print(f"  Verification trajectory: {N} points, seed={test_seed}")

    # Step 3: compute all terms along the trajectory
    Mddq_est  = np.zeros((N, nv))
    G_est     = np.zeros((N, nv))
    Cqdq_est  = np.zeros((N, nv))
    p_est     = np.zeros((N, nv))

    Mddq_true  = np.zeros((N, nv))
    G_true     = np.zeros((N, nv))
    Cqdq_true  = np.zeros((N, nv))
    CTqdq_true = np.zeros((N, nv))
    p_true     = np.zeros((N, nv))

    M_diag_est  = np.zeros((N, nv))
    M_diag_true = np.zeros((N, nv))
    M_frob_est  = np.zeros(N)
    M_frob_true = np.zeros(N)
    M_frob_err  = np.zeros(N)

    for i in range(N):
        q   = q_traj[i]
        dq  = dq_traj[i]
        ddq = ddq_traj[i]

        Y_M_b = sub_regressor_inertia(model, data, q, ddq, bp)
        Y_G_b = sub_regressor_gravity(model, data, q, bp)
        Y_C_b = sub_regressor_coriolis(model, data, q, dq, bp)
        Y_p_b = sub_regressor_inertia(model, data, q, dq, bp)

        Mddq_est[i] = Y_M_b @ pi_b_est
        G_est[i]    = Y_G_b @ pi_b_est
        Cqdq_est[i] = Y_C_b @ pi_b_est
        p_est[i]    = Y_p_b @ pi_b_est

        M_est_i  = sub_regressor_mass_matrix(model, data, q, bp, pi_b_est)
        M_true_i = ground_truth_mass_matrix(model, data, q)

        M_diag_est[i]  = np.diag(M_est_i)
        M_diag_true[i] = np.diag(M_true_i)
        M_frob_est[i]  = np.linalg.norm(M_est_i, 'fro')
        M_frob_true[i] = np.linalg.norm(M_true_i, 'fro')
        M_frob_err[i]  = np.linalg.norm(M_est_i - M_true_i, 'fro')

        Mddq_true[i]  = ground_truth_inertia_term(model, data, q, ddq)
        G_true[i]     = ground_truth_gravity(model, data, q)
        Cqdq_true[i]  = ground_truth_coriolis_term(model, data, q, dq)
        CTqdq_true[i] = ground_truth_coriolis_transpose_term(model, data, q, dq)
        p_true[i]     = ground_truth_momentum(model, data, q, dq)

    # Step 4: RMSE summary
    def nrmse(est, true):
        rms_err  = np.sqrt(np.mean((est - true) ** 2))
        rms_true = np.sqrt(np.mean(true ** 2))
        return rms_err, rms_err / max(rms_true, 1e-12)

    print("\n  RMSE summary:")
    print(f"  {'term':<25}  {'RMSE':>12}  {'norm RMSE':>12}")
    print("  " + "-" * 54)
    for label, est, true in [
        ("M(q) diag",     M_diag_est, M_diag_true),
        ("M(q)*ddq",      Mddq_est,  Mddq_true),
        ("G(q)",          G_est,     G_true),
        ("C(q,dq)*dq",    Cqdq_est,  Cqdq_true),
        ("p = M(q)*dq",   p_est,     p_true),
    ]:
        rmse, nrm = nrmse(est, true)
        print(f"  {label:<25}  {rmse:>12.4e}  {nrm:>12.4e}")

    Mddq_rms  = np.sqrt(np.mean(Mddq_true ** 2))
    Cqdq_rms  = np.sqrt(np.mean(Cqdq_true ** 2))
    CTqdq_rms = np.sqrt(np.mean(CTqdq_true ** 2))
    diff_rms  = np.sqrt(np.mean((CTqdq_true - Cqdq_true) ** 2))
    print(f"\n  Coriolis breakdown:")
    print(f"    C*dq   RMS  : {Cqdq_rms:.4e}  ({100*Cqdq_rms/max(Mddq_rms,1e-12):.1f}% of inertia)")
    print(f"    CT*dq  RMS  : {CTqdq_rms:.4e}  (needed for observer beta)")
    print(f"    |CT-C|*dq   : {diff_rms:.4e}  (error from using C instead of CT)")

    # Step 5: plot
    plot_verification(
        t_arr, Mddq_est, Mddq_true,
        G_est, G_true,
        Cqdq_est, Cqdq_true, CTqdq_true,
        p_est, p_true,
        M_diag_est, M_diag_true,
        M_frob_est, M_frob_true, M_frob_err,
        model.name,
    )

    return pi_b_est, bp, {"Cqdq_est": Cqdq_est, "Cqdq_true": Cqdq_true, "CTqdq_true": CTqdq_true}


def plot_verification(
    t, Mddq_est, Mddq_true,
    G_est, G_true,
    Cqdq_est, Cqdq_true, CTqdq_true,
    p_est, p_true,
    M_diag_est, M_diag_true,
    M_frob_est, M_frob_true, M_frob_err,
    robot_name,
):
    nv     = Mddq_true.shape[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(
        f"Dynamic Term Verification -- {robot_name}\n"
        f"Estimated (sub-regressor * pi_b)  vs  Ground Truth (Pinocchio)",
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

    overlay(axes[0, 0], Mddq_true, Mddq_est, "Inertia term  M(q)*ddq",        "N*m")
    overlay(axes[0, 1], G_true,    G_est,    "Gravity term  G(q)",              "N*m")
    overlay(axes[1, 0], p_true,    p_est,    "Generalised momentum  p=M(q)*dq", "kg*m^2/s")

    ax = axes[1, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, Cqdq_true[:, j],  color=c, lw=1.5, alpha=0.6,
                label=f"j{j+1} C*dq true")
        ax.plot(t, Cqdq_est[:, j],   color=c, lw=2.5, ls="--", alpha=1.0,
                zorder=3, label=f"j{j+1} C*dq est")
        ax.plot(t, CTqdq_true[:, j], color=c, lw=1.2, ls=":",  alpha=0.8,
                label=f"j{j+1} CT*dq true")

    Mddq_rms = np.sqrt(np.mean(Mddq_true ** 2))
    Cqdq_rms = np.sqrt(np.mean(Cqdq_true ** 2))
    ax.set_title(
        f"Coriolis  C(q,dq)*dq  est (--) vs true (solid)  |  CT*dq for beta (:)\n"
        f"C*dq RMS = {Cqdq_rms:.3e} N*m  "
        f"({100 * Cqdq_rms / max(Mddq_rms, 1e-12):.1f}% of inertia RMS)"
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("N*m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    overlay(axes[2, 0], M_diag_true, M_diag_est,
            "Mass matrix diagonal  M(q)_{ii}", "kg*m^2")

    ax = axes[2, 1]
    ax.plot(t, M_frob_true, color=colors[0], lw=1.5, alpha=0.6,
            label="||M(q)||_F  true")
    ax.plot(t, M_frob_est,  color=colors[0], lw=2.5, ls="--", alpha=1.0,
            zorder=3, label="||M(q)||_F  est")
    ax.plot(t, M_frob_err,  color=colors[1], lw=1.5,
            label="||M_est - M_true||_F")
    mean_rel = np.mean(M_frob_err / np.maximum(M_frob_true, 1e-12))
    ax.set_title(
        f"Mass matrix Frobenius norm\n"
        f"mean relative error = {mean_rel:.4e}"
    )
    ax.set_ylabel("kg*m^2"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = f"output/dynamic_terms_{robot_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.show()
