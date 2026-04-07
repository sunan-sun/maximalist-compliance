"""
Identification pipeline runner and plotting.

Orchestrates the full identification flow: base parameter discovery,
trajectory generation, regressor stacking, torque simulation, and
least-squares estimation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from maxcomp.core import RobotIdentifier, BaseParams


def print_results(pi_b_true, pi_b_est, bp: BaseParams):
    abs_err = np.abs(pi_b_est - pi_b_true)
    scale   = np.abs(pi_b_true)
    threshold = scale.max() * 1e-3

    print(f"\n  Results  (r = {bp.r} base parameters):")
    print(f"  {'#':>4}  {'true':>14}  {'estimated':>14}  {'abs err':>10}  {'rel err':>10}  note")
    print("  " + "-" * 80)
    for i in range(bp.r):
        t_val   = pi_b_true[i]
        e_val   = pi_b_est[i]
        abs_e   = abs_err[i]
        rel_e   = abs_e / max(abs(t_val), 1e-12)
        note    = ""
        if abs(t_val) < threshold:
            note = "near-zero (rel err unreliable)"
        elif rel_e > 0.05:
            note = "!! large"
        print(f"  {i:>4}  {t_val:>14.6f}  {e_val:>14.6f}  {abs_e:>10.2e}  {rel_e:>10.2e}  {note}")


def plot_identification(t, q, tau, tau_rec, pi_b_true, pi_b_est, bp: BaseParams,
                        model_name: str, nv: int, n_std: int):
    N = len(t)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Base-Parameter Identification -- {model_name}\n"
        f"r = {bp.r} base params  (from {n_std} standard)",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    for j in range(nv):
        ax.plot(t, q[:, j], label=f"q{j+1}")
    ax.set_title("Excitation Trajectory  q(t)")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    tau_arr     = tau.reshape(N, nv)
    tau_rec_arr = tau_rec.reshape(N, nv)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j in range(nv):
        ax.plot(t, tau_arr[:, j],     color=colors[j],
                label=f"tau{j+1} RNEA", alpha=0.8)
        ax.plot(t, tau_rec_arr[:, j], color=colors[j],
                ls="--", lw=1.5, label=f"tau{j+1} Y_b*pi_b", alpha=0.9)
    ax.set_title("Torques: RNEA  vs  Y_b*pi_b")
    ax.set_ylabel("N*m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    idx = np.arange(bp.r)
    ax.bar(idx - 0.2, pi_b_true, width=0.4, label="pi_b  true", alpha=0.75)
    ax.bar(idx + 0.2, pi_b_est,  width=0.4, label="pi_b  est",  alpha=0.75)
    ax.set_title(f"Base Parameters  pi_b  (r = {bp.r})")
    ax.set_xlabel("base param index"); ax.set_ylabel("value")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    abs_err  = np.abs(pi_b_est - pi_b_true)
    scale    = np.abs(pi_b_true)
    threshold = scale.max() * 1e-3
    colors_bar = [
        "steelblue" if scale[i] >= threshold else "lightgray"
        for i in idx
    ]
    ax.bar(idx, abs_err, color=colors_bar, alpha=0.85)
    ax.set_title("Absolute Error  |pi_b_est - pi_b_true|\n(gray = near-zero true value)")
    ax.set_xlabel("base param index"); ax.set_ylabel("absolute error")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = f"output/identification_{model_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.show()


def run_identification(
    ident: RobotIdentifier,
    T:            float = 30.0,
    dt:           float = 1e-2,
    omega_f:      float = 0.5,
    n_harmonics:  int   = 5,
    amplitude:    float = 0.8,
    noise_std:    float = 0.0,
    n_qr_samples: int   = 1000,
    plot:         bool  = True,
    optimize_traj: bool = False,
    n_restarts:   int   = 3,
    max_iter:     int   = 200,
) -> tuple[np.ndarray, np.ndarray, BaseParams]:
    """
    Full identification pipeline.

    Returns (pi_b_est, pi_b_true, base_params)
    """
    print("\n[1/5] Finding base parameter structure ...")
    bp = ident.find_base_params(n_samples=n_qr_samples)

    print(f"\n[2/5] Generating excitation trajectory  (T={T}s, dt={dt}s) ...")
    if optimize_traj:
        from maxcomp.excitation import ExcitationTrajectoryOptimizer
        opt = ExcitationTrajectoryOptimizer(
            ident, bp, T=T, dt=dt, omega_f=omega_f,
            n_harmonics=n_harmonics,
        )
        result = opt.optimize(
            amplitude=amplitude, n_restarts=n_restarts,
            max_iter=max_iter,
        )
        t, q, dq, ddq = result.trajectory
    else:
        t, q, dq, ddq = ident.fourier_trajectory(
            T=T, dt=dt, omega_f=omega_f,
            n_harmonics=n_harmonics, amplitude=amplitude,
        )

    print("[3/5] Computing stacked base regressor  Y_b ...")
    Y_b = ident.compute_base_regressor(q, dq, ddq, bp)
    cond = np.linalg.cond(Y_b)
    print(f"      Y_b shape : {Y_b.shape}")
    print(f"      cond(Y_b) : {cond:.3e}", end="")
    if cond > 1e8:
        print("  !!  high condition -- try longer T or more harmonics", end="")
    print()

    print("[4/5] Simulating torques via RNEA ...")
    tau = ident.simulate_torques(q, dq, ddq, noise_std=noise_std)

    print("[5/5] Estimating pi_b via least squares ...")
    pi_b_est, _, rank = ident.estimate(Y_b, tau)
    tau_rec = Y_b @ pi_b_est
    rmse    = np.sqrt(np.mean((tau_rec - tau) ** 2))
    print(f"      rank(Y_b) : {rank} / {bp.r}")
    print(f"      RMSE      : {rmse:.4e} N*m")

    pi_b_true = ident.true_base_params(bp)

    if noise_std == 0.0:
        tau_check = Y_b @ pi_b_true
        val_rmse  = np.sqrt(np.mean((tau_check - tau) ** 2))
        print(f"      Validation (Y_b*pi_b_true vs RNEA) RMSE : {val_rmse:.2e}  (-> 0)")

    print("\n      Validation on held-out trajectory (seed=99) ...")
    t_val, q_val, dq_val, ddq_val = ident.fourier_trajectory(
        T=T, dt=dt, omega_f=omega_f,
        n_harmonics=n_harmonics, amplitude=amplitude, seed=99,
    )
    Y_b_val  = ident.compute_base_regressor(q_val, dq_val, ddq_val, bp)
    tau_val  = ident.simulate_torques(q_val, dq_val, ddq_val, noise_std=noise_std)
    tau_pred = Y_b_val @ pi_b_est
    val_rmse = np.sqrt(np.mean((tau_pred - tau_val) ** 2))
    tau_norm = np.sqrt(np.mean(tau_val ** 2))
    print(f"      Validation RMSE          : {val_rmse:.4e} N*m")
    print(f"      Normalised (RMSE/||tau||_rms) : {val_rmse/tau_norm:.4e}  (-> 0 if good)")

    print_results(pi_b_true, pi_b_est, bp)

    if plot:
        plot_identification(t, q, tau, tau_rec, pi_b_true, pi_b_est, bp,
                            ident.model.name, ident.nv, ident.n_std)

    return pi_b_est, pi_b_true, bp
