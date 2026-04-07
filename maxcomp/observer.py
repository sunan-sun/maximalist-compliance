"""
Data-driven momentum observer.

Uses identified base parameters to run a momentum-based disturbance observer.
Two observers run in parallel: full (with C^T*dq) and approximate (Coriolis ignored).
"""

import os
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from maxcomp.core import BaseParams
from maxcomp.regressors import sub_regressor_inertia, sub_regressor_gravity


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
    tau_max:   float = None,
    tau_ext_mag: float = None,
):
    """
    Simulate the robot plant + two momentum observers in parallel.

    External torque schedule (joint 0):
        0  -> 2 s :  no contact
        2  -> 4 s :  step external torque tau_ext[0] = tau_ext_mag
        4  -> T   :  released
    """
    nv = model.nv
    nq = model.nq

    if q_ref is None:
        q_ref = np.zeros(nv)

    M0     = pin.crba(model, data, pin.neutral(model))
    m_diag = np.diag(M0)
    m_ref  = m_diag.mean()
    Kp_vec = Kp * (m_diag / m_ref)
    Kd_vec = Kd * (m_diag / m_ref)
    print(f"  Inertia diagonal (neutral):  min={m_diag.min():.4f}  max={m_diag.max():.4f}  mean={m_ref:.4f} kg*m^2")
    print(f"  Per-joint Kp range: [{Kp_vec.min():.2f}, {Kp_vec.max():.2f}]  "
          f"Kd range: [{Kd_vec.min():.3f}, {Kd_vec.max():.3f}]")

    q_test = pin.neutral(model)
    q_test[:nv] = 0.5
    G_scale = np.max(np.abs(pin.computeGeneralizedGravity(model, data, q_test)))
    G_scale = max(G_scale, m_ref * 9.81 * 0.05)

    if tau_max is None:
        tau_max = max(50.0 * G_scale, 1.0)
    if tau_ext_mag is None:
        tau_ext_mag = max(2.0 * G_scale, 0.01)
    print(f"  tau_ext disturbance: {tau_ext_mag:.4f} N*m  (auto-scaled)")
    print(f"  tau_max saturation:  {tau_max:.2f} N*m")

    q  = pin.neutral(model)
    dq = np.zeros(nv)

    p_obs_full   = np.zeros(nv)
    p_obs_approx = np.zeros(nv)

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
        "p":          np.zeros((N, nv)),
        "p_est":      np.zeros((N, nv)),
        "G_est":      np.zeros((N, nv)),
        "CT_dq":      np.zeros((N, nv)),
    }

    for i, t in enumerate(t_arr):
        tau_ext = np.zeros(nv)
        if 2.0 <= t < 4.0:
            tau_ext[0] = tau_ext_mag

        q_r    = q if nq == nv else q[:nv]

        Y_G_b  = sub_regressor_gravity(model, data, q_r, bp)
        G_est  = Y_G_b @ pi_b_est

        Y_p_b  = sub_regressor_inertia(model, data, q_r, dq, bp)
        p_est  = Y_p_b @ pi_b_est

        C_mat  = pin.computeCoriolisMatrix(model, data, q_r, dq)
        CT_dq  = C_mat.T @ dq

        M_true = pin.crba(model, data, q_r)
        p_true = M_true @ dq

        tau  = Kp_vec * (q_ref - q_r) - Kd_vec * dq + G_est
        tau  = np.clip(tau, -tau_max, tau_max)

        beta_full   = CT_dq - G_est
        beta_approx =       - G_est

        r_full   = K_O * (p_est - p_obs_full)
        r_approx = K_O * (p_est - p_obs_approx)

        p_obs_full   += (tau + beta_full   + r_full)   * dt
        p_obs_approx += (tau + beta_approx + r_approx) * dt

        ddq = pin.aba(model, data, q, dq, tau + tau_ext)
        if not np.isfinite(ddq).all():
            print(f"\n  !!  Simulation diverged at t={t:.3f} s -- stopping early.")
            for k in log:
                if hasattr(log[k], '__len__') and k != "t":
                    log[k] = log[k][:i]
            log["t"] = log["t"][:i]
            break
        q  = pin.integrate(model, q, dq * dt)
        dq = dq + ddq * dt

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


def plot_observer(log, robot_name, contact_window=(2.0, 4.0)):
    t  = log["t"]
    nv = log["q"].shape[1]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    t0, t1 = contact_window

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"Data-Driven Momentum Observer -- {robot_name}\n"
        f"Full observer (with CT*dq)  vs  Approx (Coriolis ignored)  |  "
        f"Shaded region = contact",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0, 0]
    for j in range(nv):
        ax.plot(t, log["q"][:, j], color=colors[j % len(colors)],
                lw=1.3, label=f"q{j+1}")
    ax.axvspan(t0, t1, alpha=0.1, color="red", label="contact")
    ax.set_title("Joint positions  q(t)")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["p"][:, j],     color=c, lw=1.5, alpha=0.6,
                label=f"j{j+1} true")
        ax.plot(t, log["p_est"][:, j], color=c, lw=2.0, ls="--",
                zorder=3, label=f"j{j+1} est")
    ax.axvspan(t0, t1, alpha=0.1, color="red")
    ax.set_title("Generalised momentum  p = M(q)*dq\ntrue (solid) vs identified (--)")
    ax.set_ylabel("kg*m^2/s"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["tau_ext"][:, j], color=c, lw=2.0, alpha=0.5,
                label=f"j{j+1} tau_ext true")
        ax.plot(t, log["r_full"][:, j],  color=c, lw=1.8, ls="--",
                zorder=3, label=f"j{j+1} r full")
        ax.plot(t, log["r_approx"][:, j], color=c, lw=5, ls=":",
                zorder=2, label=f"j{j+1} r approx")
    ax.axvspan(t0, t1, alpha=0.1, color="red")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title("Observer residual  r  vs true  tau_ext\n"
                 "true (solid)  full (--) with CT*dq   approx (:) without Coriolis")
    ax.set_ylabel("N*m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["G_est"][:, j],  color=c, lw=1.4,
                label=f"j{j+1} G(q)")
        ax.plot(t, log["CT_dq"][:, j],  color=c, lw=1.4, ls="--", alpha=0.8,
                label=f"j{j+1} CT*dq")
    ax.axvspan(t0, t1, alpha=0.1, color="red")
    ax.axhline(0, color="k", lw=0.5, ls="--")

    G_rms    = np.sqrt(np.mean(log["G_est"] ** 2))
    CT_rms   = np.sqrt(np.mean(log["CT_dq"] ** 2))
    ax.set_title(
        f"beta terms: G(q) (solid) vs CT*dq (--)\n"
        f"G RMS={G_rms:.3f} N*m   CT*dq RMS={CT_rms:.3f} N*m "
        f"({100*CT_rms/max(G_rms,1e-9):.1f}% of G)"
    )
    ax.set_ylabel("N*m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = f"output/momentum_observer_identified_{robot_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.show()
