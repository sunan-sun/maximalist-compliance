"""
Computed torque control via identified regressor.

Uses the identified base parameters pi_b to compute inverse dynamics
torques that cancel the robot's nonlinear dynamics, then applies
linear feedback to track a desired trajectory.
"""

import os
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from maxcomp.core import BaseParams


def figure_eight_trajectory(t, nv, omega=0.8, amplitude=0.4):
    """
    Smooth figure-eight-like trajectory with analytic derivatives.
    """
    q_d   = np.zeros(nv)
    dq_d  = np.zeros(nv)
    ddq_d = np.zeros(nv)

    for j in range(nv):
        phase = j * np.pi / nv
        amp = amplitude * (1.0 - 0.2 * j / max(nv - 1, 1))

        q_d[j]   =  amp * np.sin(omega * t + phase)
        dq_d[j]  =  amp * omega * np.cos(omega * t + phase)
        ddq_d[j] = -amp * omega**2 * np.sin(omega * t + phase)

    return q_d, dq_d, ddq_d


def regressor_inverse_dynamics(pin_model, pin_data, q, dq, ddq_cmd, bp, pi_b):
    """
    tau = Y_b(q, dq, ddq_cmd) * pi_b
    """
    Y_full = pin.computeJointTorqueRegressor(pin_model, pin_data, q, dq, ddq_cmd)
    Y_b = Y_full[:, bp.P[:bp.r]]
    return Y_b @ pi_b


def simulate_pinocchio(pin_model, pin_data, bp, pi_b_est, pi_b_true,
                       Kp, Kd, T=8.0, dt=1e-3):
    """
    Simulate computed torque control using Pinocchio's ABA for plant dynamics.
    Runs CTC (identified params) and PD+G (baseline) in parallel.
    """
    nv = pin_model.nv

    M0 = pin.crba(pin_model, pin_data, pin.neutral(pin_model))
    m_diag = np.diag(M0)
    m_ref = m_diag.mean()
    Kp_vec = Kp * (m_diag / m_ref)
    Kd_vec = Kd * (m_diag / m_ref)

    t_arr = np.arange(0.0, T, dt)
    N = len(t_arr)

    log = {
        "t": t_arr,
        "q_d":     np.zeros((N, nv)),
        "q_ctc":   np.zeros((N, nv)),
        "q_pd":    np.zeros((N, nv)),
        "e_ctc":   np.zeros((N, nv)),
        "e_pd":    np.zeros((N, nv)),
        "tau_ctc": np.zeros((N, nv)),
        "tau_pd":  np.zeros((N, nv)),
    }

    q_ctc  = pin.neutral(pin_model).copy()
    dq_ctc = np.zeros(nv)
    q_pd   = pin.neutral(pin_model).copy()
    dq_pd  = np.zeros(nv)

    for i, t in enumerate(t_arr):
        q_d, dq_d, ddq_d = figure_eight_trajectory(t, nv)

        q_ctc_r = q_ctc[:nv] if pin_model.nq > nv else q_ctc
        q_pd_r  = q_pd[:nv]  if pin_model.nq > nv else q_pd

        e_ctc    = q_d - q_ctc_r
        de_ctc   = dq_d - dq_ctc
        ddq_cmd  = ddq_d + Kd * de_ctc + Kp * e_ctc
        tau_ctc  = regressor_inverse_dynamics(
            pin_model, pin_data, q_ctc_r, dq_ctc, ddq_cmd, bp, pi_b_est
        )

        G_pd    = pin.computeGeneralizedGravity(pin_model, pin_data, q_pd_r)
        tau_pd  = Kp_vec * (q_d - q_pd_r) - Kd_vec * (dq_pd - dq_d) + G_pd
        tau_pd  = np.clip(tau_pd, -10.0, 10.0)

        ddq_ctc_actual = pin.aba(pin_model, pin_data, q_ctc, dq_ctc, tau_ctc)
        ddq_pd_actual  = pin.aba(pin_model, pin_data, q_pd,  dq_pd,  tau_pd)

        log["q_d"][i]     = q_d
        log["q_ctc"][i]   = q_ctc_r
        log["q_pd"][i]    = q_pd_r
        log["e_ctc"][i]   = e_ctc
        log["e_pd"][i]    = q_d - q_pd_r
        log["tau_ctc"][i] = tau_ctc
        log["tau_pd"][i]  = tau_pd

        q_ctc  = pin.integrate(pin_model, q_ctc, dq_ctc * dt)
        dq_ctc = dq_ctc + ddq_ctc_actual * dt
        q_pd   = pin.integrate(pin_model, q_pd, dq_pd * dt)
        dq_pd  = dq_pd + ddq_pd_actual * dt

    return log


def simulate_mujoco(pin_model, pin_data, bp, pi_b_est,
                    Kp, Kd, T=8.0, gui=True):
    """
    Run computed torque control on the MuJoCo plant with optional GUI.
    """
    import mujoco
    import mujoco.viewer

    mj_model = mujoco.MjModel.from_xml_path(
        "robots/my-robot/scene_single_finger_torque.xml"
    )
    mj_data = mujoco.MjData(mj_model)
    nv = pin_model.nv
    dt = mj_model.opt.timestep

    log_t, log_q, log_qd, log_e, log_tau = [], [], [], [], []

    def controller(mj_m, mj_d):
        t = mj_d.time
        q  = mj_d.qpos[:nv].copy()
        dq = mj_d.qvel[:nv].copy()

        q_d, dq_d, ddq_d = figure_eight_trajectory(t, nv)

        e     = q_d - q
        de    = dq_d - dq
        ddq_cmd = ddq_d + Kd * de + Kp * e

        tau = regressor_inverse_dynamics(pin_model, pin_data, q, dq, ddq_cmd, bp, pi_b_est)
        mj_d.ctrl[:nv] = tau

        log_t.append(t)
        log_q.append(q.copy())
        log_qd.append(q_d.copy())
        log_e.append(e.copy())
        log_tau.append(tau.copy())

    if gui:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            print("MuJoCo viewer launched. Close window to stop.")
            while viewer.is_running() and mj_data.time < T:
                controller(mj_model, mj_data)
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
    else:
        N = int(T / dt)
        for _ in range(N):
            controller(mj_model, mj_data)
            mujoco.mj_step(mj_model, mj_data)

    log = {
        "t":     np.array(log_t),
        "q":     np.array(log_q),
        "q_d":   np.array(log_qd),
        "e":     np.array(log_e),
        "tau":   np.array(log_tau),
    }
    return log


def plot_comparison(log, robot_name):
    """Plot CTC vs PD+G tracking performance (Pinocchio sim)."""
    t  = log["t"]
    nv = log["q_d"].shape[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"Computed Torque Control vs PD+G -- {robot_name}\n"
        f"CTC uses identified regressor * pi_b  |  PD+G uses per-joint gains + gravity comp",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0, 0]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["q_d"][:, j],   color=c, lw=1.2, alpha=0.5, label=f"j{j+1} desired")
        ax.plot(t, log["q_ctc"][:, j], color=c, lw=2.0, ls="--", label=f"j{j+1} CTC")
    ax.set_title("CTC: joint tracking")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["q_d"][:, j],  color=c, lw=1.2, alpha=0.5, label=f"j{j+1} desired")
        ax.plot(t, log["q_pd"][:, j], color=c, lw=2.0, ls="--", label=f"j{j+1} PD+G")
    ax.set_title("PD+G: joint tracking")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["e_ctc"][:, j], color=c, lw=1.5, label=f"j{j+1} CTC")
        ax.plot(t, log["e_pd"][:, j],  color=c, lw=1.5, ls=":", alpha=0.6, label=f"j{j+1} PD+G")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(
        f"Tracking error  e = q_d - q\n"
        f"CTC RMS: {np.sqrt(np.mean(log['e_ctc']**2)):.4e} rad  |  "
        f"PD+G RMS: {np.sqrt(np.mean(log['e_pd']**2)):.4e} rad"
    )
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["tau_ctc"][:, j], color=c, lw=1.5, label=f"j{j+1} CTC")
        ax.plot(t, log["tau_pd"][:, j],  color=c, lw=1.5, ls=":", alpha=0.6, label=f"j{j+1} PD+G")
    ax.set_title("Control torques  tau")
    ax.set_ylabel("N*m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=5, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = f"output/computed_torque_{robot_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.show()


def plot_mujoco(log, robot_name):
    """Plot MuJoCo CTC tracking results."""
    t  = log["t"]
    nv = log["q_d"].shape[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"MuJoCo Computed Torque Control -- {robot_name}",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    for j in range(nv):
        c = colors[j % len(colors)]
        ax.plot(t, log["q_d"][:, j], color=c, lw=1.2, alpha=0.5, label=f"j{j+1} desired")
        ax.plot(t, log["q"][:, j],   color=c, lw=2.0, ls="--", label=f"j{j+1} actual")
    ax.set_title("Joint tracking"); ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for j in range(nv):
        ax.plot(t, log["e"][:, j], color=colors[j % len(colors)], lw=1.5, label=f"j{j+1}")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    rms = np.sqrt(np.mean(log["e"]**2))
    ax.set_title(f"Tracking error (RMS={rms:.4e} rad)")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for j in range(nv):
        ax.plot(t, log["tau"][:, j], color=colors[j % len(colors)], lw=1.5, label=f"j{j+1}")
    ax.set_title("Control torques"); ax.set_ylabel("N*m"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = f"output/computed_torque_mujoco_{robot_name}.png"
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.show()
