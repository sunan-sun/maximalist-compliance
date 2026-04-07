"""CLI entry point for momentum observer simulation."""

import argparse
import numpy as np
import pinocchio as pin

from maxcomp.core import RobotIdentifier
from maxcomp.identification import run_identification
from maxcomp.observer import simulate, plot_observer


def main():
    p = argparse.ArgumentParser(description="Data-driven momentum observer simulation.")
    p.add_argument("urdf",       nargs="?", default="robots/my-robot/single_finger.xml")
    p.add_argument("--T_id",     type=float, default=30.0,  help="Identification duration [s]")
    p.add_argument("--T_sim",    type=float, default=6.0,   help="Simulation duration [s]")
    p.add_argument("--dt",       type=float, default=1e-3,  help="Simulation timestep [s]")
    p.add_argument("--K_O",      type=float, default=10.0,  help="Observer gain")
    p.add_argument("--Kp",       type=float, default=50.0,  help="PD proportional gain")
    p.add_argument("--Kd",       type=float, default=8.0,   help="PD derivative gain")
    p.add_argument("--noise",    type=float, default=0.0,   help="Identification torque noise [N*m]")
    p.add_argument("--tau_max",  type=float, default=0.0,   help="Torque saturation limit [N*m] (0=auto)")
    p.add_argument("--tau_ext",  type=float, default=0.0,   help="External disturbance magnitude [N*m] (0=auto)")
    args = p.parse_args()

    # Offline: identification
    print("=" * 60)
    print(" OFFLINE -- Dynamic Parameter Identification")
    print("=" * 60)
    ident = RobotIdentifier(args.urdf, verbose=True)
    pi_b_est, pi_b_true, bp = run_identification(
        ident, T=args.T_id, noise_std=args.noise, plot=False,
    )

    model = ident.model
    data  = ident.data
    nv    = ident.nv

    # Online: simulation + observer
    print("\n" + "=" * 60)
    print(" ONLINE -- Momentum Observer Simulation")
    print("=" * 60)
    print(f"  K_O={args.K_O},  Kp={args.Kp},  Kd={args.Kd},  dt={args.dt}")
    print(f"  External torque on joint 1:  auto-scaled from t=2s to t=4s")

    q_ref = pin.neutral(model)[:nv]

    log = simulate(
        model, data, bp, pi_b_est,
        K_O     = args.K_O,
        dt      = args.dt,
        T       = args.T_sim,
        Kp      = args.Kp,
        Kd      = args.Kd,
        q_ref   = q_ref,
        tau_max = args.tau_max if args.tau_max > 0 else None,
        tau_ext_mag = args.tau_ext if args.tau_ext > 0 else None,
    )

    # Summary
    print("\n  Observer residual vs true external torque (contact window 2-4 s):")
    contact = (log["t"] >= 2.0) & (log["t"] < 4.0)
    for label, r_key in [("Full (with CT*dq)", "r_full"),
                          ("Approx (no Coriolis)", "r_approx")]:
        err = log[r_key][contact] - log["tau_ext"][contact]
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"    {label:<25}  RMSE = {rmse:.4e} N*m")

    plot_observer(log, model.name)
    print("\nDone.")


if __name__ == "__main__":
    main()
