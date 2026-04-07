"""CLI entry point for computed torque control."""

import argparse
import numpy as np

from maxcomp.core import RobotIdentifier
from maxcomp.identification import run_identification
from maxcomp.control import (
    simulate_pinocchio, simulate_mujoco,
    plot_comparison, plot_mujoco,
)


def main():
    p = argparse.ArgumentParser(
        description="Computed torque control using identified regressor parameters"
    )
    p.add_argument("urdf", nargs="?", default="robots/my-robot/single_finger.xml",
                   help="Path to robot URDF or MJCF")
    p.add_argument("--Kp",    type=float, default=100.0, help="Proportional gain")
    p.add_argument("--Kd",    type=float, default=20.0,  help="Derivative gain")
    p.add_argument("--T",     type=float, default=8.0,   help="Simulation duration [s]")
    p.add_argument("--T_id",  type=float, default=30.0,  help="Identification duration [s]")
    p.add_argument("--gui",   action="store_true",        help="Run MuJoCo viewer")
    p.add_argument("--mujoco", action="store_true",       help="Use MuJoCo plant (no GUI)")
    args = p.parse_args()

    # Identification
    print("=" * 60)
    print(" STEP 1 -- Parameter Identification")
    print("=" * 60)
    ident = RobotIdentifier(args.urdf, verbose=True)
    pi_b_est, pi_b_true, bp = run_identification(ident, T=args.T_id, plot=False)

    model = ident.model
    data  = ident.data

    # Control
    print("\n" + "=" * 60)
    print(" STEP 2 -- Computed Torque Control")
    print("=" * 60)
    print(f"  Kp={args.Kp}, Kd={args.Kd}, T={args.T}s")

    if args.gui or args.mujoco:
        log = simulate_mujoco(
            model, data, bp, pi_b_est,
            Kp=args.Kp, Kd=args.Kd, T=args.T,
            gui=args.gui,
        )
        if len(log["t"]) > 0:
            rms = np.sqrt(np.mean(log["e"]**2))
            print(f"\n  MuJoCo tracking RMS error: {rms:.4e} rad")
            plot_mujoco(log, model.name)
    else:
        log = simulate_pinocchio(
            model, data, bp, pi_b_est, pi_b_true,
            Kp=args.Kp, Kd=args.Kd, T=args.T,
        )

        e_ctc_rms = np.sqrt(np.mean(log["e_ctc"]**2))
        e_pd_rms  = np.sqrt(np.mean(log["e_pd"]**2))
        print(f"\n  CTC  tracking RMS error: {e_ctc_rms:.4e} rad")
        print(f"  PD+G tracking RMS error: {e_pd_rms:.4e} rad")
        print(f"  Improvement factor:      {e_pd_rms / max(e_ctc_rms, 1e-15):.1f}x")

        plot_comparison(log, model.name)

    print("\nDone.")


if __name__ == "__main__":
    main()
