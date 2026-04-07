"""CLI entry point for robot dynamic parameter identification."""

import argparse
from maxcomp.core import RobotIdentifier
from maxcomp.identification import run_identification


def main():
    p = argparse.ArgumentParser(
        description="Plug-and-play robot dynamic identification via Pinocchio."
    )
    p.add_argument(
        "urdf", nargs="?", default="robots/my-robot/single_finger.xml",
        help="Path to robot URDF or MJCF  (default: robots/my-robot/single_finger.xml)",
    )
    p.add_argument("--T",         type=float, default=30.0, help="Trajectory duration [s]")
    p.add_argument("--dt",        type=float, default=1e-2, help="Time step [s]")
    p.add_argument("--omega",     type=float, default=0.5,  help="Fundamental frequency [rad/s]")
    p.add_argument("--harmonics", type=int,   default=5,    help="Fourier harmonics")
    p.add_argument("--amplitude", type=float, default=0.8,  help="Fourier coefficient scale")
    p.add_argument("--noise",     type=float, default=0.0,  help="Torque noise std [N*m]")
    p.add_argument("--no-plot",   action="store_true",       help="Skip plots")
    p.add_argument("--optimize-traj", action="store_true",   help="Optimize excitation trajectory")
    p.add_argument("--restarts",  type=int, default=3,       help="Multi-start restarts")
    p.add_argument("--max-iter",  type=int, default=200,     help="Max iterations per restart")
    args = p.parse_args()

    ident = RobotIdentifier(args.urdf)

    run_identification(
        ident,
        T              = args.T,
        dt             = args.dt,
        omega_f        = args.omega,
        n_harmonics    = args.harmonics,
        amplitude      = args.amplitude,
        noise_std      = args.noise,
        plot           = not args.no_plot,
        optimize_traj  = args.optimize_traj,
        n_restarts     = args.restarts,
        max_iter       = args.max_iter,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
