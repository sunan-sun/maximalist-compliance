"""CLI entry point for excitation trajectory optimization."""

import argparse
import numpy as np
from maxcomp.core import RobotIdentifier
from maxcomp.excitation import ExcitationTrajectoryOptimizer, plot_comparison
from maxcomp.identification import print_results


def main():
    p = argparse.ArgumentParser(
        description="Optimize excitation trajectory for dynamic identification."
    )
    p.add_argument("urdf", nargs="?", default="robots/sixdof_arm.urdf",
                   help="Path to robot URDF or MJCF")
    p.add_argument("--T", type=float, default=30.0, help="Trajectory duration [s]")
    p.add_argument("--dt", type=float, default=1e-2, help="Time step [s]")
    p.add_argument("--omega", type=float, default=0.5, help="Fundamental frequency [rad/s]")
    p.add_argument("--harmonics", type=int, default=5, help="Number of Fourier harmonics")
    p.add_argument("--amplitude", type=float, default=0.8, help="Initial coefficient amplitude")
    p.add_argument("--restarts", type=int, default=3, help="Multi-start restarts")
    p.add_argument("--max-iter", type=int, default=200, help="Max iterations per restart")
    p.add_argument("--n-eval", type=int, default=100, help="Time samples during optimization")
    p.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = p.parse_args()

    ident = RobotIdentifier(args.urdf)
    bp = ident.find_base_params()

    opt = ExcitationTrajectoryOptimizer(
        ident, bp,
        T=args.T, dt=args.dt,
        omega_f=args.omega,
        n_harmonics=args.harmonics,
        n_eval_points=args.n_eval,
    )

    result = opt.optimize(
        amplitude=args.amplitude,
        n_restarts=args.restarts,
        max_iter=args.max_iter,
    )

    if not args.no_plot:
        plot_comparison(ident, bp, result, T=args.T, dt=args.dt)

    # Run identification with the optimized trajectory
    print("\n" + "="*65)
    print("  Running identification with optimized trajectory...")
    print("="*65)
    t, q, dq, ddq = result.trajectory
    Yb = ident.compute_base_regressor(q, dq, ddq, bp)
    tau = ident.simulate_torques(q, dq, ddq)
    pi_est, cond, rank = ident.estimate(Yb, tau)
    pi_true = ident.true_base_params(bp)

    tau_rec = Yb @ pi_est
    rmse = np.sqrt(np.mean((tau_rec - tau) ** 2))
    print(f"  cond(Y_b) : {cond:.4e}")
    print(f"  rank(Y_b) : {rank} / {bp.r}")
    print(f"  RMSE      : {rmse:.4e} N*m")
    print_results(pi_true, pi_est, bp)

    if not args.no_plot:
        from maxcomp.identification import plot_identification
        plot_identification(t, q, tau, tau_rec, pi_true, pi_est, bp,
                            ident.model.name, ident.nv, ident.n_std)

    print("\nDone.")


if __name__ == "__main__":
    main()
