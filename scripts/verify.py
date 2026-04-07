"""CLI entry point for dynamic term verification."""

import argparse
from maxcomp.verification import verify


def main():
    p = argparse.ArgumentParser(description="Verify identified dynamic terms against Pinocchio ground truth.")
    p.add_argument("urdf", nargs="?", default="robots/my-robot/single_finger.xml")
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


if __name__ == "__main__":
    main()
