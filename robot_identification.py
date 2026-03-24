"""
Plug-and-Play Robot Dynamic Parameter Identification
=====================================================
Drop in any URDF → get π̂_b (base inertial parameters) such that

    τ = Y_b(q, q̇, q̈) · π_b

where Y_b has full column rank and π_b are the *identifiable* combinations
of the standard 10-per-body inertial parameters.

Why not τ = Y · π_standard directly?
--------------------------------------
The full regressor Y ∈ ℝ^{N·nv × 10n} is rank-deficient.  Some parameters
never appear in the equations of motion (e.g. inertia about a revolute axis),
and others always appear in fixed combinations across links.  Least-squares on
the full Y gives non-unique, physically meaningless θ̂.

The base parameter vector π_b ∈ ℝ^r  (r = rank Y ≪ 10n) contains the actual
identifiable combinations.  It is found via rank-revealing QR applied to Y
evaluated at many random configurations.

Pipeline
--------
1. Load robot from URDF via Pinocchio
2. Find base parameter structure (rank-revealing QR on random regressors)
3. Generate Fourier-series excitation trajectory
4. Compute stacked base regressor  Y_b ∈ ℝ^{N·nv × r}
5. Simulate joint torques via RNEA  (or supply real measurements)
6. Solve  π̂_b = argmin ‖Y_b · π_b − τ‖²

Usage
-----
    python robot_identification.py robots/planar_2dof.urdf
    python robot_identification.py robots/pendulum.urdf --T 60 --noise 0.05
"""

import argparse
import numpy as np
import pinocchio as pin
from scipy.linalg import qr
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Base-parameter structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseParams:
    """
    Encodes the mapping from 10n standard parameters → r base parameters.

    Given the full regressor Y ∈ ℝ^{N·nv × 10n} and any configuration (q,v,a):

        Y_b = Y[:, P[:r]] + Y[:, P[r:]] @ K.T       (r columns, full rank)

    Any π_standard satisfies:

        Y · π_standard = Y_b · π_b

    where

        π_b = π_standard[P[:r]] + K @ π_standard[P[r:]]

    Fields
    ------
    P    : column permutation index array  (length 10n)
    K    : regrouping matrix               (r × (10n - r))
    r    : number of base parameters  (= rank Y)
    n_std: total standard parameters  (= 10 * n_bodies)
    """
    P:     np.ndarray   # permutation
    K:     np.ndarray   # regrouping  (r × (n_std - r))
    r:     int          # rank
    n_std: int          # 10 * n_bodies


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class RobotIdentifier:
    """
    Plug-and-play robot dynamic identification.

    Parameters
    ----------
    urdf_path : str | Path
    verbose   : bool
    """

    # Names of the 10 standard parameters per body
    _PARAM_NAMES = ["m",
                    "mc_x", "mc_y", "mc_z",
                    "Σ_xx", "Σ_xy", "Σ_xz", "Σ_yy", "Σ_yz", "Σ_zz"]

    def __init__(self, urdf_path: str | Path, verbose: bool = True):
        self.urdf_path = str(urdf_path)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data  = self.model.createData()

        self.nq       = self.model.nq
        self.nv       = self.model.nv
        self.n_bodies = self.model.nbodies - 1       # universe excluded
        self.n_std    = 10 * self.n_bodies            # standard parameter count

        if verbose:
            print(f"\nRobot  : {self.model.name}")
            print(f"  nv              : {self.nv}  (DOF)")
            print(f"  bodies          : {self.n_bodies}")
            print(f"  standard params : {self.n_std}  ({self.n_bodies} × 10)")

    # ── Step 1 : find base parameter structure ───────────────────────────────

    def find_base_params(
        self,
        n_samples: int   = 1000,
        tol:       float = None,
    ) -> BaseParams:
        """
        Identify the base (identifiable) parameter structure.

        Evaluates the regressor at `n_samples` random configurations and
        applies rank-revealing QR (column-pivoting) to discover:
          - which standard parameters are linearly independent in the dynamics
          - how the dependent ones can be "regrouped" (absorbed) into the base

        Returns a `BaseParams` object used by every downstream method.
        """
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(n_samples):
            q   = pin.randomConfiguration(self.model)
            dq  = rng.standard_normal(self.nv)
            ddq = rng.standard_normal(self.nv)
            Y_i = pin.computeJointTorqueRegressor(
                self.model, self.data, q, dq, ddq
            )
            rows.append(Y_i)

        Y_rand = np.vstack(rows)          # (n_samples·nv, n_std)

        # Rank-revealing QR:  Y_rand[:, P] = Q @ R
        _, R, P = qr(Y_rand, pivoting=True)

        diag_R = np.abs(np.diag(R))
        if tol is None:
            tol = diag_R[0] * max(Y_rand.shape) * np.finfo(float).eps * 100

        r = int(np.sum(diag_R > tol))

        # Regrouping matrix:  dependent columns expressed in terms of base columns
        # R = [R11  R12]   →   K = R11⁻¹ R12    (r × (n_std - r))
        #     [ 0   R22]
        R11 = R[:r, :r]
        R12 = R[:r, r:]
        K   = np.linalg.solve(R11, R12)   # (r, n_std - r)

        bp = BaseParams(P=P, K=K, r=r, n_std=self.n_std)

        print(f"  standard params : {self.n_std}")
        print(f"  base params     : {r}  (identifiable)")
        print(f"  singular values (log10): {np.log10(diag_R[:r+2].clip(1e-15))}")
        self._print_base_structure(bp)

        return bp

    # ── Step 2 : trajectory generation ──────────────────────────────────────

    def fourier_trajectory(
        self,
        T:           float = 30.0,
        dt:          float = 1e-2,
        omega_f:     float = 0.5,
        n_harmonics: int   = 5,
        amplitude:   float = 0.8,
        q_center            = None,
        seed:        int   = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Band-limited Fourier-series excitation trajectory with analytic derivatives.

        For joint j, harmonic k:
            q_j(t)   = q0_j + Σ_k [ a_jk/(kω) sin(kωt) − b_jk/(kω) cos(kωt) ]
            q̇_j(t)  = Σ_k [ a_jk cos(kωt) + b_jk sin(kωt) ]
            q̈_j(t)  = Σ_k [ −a_jk kω sin(kωt) + b_jk kω cos(kωt) ]
        """
        rng = np.random.default_rng(seed)
        n   = self.nv

        a = rng.uniform(-amplitude, amplitude, (n, n_harmonics))
        b = rng.uniform(-amplitude, amplitude, (n, n_harmonics))

        if q_center is None:
            q_center = np.zeros(n)
        else:
            q_center = np.asarray(q_center, float)[:n]

        t_arr    = np.arange(0.0, T, dt)
        N        = len(t_arr)

        q_traj   = np.tile(q_center, (N, 1)).copy()
        dq_traj  = np.zeros((N, n))
        ddq_traj = np.zeros((N, n))

        for j in range(n):
            for k in range(1, n_harmonics + 1):
                kw  = k * omega_f
                sk  = np.sin(kw * t_arr)
                ck  = np.cos(kw * t_arr)
                q_traj[:, j]   +=  a[j, k-1] / kw * sk - b[j, k-1] / kw * ck
                dq_traj[:, j]  +=  a[j, k-1]      * ck + b[j, k-1]      * sk
                ddq_traj[:, j] += -a[j, k-1] * kw  * sk + b[j, k-1] * kw  * ck

        return t_arr, q_traj, dq_traj, ddq_traj

    # ── Step 3 : build stacked base regressor ────────────────────────────────

    def compute_base_regressor(
        self,
        q_traj:   np.ndarray,
        dq_traj:  np.ndarray,
        ddq_traj: np.ndarray,
        bp:       BaseParams,
    ) -> np.ndarray:
        """
        Build stacked base regressor  Y_b ∈ ℝ^{N·nv × r}.

        At each time step i:
            Y_full_i  ∈ ℝ^{nv × n_std}   (from pinocchio)
            Y_b_i     = Y_full_i[:, P[:r]]  +  Y_full_i[:, P[r:]] @ K.T
                      ∈ ℝ^{nv × r}

        The projection absorbs the dependent columns into the base ones so that
        Y_b has full column rank by construction.
        """
        N = len(q_traj)
        Y_b = np.zeros((N * self.nv, bp.r))

        for i in range(N):
            Y_full_i = pin.computeJointTorqueRegressor(
                self.model, self.data,
                q_traj[i], dq_traj[i], ddq_traj[i],
            )
            # Y_b = Y_full[:, P[:r]]  — select the r base columns only.
            # The dependent columns (P[r:]) are already encoded in π_b via
            #   π_b = π_std[P[:r]]  +  K @ π_std[P[r:]]
            # Adding them back here would double-count their contribution.
            Y_b[i * self.nv : (i + 1) * self.nv, :] = Y_full_i[:, bp.P[:bp.r]]

        return Y_b

    # ── Step 4 : torque simulation ───────────────────────────────────────────

    def simulate_torques(
        self,
        q_traj:   np.ndarray,
        dq_traj:  np.ndarray,
        ddq_traj: np.ndarray,
        noise_std: float = 0.0,
    ) -> np.ndarray:
        """
        Compute inverse-dynamics torques via RNEA.
        Replace this with real torque measurements on a physical robot.
        Returns τ ∈ ℝ^{N·nv}.
        """
        N   = len(q_traj)
        tau = np.zeros(N * self.nv)

        for i in range(N):
            tau_i = pin.rnea(
                self.model, self.data,
                q_traj[i], dq_traj[i], ddq_traj[i],
            )
            tau[i * self.nv : (i + 1) * self.nv] = tau_i

        if noise_std > 0.0:
            tau += np.random.default_rng(1).normal(0.0, noise_std, tau.shape)

        return tau

    # ── Step 5 : estimate ────────────────────────────────────────────────────

    def estimate(
        self,
        Y_b:  np.ndarray,
        tau:  np.ndarray,
    ) -> tuple[np.ndarray, float, int]:
        """
        π̂_b = argmin ‖Y_b · π_b − τ‖²   (ordinary least squares)

        Returns (pi_b_est, condition_number, rank)
        """
        cond = np.linalg.cond(Y_b)
        pi_b_est, _, rank, _ = np.linalg.lstsq(Y_b, tau, rcond=None)
        return pi_b_est, cond, rank

    # ── True base parameters from pinocchio model ────────────────────────────

    def true_standard_params(self) -> np.ndarray:
        """
        Extract the full 10n standard parameter vector from the Pinocchio model.

        For body i:
            [m,  mc_x, mc_y, mc_z,  Σ_xx, Σ_xy, Σ_xz, Σ_yy, Σ_yz, Σ_zz]

        where
            mc = m · c               (first moment, c = COM in body frame)
            Σ  = I_c + m(‖c‖²I₃ − ccᵀ)  (inertia about joint-frame origin)
        """
        pi = []
        for i in range(1, self.model.nbodies):
            inertia = self.model.inertias[i]
            m   = inertia.mass
            c   = inertia.lever
            I_c = inertia.inertia

            mc    = m * c
            Sigma = I_c + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))

            pi.extend([
                m,
                mc[0], mc[1], mc[2],
                Sigma[0, 0], Sigma[0, 1], Sigma[0, 2],
                Sigma[1, 1], Sigma[1, 2], Sigma[2, 2],
            ])
        return np.array(pi)

    def true_base_params(self, bp: BaseParams) -> np.ndarray:
        """
        Project the true standard parameter vector onto the base parameter space.

        π_b_true = π_std[P[:r]]  +  K @ π_std[P[r:]]

        These are the actual identifiable combinations that appear in the dynamics.
        """
        pi_std = self.true_standard_params()
        pi_p   = pi_std[bp.P]
        return pi_p[:bp.r] + bp.K @ pi_p[bp.r:]

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        T:            float = 30.0,
        dt:           float = 1e-2,
        omega_f:      float = 0.5,
        n_harmonics:  int   = 5,
        amplitude:    float = 0.8,
        noise_std:    float = 0.0,
        n_qr_samples: int   = 1000,
        plot:         bool  = True,
    ) -> tuple[np.ndarray, np.ndarray, BaseParams]:
        """
        Full identification pipeline.

        Returns (pi_b_est, pi_b_true, base_params)
        """
        print("\n[1/5] Finding base parameter structure ...")
        bp = self.find_base_params(n_samples=n_qr_samples)

        print(f"\n[2/5] Generating excitation trajectory  (T={T}s, dt={dt}s) ...")
        t, q, dq, ddq = self.fourier_trajectory(
            T=T, dt=dt, omega_f=omega_f,
            n_harmonics=n_harmonics, amplitude=amplitude,
        )

        print("[3/5] Computing stacked base regressor  Y_b ...")
        Y_b = self.compute_base_regressor(q, dq, ddq, bp)
        cond = np.linalg.cond(Y_b)
        print(f"      Y_b shape : {Y_b.shape}")
        print(f"      cond(Y_b) : {cond:.3e}", end="")
        if cond > 1e8:
            print("  ⚠  high condition — try longer T or more harmonics", end="")
        print()

        print("[4/5] Simulating torques via RNEA ...")
        tau = self.simulate_torques(q, dq, ddq, noise_std=noise_std)

        print("[5/5] Estimating π_b via least squares ...")
        pi_b_est, _, rank = self.estimate(Y_b, tau)
        tau_rec = Y_b @ pi_b_est
        rmse    = np.sqrt(np.mean((tau_rec - tau) ** 2))
        print(f"      rank(Y_b) : {rank} / {bp.r}")
        print(f"      RMSE      : {rmse:.4e} N·m")

        pi_b_true = self.true_base_params(bp)

        # Sanity check: Y_b @ pi_b_true should reproduce RNEA torques
        if noise_std == 0.0:
            tau_check = Y_b @ pi_b_true
            val_rmse  = np.sqrt(np.mean((tau_check - tau) ** 2))
            print(f"      Validation (Y_b·π_b_true vs RNEA) RMSE : {val_rmse:.2e}  (→ 0)")

        # ── Held-out validation trajectory (different seed) ──────────────────
        print("\n      Validation on held-out trajectory (seed=99) ...")
        t_val, q_val, dq_val, ddq_val = self.fourier_trajectory(
            T=T, dt=dt, omega_f=omega_f,
            n_harmonics=n_harmonics, amplitude=amplitude, seed=99,
        )
        Y_b_val  = self.compute_base_regressor(q_val, dq_val, ddq_val, bp)
        tau_val  = self.simulate_torques(q_val, dq_val, ddq_val, noise_std=noise_std)
        tau_pred = Y_b_val @ pi_b_est
        val_rmse = np.sqrt(np.mean((tau_pred - tau_val) ** 2))
        tau_norm = np.sqrt(np.mean(tau_val ** 2))
        print(f"      Validation RMSE          : {val_rmse:.4e} N·m")
        print(f"      Normalised (RMSE/‖τ‖_rms) : {val_rmse/tau_norm:.4e}  (→ 0 if good)")

        self._print_results(pi_b_true, pi_b_est, bp)

        if plot:
            self._plot(t, q, tau, tau_rec, pi_b_true, pi_b_est, bp)

        return pi_b_est, pi_b_true, bp

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _param_label(self, standard_idx: int) -> str:
        """Human-readable label for a standard parameter index."""
        body  = standard_idx // 10 + 1
        param = self._PARAM_NAMES[standard_idx % 10]
        return f"b{body}:{param}"

    def _print_base_structure(self, bp: BaseParams):
        """
        For each base parameter, show which standard parameters it combines.
        """
        print(f"\n  Base parameter breakdown  (r = {bp.r}):")
        print(f"  {'π_b idx':>7}  {'expression (top contributions)':}")
        print("  " + "-" * 60)
        for bi in range(bp.r):
            # Primary term
            primary_std_idx = bp.P[bi]
            label = f"{self._param_label(primary_std_idx)}"
            # Absorbed dependent terms (only show significant ones)
            absorbed = []
            for di in range(bp.n_std - bp.r):
                coeff = bp.K[bi, di]
                if abs(coeff) > 1e-6:
                    dep_std_idx = bp.P[bp.r + di]
                    absorbed.append(f"{coeff:+.4f}·{self._param_label(dep_std_idx)}")
            if absorbed:
                label += "  +  " + "  +  ".join(absorbed[:3])
                if len(absorbed) > 3:
                    label += f"  (+{len(absorbed)-3} more)"
            print(f"  {bi:>7}  {label}")

    def _print_results(self, pi_b_true, pi_b_est, bp: BaseParams):
        abs_err = np.abs(pi_b_est - pi_b_true)
        scale   = np.abs(pi_b_true)
        # Relative error is only meaningful when the true value is significant
        # relative to the largest parameter in the vector.
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
                note = "⚠ large"
            print(f"  {i:>4}  {t_val:>14.6f}  {e_val:>14.6f}  {abs_e:>10.2e}  {rel_e:>10.2e}  {note}")

    def _plot(self, t, q, tau, tau_rec, pi_b_true, pi_b_est, bp: BaseParams):
        N, nv = len(t), self.nv

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(
            f"Base-Parameter Identification — {self.model.name}\n"
            f"r = {bp.r} base params  (from {self.n_std} standard)",
            fontsize=12, fontweight="bold",
        )

        # Joint trajectories
        ax = axes[0, 0]
        for j in range(nv):
            ax.plot(t, q[:, j], label=f"q{j+1}")
        ax.set_title("Excitation Trajectory  q(t)")
        ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Torques: RNEA vs reconstructed
        ax = axes[0, 1]
        tau_arr     = tau.reshape(N, nv)
        tau_rec_arr = tau_rec.reshape(N, nv)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for j in range(nv):
            ax.plot(t, tau_arr[:, j],     color=colors[j],
                    label=f"τ{j+1} RNEA", alpha=0.8)
            ax.plot(t, tau_rec_arr[:, j], color=colors[j],
                    ls="--", lw=1.5, label=f"τ{j+1} Y_b·π̂_b", alpha=0.9)
        ax.set_title("Torques: RNEA  vs  Y_b·π̂_b")
        ax.set_ylabel("N·m"); ax.set_xlabel("t [s]")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

        # Base parameter comparison
        ax = axes[1, 0]
        idx = np.arange(bp.r)
        ax.bar(idx - 0.2, pi_b_true, width=0.4, label="π_b  true", alpha=0.75)
        ax.bar(idx + 0.2, pi_b_est,  width=0.4, label="π̂_b  est",  alpha=0.75)
        ax.set_title(f"Base Parameters  π_b  (r = {bp.r})")
        ax.set_xlabel("base param index"); ax.set_ylabel("value")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Absolute error per base parameter
        # (relative error is misleading for near-zero parameters)
        ax = axes[1, 1]
        abs_err  = np.abs(pi_b_est - pi_b_true)
        scale    = np.abs(pi_b_true)
        threshold = scale.max() * 1e-3          # "near-zero" threshold
        colors_bar = [
            "steelblue" if scale[i] >= threshold else "lightgray"
            for i in idx
        ]
        ax.bar(idx, abs_err, color=colors_bar, alpha=0.85)
        ax.set_title("Absolute Error  |π̂_b − π_b|\n(gray = near-zero true value, rel err unreliable)")
        ax.set_xlabel("base param index"); ax.set_ylabel("absolute error")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = f"identification_{self.model.name}.png"
        plt.savefig(out, dpi=150)
        print(f"\n  Plot saved: {out}")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plug-and-play robot dynamic identification via Pinocchio."
    )
    p.add_argument(
        "urdf", nargs="?", default="robots/sixdof_arm.urdf",
        help="Path to robot URDF  (default: robots/sixdof_arm.urdf)",
    )
    p.add_argument("--T",         type=float, default=30.0,
                   help="Trajectory duration [s]")
    p.add_argument("--dt",        type=float, default=1e-2,
                   help="Time step [s]")
    p.add_argument("--omega",     type=float, default=0.5,
                   help="Fundamental frequency [rad/s]")
    p.add_argument("--harmonics", type=int,   default=5,
                   help="Fourier harmonics")
    p.add_argument("--amplitude", type=float, default=0.8,
                   help="Fourier coefficient scale")
    p.add_argument("--noise",     type=float, default=0.0,
                   help="Torque noise std [N·m]")
    p.add_argument("--no-plot",   action="store_true",
                   help="Skip plots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ident = RobotIdentifier(args.urdf)

    pi_b_est, pi_b_true, bp = ident.run(
        T           = args.T,
        dt          = args.dt,
        omega_f     = args.omega,
        n_harmonics = args.harmonics,
        amplitude   = args.amplitude,
        noise_std   = args.noise,
        plot        = not args.no_plot,
    )

    print("\nDone.")
