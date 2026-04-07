"""
Optimal excitation trajectory generation.

Optimizes Fourier-series coefficients so that the resulting base regressor
matrix Y_b has minimum condition number, making identified dynamic
parameters robust against measurement noise.
"""

import os
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from dataclasses import dataclass
import time


@dataclass
class FourierParams:
    """Fourier series parameterization for joint trajectories."""
    a: np.ndarray          # (nv, n_harmonics) -- sine coefficients
    b: np.ndarray          # (nv, n_harmonics) -- cosine coefficients
    q0: np.ndarray         # (nv,) -- joint offsets
    omega_f: float         # fundamental frequency [rad/s]
    n_harmonics: int       # number of harmonics L


@dataclass
class OptimizationResult:
    """Result of the excitation trajectory optimization."""
    trajectory: tuple      # (t, q, dq, ddq)
    fourier_params: FourierParams
    cond_initial: float
    cond_final: float
    rc_initial: float
    rc_final: float
    n_iterations: int
    elapsed_s: float


class ExcitationTrajectoryOptimizer:
    """
    Optimize Fourier-series coefficients to minimize the condition number
    of the base regressor Y_b, subject to joint kinematic limits.
    """

    def __init__(
        self,
        identifier,
        bp,
        T: float = 30.0,
        dt: float = 1e-2,
        omega_f: float = 0.5,
        n_harmonics: int = 5,
        n_eval_points: int = 100,
    ):
        self.ident = identifier
        self.bp = bp
        self.model = identifier.model
        self.data = identifier.model.createData()
        self.nv = identifier.nv
        self.T = T
        self.dt = dt
        self.omega_f = omega_f
        self.n_harmonics = n_harmonics
        self.n_eval_points = n_eval_points

        q_min = np.array(self.model.lowerPositionLimit[:self.nv])
        q_max = np.array(self.model.upperPositionLimit[:self.nv])
        dq_max = np.array(self.model.velocityLimit[:self.nv])

        for i in range(self.nv):
            if not np.isfinite(q_min[i]):
                q_min[i] = -np.pi
            if not np.isfinite(q_max[i]):
                q_max[i] = np.pi

        for i in range(self.nv):
            if not np.isfinite(dq_max[i]) or dq_max[i] <= 0:
                dq_max[i] = (q_max[i] - q_min[i]) / 1.0

        self.q_min = q_min
        self.q_max = q_max
        self.dq_max = dq_max
        self.ddq_max = self.dq_max * 2.0

        self.t_eval = np.linspace(0, T, n_eval_points, endpoint=False)
        self.t_full = np.arange(0.0, T, dt)

        self.q_center = 0.5 * (self.q_min + self.q_max)

        self._margin = 0.9
        self.q_min_safe = self.q_center + self._margin * (self.q_min - self.q_center)
        self.q_max_safe = self.q_center + self._margin * (self.q_max - self.q_center)
        self.dq_max_safe = self._margin * self.dq_max
        self.ddq_max_safe = self._margin * self.ddq_max

        self._kw = np.array([(k + 1) * omega_f for k in range(n_harmonics)])

        self._iter = 0
        self._best_rc = np.inf

    def _n_vars(self) -> int:
        return 2 * self.nv * self.n_harmonics

    def _pack(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([a.ravel(), b.ravel()])

    def _unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.nv * self.n_harmonics
        a = x[:n].reshape(self.nv, self.n_harmonics)
        b = x[n:].reshape(self.nv, self.n_harmonics)
        return a, b

    def eval_trajectory(
        self,
        a: np.ndarray,
        b: np.ndarray,
        t_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate Fourier trajectory and its analytic derivatives.
        Returns (q, dq, ddq) each of shape (N, nv).
        """
        N = len(t_arr)
        nv = self.nv
        q = np.tile(self.q_center, (N, 1)).copy()
        dq = np.zeros((N, nv))
        ddq = np.zeros((N, nv))

        for k in range(self.n_harmonics):
            kw = self._kw[k]
            s = np.sin(kw * t_arr)
            c = np.cos(kw * t_arr)
            for j in range(nv):
                q[:, j] += a[j, k] / kw * s - b[j, k] / kw * c
                dq[:, j] += a[j, k] * c + b[j, k] * s
                ddq[:, j] += -a[j, k] * kw * s + b[j, k] * kw * c

        return q, dq, ddq

    def _compute_Yb(self, q, dq, ddq):
        N = len(q)
        bp = self.bp
        nv = self.nv
        Yb = np.zeros((N * nv, bp.r))

        for i in range(N):
            Y_full = pin.computeJointTorqueRegressor(
                self.model, self.data,
                q[i], dq[i], ddq[i],
            )
            Yb[i * nv:(i + 1) * nv, :] = Y_full[:, bp.P[:bp.r]]

        return Yb

    def _rc_criterion(self, Yb: np.ndarray) -> float:
        N = Yb.shape[0] // self.nv
        Hb = Yb.T @ Yb / N
        try:
            Hb_inv = np.linalg.inv(Hb)
        except np.linalg.LinAlgError:
            return np.inf

        fro_Hb = np.linalg.norm(Hb, 'fro')
        fro_Hb_inv = np.linalg.norm(Hb_inv, 'fro')
        return 0.5 * (fro_Hb + fro_Hb_inv)

    def _kinematic_constraints(self, x: np.ndarray) -> np.ndarray:
        a, b = self._unpack(x)
        q, dq, ddq = self.eval_trajectory(a, b, self.t_eval)

        constraints = []
        constraints.append((self.q_max_safe - q).ravel())
        constraints.append((q - self.q_min_safe).ravel())
        constraints.append((self.dq_max_safe - np.abs(dq)).ravel())
        constraints.append((self.ddq_max_safe - np.abs(ddq)).ravel())

        return np.concatenate(constraints)

    def _objective(self, x: np.ndarray) -> float:
        a, b = self._unpack(x)
        q, dq, ddq = self.eval_trajectory(a, b, self.t_eval)
        Yb = self._compute_Yb(q, dq, ddq)

        sv = np.linalg.svd(Yb, compute_uv=False)
        if sv[-1] < 1e-15:
            obj = 1e15
        else:
            obj = np.log(sv[0] / sv[-1])

        self._iter += 1
        if obj < self._best_rc:
            self._best_rc = obj
        if self._iter % 20 == 0:
            cond = np.exp(obj) if obj < 30 else np.inf
            print(f"    iter {self._iter:4d}  log(cond) = {obj:.4f}  "
                  f"cond(Y_b) = {cond:.2e}")

        return obj

    def optimize(
        self,
        amplitude: float = 0.8,
        n_restarts: int = 3,
        max_iter: int = 200,
        seed: int = 42,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Optimize Fourier excitation trajectory with kinematic constraints.
        """
        rng = np.random.default_rng(seed)
        t0 = time.time()

        if verbose:
            print(f"\n{'='*65}")
            print(f"  Optimal Excitation Trajectory Generation")
            print(f"  nv={self.nv}  L={self.n_harmonics}  "
                  f"omega_f={self.omega_f}  T={self.T}s  "
                  f"n_eval={self.n_eval_points}")
            print(f"{'='*65}")

        a_init = rng.uniform(-amplitude, amplitude,
                             (self.nv, self.n_harmonics))
        b_init = rng.uniform(-amplitude, amplitude,
                             (self.nv, self.n_harmonics))
        x_init = self._pack(a_init, b_init)

        q_init, dq_init, ddq_init = self.eval_trajectory(
            a_init, b_init, self.t_eval)
        Yb_init = self._compute_Yb(q_init, dq_init, ddq_init)
        rc_initial = self._rc_criterion(Yb_init)
        cond_initial = np.linalg.cond(Yb_init)

        if verbose:
            print(f"\n  Initial (random):  cond(Y_b) = {cond_initial:.4e}  "
                  f"rc = {rc_initial:.4e}")

        if verbose:
            print(f"\n  Step 1: Optimizing with kinematic constraints "
                  f"({n_restarts} restarts, max {max_iter} iter each)")

        best_result = None
        best_rc = np.inf

        kin_constraint = {
            'type': 'ineq',
            'fun': self._kinematic_constraints,
        }

        for restart in range(n_restarts):
            if restart == 0:
                x0 = x_init.copy()
            else:
                a_r = rng.uniform(-amplitude, amplitude,
                                  (self.nv, self.n_harmonics))
                b_r = rng.uniform(-amplitude, amplitude,
                                  (self.nv, self.n_harmonics))
                x0 = self._pack(a_r, b_r)

            self._iter = 0
            self._best_rc = np.inf

            if verbose:
                print(f"\n  --- Restart {restart + 1}/{n_restarts} ---")

            result = minimize(
                self._objective,
                x0,
                method='SLSQP',
                constraints=kin_constraint,
                options={
                    'maxiter': max_iter,
                    'ftol': 1e-10,
                    'disp': False,
                },
            )

            rc_val = result.fun
            if verbose:
                cond_r = np.exp(rc_val) if rc_val < 30 else np.inf
                print(f"  Result: cond(Y_b) = {cond_r:.4e}  "
                      f"success={result.success}  ({result.nit} iter)")

            if rc_val < best_rc:
                best_rc = rc_val
                best_result = result

        a_opt, b_opt = self._unpack(best_result.x)

        q_full, dq_full, ddq_full = self.eval_trajectory(
            a_opt, b_opt, self.t_full)

        Yb_final = self._compute_Yb(q_full, dq_full, ddq_full)
        rc_final = self._rc_criterion(Yb_final)
        cond_final = np.linalg.cond(Yb_final)

        elapsed = time.time() - t0

        if verbose:
            print(f"\n{'='*65}")
            print(f"  Optimization complete in {elapsed:.1f}s")
            print(f"  cond(Y_b):  {cond_initial:.4e}  ->  {cond_final:.4e}"
                  f"  ({cond_initial/cond_final:.1f}x improvement)")
            print(f"  rc:         {rc_initial:.4e}  ->  {rc_final:.4e}")
            self._print_limit_check(q_full, dq_full, ddq_full)
            print(f"{'='*65}\n")

        fourier_params = FourierParams(
            a=a_opt, b=b_opt, q0=self.q_center,
            omega_f=self.omega_f, n_harmonics=self.n_harmonics,
        )

        return OptimizationResult(
            trajectory=(self.t_full, q_full, dq_full, ddq_full),
            fourier_params=fourier_params,
            cond_initial=cond_initial,
            cond_final=cond_final,
            rc_initial=rc_initial,
            rc_final=rc_final,
            n_iterations=best_result.nit,
            elapsed_s=elapsed,
        )

    def _print_limit_check(self, q, dq, ddq):
        print(f"\n  Joint limit utilization (% of safe limit):")
        print(f"  {'joint':>6}  {'q range':>12}  {'|dq| max':>12}  {'|ddq| max':>12}")
        for j in range(self.nv):
            q_range = q[:, j].max() - q[:, j].min()
            q_limit_range = self.q_max_safe[j] - self.q_min_safe[j]
            dq_pct = np.abs(dq[:, j]).max() / self.dq_max_safe[j] * 100
            ddq_pct = np.abs(ddq[:, j]).max() / self.ddq_max_safe[j] * 100
            q_pct = q_range / q_limit_range * 100
            print(f"  {j+1:>6}  {q_pct:>10.1f}%  {dq_pct:>10.1f}%  {ddq_pct:>10.1f}%")


def plot_comparison(ident, bp, result: OptimizationResult, T=30.0, dt=1e-2):
    """Plot optimized vs random trajectory and their identification results."""
    import matplotlib.pyplot as plt

    t_opt, q_opt, dq_opt, ddq_opt = result.trajectory
    t_rnd, q_rnd, dq_rnd, ddq_rnd = ident.fourier_trajectory(T=T, dt=dt)

    Yb_opt = ident.compute_base_regressor(q_opt, dq_opt, ddq_opt, bp)
    Yb_rnd = ident.compute_base_regressor(q_rnd, dq_rnd, ddq_rnd, bp)

    sv_opt = np.linalg.svd(Yb_opt, compute_uv=False)
    sv_rnd = np.linalg.svd(Yb_rnd, compute_uv=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Optimized vs Random Excitation Trajectory", fontsize=13,
                 fontweight="bold")

    ax = axes[0, 0]
    for j in range(ident.nv):
        ax.plot(t_opt, q_opt[:, j], label=f"q{j+1}")
    ax.set_title(f"Optimized Trajectory  (cond = {result.cond_final:.2e})")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for j in range(ident.nv):
        ax.plot(t_rnd, q_rnd[:, j], label=f"q{j+1}")
    ax.set_title(f"Random Trajectory  (cond = {result.cond_initial:.2e})")
    ax.set_ylabel("rad"); ax.set_xlabel("t [s]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    idx = np.arange(1, len(sv_opt) + 1)
    ax.semilogy(idx, sv_opt, 'o-', label="Optimized", markersize=4)
    ax.semilogy(idx[:len(sv_rnd)], sv_rnd[:len(idx)], 's--',
                label="Random", markersize=4)
    ax.set_title("Singular Values of Y_b")
    ax.set_xlabel("index"); ax.set_ylabel("singular value")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    pi_b_true = ident.true_base_params(bp)

    tau_opt = ident.simulate_torques(q_opt, dq_opt, ddq_opt)
    pi_opt, _, _ = ident.estimate(Yb_opt, tau_opt)

    tau_rnd = ident.simulate_torques(q_rnd, dq_rnd, ddq_rnd)
    pi_rnd, _, _ = ident.estimate(Yb_rnd, tau_rnd)

    err_opt = np.abs(pi_opt - pi_b_true)
    err_rnd = np.abs(pi_rnd - pi_b_true)
    idx_p = np.arange(bp.r)

    ax.bar(idx_p - 0.2, err_opt, width=0.4, label="Optimized", alpha=0.75)
    ax.bar(idx_p + 0.2, err_rnd, width=0.4, label="Random", alpha=0.75)
    ax.set_title("|pi_b_est - pi_b_true|")
    ax.set_xlabel("base param index"); ax.set_ylabel("absolute error")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = f"output/excitation_optimization_{ident.model.name}.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved: {out}")
    plt.show()
